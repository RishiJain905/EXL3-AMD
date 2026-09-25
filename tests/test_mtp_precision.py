import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from quantlab.methods.exl3 import mtp_precision as m

try:
    import torch
except ImportError:
    torch = None

@unittest.skipIf(torch is None, 'CPU PyTorch unavailable')
class BF16ProjectionTests(unittest.TestCase):
    def inner(self):
        # The small nonzero value is lost in FP16, but must remain in BF16.
        return SimpleNamespace(weight=torch.tensor([[2**-30, 2], [0, -1]], dtype=torch.bfloat16),
                               in_features=2, out_features=2, bias=None,
                               out_dtype=torch.float32, bf16_calls=0)

    def test_preserves_small_bf16_weight_and_fp32_interface(self):
        inner = self.inner()
        x = torch.tensor([[[1, 0]]], dtype=torch.float16)
        before = x.clone()
        y = m.bf16_projection_forward(inner, x, {})
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(y.shape, (1, 1, 2))
        self.assertEqual(y[0, 0, 0].item(), 2**-30)
        self.assertEqual(inner.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(x, before))
        self.assertEqual(inner.bf16_calls, 1)

    def test_explicit_output_dtype_and_noncontiguous_batch(self):
        inner = self.inner()
        x = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2).transpose(0, 1)
        y = m.bf16_projection_forward(inner, x, {}, torch.float16)
        expected = (x.to(torch.bfloat16) @ inner.weight).to(torch.float16)
        self.assertEqual(y.dtype, torch.float16)
        self.assertTrue(torch.equal(y, expected))

    def test_default_half_interface_and_bias(self):
        inner = self.inner()
        inner.out_dtype = None
        inner.bias = torch.tensor([1, -2], dtype=torch.bfloat16)
        x = torch.tensor([[1, 1]], dtype=torch.float32)
        self.assertTrue(torch.equal(m.bf16_projection_forward(inner, x, {}),
                                   torch.tensor([[1, -1]], dtype=torch.float16)))


@unittest.skipIf(torch is None, 'CPU PyTorch unavailable')
class BF16InstallTests(unittest.TestCase):
    def setUp(self):
        class LinearFP16: pass
        class Linear: pass
        class RMSNorm: pass
        class Qwen3_5MTPModel:
            def __iter__(self): return iter(self.modules)
        self.draft = Qwen3_5MTPModel()
        self.source = torch.tensor([[2**-30,1],[2,3]],dtype=torch.bfloat16)
        self.meta = {}
        self.loads = []
        self.draft.modules = []
        for i in range(8):
            layer = Linear()
            layer.key = f'mtp.projection{i}'
            layer.inner = LinearFP16()
            layer.inner.weight = self.source.half().clone()
            layer.inner.bias = None
            layer.inner.bc = object()
            layer.inner.out_dtype = torch.float32
            layer.inner.in_features = layer.inner.out_features = 2
            layer.device = 'cpu'
            layer.in_features = layer.out_features = 2
            layer.is_sliced = layer.used_alt_key = False
            layer.transposed_load = True
            layer.weight_scale = 1.0
            self.meta[layer.key+'.weight'] = dict(dtype='torch.bfloat16',shape=[2,2],n_bytes=8)
            self.draft.modules.append(layer)
        for i in range(7):
            norm = RMSNorm()
            norm.weight = torch.ones(2,dtype=torch.bfloat16)
            self.draft.modules.append(norm)
        def get_tensor(key,device,**kwargs):
            self.loads.append((key,device,kwargs))
            return self.source.T.contiguous()
        stc = SimpleNamespace(list_tensors=lambda *a,**kw:self.meta,get_tensor=get_tensor)
        self.draft.config = SimpleNamespace(mtp_num_hidden_layers=1,stc=stc)
        definitions = {
            'exllamav3.architecture.qwen3_5_mtp': dict(Qwen3_5MTPModel=Qwen3_5MTPModel),
            'exllamav3.modules': dict(Linear=Linear,RMSNorm=RMSNorm),
            'exllamav3.modules.quant.fp16': dict(LinearFP16=LinearFP16),
        }
        fake = {}
        for name,attrs in definitions.items():
            fake[name] = ModuleType(name)
            fake[name].__dict__.update(attrs)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules,fake).start()
        self.prepare = patch('quantlab.methods.exl3.compat.prepare_loaded_module').start()

    def test_reloads_original_values_without_fp16_roundtrip(self):
        report = m.preserve_mtp_bf16(self.draft)
        self.assertEqual(report['projection_bytes'],64)
        self.assertEqual(len(self.loads),8)
        self.assertTrue(all(kwargs==dict(allow_bf16=True,transpose=True,no_defer=True)
                            for _,_,kwargs in self.loads))
        for layer in self.draft.modules[:8]:
            self.assertTrue(torch.equal(layer.inner.weight,self.source.T))
            self.assertIsNone(layer.inner.bc)
            self.assertEqual(layer.quant_type,'bf16')
        with self.assertRaisesRegex(ValueError,'already installed'):
            m.preserve_mtp_bf16(self.draft)

    def test_wrong_storage_rejected_before_any_reload(self):
        self.meta['mtp.projection7.weight']['dtype']='torch.float16'
        with self.assertRaisesRegex(ValueError,'ordinary unsliced BF16'):
            m.preserve_mtp_bf16(self.draft)
        self.assertEqual(self.loads,[])
        self.prepare.assert_not_called()

    def test_packed_or_wrong_shape_rejected(self):
        self.draft.modules[0].inner = object()
        with self.assertRaisesRegex(ValueError,'ordinary unsliced BF16'):
            m.preserve_mtp_bf16(self.draft)
        self.assertEqual(self.loads,[])

    def test_nonfinite_source_fails_before_inference(self):
        self.source[0,0] = float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            m.preserve_mtp_bf16(self.draft)

    def test_norm_dtype_and_budget_fail_closed(self):
        self.draft.modules[-1].weight = self.draft.modules[-1].weight.half()
        with self.assertRaisesRegex(ValueError,'norm weights'):
            m.preserve_mtp_bf16(self.draft)
        self.draft.modules[-1].weight = self.draft.modules[-1].weight.bfloat16()
        self.meta['mtp.projection0.weight']['n_bytes'] = 513*1024**2
        with self.assertRaisesRegex(ValueError,'512 MiB'):
            m.preserve_mtp_bf16(self.draft)


if __name__ == '__main__':
    unittest.main()
