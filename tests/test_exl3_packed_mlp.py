"""CPU dispatch contracts; numerical equivalence lives in check_mlp_pair.py."""
import math
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from quantlab.methods.exl3.packed_mlp import eligible_pair, install_mlp_pair


class Tensor:
    def __init__(self, shape, dtype='half', device='cuda', contiguous=True, pointer=128):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim, self.is_cuda = len(shape), device == 'cuda'
        self.contiguous, self.pointer = contiguous, pointer

    def numel(self): return math.prod(self.shape)
    def is_contiguous(self): return self.contiguous
    def data_ptr(self): return self.pointer
    def view(self, *shape): return Tensor(shape, self.dtype, self.device)


class PackedMlpTests(unittest.TestCase):
    def setUp(self):
        self.Linear = type('Linear', (SimpleNamespace,), {})
        self.Packed = type('LinearEXL3', (SimpleNamespace,), {})
        class GatedMLP(SimpleNamespace):
            def forward(self, x, params, out_dtype=None):
                return ('fallback', x, params, out_dtype)
        self.GatedMLP = GatedMLP
        def projection():
            return self.Linear(inner=self.Packed(K=5, mul1=True, mcg=False, bias=None,
                default_out_dtype='half', in_features=128, out_features=256,
                trellis=Tensor((8, 16, 80)), suh=Tensor((128,)), svh=Tensor((256,))),
                lora_a_tensors={}, pre_scale=1., post_scale=1., softcap=0.,
                trim_padded_out=False, out_features=256, out_features_unpadded=256)
        self.module = GatedMLP(key='mlp', num_slices=1, tp_reduce=False, activation_fn='silu',
            act_limit=0., interm_dtype='half', out_dtype='float', gates=[projection()],
            ups=[projection()], downs=[SimpleNamespace(forward=Mock(return_value='down'))])
        self.native = Mock()
        specs = {
            'torch': dict(half='half', empty=lambda shape, **kw: Tensor(shape, **kw)),
            'exllamav3.ext': dict(exllamav3_ext=SimpleNamespace(exl3_mlp_gate_up=self.native)),
            'exllamav3.modules': dict(Linear=self.Linear),
            'exllamav3.modules.mlp': dict(GatedMLP=GatedMLP),
            'exllamav3.modules.quant.exl3': dict(LinearEXL3=self.Packed),
            'exllamav3.util.tensor': dict(to2=lambda result, override, default: (result, override or default)),
        }
        modules = {}
        for name, attrs in specs.items():
            modules[name] = ModuleType(name)
            modules[name].__dict__.update(attrs)
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        env = patch.dict(os.environ, EXL3_SMALLM_WMMA='0', EXL3_GEMV_LDS='0')
        env.start()
        self.addCleanup(env.stop)

    def eligible(self):
        return eligible_pair(self.module, self.Linear, self.Packed, 'half')

    def test_incompatible_semantics_decline(self):
        self.assertTrue(self.eligible())
        for obj, changes in (
            (self.module, dict(num_slices=0, tp_reduce=True, activation_fn='gelu', act_limit=1., interm_dtype='float')),
            (self.module.gates[0], dict(pre_scale=2., post_scale=2., softcap=1., lora_a_tensors={'a': 1})),
            (self.module.gates[0].inner, dict(bias=object(), mcg=True, K=7, mul1=False,
                 in_features=127, out_features=0, default_out_dtype='float')),
        ):
            for name, value in changes.items():
                old = getattr(obj, name)
                setattr(obj, name, value)
                self.assertFalse(self.eligible(), name)
                setattr(obj, name, old)
        self.module.gates[0].trim_padded_out = True
        self.module.gates[0].out_features_unpadded = 255
        self.assertFalse(self.eligible())

    def test_native_dispatch_and_disable_do_not_nest(self):
        stats = install_mlp_pair([self.module], enabled=True)
        original_wrapper = self.module.forward
        for rows in (1, 2, 3, 5):
            x = Tensor((1, rows, 128))
            self.assertEqual(self.module.forward(x, {}, 'half'), ('down', 'half'))
            args = self.native.call_args.args
            self.assertEqual(args[7].shape, (2, rows, 128))
            self.assertEqual(args[8].shape, (2, rows, 256))
            self.assertEqual(args[9].shape, (rows, 256))
            self.assertEqual(args[-2:], (5, True))
            self.assertIs(args[1], self.module.gates[0].inner.trellis)
            self.assertIs(args[2], self.module.ups[0].inner.trellis)
        self.assertEqual(stats[0]['paired_calls'], 4)
        install_mlp_pair([self.module], enabled=False)
        self.assertIs(self.module.forward, original_wrapper)
        self.assertEqual(self.module.forward(x, {})[0], 'fallback')
        install_mlp_pair([self.module], enabled=True)
        self.assertIs(self.module.forward, original_wrapper)

    def test_existing_fusion_and_custom_projection_are_preserved(self):
        custom = Mock(return_value='existing fusion')
        self.module.forward = custom
        self.assertEqual(install_mlp_pair([self.module], enabled=True), [])
        self.assertIs(self.module.forward, custom)
        del self.module.forward
        self.module.gates[0].inner.forward = Mock()
        self.assertEqual(install_mlp_pair([self.module], enabled=True), [])
        self.native.assert_not_called()

    def test_diagnostic_kernels_keep_the_separate_path(self):
        for env in ({'EXL3_SMALLM_WMMA': '1'}, {'EXL3_SMALLM_WMMA': '2'}, {'EXL3_GEMV_LDS': '1'}):
            with patch.dict(os.environ, env):
                self.assertEqual(install_mlp_pair([self.module], enabled=True), [])
                self.assertEqual(self.module.forward(Tensor((1, 3, 128)), {})[0], 'fallback')

    def test_unload_releases_weights_and_reload_requires_reinstallation(self):
        import copy
        import gc
        import weakref
        reference = weakref.ref(self.module.gates[0].inner)
        install_mlp_pair([self.module], enabled=True)
        self.module.gates[0].inner = None
        gc.collect()
        self.assertIsNone(reference())
        x = Tensor((1, 3, 128))
        self.assertEqual(self.module.forward(x, {})[0], 'fallback')
        self.module.gates[0].inner = copy.copy(self.module.ups[0].inner)
        self.assertEqual(self.module.forward(x, {})[0], 'fallback')
        install_mlp_pair([self.module], enabled=True)
        self.assertEqual(self.module.forward(x, {}), ('down', 'float'))

    def test_runtime_fallback_keeps_original_arguments(self):
        install_mlp_pair([self.module], enabled=True)
        for shape, kw, params in (
            ((1, 4, 128), {}, {}), ((1, 9, 128), {}, {}), ((3, 128), {}, {}),
            ((1, 3, 127), {}, {}), ((1, 3, 128), {'dtype': 'float'}, {}),
            ((1, 3, 128), {'device': 'cpu'}, {}), ((1, 3, 128), {'contiguous': False}, {}),
            ((1, 3, 128), {'pointer': 130}, {}),
            *(( (1, 3, 128), {}, {name: True}) for name in ('reconstruct', 'capture', 'ovr', 'q_mlp_slice')),
        ):
            x = Tensor(shape, **kw)
            self.assertEqual(self.module.forward(x, params, 'half'), ('fallback', x, params, 'half'))
        self.native.assert_not_called()
        self.module.gates[0].lora_a_tensors['adapter'] = object()
        self.assertEqual(self.module.forward(Tensor((1, 3, 128)), {})[0], 'fallback')


if __name__ == '__main__':
    unittest.main()
