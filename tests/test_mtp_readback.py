"""Exercise the MTP loop with real CPU tensors, without loading the native backend."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import textwrap
import time
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / 'vendor/rocm-exl3/exllamav3/generator/generator.py'
CALIBRATOR = GENERATOR.with_name('draft_confidence.py')


def load_loop():
    source = GENERATOR.read_text(encoding='utf-8')
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == 'iterate_draftmodel_mtp_gen')
    namespace = dict(torch=torch, time=time, PAGE_SIZE=256, cuda_sync_active=lambda: None)
    exec(compile(textwrap.dedent(ast.get_source_segment(source, method)), str(GENERATOR), 'exec'), namespace)
    return namespace[method.name]


@unittest.skipIf(torch is None, 'CPU Torch required')
class MtpReadbackTests(unittest.TestCase):
    def run_loop(self, *, calibrated=False, missing=(), gpu_draft=False, fixed=False):
        spec = importlib.util.spec_from_file_location('readback_calibrator', CALIBRATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cal = None if fixed else module.DraftConfidenceCalibrator(.6)
        if calibrated:
            for _ in range(64):
                cal.add_label(3.0, False)
        jobs = []
        for token in (10, 20):
            seq = SimpleNamespace(block_index_tensor=torch.zeros((1, 1), dtype=torch.int32), kv_position=5)
            jobs.append(SimpleNamespace(is_prefill_done=lambda: True, get_max_seq_len=lambda: 5,
                        sequences=[seq], mtp_last_hidden=torch.zeros((1, 1, 4)), time_first_token=1,
                        get_input_ids_list=lambda token=token: [torch.tensor([[token]])]))
        inputs = []
        def forward(ids, params):
            inputs.append(ids.clone())
            return torch.zeros((2, 1, 4))
        def sample(state, params):
            if params.get('export_draft_conf') and len(inputs)-1 not in missing:
                params['draft_conf'] = torch.tensor([[3.0], [3.5]])
            return inputs[-1] + 1
        draft = SimpleNamespace(forward=forward, sample_from_state=sample)
        head = SimpleNamespace(prepare_for_device=lambda state, params: state)
        gen = SimpleNamespace(active_jobs=jobs, num_draft_tokens=4, draft_calibrator=cal,
                  draft_model=draft, draft_cache=None, model=SimpleNamespace(modules=[head], logit_layer_idx=0),
                  draft_input_ids_pinned=torch.zeros((2, 1), dtype=torch.long),
                  draft_ids_pinned=torch.zeros((2, 4), dtype=torch.long),
                  quantlab_gpu_draft=gpu_draft, quantlab_gpu_draft_metadata=gpu_draft)
        copies = []
        original = torch.Tensor.cpu
        def cpu(tensor, *args, **kwargs):
            copies.append(tensor.shape)
            return original(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, 'cpu', cpu):
            result = load_loop()(gen, [])
        return gen, result, inputs, copies

    def test_warmup_keeps_ids_and_scores_with_one_confidence_readback(self):
        gen, result, inputs, copies = self.run_loop()
        self.assertEqual(result.tolist(), [[11, 12, 13, 14], [21, 22, 23, 24]])
        self.assertEqual([x[:, 0].tolist() for x in inputs], [[10, 20], [11, 21], [12, 22], [13, 23]])
        self.assertEqual(gen._draft_conf_round['conf'].tolist(), [[3.0]*4, [3.5]*4])
        self.assertEqual(gen._draft_conf_round['conf'].device.type, 'cpu')
        self.assertEqual(copies, [torch.Size([2, 4])])

    def test_calibrated_low_confidence_stops_after_first_probe(self):
        gen, result, inputs, copies = self.run_loop(calibrated=True)
        self.assertEqual(result.tolist(), [[11], [21]])
        self.assertEqual(len(inputs), 1)
        self.assertEqual(gen._draft_conf_round['conf'].tolist(), [[3.0], [3.5]])
        self.assertEqual(copies, [torch.Size([2, 1])])

    def test_missing_confidence_does_not_misalign_labels(self):
        for missing in ((1,), (0, 1, 2, 3)):
            with self.subTest(missing=missing):
                gen, result, _, copies = self.run_loop(missing=missing)
                self.assertEqual(result.shape, (2, 4))
                self.assertIsNone(gen._draft_conf_round)
                self.assertEqual(copies, [])

    def test_fixed_and_gpu_bookkeeping_preserve_draft_ids(self):
        for gpu_draft in (False, True):
            with self.subTest(gpu_draft=gpu_draft):
                gen, result, _, copies = self.run_loop(fixed=True, gpu_draft=gpu_draft)
                self.assertEqual(result.tolist(), [[11, 12, 13, 14], [21, 22, 23, 24]])
                self.assertIsNone(gen._draft_conf_round)
                self.assertEqual(copies, [])


if __name__ == '__main__':
    unittest.main()
