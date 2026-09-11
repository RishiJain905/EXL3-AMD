"""Smoke execution gates must fail without importing a GPU runtime."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_exl3_model_smoke.py"
sys.path.insert(0, str(SCRIPT.parent))
from run_exl3_model_smoke import inspect_logits, observe_sampler_input, observe_mtp_head
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU Torch required for actual tensor boundary checks")
class StreamedLogitBoundaryTests(unittest.TestCase):
    def test_finite_logits_counted(self):
        self.assertEqual(inspect_logits(torch.zeros((1, 2, 4)), 2, 4),
                         {"checked_logit_count": 8, "checked_logit_rows": 2, "logits_finite": True,
                          "nan_count": 0, "positive_infinity_count": 0, "negative_infinity_count": 0,
                          "logits_valid": True})

    def test_exact_padding_mask_is_valid_but_not_raw_finite(self):
        logits = torch.zeros((1, 2, 5))
        logits[..., 3:] = -float("inf")
        checked = inspect_logits(logits, 2, 5, 3)
        self.assertTrue(checked["logits_valid"])
        self.assertFalse(checked["logits_finite"])
        self.assertEqual(checked["masked_tail_count"], 4)
        self.assertEqual(checked["expected_masked_tail_count"], 4)
        self.assertFalse(inspect_logits(logits, 2, 5)["logits_valid"])

    def test_returned_nonfinite_inside_vocab_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            logits = torch.zeros((1, 1, 5))
            logits[..., 3:] = -float("inf")
            logits[..., 1] = value
            self.assertFalse(inspect_logits(logits, 1, 5, 3)["logits_valid"])

    def test_returned_tail_requires_only_negative_infinity(self):
        for value in (float("nan"), float("inf"), 0.0):
            logits = torch.zeros((1, 1, 5))
            logits[..., 3:] = -float("inf")
            logits[..., 4] = value
            self.assertFalse(inspect_logits(logits, 1, 5, 3)["logits_valid"])
        self.assertTrue(inspect_logits(torch.zeros((1, 1, 5)), 1, 5, 5)["logits_valid"])

    def test_empty_or_invalid_valid_vocab_rejected(self):
        for size in (0, -1, 6):
            with self.assertRaises(ValueError):
                inspect_logits(torch.zeros((1, 1, 5)), 1, 5, size)
        with self.assertRaises(ValueError):
            inspect_logits(torch.zeros((1, 0, 5)), 0, 5, 3)

    def test_sampler_observes_before_mutation_and_preserves_call(self):
        seen = []
        logits = torch.zeros((1, 3, 5))
        marker = object()
        class Sampler:
            def forward(self, actual, arg, *, option):
                self_call = (actual, arg, option)
                self.asserted = self_call
                actual[..., 3:] = -float("inf")
                return marker
        sampler = Sampler()
        observe_sampler_input(sampler, lambda x: seen.append(inspect_logits(x, 3, 5)))
        self.assertIs(sampler.forward(logits, marker, option=7), marker)
        self.assertIs(sampler.asserted[0], logits)
        self.assertEqual(sampler.asserted[1:], (marker, 7))
        self.assertTrue(seen[0]["logits_valid"])
        self.assertEqual(seen[0]["checked_logit_rows"], 3)
        self.assertTrue(inspect_logits(logits, 3, 5, 3)["logits_valid"])

    def test_mtp_observer_preserves_head_and_restores_after_failure(self):
        logits = torch.zeros((1, 3, 5))
        head = SimpleNamespace(forward=lambda state, params: state)
        model = SimpleNamespace(modules=[head], logit_layer_idx=0)
        original = head.forward
        draft = SimpleNamespace(sample_from_state=lambda state, params: head.forward(state, params).argmax(-1))
        seen = []
        observe_mtp_head(draft, model, lambda x: seen.append(inspect_logits(x, 3, 5)))
        self.assertTrue(torch.equal(draft.sample_from_state(logits, {}), logits.argmax(-1)))
        self.assertIs(head.forward, original)
        self.assertEqual(len(seen), 1)
        def reject(x):
            raise RuntimeError("raw rejection")
        observe_mtp_head(draft, model, reject)
        with self.assertRaisesRegex(RuntimeError, "raw rejection"):
            draft.sample_from_state(logits, {})
        self.assertIs(head.forward, original)

    def test_nan_and_infinity_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            logits = torch.zeros((1, 1, 4))
            logits[0, 0, 2] = value
            self.assertFalse(inspect_logits(logits, 1, 4)["logits_finite"])
            diagnostic = inspect_logits(logits, 1, 4)
            self.assertEqual(sum(diagnostic[key] for key in
                                 ("nan_count", "positive_infinity_count", "negative_infinity_count")), 1)

    def test_missing_or_mismatched_logits_rejected(self):
        for logits in (None, torch.zeros((1, 2, 4)), torch.zeros((1, 1, 3))):
            with self.assertRaisesRegex(ValueError, "streamed logits shape"):
                inspect_logits(logits, 1, 4)


class ModelSmokeBoundaryTests(unittest.TestCase):
    def invoke(self, root, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(root / "local.toml"),
             "--candidate", str(root / "candidate"), "--source-dir", str(root / "absent"),
             "--extension-dir", str(root / "absent"), "--expected-extension-sha256", "0" * 64,
             "--output", str(root / "out"), *extra],
            capture_output=True, text=True, timeout=10)

    def config(self, root, probe=True, inference=True):
        (root / "local.toml").write_text(
            f"[execution]\nallow_backend_probes={str(probe).lower()}\n"
            f"allow_local_inference={str(inference).lower()}\n")

    def test_permissions_and_execute_required_before_candidate_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for probe, inference, execute in [(True, True, False), (False, True, True), (True, False, True)]:
                self.config(root, probe, inference)
                result = self.invoke(root, *(["--execute"] if execute else []))
                self.assertEqual(result.returncode, 2)
                self.assertIn("both configured local permissions", result.stderr)
                self.assertFalse((root / "out").exists())

    def test_invalid_budget_rejected_before_config_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for budget in ("0", "14.1", "nan"):
                result = self.invoke(root, "--gpu-budget-gib", budget)
                self.assertEqual(result.returncode, 2)
                self.assertIn("at most 14", result.stderr)
                self.assertFalse((root / "out").exists())

    def test_incomplete_candidate_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.config(root)
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "config.json").write_text("{}")
            result = self.invoke(root, "--execute")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing quantization_config.json", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_existing_output_preserved_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.config(root)
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "exl3"}}))
            for name in ("quantization_config.json", "tokenizer.json", "model.safetensors"):
                (candidate / name).write_bytes(b"")
            output = root / "out"
            output.mkdir()
            sentinel = output / "retained"
            sentinel.write_bytes(b"old evidence")
            result = self.invoke(root, "--execute")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(list(output.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_bytes(), b"old evidence")
