"""CPU boundaries for the small-M packed operator harness; no GPU work."""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_exl3_smallm.py"
sys.path.insert(0, str(SCRIPT.parent))
from benchmark_exl3_smallm import (
    check_output_exclusive,
    compare_pair,
    execution_allowed,
    infer_dimensions,
    layer_artifact,
    mark_failed,
    median_values,
    parse_batches_text,
    parse_timing_repeats_value,
    validate_encoding_meta,
)

try:
    import torch
except ImportError:
    torch = None


class BatchParsingTests(unittest.TestCase):
    def test_valid_subsets_sorted(self):
        self.assertEqual(parse_batches_text("1"), [1])
        self.assertEqual(parse_batches_text("2,3"), [2, 3])
        self.assertEqual(parse_batches_text("3,1"), [1, 3])
        self.assertEqual(parse_batches_text("1,2,3"), [1, 2, 3])
        self.assertEqual(parse_batches_text("4"), [4])
        self.assertEqual(parse_batches_text("5"), [5])
        self.assertEqual(parse_batches_text("5,4"), [4, 5])
        self.assertEqual(parse_batches_text("1,2,3,4,5"), [1, 2, 3, 4, 5])
        self.assertEqual(parse_batches_text("6"), [6])
        self.assertEqual(parse_batches_text("9"), [9])
        self.assertEqual(parse_batches_text("9,6"), [6, 9])
        self.assertEqual(parse_batches_text("1,2,3,4,5,6,7,8,9"), [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_invalid_rejected(self):
        for bad in ("", " ", None, "0", "10", "1,10", "a", "1,,2", "1,1", "1.0", "4,4"):
            with self.assertRaisesRegex(ValueError, "subset of 1,2,3,4,5,6,7,8,9", msg=repr(bad)):
                parse_batches_text(bad)


class TimingRepeatsTests(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(parse_timing_repeats_value(1), 1)
        self.assertEqual(parse_timing_repeats_value(20), 20)
        self.assertEqual(parse_timing_repeats_value(100), 100)

    def test_out_of_bounds_rejected(self):
        for bad in (0, 101, -1, "a", None):
            with self.assertRaisesRegex(ValueError, "timing-repeats", msg=repr(bad)):
                parse_timing_repeats_value(bad)


class ExecutionGateTests(unittest.TestCase):
    def test_requires_execute_and_both_true(self):
        good = {"execution": {"allow_local_inference": True, "allow_backend_probes": True}}
        self.assertTrue(execution_allowed(good, True))
        self.assertFalse(execution_allowed(good, False))
        for config in ({"execution": {"allow_local_inference": False, "allow_backend_probes": True}},
                       {"execution": {"allow_local_inference": True, "allow_backend_probes": False}},
                       {"execution": {"allow_local_inference": 1, "allow_backend_probes": True}},
                       {}, {"execution": None}, {"execution": "yes"}):
            self.assertFalse(execution_allowed(config, True), msg=repr(config))


class OutputExclusivityTests(unittest.TestCase):
    def test_layer_and_head_names_require_one_unambiguous_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            head = root / 'head.safetensors'
            head.write_bytes(b'head')
            self.assertEqual(layer_artifact(root), head)
            (root / 'layer.safetensors').write_bytes(b'layer')
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                layer_artifact(root)

    def test_separate_siblings_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            check_output_exclusive(root / "out", [root / "in1", root / "in2"])

    def test_equal_or_nested_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "nonnested"):
                check_output_exclusive(root / "same", [root / "same"])
            with self.assertRaisesRegex(ValueError, "nonnested"):
                check_output_exclusive(root / "in" / "out", [root / "in"])
            with self.assertRaisesRegex(ValueError, "nonnested"):
                check_output_exclusive(root / "out", [root / "out" / "sub"])


class InferDimensionsTests(unittest.TestCase):
    def test_valid_shapes(self):
        self.assertEqual(infer_dimensions(256, 128, (16, 8, 48), 3), (256, 128))
        self.assertEqual(infer_dimensions(128, 128, (8, 8, 32), 2), (128, 128))
        self.assertEqual(infer_dimensions(512, 256, (32, 16, 64), 4), (512, 256))

    def test_mismatches_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive multiple of 128"):
            infer_dimensions(100, 128, (6, 8, 48), 3)
        with self.assertRaisesRegex(ValueError, "positive multiple of 128"):
            infer_dimensions(128, 0, (8, 0, 32), 2)
        with self.assertRaisesRegex(ValueError, "bits must be"):
            infer_dimensions(128, 128, (8, 8, 32), 5)
        with self.assertRaisesRegex(ValueError, "mismatches"):
            infer_dimensions(256, 128, (16, 8, 32), 3)


class EncodingMetaTests(unittest.TestCase):
    def test_valid_with_and_without_key(self):
        sha = "ab" * 32
        self.assertEqual(validate_encoding_meta({"bits": 3, "artifact_sha256": sha, "key": "k"}),
                         ("k", 3, sha))
        self.assertEqual(validate_encoding_meta({"bits": 2, "artifact_sha256": sha.upper()}),
                         (None, 2, sha))

    def test_invalid_rejected(self):
        sha = "ab" * 32
        for meta in (None, [], {"bits": 1, "artifact_sha256": sha},
                     {"bits": 3}, {"bits": 3, "artifact_sha256": "short"},
                     {"bits": 3, "artifact_sha256": "zz" * 32},
                     {"bits": 3, "artifact_sha256": sha, "key": 7}):
            with self.assertRaises(ValueError, msg=repr(meta)):
                validate_encoding_meta(meta)


class MedianTests(unittest.TestCase):
    def test_values(self):
        self.assertEqual(median_values([2.0]), 2.0)
        self.assertEqual(median_values([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(median_values([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_empty_or_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            median_values([])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            median_values([float("inf")])


class FailureRecordingTests(unittest.TestCase):
    def test_mark_failed_preserves_partial_cases(self):
        status = {"status": "running", "cases": [{"batch": 1}], "layers": [{"index": 0}]}
        mark_failed(status, ValueError("boom"))
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error_type"], "ValueError")
        self.assertEqual(status["error"], "boom")
        self.assertEqual(status["cases"], [{"batch": 1}])
        self.assertEqual(status["layers"], [{"index": 0}])
        json.dumps(status, allow_nan=False)


@unittest.skipIf(torch is None, "CPU Torch required for metric boundary checks")
class CompareMetricTests(unittest.TestCase):
    def assertJsonSafe(self, payload):
        json.dumps(payload, allow_nan=False)
        if payload['relative_l2'] is not None:
            self.assertTrue(math.isfinite(payload["relative_l2"]))
        if payload["max_abs"] is not None:
            self.assertTrue(math.isfinite(payload["max_abs"]))

    def test_identical_zeros_yield_zero_error(self):
        result = compare_pair(torch.zeros(2, 4), torch.zeros(2, 4))
        self.assertTrue(result["shape_match"] and result["both_finite"] and result["exact_match"])
        self.assertEqual(result["relative_l2"], 0.0)
        self.assertEqual(result["max_abs"], 0.0)
        self.assertJsonSafe(result)

    def test_zero_reference_with_error_is_finite_failure(self):
        result = compare_pair(torch.ones(1, 4), torch.zeros(1, 4))
        self.assertTrue(result["both_finite"])
        self.assertFalse(result["exact_match"])
        self.assertIsNone(result["relative_l2"])
        self.assertEqual(result["max_abs"], 1.0)
        self.assertIsNone(result["relative_l2"])
        self.assertJsonSafe(result)

    def test_small_and_large_errors_separate_gates(self):
        ref = torch.ones(1, 8)
        small = compare_pair(ref + 1e-4, ref)
        self.assertLess(small["relative_l2"], 1e-3)
        large = compare_pair(2.0 * ref, ref)
        self.assertGreater(large["relative_l2"], 5e-3)
        self.assertFalse(large["exact_match"])
        self.assertJsonSafe(small)
        self.assertJsonSafe(large)

    def test_nonfinite_and_shape_mismatch_use_safe_placeholders(self):
        bad = torch.zeros(1, 4)
        bad[0, 0] = float("nan")
        result = compare_pair(bad, torch.zeros(1, 4))
        self.assertFalse(result["both_finite"] or result["exact_match"])
        self.assertIsNone(result["relative_l2"])
        self.assertIsNone(result["max_abs"])
        self.assertJsonSafe(result)
        mismatched = compare_pair(torch.zeros(1, 4), torch.zeros(1, 3))
        self.assertFalse(mismatched["shape_match"] or mismatched["exact_match"])
        self.assertIsNone(mismatched["relative_l2"])
        self.assertIsNone(mismatched["max_abs"])
        self.assertJsonSafe(mismatched)


class CliBoundaryTests(unittest.TestCase):
    def base(self, root):
        return [sys.executable, str(SCRIPT), "--config", str(root / "local.toml"),
                "--input", str(root / "absent-in"), "--source-dir", str(root / "absent-src"),
                "--extension-dir", str(root / "absent-ext"),
                "--expected-extension-sha256", "0" * 64,
                "--output", str(root / "out")]

    def config(self, root, probe=True, inference=True):
        (root / "local.toml").write_text(
            f"[execution]\nallow_backend_probes={str(probe).lower()}\n"
            f"allow_local_inference={str(inference).lower()}\n")

    def test_help_works_without_runtime(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        for flag in ("--config", "--input", "--source-dir", "--extension-dir",
                     "--expected-extension-sha256", "--output", "--batches",
                     "--timing-repeats", "--correctness-only", "--execute"):
            self.assertIn(flag, result.stdout)

    def test_denied_gates_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for probe, inference, execute in [(True, True, False), (False, True, True), (True, False, True)]:
                self.config(root, probe, inference)
                command = self.base(root) + (["--execute"] if execute else [])
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("configured local permissions", result.stderr)
                self.assertFalse((root / "out").exists())

    def test_invalid_batches_and_repeats_rejected_before_config_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for bad in ("0", "10", "1,1"):
                command = self.base(root) + ["--batches", bad, "--execute"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, msg=bad)
                self.assertIn("subset of 1,2,3,4,5,6,7,8,9", result.stderr)
                self.assertFalse((root / "out").exists())
            for bad in ("0", "101"):
                command = self.base(root) + ["--timing-repeats", bad, "--execute"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, msg=bad)
                self.assertIn("timing-repeats", result.stderr)
                self.assertFalse((root / "out").exists())

    def test_nested_output_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.config(root)
            nested = [("--input", str(root / "in"), "--output", str(root / "in" / "out")),
                      ("--input", str(root / "out" / "sub"), "--output", str(root / "out"))]
            for in_flag, in_val, out_flag, out_val in nested:
                command = [sys.executable, str(SCRIPT), "--config", str(root / "local.toml"),
                           in_flag, in_val, "--source-dir", str(root / "absent-src"),
                           "--extension-dir", str(root / "absent-ext"),
                           "--expected-extension-sha256", "0" * 64,
                           out_flag, out_val, "--execute"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("nonnested", result.stderr)
                self.assertFalse(Path(out_val).exists())

    def test_missing_layer_files_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.config(root)
            layer = root / "layer"
            layer.mkdir()
            command = [sys.executable, str(SCRIPT), "--config", str(root / "local.toml"),
                       "--input", str(layer), "--source-dir", str(root / "absent-src"),
                       "--extension-dir", str(root / "absent-ext"),
                       "--expected-extension-sha256", "0" * 64,
                       "--output", str(root / "out"), "--execute"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing layer.safetensors", result.stderr)
            self.assertFalse((root / "out").exists())
            (layer / "layer.safetensors").write_bytes(b"")
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing encoding.json", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_existing_output_preserved_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.config(root)
            layer = root / "layer"
            layer.mkdir()
            (layer / "layer.safetensors").write_bytes(b"")
            (layer / "encoding.json").write_bytes(b"{}")
            output = root / "out"
            output.mkdir()
            sentinel = output / "retained"
            sentinel.write_bytes(b"old evidence")
            command = [sys.executable, str(SCRIPT), "--config", str(root / "local.toml"),
                       "--input", str(layer), "--source-dir", str(root / "absent-src"),
                       "--extension-dir", str(root / "absent-ext"),
                       "--expected-extension-sha256", "0" * 64,
                       "--output", str(output), "--execute"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(list(output.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_bytes(), b"old evidence")


if __name__ == "__main__":
    unittest.main()
