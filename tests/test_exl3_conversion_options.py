"""CPU-only argument plumbing checks; no EXL3 or Torch import."""
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import Mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("exl3_conversion_runner", SCRIPTS / "run_exl3_conversion.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ConversionOptionsTests(unittest.TestCase):
    def test_resume_content_changes_fail_without_overwriting_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = dict(source_files={"weights.safetensors": "hashA"}, calibration_sha256="tokensA",
                            out_scales="always", compact_cpu_buffers=True)
            runner.freeze_identity(tmp, identity, resume=False)
            path = Path(tmp) / "conversion_identity.json"
            original = path.read_bytes()
            runner.freeze_identity(tmp, identity, resume=True)
            for key, value in (("source_files", {"weights.safetensors": "hashB"}),
                               ("calibration_sha256", "tokensB"), ("out_scales", "never"),
                               ("compact_cpu_buffers", False)):
                with self.assertRaisesRegex(ValueError, key):
                    runner.freeze_identity(tmp, {**identity, key: value}, resume=True)
                self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                runner.freeze_identity(tmp, identity, resume=False)

    def test_unpinned_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "lacks conversion_identity"):
                runner.freeze_identity(tmp, {}, resume=True)

    def test_source_identity_detects_payload_replacement_at_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"weights.safetensors"
            path.write_bytes(b"original")
            first = runner.source_identity(tmp)
            path.write_bytes(b"changed")
            self.assertNotEqual(first, runner.source_identity(tmp))

    def test_compact_flag_set_on_factory_result_before_caller_receives_it(self):
        factory = Mock(return_value={"K": 4, "seed": 12})
        wrapped = runner.compact_quant_args(factory)
        result = wrapped("job", 66, K=4)
        factory.assert_called_once_with("job", 66, K=4)
        self.assertEqual(result, {"K": 4, "seed": 12, "compact_cpu_buffers": True})
