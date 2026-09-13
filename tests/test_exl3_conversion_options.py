"""CPU-only argument plumbing checks; no EXL3 or Torch import."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("exl3_conversion_runner", SCRIPTS / "run_exl3_conversion.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ConversionOptionsTests(unittest.TestCase):
    def test_compact_flag_set_on_factory_result_before_caller_receives_it(self):
        factory = Mock(return_value={"K": 4, "seed": 12})
        wrapped = runner.compact_quant_args(factory)
        result = wrapped("job", 66, K=4)
        factory.assert_called_once_with("job", 66, K=4)
        self.assertEqual(result, {"K": 4, "seed": 12, "compact_cpu_buffers": True})
