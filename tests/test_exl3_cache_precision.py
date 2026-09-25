"""CPU-only cache-precision plumbing; no Torch, GPU, model, or network."""
import argparse
import contextlib
import importlib.util
import io
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from quantlab.methods.exl3 import cache_precision  # noqa: E402

_eval_spec = importlib.util.spec_from_file_location("evaluate_under_test", SCRIPTS / "evaluate_exl3_candidate.py")
evaluate = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(evaluate)

_serve_spec = importlib.util.spec_from_file_location("serve_under_test", SCRIPTS / "serve_exl3.py")
serve = importlib.util.module_from_spec(_serve_spec)
_serve_spec.loader.exec_module(serve)


def _namespace(**overrides):
    base = dict(cache_type=None, cache_type_k=None, cache_type_v=None)
    base.update(overrides)
    return SimpleNamespace(**base)


class ResolveTests(unittest.TestCase):
    def test_no_flag_preserves_f16(self):
        self.assertEqual(cache_precision.resolve_cache_types(_namespace()), ("f16", "f16"))

    def test_shorthand_applies_to_both_sides(self):
        for shorthand in ("f16", "q8", "q4"):
            with self.subTest(shorthand=shorthand):
                self.assertEqual(cache_precision.resolve_cache_types(_namespace(cache_type=shorthand)),
                                 (shorthand, shorthand))

    def test_mixed_q8_q4_accepted(self):
        self.assertEqual(cache_precision.resolve_cache_types(_namespace(cache_type_k="q8", cache_type_v="q4")),
                         ("q8", "q4"))
        self.assertEqual(cache_precision.resolve_cache_types(_namespace(cache_type_k="q4", cache_type_v="q8")),
                         ("q4", "q8"))

    def test_consistent_shorthand_coexists(self):
        self.assertEqual(cache_precision.resolve_cache_types(
            _namespace(cache_type="q8", cache_type_k="q8", cache_type_v="q8")), ("q8", "q8"))

    def test_conflicting_shorthand_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            cache_precision.resolve_cache_types(_namespace(cache_type="q8", cache_type_k="q4"))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            cache_precision.resolve_cache_types(_namespace(cache_type="q8", cache_type_v="q4"))

    def test_f16_quant_mixing_rejected(self):
        for k, v in (("f16", "q8"), ("q8", "f16"), ("f16", "q4"), ("q4", "f16")):
            with self.subTest(k=k, v=v):
                with self.assertRaisesRegex(ValueError, "mix"):
                    cache_precision.resolve_cache_types(_namespace(cache_type_k=k, cache_type_v=v))

    def test_single_side_quant_mixes_with_default_f16(self):
        with self.assertRaisesRegex(ValueError, "mix"):
            cache_precision.resolve_cache_types(_namespace(cache_type_k="q8"))

    def test_unknown_precision_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown cache precision"):
            cache_precision.resolve_cache_types(_namespace(cache_type_k="fp8"))

    def test_is_quantized(self):
        self.assertFalse(cache_precision.is_quantized("f16", "f16"))
        self.assertTrue(cache_precision.is_quantized("q8", "q8"))
        self.assertTrue(cache_precision.is_quantized("q8", "q4"))


class ParserSyntaxTests(unittest.TestCase):
    def _parser(self):
        parser = argparse.ArgumentParser()
        cache_precision.add_cache_precision_args(parser)
        return parser

    def test_short_and_long_syntax(self):
        args = self._parser().parse_args(["-ctk", "q8", "-ctv", "q4"])
        self.assertEqual((args.cache_type_k, args.cache_type_v), ("q8", "q4"))
        self.assertEqual(cache_precision.resolve_cache_types(args), ("q8", "q4"))
        args = self._parser().parse_args(["--cache-type-k", "q8", "--cache-type-v", "q4"])
        self.assertEqual((args.cache_type_k, args.cache_type_v), ("q8", "q4"))
        self.assertEqual(cache_precision.resolve_cache_types(args), ("q8", "q4"))

    def test_shorthand_syntax(self):
        args = self._parser().parse_args(["--cache-type", "q4"])
        self.assertEqual(cache_precision.resolve_cache_types(args), ("q4", "q4"))

    def test_no_flag_syntax(self):
        self.assertEqual(cache_precision.resolve_cache_types(self._parser().parse_args([])), ("f16", "f16"))

    def test_fp8_rejected_by_choices(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self._parser().parse_args(["--cache-type", "fp8"])
        self.assertEqual(caught.exception.code, 2)

    def test_attention_profile_defaults_to_default(self):
        self.assertEqual(self._parser().parse_args([]).attention_profile, "default")

    def test_attention_profile_long(self):
        args = self._parser().parse_args(["--attention-profile", "long"])
        self.assertEqual(args.attention_profile, "long")

    def test_attention_profile_malformed_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self._parser().parse_args(["--attention-profile", "bogus"])
        self.assertEqual(caught.exception.code, 2)


EVAL_BASE = ["--config", "c", "--candidate", "m", "--source-dir", "s", "--extension-dir", "e",
             "--suite", "t", "--output", "o", "--expected-extension-sha256", "h", "--mode", "speed"]
SERVE_BASE = ["--config", "c", "--candidate", "m", "--source-dir", "s", "--extension-dir", "e",
              "--output", "o", "--expected-extension-sha256", "h"]


class DownstreamParserTests(unittest.TestCase):
    def test_mtp_precision_guards_run_before_config_or_model_access(self):
        invalid = (
            ['--mtp-dtype', 'bf16'],
            ['--mtp', '--mtp-dtype', 'bf16', '--decode-fusions', 'gdn'],
            ['--mtp', '--mtp-dtype', 'bf16', '--native-attention'],
            ['--mtp', '--mtp-dtype', 'bf16', '--cache-mtp', 'fc'],
            ['--verify-attention', 'rowwise', '--native-attention'],
            ['--verify-attention', 'rowwise', '--cache-type', 'q8'],
        )
        for module, base in ((evaluate, EVAL_BASE), (serve, SERVE_BASE)):
            for flags in invalid:
                with self.subTest(module=module.__name__, flags=flags):
                    with patch.object(sys, 'argv', ['test', *base, '--decode-fusions', 'off', *flags]), \
                            patch.object(Path, 'read_text', side_effect=AssertionError('config accessed')), \
                            contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as caught:
                            module.main()
                    self.assertEqual(caught.exception.code, 2)

    def test_evaluator_parser(self):
        self.assertEqual(cache_precision.resolve_cache_types(evaluate.parser().parse_args(EVAL_BASE)), ("f16", "f16"))
        args = evaluate.parser().parse_args(EVAL_BASE + ["--cache-type", "q8"])
        self.assertEqual(cache_precision.resolve_cache_types(args), ("q8", "q8"))
        args = evaluate.parser().parse_args(EVAL_BASE + ["-ctk", "q8", "-ctv", "q4"])
        self.assertEqual(cache_precision.resolve_cache_types(args), ("q8", "q4"))

    def test_server_parser(self):
        self.assertEqual(cache_precision.resolve_cache_types(serve.parser().parse_args(SERVE_BASE)), ("f16", "f16"))
        args = serve.parser().parse_args(SERVE_BASE + ["--cache-type", "q4"])
        self.assertEqual(cache_precision.resolve_cache_types(args), ("q4", "q4"))
        args = serve.parser().parse_args(SERVE_BASE + ["--cache-type-k", "q4", "--cache-type-v", "q8"])
        self.assertEqual(cache_precision.resolve_cache_types(args), ("q4", "q8"))

    def test_evaluator_attention_profile(self):
        self.assertEqual(evaluate.parser().parse_args(EVAL_BASE).attention_profile, "default")
        args = evaluate.parser().parse_args(EVAL_BASE + ["--attention-profile", "long"])
        self.assertEqual(args.attention_profile, "long")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                evaluate.parser().parse_args(EVAL_BASE + ["--attention-profile", "bogus"])
        self.assertEqual(caught.exception.code, 2)

    def test_server_attention_profile(self):
        self.assertEqual(serve.parser().parse_args(SERVE_BASE).attention_profile, "default")
        args = serve.parser().parse_args(SERVE_BASE + ["--attention-profile", "long"])
        self.assertEqual(args.attention_profile, "long")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                serve.parser().parse_args(SERVE_BASE + ["--attention-profile", "bogus"])
        self.assertEqual(caught.exception.code, 2)


class CacheKwargsTests(unittest.TestCase):
    def test_f16_preserves_default(self):
        self.assertEqual(cache_precision.cache_kwargs("f16", "f16"), {})

    def test_quant_forwards_bits_without_compander(self):
        sentinel = type("CacheLayer_quant", (), {})
        kwargs = cache_precision.cache_kwargs("q8", "q4", quant_layer_type=sentinel)
        self.assertIs(kwargs["layer_type"], sentinel)
        self.assertEqual((kwargs["k_bits"], kwargs["v_bits"], kwargs["compand_a"]), (8, 4, 0))

    def test_q4_pair(self):
        sentinel = type("CacheLayer_quant", (), {})
        kwargs = cache_precision.cache_kwargs("q4", "q4", quant_layer_type=sentinel)
        self.assertEqual((kwargs["k_bits"], kwargs["v_bits"], kwargs["compand_a"]), (4, 4, 0))

    def test_mixing_rejected(self):
        with self.assertRaisesRegex(ValueError, "mix"):
            cache_precision.cache_kwargs("f16", "q8", quant_layer_type=object())

    def test_unknown_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown cache precision"):
            cache_precision.cache_kwargs("q8", "fp8", quant_layer_type=object())


class FakeTensor:
    """Torch tensor interface subset used for byte accounting (numel/element_size/shape)."""

    def __init__(self, numel, element_size, shape):
        self._numel = numel
        self._element_size = element_size
        self.shape = shape

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class FakeQuantLayer:
    def __init__(self, tensors):
        self._tensors = tensors

    def get_tensors(self):
        return self._tensors


class FakeCache:
    def __init__(self, layers):
        self.layers = layers


class StorageTests(unittest.TestCase):
    def test_absent_is_explicit_not_unknown(self):
        self.assertEqual(cache_precision.cache_storage(None, cache_k="q8", cache_v="q8"),
                         {"present": False, "cache_k": None, "cache_v": None,
                          "attention_layers": 0, "tensor_bytes": 0, "layers": []})

    def test_sums_measured_tensor_bytes(self):
        cache = FakeCache({0: FakeQuantLayer([FakeTensor(100, 4, (10, 10)), FakeTensor(50, 2, (50,)), None]),
                           1: FakeQuantLayer([FakeTensor(200, 1, (200,))])})
        storage = cache_precision.cache_storage(cache, cache_k="q8", cache_v="q4")
        self.assertTrue(storage["present"])
        self.assertEqual((storage["cache_k"], storage["cache_v"]), ("q8", "q4"))
        self.assertEqual(storage["attention_layers"], 2)
        self.assertEqual(storage["tensor_bytes"], 100 * 4 + 50 * 2 + 200 * 1)
        self.assertEqual(storage["layers"][0]["class"], "FakeQuantLayer")
        self.assertEqual(storage["layers"][0]["tensor_shapes"], [[10, 10], [50]])
        self.assertEqual(storage["layers"][0]["tensor_bytes"], 100 * 4 + 50 * 2)
        self.assertEqual(storage["layers"][1]["tensor_bytes"], 200)

    def test_unallocated_layers_report_zero_bytes(self):
        cache = FakeCache({0: FakeQuantLayer([None, None])})
        storage = cache_precision.cache_storage(cache, cache_k="f16", cache_v="f16")
        self.assertTrue(storage["present"])
        self.assertEqual(storage["attention_layers"], 1)
        self.assertEqual(storage["tensor_bytes"], 0)
        self.assertEqual(storage["layers"][0]["tensor_shapes"], [])


class AttentionProfileStatusTests(unittest.TestCase):
    MODULE = "exllamav3.modules.attention_fn.triton_paged"

    def test_unloaded_reports_absence_without_importing(self):
        sentinel = self.MODULE in sys.modules
        prior = sys.modules.get(self.MODULE)
        torch_before = "torch" in sys.modules
        sys.modules.pop(self.MODULE, None)
        try:
            self.assertEqual(cache_precision.attention_profile_status(),
                             {"profile": None, "applied_calls": None})
            self.assertNotIn(self.MODULE, sys.modules)
            self.assertEqual("torch" in sys.modules, torch_before)
        finally:
            if prior is not None or sentinel:
                sys.modules[self.MODULE] = prior

    def test_loaded_reads_actual_values(self):
        module = ModuleType(self.MODULE)
        module.qc_decode_profile_status = lambda: {"profile": "long", "applied_calls": 1234}
        sentinel = self.MODULE in sys.modules
        prior = sys.modules.get(self.MODULE)
        sys.modules[self.MODULE] = module
        try:
            self.assertEqual(cache_precision.attention_profile_status(),
                             {"profile": "long", "applied_calls": 1234})
            module.qc_decode_profile_status = lambda: {"profile": "default", "applied_calls": 0}
            self.assertEqual(cache_precision.attention_profile_status(),
                             {"profile": "default", "applied_calls": 0})
        finally:
            if prior is not None or sentinel:
                sys.modules[self.MODULE] = prior
            else:
                sys.modules.pop(self.MODULE, None)

    def test_missing_status_function_reports_absence(self):
        module = ModuleType(self.MODULE)
        sentinel = self.MODULE in sys.modules
        prior = sys.modules.get(self.MODULE)
        sys.modules[self.MODULE] = module
        try:
            self.assertEqual(cache_precision.attention_profile_status(),
                             {"profile": None, "applied_calls": None})
        finally:
            if prior is not None or sentinel:
                sys.modules[self.MODULE] = prior
            else:
                sys.modules.pop(self.MODULE, None)


if __name__ == "__main__":
    unittest.main()
