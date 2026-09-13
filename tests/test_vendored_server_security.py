"""Dependency-free security contracts for the vendored EXL3 HTTP server."""

import ast
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "vendor" / "rocm-exl3" / "rocm_tools" / "exl3_server"
SERVER_PATH = SERVER_DIR / "server.py"
DRY_PATH = SERVER_DIR / "dry_sampler.py"
LIMITS_PATH = SERVER_DIR / "request_validation.py"


def _load_limits():
    spec = importlib.util.spec_from_file_location("request_validation", LIMITS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


limits = _load_limits()


class RequestLimitTests(unittest.TestCase):
    def test_security_envelopes_are_fixed(self):
        self.assertEqual(limits.MAX_COMPLETIONS_PER_REQUEST, 16)
        self.assertEqual(limits.MAX_DRY_BREAKERS, 64)
        self.assertEqual(limits.MAX_DRY_BREAKER_BYTES, 256)
        self.assertEqual(limits.MAX_DRY_BREAKERS_TOTAL_BYTES, 4096)
        self.assertEqual(limits.MAX_DRY_BREAKER_CACHE_ENTRIES, 32)

    def test_breakers_accept_list_and_legacy_json_and_canonicalize(self):
        expected = ("a", "b")
        self.assertEqual(limits.normalize_dry_breakers(["b", "", "a", "b"]), expected)
        self.assertEqual(
            limits.normalize_dry_breakers(json.dumps(["b", "a", "b", ""])),
            expected,
        )

    def test_breaker_boundaries_are_measured_as_utf8_bytes(self):
        self.assertEqual(limits.normalize_dry_breakers(["é" * 128]), ("é" * 128,))
        with self.assertRaisesRegex(ValueError, "at most 256 UTF-8 bytes"):
            limits.normalize_dry_breakers(["a" * 257])

        exact_total = [f"{i:02x}" + "a" * 254 for i in range(16)]
        self.assertEqual(len(limits.normalize_dry_breakers(exact_total)), 16)
        with self.assertRaisesRegex(ValueError, "total at most 4096 UTF-8 bytes"):
            limits.normalize_dry_breakers([*exact_total, "x"])

    def test_breaker_count_and_types_are_bounded_before_deduplication(self):
        self.assertEqual(len(limits.normalize_dry_breakers([str(i) for i in range(64)])), 64)
        with self.assertRaisesRegex(ValueError, "at most 64 entries"):
            limits.normalize_dry_breakers(["x"] * 65)
        for invalid in (["ok", 1], '{"not":"an array"}', "not-json", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "JSON array of strings"):
                    limits.normalize_dry_breakers(invalid)

    def test_breaker_errors_do_not_echo_input(self):
        marker = "private-marker-should-not-be-returned"
        with self.assertRaises(ValueError) as caught:
            limits.normalize_dry_breakers([marker * 20])
        self.assertNotIn(marker, str(caught.exception))


class _FakeTensor:
    def __init__(self, values=()):
        self.values = list(values)

    def flatten(self):
        return self

    def tolist(self):
        return list(self.values)


class _FakeTokenizer:
    def __init__(self):
        self.scans = 0
        self.encodes = 0

    def get_id_to_piece_list(self):
        self.scans += 1
        return ["a", "b", "ab", "other"]

    def encode(self, value, **unused):
        self.encodes += 1
        return _FakeTensor([len(value)])


def _load_dry_sampler():
    torch = types.ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.long = object()
    torch.empty = lambda *unused, **kwargs: _FakeTensor()
    torch.tensor = lambda values, **kwargs: _FakeTensor(values)

    custom = types.ModuleType("exllamav3.generator.sampler.custom")
    custom.SS_Base = type("SS_Base", (), {})
    custom.SS = type("SS", (), {"INIT": 0, "LOGITS": 1})
    custom.SamplingState = type("SamplingState", (), {})

    modules = {
        "torch": torch,
        "request_validation": limits,
        "exllamav3": types.ModuleType("exllamav3"),
        "exllamav3.generator": types.ModuleType("exllamav3.generator"),
        "exllamav3.generator.sampler": types.ModuleType("exllamav3.generator.sampler"),
        "exllamav3.generator.sampler.custom": custom,
    }
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("vendored_dry_sampler_test", DRY_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class BreakerCacheTests(unittest.TestCase):
    def setUp(self):
        self.dry = _load_dry_sampler()

    def test_permutations_share_one_cache_entry(self):
        tokenizer = _FakeTokenizer()
        first = self.dry.breaker_token_ids(tokenizer, ("b", "a", ""))
        second = self.dry.breaker_token_ids(tokenizer, ("a", "b", "a"))
        self.assertIs(first, second)
        self.assertEqual(tokenizer.scans, 1)
        self.assertEqual(len(self.dry._breaker_cache), 1)

    def test_cache_is_lru_bounded_and_hits_refresh_recency(self):
        tokenizer = _FakeTokenizer()
        for index in range(32):
            self.dry.breaker_token_ids(tokenizer, (f"b{index}",))
        self.dry.breaker_token_ids(tokenizer, ("b0",))
        scans_before_eviction = tokenizer.scans

        self.dry.breaker_token_ids(tokenizer, ("b32",))
        self.assertEqual(len(self.dry._breaker_cache), 32)
        self.dry.breaker_token_ids(tokenizer, ("b0",))
        self.assertEqual(tokenizer.scans, scans_before_eviction + 1)
        self.dry.breaker_token_ids(tokenizer, ("b1",))
        self.assertEqual(tokenizer.scans, scans_before_eviction + 2)

    def test_direct_callers_cannot_bypass_limits(self):
        with self.assertRaisesRegex(ValueError, "at most 64 entries"):
            self.dry.breaker_token_ids(_FakeTokenizer(), tuple("x" for _ in range(65)))


class SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server_source = SERVER_PATH.read_text(encoding="utf-8")
        cls.server_tree = ast.parse(cls.server_source, filename=str(SERVER_PATH))

    def test_both_completion_schemas_bound_n(self):
        classes = {
            node.name: node
            for node in self.server_tree.body
            if isinstance(node, ast.ClassDef)
        }
        for name in ("ChatCompletionRequest", "CompletionRequest"):
            with self.subTest(request=name):
                assignments = [
                    node for node in classes[name].body
                    if isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "n"
                ]
                self.assertEqual(len(assignments), 1)
                rendered = ast.unparse(assignments[0].value)
                self.assertEqual(
                    rendered,
                    "Field(default=1, ge=1, le=MAX_COMPLETIONS_PER_REQUEST)",
                )

    def test_props_authenticates_and_never_returns_model_directory(self):
        props, = [
            node for node in self.server_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "props"
        ]
        rendered = ast.unparse(props)
        self.assertIn("check_auth(request)", rendered)
        self.assertNotIn("state.args.model_dir", rendered)
        self.assertIn("'model_path': ''", rendered)

    def test_advertised_fallback_names_are_machine_neutral(self):
        self.assertIn('args.served_model_name or "exl3-model"', self.server_source)
        primary = (ROOT / "scripts" / "serve_exl3.py").read_text(encoding="utf-8")
        self.assertIn("args.alias or 'exl3'", primary)
        self.assertNotIn("args.alias or args.candidate.name", primary)

    def test_changed_python_and_documented_limits_are_consistent(self):
        for path in (SERVER_PATH, DRY_PATH, LIMITS_PATH):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        readme = (SERVER_DIR / "README.md").read_text(encoding="utf-8")
        for text in ("1–16", "64 strings", "256 UTF-8 bytes", "4096 UTF-8 bytes"):
            self.assertIn(text, readme)


if __name__ == "__main__":
    unittest.main()
