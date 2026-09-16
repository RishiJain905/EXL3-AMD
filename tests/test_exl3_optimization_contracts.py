"""Bounded CPU contracts for opt80 extensions; no GPU, model, or network."""
import ast
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("launch_opt80_contracts", SCRIPTS / "launch_runtime.py")
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)

from quantlab.methods.exl3.optimizations import configure_native  # noqa: E402
from quantlab.methods.exl3.shortlist import install_draft_shortlist  # noqa: E402


class ExecutableSyntaxTests(unittest.TestCase):
    def test_all_scripts_parse(self):
        files = sorted(SCRIPTS.glob("*.py"))
        self.assertGreater(len(files), 0)
        for path in files:
            with self.subTest(script=path.name):
                try:
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                except SyntaxError as error:
                    self.fail(f"{path.name}:{error.lineno}: {error.msg}")


class LaunchExtensionRejectionTests(unittest.TestCase):
    def _write_config(self, path):
        path.write_text("[execution]\nallow_local_inference = true\nallow_backend_probes = true\n", encoding="utf-8")

    def _assert_rejects(self, argv, output, needle):
        err = io.StringIO()
        with patch.object(sys, "argv", argv), patch("subprocess.Popen") as popen, \
                patch.object(launch, "write_json") as written, contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                launch.main()
        self.assertEqual(caught.exception.code, 2)
        popen.assert_not_called()
        written.assert_not_called()
        self.assertFalse(output.exists())
        self.assertIn(needle, err.getvalue())

    def test_shortlist_requires_fixed_mtp(self):
        cases = [
            (["--shortlist-groups", "16"], "Draft shortlist requires"),
            (["--mtp", "4", "--shortlist-groups", "16", "--draft-confidence", "0.4"], "Draft shortlist requires"),
        ]
        for extra, needle in cases:
            with self.subTest(extra=extra):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    cfg = tmp / "local.toml"
                    self._write_config(cfg)
                    out = tmp / "run-out"
                    self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                          "--output", str(out), *extra], out, needle)

    def test_draft_step_graph_requires_metadata_and_full_head(self):
        cases = [
            (["--mtp", "4", "--gpu-embedding", "--gpu-draft", "--draft-step-graph"], "Draft step graph requires"),
            (["--mtp", "4", "--gpu-embedding", "--gpu-draft", "--gpu-draft-metadata",
              "--draft-step-graph", "--shortlist-groups", "16"], "Draft step graph requires"),
        ]
        for extra, needle in cases:
            with self.subTest(extra=extra):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    cfg = tmp / "local.toml"
                    self._write_config(cfg)
                    out = tmp / "run-out"
                    self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                          "--output", str(out), *extra], out, needle)


class NativeAbiTests(unittest.TestCase):
    def test_smallm_capability_uses_loaded_binary_and_preserves_legacy(self):
        with patch('ctypes.CDLL', return_value=SimpleNamespace()):
            self.assertEqual(configure_native('old.so', native_smallm=True)['smallm_codebooks'], [0])
        for mask, expected in ((1, [0]), (5, [0, 2])):
            def capability():
                return mask
            with patch('ctypes.CDLL', return_value=SimpleNamespace(quantlab_exl3_smallm_codebooks=capability)):
                self.assertEqual(configure_native('new.so', native_smallm=True)['smallm_codebooks'], expected)
        for mask in (0, 2, 4, 7, -1):
            def capability():
                return mask
            with patch('ctypes.CDLL', return_value=SimpleNamespace(quantlab_exl3_smallm_codebooks=capability)):
                with self.assertRaisesRegex(ValueError, 'codebook capability'):
                    configure_native('unknown.so', native_smallm=True)

    def _abi_lib(self, abi):
        def version():
            return abi
        return SimpleNamespace(quantlab_exl3_optimization_abi=version)

    def test_dot_default_does_not_load_cdll(self):
        with patch("ctypes.CDLL") as lib, patch.dict(os.environ, {}, clear=True):
            for kwargs in ({}, {"smallm_kernel": "dot"}):
                result = configure_native("unused", **kwargs)
                self.assertIsNone(result["optimization_abi"])
                self.assertEqual(os.environ["EXL3_SMALLM_WMMA"], "0")
            lib.assert_not_called()

    def test_wmma_register_gated_on_abi2(self):
        with patch("ctypes.CDLL", return_value=self._abi_lib(1)), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ABI"):
                configure_native("new.so", smallm_kernel="wmma-register")
            self.assertEqual(configure_native("new.so", smallm_kernel="wmma")["optimization_abi"], 1)
        with patch("ctypes.CDLL", return_value=self._abi_lib(2)), patch.dict(os.environ, {}, clear=True):
            result = configure_native("new.so", smallm_kernel="wmma-register")
            self.assertEqual(result["optimization_abi"], 2)
            self.assertEqual(os.environ["EXL3_SMALLM_WMMA"], "2")

    def test_prefill_rejects_missing_or_unknown_abi(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('ctypes.CDLL', return_value=SimpleNamespace()):
                with self.assertRaisesRegex(ValueError, 'prefill GEMM extension'):
                    configure_native('old.so', prefill_gemm='wmma')
            def version():
                return 2
            library = SimpleNamespace(quantlab_exl3_hgemm_abi=version)
            with patch('ctypes.CDLL', return_value=library):
                with self.assertRaisesRegex(ValueError, 'prefill GEMM ABI'):
                    configure_native('unknown.so', prefill_gemm='wmma')
            self.assertNotIn('EXL3_HGEMM_IMPL', os.environ)

    def test_prefill_cli_setting_overrides_inherited_environment(self):
        def version():
            return 1
        library = SimpleNamespace(quantlab_exl3_hgemm_abi=version)
        with patch('ctypes.CDLL', return_value=library), \
                patch.dict(os.environ, {'EXL3_HGEMM_IMPL': 'unexpected'}, clear=True):
            result = configure_native('new.so', prefill_gemm='wmma')
            self.assertEqual(result['prefill_gemm_abi'], 1)
            self.assertEqual(os.environ['EXL3_HGEMM_IMPL'], 'wmma')
            configure_native('new.so')
            self.assertEqual(os.environ['EXL3_HGEMM_IMPL'], 'blas')

    def test_invalid_prefill_setting_rejects_before_loading_library(self):
        with patch('ctypes.CDLL') as library:
            with self.assertRaisesRegex(ValueError, 'prefill GEMM setting'):
                configure_native('unused', prefill_gemm='unknown')
            library.assert_not_called()


class ShortlistValidationTests(unittest.TestCase):
    BLOCKED = {"torch": None, "exllamav3": None}

    def test_missing_draft_rejects_before_torch(self):
        with patch.dict(sys.modules, self.BLOCKED):
            with self.assertRaisesRegex(ValueError, "must not be None"):
                install_draft_shortlist(object(), None, 16, "packed")
            with self.assertRaisesRegex(ValueError, "must not be None"):
                install_draft_shortlist(None, object(), 16, "packed")

    def test_invalid_groups_and_mode_reject_before_torch(self):
        with patch.dict(sys.modules, self.BLOCKED):
            for groups in (0, 7, 129, True, "16", None):
                with self.subTest(groups=groups):
                    with self.assertRaisesRegex(ValueError, "groups|integer"):
                        install_draft_shortlist(object(), object(), groups, "packed")
            with self.assertRaisesRegex(ValueError, "mode"):
                install_draft_shortlist(object(), object(), 16, "bogus")

    def test_guard_blocks_post_validation_import(self):
        with patch.dict(sys.modules, self.BLOCKED):
            with self.assertRaises(ImportError):
                install_draft_shortlist(object(), object(), 16, "packed")


if __name__ == "__main__":
    unittest.main()
