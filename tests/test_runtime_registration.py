"""CPU-only runtime registration tests; no installs, builds, GPU, or network."""
import contextlib
import importlib.util
import io
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("register_under_test", SCRIPTS / "register_runtime.py")
register = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(register)

VALID_SHA = "ab" * 32


def _source_text(**overrides):
    infer = overrides.get("infer", True)
    probes = overrides.get("probes", True)
    lines = ["[execution]"]
    if infer is not None:
        lines.append(f"allow_local_inference = {'true' if infer else 'false'}")
    if probes is not None:
        lines.append(f"allow_backend_probes = {'true' if probes else 'false'}")
    lines += ["", "[limits]", "max_capture_seconds = 60",
              "", "[runtime]", "distribution = 'Ubuntu'",
              "python = '/opt/rocm-venv/bin/python'", "sdk = '/opt/rocm'",
              "torch_lib = '/opt/rocm-venv/lib/torch'", "hsa_preload = '/opt/rocm/lib/libhsa.so'",
              "source_dir = 'vendor/rocm-exl3'", "extension_dir = '.runtime/lib'",
              f"extension_sha256 = '{overrides.get('sha', VALID_SHA)}'",
              "gpu_arch = 'gfx1101'", "native_smallm_max_rows = 9",
              "lease_file = 'artifacts/locks/local-gpu.lock'",
              "server_deps = '.runtime/server-deps'",
              "candidate = '/models/private-exl3'",
              "candidate_manifest = 'configs/model-manifest.local.json'",
              "", "[paths]", "scratch = '/tmp/scratch'",
              "", "[model]", "name = 'private-model'",
              "", "[targets]", "primary = 'cuda:0'"]
    return "\n".join(lines) + "\n"


class RegistrationTests(unittest.TestCase):
    def _register(self, source, output):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = register.main([str(source), "--output", str(output)])
        self.assertEqual(code, 0)
        return tomllib.loads(output.read_text(encoding="utf-8"))

    def _assert_rejects(self, argv, output):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                register.main(argv)
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(output.exists(), "rejected registration must not write output")

    def test_model_and_private_sections_stripped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "local.toml"
            before = _source_text()
            source.write_text(before, encoding="utf-8")
            output = tmp / "installation.toml"
            installed = self._register(source, output)
            after = source.read_text(encoding="utf-8")
        self.assertEqual(set(installed), {"execution", "limits", "runtime"})
        runtime = installed["runtime"]
        self.assertNotIn("candidate", runtime)
        self.assertNotIn("candidate_manifest", runtime)
        self.assertEqual(runtime["extension_sha256"], VALID_SHA)
        self.assertEqual(runtime["native_smallm_max_rows"], 9)
        self.assertEqual(after, before)

    def test_enabled_permissions_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "local.toml"
            source.write_text(_source_text(), encoding="utf-8")
            installed = self._register(source, tmp / "installation.toml")
        self.assertIs(installed["execution"]["allow_local_inference"], True)
        self.assertIs(installed["execution"]["allow_backend_probes"], True)

    def test_disabled_and_missing_permissions_stay_false(self):
        variants = [dict(infer=False, probes=True), dict(infer=True, probes=False),
                    dict(infer=False, probes=False),
                    dict(infer=None, probes=None)]
        for kwargs in variants:
            with self.subTest(**kwargs):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    source = tmp / "local.toml"
                    source.write_text(_source_text(**kwargs), encoding="utf-8")
                    installed = self._register(source, tmp / "installation.toml")
                self.assertIs(installed["execution"]["allow_local_inference"],
                              kwargs["infer"] is True)
                self.assertIs(installed["execution"]["allow_backend_probes"],
                              kwargs["probes"] is True)

    def test_default_output_registers_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "local.toml"
            source.write_text(_source_text(), encoding="utf-8")
            with patch.object(register, "ROOT", root), \
                    patch.object(register, "DEFAULT_OUTPUT",
                                 root / ".runtime" / "installation.toml"), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(register.main([str(source)]), 0)
            installed = tomllib.loads((root / ".runtime" / "installation.toml").read_text(
                encoding="utf-8"))
        self.assertEqual(installed["runtime"]["distribution"], "Ubuntu")

    def test_exclusive_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "local.toml"
            source.write_text(_source_text(), encoding="utf-8")
            output = tmp / "installation.toml"
            output.write_text("[runtime]\n", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    register.main([str(source), "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "[runtime]\n")

    def test_invalid_sources_reject_without_output(self):
        cases = {
            "malformed": "[runtime\n",
            "missing_runtime": "[execution]\nallow_local_inference = true\n",
            "short_hash": _source_text(sha="abc"),
            "nonhex_hash": _source_text(sha="zz" * 32),
            "missing_arch": _source_text().replace("gpu_arch = 'gfx1101'\n", ""),
            "bad_rows": _source_text().replace("native_smallm_max_rows = 9", "native_smallm_max_rows = true"),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    source = tmp / "local.toml"
                    source.write_text(text, encoding="utf-8")
                    self._assert_rejects([str(source), "--output", str(tmp / "out.toml")],
                                         tmp / "out.toml")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._assert_rejects([str(tmp / "missing.toml"), "--output",
                                  str(tmp / "out.toml")], tmp / "out.toml")

    def test_invalid_row_count_rejects(self):
        for rows in ("2", "4", "6", "10", "'5'"):
            with self.subTest(rows=rows):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    source = tmp / "local.toml"
                    text = _source_text().replace("native_smallm_max_rows = 9",
                                                  f"native_smallm_max_rows = {rows}")
                    source.write_text(text, encoding="utf-8")
                    self._assert_rejects([str(source), "--output", str(tmp / "out.toml")],
                                         tmp / "out.toml")


if __name__ == "__main__":
    unittest.main()
