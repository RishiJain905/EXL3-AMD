"""Execution must fail before runtime import or artifact creation on denied gates."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_exl3_layer.py"


class Exl3LayerBoundaryTests(unittest.TestCase):
    def test_compact_reload_rejected_before_input_or_runtime_access(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--config", "absent.toml",
                                 "--runtime", "absent", "--output", "absent-output",
                                 "--mode", "reload", "--compact-cpu-buffers"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("only valid with --mode encode", result.stderr)

    def test_compact_requires_patched_source_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "local.toml"
            config.write_text("[execution]\nallow_backend_probes=true\nallow_local_inference=true\n")
            result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config),
                                     "--runtime", str(root / "unpatched"), "--output", str(root / "out"),
                                     "--mode", "encode", "--compact-cpu-buffers", "--execute"],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires the patched quantizer source", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_denied_gates_do_not_import_runtime_or_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "local.toml"
            for probe, inference, execute in [(True, True, False), (False, True, True), (True, False, True)]:
                config.write_text(f"[execution]\nallow_backend_probes={str(probe).lower()}\nallow_local_inference={str(inference).lower()}\n")
                command = [sys.executable, str(SCRIPT), "--config", str(config), "--runtime", str(root / "absent"),
                           "--output", str(root / "out"), "--mode", "encode"]
                if execute:
                    command.append("--execute")
                result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("configured local permissions", result.stderr)
                self.assertFalse((root / "out").exists())

    def test_existing_output_preserved_before_runtime_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "local.toml"
            config.write_text("[execution]\nallow_backend_probes=true\nallow_local_inference=true\n")
            output = root / "out"
            output.mkdir()
            sentinel = output / "retained"
            sentinel.write_bytes(b"old evidence")
            result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config), "--runtime", str(root / "absent"),
                                     "--output", str(output), "--mode", "encode", "--execute"],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(list(output.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_bytes(), b"old evidence")
