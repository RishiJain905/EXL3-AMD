"""Smoke-runner execution boundaries; no third-party imports or GPU work."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "scripts/smoke_exl3_runtime.py"
BOOTSTRAP = """
import importlib.abc
from pathlib import Path
# Python 3.12 platform probes optional Windows stdlib modules even on Linux.
# Resolve that stdlib probe before installing the third-party import sentinel.
import platform
import runpy
import sys
marker = Path(sys.argv.pop(1))
class RejectThirdParty(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in sys.stdlib_module_names:
            marker.write_text(fullname, encoding='utf-8')
            raise RuntimeError('Third-party import attempted: ' + fullname)
sys.meta_path.insert(0, RejectThirdParty())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


class Exl3SmokePermissionTests(unittest.TestCase):
    def run_smoke(self, root, execute=True):
        command = [sys.executable, "-S", "-c", BOOTSTRAP,
                   str(root / "import-attempt"), str(RUNNER),
                   "--config", str(root / "config.toml"),
                   "--extension-dir", str(root / "missing-extension"),
                   "--source-dir", str(root / "missing-source"),
                   "--output", str(root / "output.jsonl")]
        if execute:
            command.append("--execute")
        return subprocess.run(command, capture_output=True, text=True, timeout=10)

    def test_denied_execution_precedes_third_party_imports(self):
        cases = [
            (False, "allow_backend_probes = true\nallow_local_inference = true\n"),
            (True, ""),
            (True, "allow_backend_probes = false\nallow_local_inference = true\n"),
            (True, "allow_backend_probes = true\nallow_local_inference = false\n"),
            (True, 'allow_backend_probes = "true"\nallow_local_inference = true\n'),
            (True, 'allow_backend_probes = true\nallow_local_inference = "true"\n'),
        ]
        for execute, settings in cases:
            with self.subTest(execute=execute, settings=settings), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "config.toml").write_text("[execution]\n" + settings, encoding="utf-8")
                result = self.run_smoke(root, execute)
                self.assertEqual(result.returncode, 1, result.stderr)
                records = [json.loads(line) for line in (root / "output.jsonl").read_text().splitlines()]
                self.assertEqual([record["stage"] for record in records], ["started", "failed"])
                self.assertEqual(records[-1]["error"], "Requires --execute and both local execution permissions")
                self.assertFalse((root / "import-attempt").exists(), result.stderr)

    def test_existing_output_is_preserved_before_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = b"prior evidence\n\x00\xff"
            output = root / "output.jsonl"
            output.write_bytes(original)
            # No config: exclusive output creation must fail even before config access.
            result = self.run_smoke(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(output.read_bytes(), original)
            self.assertFalse((root / "import-attempt").exists(), result.stderr)


if __name__ == "__main__":
    unittest.main()
