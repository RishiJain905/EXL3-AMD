"""The public kernel validator must retain registration, hash and lease guards."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_exl3_kernels as validator


class KernelValidatorTests(unittest.TestCase):
    def test_binary_hash_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            binary = directory / 'exllamav3_ext.so'
            binary.write_bytes(b'fixture, never loaded')
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            self.assertEqual(validator.verified_binary(directory, digest.upper()), binary)
            for bad in ('0' * 64, 'invalid', None):
                with self.subTest(expected=bad), self.assertRaises(ValueError):
                    validator.verified_binary(directory, bad)
            (directory / 'exllamav3_ext.second.so').write_bytes(b'another')
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                validator.verified_binary(directory, digest)

    def test_missing_permission_stops_before_native_loading(self):
        for permissions in ('', 'allow_backend_probes = true', 'allow_local_inference = true',
                            'allow_backend_probes = "true"\nallow_local_inference = true'):
            with self.subTest(permissions=permissions), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = root / 'config.toml'
                config.write_text('[execution]\n' + permissions)
                with patch.object(sys, 'platform', 'linux'), patch.object(validator, 'verified_binary') as verified, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    validator.main(['--config', str(config), '--output', str(root / 'results')])
                verified.assert_not_called()
                self.assertFalse((root / 'results').exists())

    def test_candidate_requires_hash_pair(self):
        for extra in (['--extension-dir', 'candidate'], ['--expected-extension-sha256', '0' * 64],
                      ['--wmma-probe', 'probe'], ['--expected-wmma-sha256', '0' * 64]):
            with patch.object(sys, 'platform', 'linux'), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                validator.main(['--output', 'unused', *extra])

    def test_supervisor_holds_shared_lease_and_bounds_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'exllamav3_ext.so'
            binary.write_bytes(b'fixture, never loaded')
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            lock = root / 'gpu.lock'
            config = root / 'config.toml'
            config.write_text('\n'.join([
                '[execution]', 'allow_backend_probes=true', 'allow_local_inference=true',
                '[limits]', 'max_capture_seconds=1200', '[runtime]',
                "python='/usr/bin/python3'", "sdk='/sdk'", "torch_lib='/torch'", "hsa_preload='/hsa.so'",
                "gpu_arch='gfx1201'", f"extension_dir='{root.as_posix()}'",
                f"extension_sha256='{digest}'", f"lease_file='{lock.as_posix()}'",
            ]))
            def worker(request_file):
                self.assertTrue(lock.exists())
                request = json.loads(request_file.read_text())
                self.assertEqual(request['timeout'], 900)
                self.assertEqual(request['expected_sha256'], digest)
                self.assertEqual(request['runtime']['gpu_arch'], 'gfx1201')
                return 7  # Preserve failures from the monitored child.
            # Keep temp paths native in this CPU-only cross-platform fixture.
            with patch.object(sys, 'platform', 'linux'), \
                    patch.object(validator.launcher, 'linux_path', side_effect=str), \
                    patch.object(validator.launcher, 'worker', side_effect=worker):
                self.assertEqual(validator.main(['--config', str(config), '--output', str(root / 'results')]), 7)
            self.assertFalse(lock.exists())


if __name__ == '__main__':
    unittest.main()
