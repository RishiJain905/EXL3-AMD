"""Tests for checkpoint rename helper (stdlib only, no torch)."""

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HELPER_PATH = (
    Path(__file__).resolve().parent.parent
    / "vendor"
    / "rocm-exl3"
    / "exllamav3"
    / "conversion"
    / "checkpoint_io.py"
)


def _load_helper():
    spec = importlib.util.spec_from_file_location("checkpoint_io_under_test", _HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helper = _load_helper()
rename_checkpoint_dir = _helper.rename_checkpoint_dir

EXPECTED_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _write_bytes(path, data):
    with open(path, "wb") as handle:
        handle.write(data)


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


class RenameCheckpointDirTests(unittest.TestCase):
    def test_retry_delays_contract(self):
        self.assertEqual(_helper.RETRY_DELAYS, EXPECTED_DELAYS)
        self.assertEqual(len(_helper.RETRY_DELAYS), 10)
        self.assertEqual(len(_helper.RETRY_DELAYS) + 1, 11)
        self.assertAlmostEqual(sum(_helper.RETRY_DELAYS), 7.5)

    def test_ordinary_rename_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            os.mkdir(os.path.join(src, "sub"))
            _write_bytes(os.path.join(src, "a.safetensors"), b"ckpt-bytes-12345")
            _write_bytes(os.path.join(src, "sub", "b.json"), b'{"step": 3}')
            with mock.patch("builtins.print") as mprint:
                rename_checkpoint_dir(src, dst)
            mprint.assert_not_called()
            self.assertFalse(os.path.exists(src))
            self.assertTrue(os.path.isdir(dst))
            self.assertEqual(_read_bytes(os.path.join(dst, "a.safetensors")), b"ckpt-bytes-12345")
            self.assertEqual(_read_bytes(os.path.join(dst, "sub", "b.json")), b'{"step": 3}')
            self.assertEqual(sorted(os.listdir(parent)), ["ckpt"])

    def test_transient_permission_error_retries_then_succeeds(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            _write_bytes(os.path.join(src, "state.safetensors"), b"quant-block-bytes")
            real_rename = os.rename
            calls = []

            def fake_rename(first, second):
                calls.append((first, second))
                if len(calls) <= 2:
                    raise PermissionError("locked")
                return real_rename(first, second)

            with (
                mock.patch("os.rename", side_effect=fake_rename),
                mock.patch("time.sleep") as msleep,
                mock.patch("builtins.print") as mprint,
            ):
                rename_checkpoint_dir(src, dst)
            self.assertEqual(len(calls), 3)
            for first, second in calls:
                self.assertEqual((first, second), (src, dst))
            self.assertEqual([call.args[0] for call in msleep.call_args_list], [0.1, 0.2])
            self.assertEqual(mprint.call_count, 3)
            retry_text = mprint.call_args_list[0].args[0]
            self.assertIn("attempt 1", retry_text)
            self.assertIn("0.1s", retry_text)
            success_text = mprint.call_args_list[-1].args[0]
            self.assertIn("after 2 retries", success_text)
            self.assertFalse(os.path.exists(src))
            self.assertEqual(
                _read_bytes(os.path.join(dst, "state.safetensors")), b"quant-block-bytes"
            )
            self.assertEqual(sorted(os.listdir(parent)), ["ckpt"])

    def test_exhaustion_exactly_11_calls_10_sleeps(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            _write_bytes(os.path.join(src, "job.json"), b'{"idx": 7}')
            with (
                mock.patch("os.rename", side_effect=PermissionError("locked")) as mrename,
                mock.patch("time.sleep") as msleep,
                mock.patch("builtins.print") as mprint,
            ):
                with self.assertRaises(PermissionError):
                    rename_checkpoint_dir(src, dst)
            self.assertEqual(mrename.call_count, 11)
            self.assertEqual(msleep.call_count, 10)
            self.assertEqual(
                [call.args[0] for call in msleep.call_args_list], list(EXPECTED_DELAYS)
            )
            self.assertEqual(mprint.call_count, 10)
            self.assertTrue(os.path.isdir(src))
            self.assertFalse(os.path.lexists(dst))
            self.assertEqual(_read_bytes(os.path.join(src, "job.json")), b'{"idx": 7}')

    def test_unrelated_oserror_no_retry(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            with (
                mock.patch("os.rename", side_effect=OSError("boom")) as mrename,
                mock.patch("time.sleep") as msleep,
                mock.patch("builtins.print") as mprint,
            ):
                with self.assertRaises(OSError):
                    rename_checkpoint_dir(src, dst)
            self.assertEqual(mrename.call_count, 1)
            msleep.assert_not_called()
            mprint.assert_not_called()
            self.assertTrue(os.path.isdir(src))
            self.assertFalse(os.path.lexists(dst))

    def test_existing_target_not_overwritten(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            os.mkdir(src)
            _write_bytes(os.path.join(src, "job.json"), b"new-bytes")
            for name in ("ckpt", "ckpt_file"):
                target = os.path.join(parent, name)
                if name == "ckpt":
                    os.mkdir(target)
                    _write_bytes(os.path.join(target, "job.json"), b"old-bytes")
                else:
                    _write_bytes(target, b"old-file-bytes")
                with self.subTest(target=name):
                    with (
                        mock.patch("os.rename") as mrename,
                        mock.patch("time.sleep") as msleep,
                    ):
                        with self.assertRaises(FileExistsError):
                            rename_checkpoint_dir(src, target)
                    mrename.assert_not_called()
                    msleep.assert_not_called()
                    self.assertTrue(os.path.isdir(src))
                    self.assertEqual(
                        _read_bytes(os.path.join(src, "job.json")), b"new-bytes"
                    )
                    if name == "ckpt":
                        self.assertEqual(
                            _read_bytes(os.path.join(target, "job.json")), b"old-bytes"
                        )
                    else:
                        self.assertEqual(_read_bytes(target), b"old-file-bytes")

    def test_dangling_symlink_target_not_overwritten(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            _write_bytes(os.path.join(src, "job.json"), b"new-bytes")
            try:
                os.symlink("nonexistent-target-12345", dst)
            except OSError as exc:
                self.skipTest(f"symlink creation not permitted: {exc}")
            with (
                mock.patch("os.rename") as mrename,
                mock.patch("time.sleep") as msleep,
            ):
                with self.assertRaises(FileExistsError):
                    rename_checkpoint_dir(src, dst)
            mrename.assert_not_called()
            msleep.assert_not_called()
            self.assertTrue(os.path.lexists(dst))
            self.assertFalse(os.path.exists(dst))
            self.assertEqual(_read_bytes(os.path.join(src, "job.json")), b"new-bytes")

    def test_source_absent_stops_immediately(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            with (
                mock.patch("os.rename") as mrename,
                mock.patch("time.sleep") as msleep,
            ):
                with self.assertRaises(FileNotFoundError):
                    rename_checkpoint_dir(src, dst)
            mrename.assert_not_called()
            msleep.assert_not_called()
            self.assertFalse(os.path.lexists(dst))

    def test_target_appearing_after_error_stops_immediately(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            _write_bytes(os.path.join(src, "job.json"), b"new-bytes")

            def fake_rename(first, second):
                os.mkdir(second)
                _write_bytes(os.path.join(second, "sentinel.json"), b"other-writer")
                raise PermissionError("locked")

            with (
                mock.patch("os.rename", side_effect=fake_rename) as mrename,
                mock.patch("time.sleep") as msleep,
                mock.patch("builtins.print") as mprint,
            ):
                with self.assertRaises(PermissionError):
                    rename_checkpoint_dir(src, dst)
            self.assertEqual(mrename.call_count, 1)
            msleep.assert_not_called()
            mprint.assert_not_called()
            self.assertTrue(os.path.isdir(src))
            self.assertEqual(_read_bytes(os.path.join(src, "job.json")), b"new-bytes")
            self.assertEqual(
                _read_bytes(os.path.join(dst, "sentinel.json")), b"other-writer"
            )
            self.assertFalse(os.path.exists(os.path.join(dst, "job.json")))

    def test_source_absent_after_error_stops_immediately(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            _write_bytes(os.path.join(src, "job.json"), b"new-bytes")

            def fake_rename(first, second):
                shutil.rmtree(first)
                raise PermissionError("locked")

            with (
                mock.patch("os.rename", side_effect=fake_rename) as mrename,
                mock.patch("time.sleep") as msleep,
                mock.patch("builtins.print") as mprint,
            ):
                with self.assertRaises(PermissionError):
                    rename_checkpoint_dir(src, dst)
            self.assertEqual(mrename.call_count, 1)
            msleep.assert_not_called()
            mprint.assert_not_called()
            self.assertFalse(os.path.lexists(src))
            self.assertFalse(os.path.lexists(dst))

    def test_different_parent_rejected(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            src = os.path.join(first, "ckpt_new")
            dst = os.path.join(second, "ckpt")
            os.mkdir(src)
            with (
                mock.patch("os.rename") as mrename,
                mock.patch("time.sleep") as msleep,
            ):
                with self.assertRaises(ValueError):
                    rename_checkpoint_dir(src, dst)
            mrename.assert_not_called()
            msleep.assert_not_called()
            self.assertTrue(os.path.isdir(src))
            self.assertFalse(os.path.lexists(dst))

    def test_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            real = os.path.join(parent, "real_ckpt")
            os.mkdir(real)
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            try:
                if os.name == "nt":
                    os.symlink(real, src, target_is_directory=True)
                else:
                    os.symlink(real, src)
            except OSError as exc:
                self.skipTest(f"symlink creation not permitted: {exc}")
            with (
                mock.patch("os.rename") as mrename,
                mock.patch("time.sleep") as msleep,
            ):
                with self.assertRaises(NotADirectoryError):
                    rename_checkpoint_dir(src, dst)
            mrename.assert_not_called()
            msleep.assert_not_called()
            self.assertTrue(os.path.islink(src))
            self.assertFalse(os.path.lexists(dst))

    def test_source_file_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            _write_bytes(src, b"not-a-dir")
            with (
                mock.patch("os.rename") as mrename,
                mock.patch("time.sleep") as msleep,
            ):
                with self.assertRaises(NotADirectoryError):
                    rename_checkpoint_dir(src, dst)
            mrename.assert_not_called()
            msleep.assert_not_called()
            self.assertEqual(_read_bytes(src), b"not-a-dir")
            self.assertFalse(os.path.lexists(dst))

    def test_destination_appearing_during_backoff_not_replaced(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            _write_bytes(os.path.join(src, "job.json"), b"new-bytes")

            def on_sleep(delay):
                os.mkdir(dst)
                _write_bytes(os.path.join(dst, "sentinel.json"), b"other-writer")

            with (
                mock.patch("os.rename", side_effect=PermissionError("locked")) as mrename,
                mock.patch("time.sleep", side_effect=on_sleep) as msleep,
                mock.patch("builtins.print") as mprint,
            ):
                with self.assertRaises(FileExistsError):
                    rename_checkpoint_dir(src, dst)
            self.assertEqual(mrename.call_count, 1)
            self.assertEqual(msleep.call_count, 1)
            self.assertEqual(msleep.call_args_list[0].args[0], 0.1)
            self.assertEqual(mprint.call_count, 1)
            self.assertTrue(os.path.isdir(src))
            self.assertEqual(_read_bytes(os.path.join(src, "job.json")), b"new-bytes")
            self.assertEqual(
                _read_bytes(os.path.join(dst, "sentinel.json")), b"other-writer"
            )
            self.assertFalse(os.path.exists(os.path.join(dst, "job.json")))

    def test_source_replaced_by_file_during_backoff_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            _write_bytes(os.path.join(src, "job.json"), b"new-bytes")

            def on_sleep(delay):
                shutil.rmtree(src)
                _write_bytes(src, b"not-a-dir-now")

            with (
                mock.patch("os.rename", side_effect=PermissionError("locked")) as mrename,
                mock.patch("time.sleep", side_effect=on_sleep) as msleep,
                mock.patch("builtins.print") as mprint,
            ):
                with self.assertRaises(NotADirectoryError):
                    rename_checkpoint_dir(src, dst)
            self.assertEqual(mrename.call_count, 1)
            self.assertEqual(msleep.call_count, 1)
            self.assertEqual(mprint.call_count, 1)
            self.assertTrue(os.path.isfile(src))
            self.assertFalse(os.path.isdir(src))
            self.assertEqual(_read_bytes(src), b"not-a-dir-now")
            self.assertFalse(os.path.lexists(dst))

    def test_source_replaced_by_symlink_during_backoff_rejected(self):
        with tempfile.TemporaryDirectory() as parent:
            real = os.path.join(parent, "real_ckpt")
            os.mkdir(real)
            probe = os.path.join(parent, "probe_link")
            try:
                if os.name == "nt":
                    os.symlink(real, probe, target_is_directory=True)
                else:
                    os.symlink(real, probe)
            except OSError as exc:
                self.skipTest(f"symlink creation not permitted: {exc}")
            os.unlink(probe)
            src = os.path.join(parent, "ckpt_new")
            dst = os.path.join(parent, "ckpt")
            os.mkdir(src)
            _write_bytes(os.path.join(src, "job.json"), b"new-bytes")

            def on_sleep(delay):
                shutil.rmtree(src)
                if os.name == "nt":
                    os.symlink(real, src, target_is_directory=True)
                else:
                    os.symlink(real, src)

            with (
                mock.patch("os.rename", side_effect=PermissionError("locked")) as mrename,
                mock.patch("time.sleep", side_effect=on_sleep) as msleep,
                mock.patch("builtins.print") as mprint,
            ):
                with self.assertRaises(NotADirectoryError):
                    rename_checkpoint_dir(src, dst)
            self.assertEqual(mrename.call_count, 1)
            self.assertEqual(msleep.call_count, 1)
            self.assertEqual(mprint.call_count, 1)
            self.assertTrue(os.path.islink(src))
            self.assertFalse(os.path.lexists(dst))


if __name__ == "__main__":
    unittest.main()
