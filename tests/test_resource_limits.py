"""CPU-only resource policy and monitor tests; no WSL, GPU, model files, or network."""
import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import exl3_resources as resources

_spec = importlib.util.spec_from_file_location("launcher_under_test", ROOT / "scripts" / "launch_runtime.py")
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)

_eval_spec = importlib.util.spec_from_file_location("evaluate_under_test", ROOT / "scripts" / "evaluate_exl3_candidate.py")
evaluate = importlib.util.module_from_spec(_eval_spec)
_eval_spec.loader.exec_module(evaluate)

_serve_spec = importlib.util.spec_from_file_location("serve_under_test", ROOT / "scripts" / "serve_exl3.py")
serve = importlib.util.module_from_spec(_serve_spec)
_serve_spec.loader.exec_module(serve)

GIB = 2 ** 30


class ValidationTests(unittest.TestCase):
    def test_fraction_accepts_interior_values(self):
        for value in (0.01, 0.25, 0.5, 0.9, 0.95, 0.99):
            self.assertEqual(resources.validate_fraction(value, "--flag"), value)

    def test_fraction_rejects_boundaries(self):
        for value in (0.0, 1.0, -0.1, 1.5, 2.0):
            with self.assertRaises(ValueError, msg=value):
                resources.validate_fraction(value, "--flag")

    def test_fraction_rejects_nonfinite(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError, msg=value):
                resources.validate_fraction(value, "--flag")

    def test_fraction_rejects_nonnumeric(self):
        for value in (None, "x", object()):
            with self.assertRaises(ValueError, msg=value):
                resources.validate_fraction(value, "--flag")

    def test_gib_accepts_positive(self):
        for value in (0.25, 0.5, 1.0, 4.0, 120.0):
            self.assertEqual(resources.validate_positive_gib(value, "--flag"), value)

    def test_gib_rejects_nonpositive(self):
        for value in (0.0, -1.0, -0.5):
            with self.assertRaises(ValueError, msg=value):
                resources.validate_positive_gib(value, "--flag")

    def test_gib_rejects_nonfinite_and_nonnumeric(self):
        for value in (math.nan, math.inf, -math.inf, None, "x", object()):
            with self.assertRaises(ValueError, msg=value):
                resources.validate_positive_gib(value, "--flag")


class DerivedSizeTests(unittest.TestCase):
    def test_public_defaults_unchanged(self):
        self.assertEqual(resources.GPU_MEMORY_FRACTION_DEFAULT, 0.90)
        self.assertEqual(resources.MAX_GPU_MEMORY_FRACTION_DEFAULT, 0.95)
        self.assertEqual(resources.MAX_HOST_MEMORY_FRACTION_DEFAULT, 0.50)
        self.assertEqual(resources.MIN_FREE_RAM_GIB_DEFAULT, 1.0)
        self.assertEqual(resources.MIN_FREE_DISK_GIB_DEFAULT, 1.0)

    def test_allocator_scales_with_device_total(self):
        for total_gib in (4, 8, 16, 24, 48):
            for fraction in (0.5, 0.75, 0.9):
                self.assertEqual(resources.allocator_bytes(fraction, total_gib * GIB),
                                 int(fraction * total_gib * GIB))

    def test_rss_limit_scales_with_memtotal(self):
        for total_gib in (4, 8, 16, 32, 64):
            for fraction in (0.25, 0.5, 0.8):
                self.assertEqual(resources.rss_limit_bytes(fraction, total_gib * GIB),
                                 int(fraction * total_gib * GIB))

    def test_free_bytes_converts_gib(self):
        self.assertEqual(resources.free_bytes(1.0), GIB)
        self.assertEqual(resources.free_bytes(0.5), GIB // 2)
        self.assertEqual(resources.free_bytes(2.5), int(2.5 * GIB))

    def test_limits_from_args_roundtrip(self):
        args = SimpleNamespace(max_host_memory_fraction=0.4, max_gpu_memory_fraction=0.8,
                               min_free_ram_gib=2.0, min_free_disk_gib=3.0)
        self.assertEqual(resources.limits_from_args(args),
                         {"max_host_memory_fraction": 0.4, "max_gpu_memory_fraction": 0.8,
                          "min_free_ram_gib": 2.0, "min_free_disk_gib": 3.0})

    def test_wsl_limits_old_request_defaults(self):
        for request in ({}, {"run": "x"}, {"limits": None}, {"limits": "nope"}):
            self.assertEqual(resources.wsl_limits(request),
                             {"max_host_memory_fraction": 0.50, "max_gpu_memory_fraction": 0.95,
                              "min_free_ram_gib": 1.0, "min_free_disk_gib": 1.0})

    def test_wsl_limits_preserves_valid_saved_values(self):
        saved = {"max_host_memory_fraction": 0.3, "max_gpu_memory_fraction": 0.7,
                 "min_free_ram_gib": 2.0, "min_free_disk_gib": 5.0}
        self.assertEqual(resources.wsl_limits({"limits": saved}), saved)

    def test_wsl_limits_sanitizes_bad_saved_values(self):
        saved = {"max_host_memory_fraction": 5.0, "max_gpu_memory_fraction": 0.0,
                 "min_free_ram_gib": -1.0, "min_free_disk_gib": math.nan}
        self.assertEqual(resources.wsl_limits({"limits": saved}),
                         {"max_host_memory_fraction": 0.50, "max_gpu_memory_fraction": 0.95,
                          "min_free_ram_gib": 1.0, "min_free_disk_gib": 1.0})

    def test_parse_meminfo_variants(self):
        total, available = resources.parse_meminfo("MemTotal: 8192 kB\nMemAvailable: 4096 kB\n")
        self.assertEqual((total, available), (8192 * 1024, 4096 * 1024))
        self.assertEqual(resources.parse_meminfo("MemAvailable: 4096 kB\n"), (None, 4096 * 1024))
        self.assertEqual(resources.parse_meminfo("MemTotal: 8192 kB\n"), (8192 * 1024, None))
        self.assertEqual(resources.parse_meminfo(""), (None, None))
        self.assertEqual(resources.parse_meminfo("MemTotal: nope kB\nMemAvailable: kB\n"), (None, None))


class AllocatorForwardingTests(unittest.TestCase):
    def test_launcher_parser_resource_defaults(self):
        args = launch.parser().parse_args([])
        self.assertEqual(args.gpu_memory_fraction, 0.90)
        self.assertEqual(args.max_gpu_memory_fraction, 0.95)
        self.assertEqual(args.max_host_memory_fraction, 0.50)
        self.assertEqual(args.min_free_ram_gib, 1.0)
        self.assertEqual(args.min_free_disk_gib, 1.0)
        self.assertIsNone(args.telemetry_gpu)

    def test_evaluate_and_serve_expose_allocator_flag(self):
        cases = [
            (evaluate, ["--config", "c", "--candidate", "d", "--source-dir", "s",
                        "--extension-dir", "e", "--suite", "t", "--output", "o",
                        "--expected-extension-sha256", "ab", "--mode", "speed"]),
            (serve, ["--config", "c", "--candidate", "d", "--source-dir", "s",
                    "--extension-dir", "e", "--output", "o",
                    "--expected-extension-sha256", "ab"]),
        ]
        for module, argv in cases:
            with self.subTest(module=module.__name__):
                args = module.parser().parse_args(argv)
                self.assertEqual(args.gpu_memory_fraction, 0.90)

    def test_telemetry_gpu_help_selects_monitoring_not_torch(self):
        text = " ".join(launch.parser().format_help().split())
        self.assertIn("--telemetry-gpu", text)
        self.assertIn("not the Torch device", text)


def _runtime(**overrides):
    base = {"sdk": "/opt/rocm", "hsa_preload": "/opt/rocm/lib/libhsa-runtime64.so",
            "gpu_arch": "gfx942", "python": "/opt/rocm/bin/python",
            "torch_lib": "/opt/torch/lib"}
    base.update(overrides)
    return base


class WslMonitorTests(unittest.TestCase):
    def _worker_run(self, timeout, *, elapsed, rss=1024, mem_gib=8, disk_gib=200,
                    iters=1, pre_stop=False, mode=None, total_gib=16, limits=None,
                    meminfo=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run = Path(temporary.name) / "run"
        run.mkdir()
        (run / "result").mkdir()
        request = dict(run=str(run), root="/fake", runtime=_runtime(), argv=["fake"],
                       mode=mode or ("serve" if timeout is None else "speed"), timeout=timeout)
        if limits is not None:
            request["limits"] = dict(limits)
        req = run / "request.json"
        req.write_text(json.dumps(request), encoding="utf-8")
        if pre_stop:
            (run / "stop").touch()
        child = MagicMock()
        child.pid = 4321
        child.poll.side_effect = [None] * iters + [0, 0]
        child.wait.return_value = 0
        child.returncode = 0
        meminfo = meminfo if meminfo is not None else (
            f"MemTotal: {total_gib * 1024 * 1024} kB\n"
            f"MemAvailable: {mem_gib * 1024 * 1024} kB\n")
        real_read = Path.read_text

        def fake_read(self, *a, **k):
            if "meminfo" in str(self):
                return meminfo
            return real_read(self, *a, **k)

        disk = MagicMock()
        disk.free = disk_gib * GIB
        monotonic = [1000.0] + [1000.0 + elapsed] * (iters + 2)
        with patch("subprocess.Popen", return_value=child), \
                patch.object(launch, "group_rss", return_value=rss), \
                patch.object(launch, "runtime_environment", return_value={}), \
                patch.object(launch.time, "monotonic", side_effect=monotonic), \
                patch.object(launch.time, "sleep"), \
                patch("shutil.disk_usage", return_value=disk), \
                patch.object(Path, "read_text", fake_read), \
                patch.object(launch, "terminate_group") as term:
            code = launch.worker(str(req))
        monitor = json.loads((run / "monitor.json").read_text(encoding="utf-8"))
        return code, monitor, term

    def test_group_rss_threshold_scales_with_memtotal(self):
        for total_gib in (4, 8, 32, 64):
            limit = int(0.50 * total_gib * GIB)
            with self.subTest(total_gib=total_gib):
                _, monitor, term = self._worker_run(60, elapsed=0, rss=limit - GIB,
                                                    total_gib=total_gib, mem_gib=total_gib // 2)
                term.assert_not_called()
                self.assertIsNone(monitor["stop_reason"])
                _, monitor, term = self._worker_run(60, elapsed=0, rss=limit + 1,
                                                    total_gib=total_gib, mem_gib=total_gib // 2)
                term.assert_called_once()
                self.assertEqual(monitor["stop_reason"], "group_rss")

    def test_custom_host_fraction_threshold(self):
        limits = {"max_host_memory_fraction": 0.25, "max_gpu_memory_fraction": 0.9,
                  "min_free_ram_gib": 1.0, "min_free_disk_gib": 1.0}
        _, monitor, term = self._worker_run(60, elapsed=0, rss=3 * GIB,
                                            total_gib=16, limits=limits)
        term.assert_not_called()
        self.assertIsNone(monitor["stop_reason"])
        _, monitor, term = self._worker_run(60, elapsed=0, rss=5 * GIB,
                                            total_gib=16, limits=limits)
        term.assert_called_once()
        self.assertEqual(monitor["stop_reason"], "group_rss")

    def test_available_ram_guard(self):
        limits = {"max_host_memory_fraction": 0.9, "max_gpu_memory_fraction": 0.9,
                  "min_free_ram_gib": 2.0, "min_free_disk_gib": 1.0}
        _, monitor, term = self._worker_run(60, elapsed=0, mem_gib=1,
                                            total_gib=16, limits=limits)
        term.assert_called_once()
        self.assertEqual(monitor["stop_reason"], "wsl_available_ram")
        _, monitor, term = self._worker_run(60, elapsed=0, mem_gib=3,
                                            total_gib=16, limits=limits)
        term.assert_not_called()
        self.assertIsNone(monitor["stop_reason"])

    def test_artifact_disk_guard(self):
        limits = {"max_host_memory_fraction": 0.9, "max_gpu_memory_fraction": 0.9,
                  "min_free_ram_gib": 1.0, "min_free_disk_gib": 50.0}
        _, monitor, term = self._worker_run(60, elapsed=0, disk_gib=10, limits=limits)
        term.assert_called_once()
        self.assertEqual(monitor["stop_reason"], "artifact_disk")
        _, monitor, term = self._worker_run(60, elapsed=0, disk_gib=60, limits=limits)
        term.assert_not_called()
        self.assertIsNone(monitor["stop_reason"])

    def test_old_request_uses_portable_defaults(self):
        _, monitor, term = self._worker_run(60, elapsed=0, rss=5 * GIB, total_gib=8)
        term.assert_called_once()
        self.assertEqual(monitor["stop_reason"], "group_rss")
        _, monitor, term = self._worker_run(60, elapsed=0, rss=3 * GIB, total_gib=8)
        term.assert_not_called()
        self.assertIsNone(monitor["stop_reason"])

    def test_bad_saved_limits_fall_back_to_defaults(self):
        limits = {"max_host_memory_fraction": 5.0, "max_gpu_memory_fraction": 0.0,
                  "min_free_ram_gib": -1.0, "min_free_disk_gib": math.nan}
        _, monitor, term = self._worker_run(60, elapsed=0, rss=5 * GIB,
                                            total_gib=8, limits=limits)
        term.assert_called_once()
        self.assertEqual(monitor["stop_reason"], "group_rss")

    def test_missing_memtotal_disables_rss_stop(self):
        _, monitor, term = self._worker_run(60, elapsed=0, rss=100 * GIB,
                                            meminfo="MemAvailable: 8388608 kB\n")
        term.assert_not_called()
        self.assertIsNone(monitor["stop_reason"])
        self.assertIsNone(monitor["error"])


class _FakeTelemetry:
    def __init__(self, *args, **kwargs):
        self.gpu_total_dedicated_bytes = 0
        self._samples = []

    def start(self):
        pass

    def stop(self):
        pass

    def sample_now(self):
        return {}

    def summary(self):
        return {}

    def write_csv(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")


class WindowsAdapterTests(unittest.TestCase):
    def _main_run(self, mode, argv_extra, *, elapsed, sample, iters=1,
                  gpu_total_gib=16, disk_gib=200, collector=None):
        if os.name != "nt":
            self.skipTest("Windows monitor requires Windows")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp = Path(temporary.name)
        cfg = tmp / "local.toml"
        out = tmp / "run-out"
        cfg.write_text("\n".join(["[execution]", "allow_local_inference = true",
            "allow_backend_probes = true", "[runtime]", "python = '/fake/python'",
            "sdk = '/fake/sdk'", "torch_lib = '/fake/torch'", "hsa_preload = '/fake/preload.so'",
            "source_dir = '/fake/src'", "extension_dir = '/fake/ext'",
            "candidate = '/fake/candidate'", "distribution = 'fake'",
            "extension_sha256 = 'abc'", f"lease_file = '{tmp / 'lock'}'",
            "[limits]", "max_capture_seconds = 60"]) + "\n", encoding="utf-8")
        child = MagicMock()
        child.poll.side_effect = [None] * iters + [0, 0]
        child.wait.return_value = 0
        telemetry = collector if collector is not None else _FakeTelemetry()
        telemetry.gpu_total_dedicated_bytes = gpu_total_gib * GIB
        telemetry.sample_now = lambda: dict(sample)
        telemetry_module = ModuleType("quantlab.telemetry")
        telemetry_module.TelemetryCollector = (telemetry if callable(telemetry)
                                               else lambda *a, **k: telemetry)
        quantlab_package = ModuleType("quantlab")
        quantlab_package.__path__ = []
        disk = MagicMock()
        disk.free = disk_gib * GIB
        monotonic = [1000.0] + [1000.0 + elapsed] * (iters + 2)
        stdout = io.StringIO()
        argv = ["launch", mode, "--config", str(cfg), "--output", str(out),
                "--execute", *argv_extra]
        with patch.object(sys, "argv", argv), \
                patch("subprocess.Popen", return_value=child), \
                patch.dict(sys.modules, {"quantlab": quantlab_package,
                                         "quantlab.telemetry": telemetry_module}), \
                patch.object(launch.time, "monotonic", side_effect=monotonic), \
                patch.object(launch.time, "sleep"), \
                patch("shutil.disk_usage", return_value=disk), \
                patch.dict(os.environ, {"SystemDrive": "C:"}), \
                contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                launch.main()
        self.assertEqual(caught.exception.code, 0)
        request = json.loads((out / "request.json").read_text(encoding="utf-8"))
        return request, out, stdout.getvalue()

    def test_allocator_fraction_forwarded_into_child_argv(self):
        request, _, _ = self._main_run("generate", ["--prompt", "hi",
                                        "--gpu-memory-fraction", "0.75",
                                        "--max-gpu-memory-fraction", "0.8",
                                        "--max-host-memory-fraction", "0.4",
                                        "--min-free-ram-gib", "2",
                                        "--min-free-disk-gib", "3"],
                                       elapsed=0, sample={}, iters=0)
        argv = request["argv"]
        index = argv.index("--gpu-memory-fraction")
        self.assertEqual(argv[index + 1], "0.75")
        self.assertEqual(request["limits"],
                         {"max_host_memory_fraction": 0.4, "max_gpu_memory_fraction": 0.8,
                          "min_free_ram_gib": 2.0, "min_free_disk_gib": 3.0})

    def test_adapter_stop_threshold(self):
        total = 8 * GIB
        limit = int(0.95 * total)
        _, out, _ = self._main_run("speed", [], elapsed=0,
                                   sample={"dedicated_gpu_bytes": limit - 1},
                                   gpu_total_gib=8)
        self.assertFalse((out / "stop").exists())
        _, out, _ = self._main_run("speed", [], elapsed=0,
                                   sample={"dedicated_gpu_bytes": limit + 1},
                                   gpu_total_gib=8)
        self.assertTrue((out / "stop").exists())
        stop = json.loads((out / "external-stop.json").read_text(encoding="utf-8"))
        self.assertEqual(stop["reason"], "adapter_dedicated")

    def test_adapter_unknown_capacity_never_stops(self):
        _, out, _ = self._main_run("speed", [], elapsed=0,
                                   sample={"dedicated_gpu_bytes": 1 << 40},
                                   gpu_total_gib=0)
        self.assertFalse((out / "stop").exists())
        self.assertFalse((out / "external-stop.json").exists())

    def test_custom_max_gpu_fraction_threshold(self):
        _, out, _ = self._main_run("speed", ["--max-gpu-memory-fraction", "0.5"],
                                   elapsed=0, sample={"dedicated_gpu_bytes": 3 * GIB},
                                   gpu_total_gib=8)
        self.assertFalse((out / "stop").exists())
        _, out, _ = self._main_run("speed", ["--max-gpu-memory-fraction", "0.5"],
                                   elapsed=0, sample={"dedicated_gpu_bytes": 5 * GIB},
                                   gpu_total_gib=8)
        stop = json.loads((out / "external-stop.json").read_text(encoding="utf-8"))
        self.assertEqual(stop["reason"], "adapter_dedicated")

    def test_telemetry_gpu_filter_reaches_collector(self):
        seen = {}

        class _Spy(_FakeTelemetry):
            def __call__(self, *args, **kwargs):
                seen.update(kwargs)
                return self

        spy = _Spy()
        self._main_run("generate", ["--prompt", "hi", "--telemetry-gpu", "Radeon"],
                       elapsed=0, sample={}, iters=0, collector=spy)
        self.assertEqual(seen.get("target_gpu_substring"), "Radeon")

    def test_telemetry_gpu_defaults_to_no_constraint(self):
        seen = {}

        class _Spy(_FakeTelemetry):
            def __call__(self, *args, **kwargs):
                seen.update(kwargs)
                return self

        spy = _Spy()
        self._main_run("generate", ["--prompt", "hi"],
                       elapsed=0, sample={}, iters=0, collector=spy)
        self.assertIn("target_gpu_substring", seen)
        self.assertIsNone(seen.get("target_gpu_substring"))


if __name__ == "__main__":
    unittest.main()
