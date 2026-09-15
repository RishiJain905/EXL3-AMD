"""CPU-only launcher boundary tests; no WSL, GPU, model files, or network."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("launcher_under_test", SCRIPTS / "launch_runtime.py")
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)


def _runtime(**overrides):
    base = {
        "sdk": "/opt/rocm",
        "hsa_preload": "/opt/rocm/lib/libhsa-runtime64.so",
        "gpu_arch": "gfx942",
        "python": "/opt/rocm/bin/python",
        "torch_lib": "/opt/torch/lib",
    }
    base.update(overrides)
    return base


class LinuxPathTests(unittest.TestCase):
    def test_windows_drive_with_spaces_converts_to_mnt(self):
        self.assertEqual(launch.linux_path(r"C:\My Models\foo bar\baz"),
                         "/mnt/c/My Models/foo bar/baz")
        self.assertEqual(launch.linux_path("F:/plain/path"), "/mnt/f/plain/path")

    def test_drive_letter_lowercased_rest_preserved(self):
        self.assertEqual(launch.linux_path("D:/Mixed/CASE"), "/mnt/d/Mixed/CASE")

    def test_linux_absolute_passthrough(self):
        self.assertEqual(launch.linux_path("/mnt/f/models/foo"), "/mnt/f/models/foo")
        self.assertEqual(launch.linux_path("/tmp/x"), "/tmp/x")

    def test_relative_anchored_to_root(self):
        self.assertEqual(launch.linux_path("artifacts/run", root=Path("C:/fake/root")),
                         "/mnt/c/fake/root/artifacts/run")


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_http_overlay_only_when_explicitly_configured(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/untrusted/inherited"}):
            self.assertNotIn("PYTHONPATH", launch.runtime_environment(_runtime()))
            env = launch.runtime_environment(_runtime(server_deps="/isolated/server-deps"))
            self.assertEqual(env["PYTHONPATH"], "/isolated/server-deps")

    def test_clears_inherited_exl3_without_mutating_environ(self):
        before = dict(os.environ)
        with patch.dict(os.environ, {"EXL3_FOO": "inherited", "EXL3_BACKEND": "legacy",
                                     "OTHER_KEEP": "1"}):
            snapshot = dict(os.environ)
            env = launch.runtime_environment(_runtime())
            self.assertNotIn("EXL3_FOO", env)
            self.assertEqual(env["EXL3_BACKEND"], "rocm")
            self.assertEqual(env["OTHER_KEEP"], "1")
            self.assertEqual(os.environ["EXL3_FOO"], "inherited")
            self.assertEqual(dict(os.environ), snapshot)
        self.assertEqual(dict(os.environ), before)

    def test_selects_configured_sdk_python_libs(self):
        runtime = _runtime(sdk="/opt/custom-rocm", python="/opt/custom-rocm/bin/python3",
                           torch_lib="/opt/custom-torch/lib",
                           hsa_preload="/opt/custom-rocm/lib/preload.so", gpu_arch="gfx1100")
        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            env = launch.runtime_environment(runtime)
        self.assertEqual(env["LD_PRELOAD"], runtime["hsa_preload"])
        self.assertEqual(env["ROCM_PATH"], runtime["sdk"])
        self.assertEqual(env["ROCM_HOME"], runtime["sdk"])
        self.assertEqual(env["HIP_PATH"], runtime["sdk"])
        self.assertEqual(env["PYTORCH_ROCM_ARCH"], runtime["gpu_arch"])
        self.assertEqual(env["GPU_ARCHS"], runtime["gpu_arch"])
        self.assertTrue(env["PATH"].startswith(str(Path(runtime["python"]).parent)))
        self.assertIn(runtime["sdk"] + "/bin", env["PATH"])
        self.assertIn(runtime["sdk"] + "/llvm/bin", env["PATH"])
        self.assertTrue(env["PATH"].endswith("/usr/bin:/bin"))
        for entry in (runtime["torch_lib"], runtime["sdk"] + "/lib",
                      str(Path(runtime["hsa_preload"]).parent)):
            self.assertIn(entry, env["LD_LIBRARY_PATH"])


class LeaseTests(unittest.TestCase):
    def test_clean_server_stop_requires_shutdown_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run/'result').mkdir()
            request = {'mode': 'serve'}
            check = lambda code=-15, reason='external_stop', error=None: launch.clean_server_stop(request, run, code, reason, error)
            self.assertFalse(check())
            state = dict(ready=False, active=False, failed=0, model_load_count=1)
            (run/'result/shutdown.json').write_text(json.dumps(state))
            self.assertTrue(check())
            for reason in ('timeout', 'group_rss', 'wsl_available_ram', 'artifact_disk'):
                self.assertFalse(check(reason=reason))
            self.assertFalse(check(code=-9))
            self.assertFalse(check(error='monitor failed'))
            (run/'external-stop.json').write_text('{"reason":"adapter_dedicated"}')
            self.assertFalse(check())
            (run/'external-stop.json').unlink()
            (run/'result/shutdown.json').write_text(json.dumps(dict(state, active=True)))
            self.assertFalse(check())

    def test_preexisting_owner_lock_survives_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "locks" / "gpu.lock"
            lock.parent.mkdir(parents=True)
            owner = {"pid": 1234, "run": "owner-run", "nonce": "owner-nonce"}
            lock.write_text(json.dumps(owner), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                with launch.lease(lock, "contender-run"):
                    pass
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8")), owner)

    def test_contender_fails_while_owner_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "gpu.lock"
            with launch.lease(lock, "run-a"):
                held = json.loads(lock.read_text(encoding="utf-8"))
                self.assertEqual(held["run"], "run-a")
                with self.assertRaises(FileExistsError):
                    with launch.lease(lock, "run-b"):
                        pass
                self.assertEqual(json.loads(lock.read_text(encoding="utf-8")), held)
            self.assertFalse(lock.exists())

    def test_nonce_replacement_preserved_on_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "gpu.lock"
            with launch.lease(lock, "run-a"):
                original = json.loads(lock.read_text(encoding="utf-8"))
                replacement = {"pid": 9999, "run": "run-b", "nonce": "replacement-nonce"}
                self.assertNotEqual(replacement["nonce"], original["nonce"])
                lock.write_text(json.dumps(replacement), encoding="utf-8")
            self.assertTrue(lock.exists())
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8")), replacement)

    def test_owner_cleanup_removes_own_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "nested" / "gpu.lock"
            with launch.lease(lock, "run-a"):
                self.assertTrue(lock.exists())
            self.assertFalse(lock.exists())


class MainRejectionTests(unittest.TestCase):
    def test_gpu_draft_dependencies_rejected_before_config_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out'
            for flags in (['--gpu-draft'],['--gpu-draft','--gpu-embedding'],
                          ['--gpu-draft','--mtp','4'],['--gpu-draft-metadata']):
                self._assert_rejects(['launch','speed','--config',str(Path(tmp)/'missing.toml'),
                                      '--output',str(out),*flags],out)

    def test_server_rejects_public_bind_before_launch(self):
        with patch("subprocess.Popen") as child, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                launch.parser().parse_args(["serve", "--host", "0.0.0.0"])
        self.assertEqual(result.exception.code, 2)
        child.assert_not_called()

    def _write_config(self, path, infer=True, probes=True, omit=()):
        lines = ["[execution]"]
        if "allow_local_inference" not in omit:
            lines.append(f"allow_local_inference = {'true' if infer else 'false'}")
        if "allow_backend_probes" not in omit:
            lines.append(f"allow_backend_probes = {'true' if probes else 'false'}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _assert_rejects(self, argv, output):
        with patch.object(sys, "argv", argv), \
                patch("subprocess.Popen") as mock_popen, \
                patch.object(launch, "write_json") as mock_write, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                launch.main()
        self.assertEqual(caught.exception.code, 2)
        mock_popen.assert_not_called()
        mock_write.assert_not_called()
        self.assertFalse(output.exists(), "rejected run must not create output")

    def test_removed_execute_flag_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out), "--execute"], out)


    def test_unpermitted_config_rejects(self):
        variants = [dict(infer=False), dict(probes=False),
                    dict(omit=("allow_local_inference",)),
                    dict(omit=("allow_backend_probes",))]
        for kwargs in variants:
            with self.subTest(**kwargs):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    cfg = tmp / "local.toml"
                    self._write_config(cfg, **kwargs)
                    out = tmp / "run-out"
                    self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                          "--output", str(out)], out)

    def test_invalid_context_rejects(self):
        for context in (512, 4097, 0, 1025):
            with self.subTest(context=context):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    cfg = tmp / "local.toml"
                    self._write_config(cfg)
                    out = tmp / "run-out"
                    self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                          "--output", str(out),
                                          "--context", str(context)], out)

    def test_generate_without_prompt_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "generate", "--config", str(cfg),
                                  "--output", str(out)], out)

    def test_both_prompt_and_prompt_file_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            prompt_file = tmp / "p.txt"
            prompt_file.write_text("hi", encoding="utf-8")
            out = tmp / "run-out"
            self._assert_rejects(["launch", "generate", "--config", str(cfg),
                                  "--output", str(out),
                                  "--prompt", "hi", "--prompt-file", str(prompt_file)], out)

    def test_invalid_max_tokens_rejects(self):
        for tokens in (0, 8193):
            with self.subTest(max_tokens=tokens):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    cfg = tmp / "local.toml"
                    self._write_config(cfg)
                    out = tmp / "run-out"
                    self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                          "--output", str(out),
                                          "--max-tokens", str(tokens)], out)

    def test_cache_mtp_without_mtp_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--cache-mtp", "fc"], out)

    def test_cache_mtp_with_gdn_mlp_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--mtp", "4", "--decode-fusions", "gdn-mlp",
                                  "--cache-mtp", "mlp"], out)

    def test_draft_confidence_without_mtp_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--draft-confidence", "0.4"], out)

    def test_draft_confidence_invalid_values_reject(self):
        for value in ("0", "1", "-0.1", "1.5", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    cfg = tmp / "local.toml"
                    self._write_config(cfg)
                    out = tmp / "run-out"
                    self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                          "--output", str(out),
                                          "--mtp", "4", "--draft-confidence", value], out)

    def test_draft_confidence_with_gpu_draft_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--mtp", "4", "--gpu-embedding", "--gpu-draft",
                                  "--draft-confidence", "0.4"], out)

    def test_quant_k_without_v_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--cache-type-k", "q8"], out)

    def test_explicit_f16_quant_mix_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--cache-type-k", "f16", "--cache-type-v", "q4"], out)

    def test_conflicting_shorthand_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--cache-type", "q8", "--cache-type-k", "q4"], out)

    def test_fp8_cache_type_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--cache-type", "fp8"], out)

    def test_malformed_attention_profile_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--attention-profile", "bogus"], out)

    def test_quantized_cache_rejects_native_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--cache-type", "q8", "--native-attention"], out)

    def test_quantized_cache_rejects_draft_step_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            self._write_config(cfg)
            out = tmp / "run-out"
            self._assert_rejects(["launch", "speed", "--config", str(cfg),
                                  "--output", str(out),
                                  "--mtp", "4", "--gpu-embedding", "--gpu-draft",
                                  "--gpu-draft-metadata", "--draft-step-graph",
                                  "--cache-type", "q8"], out)

class _FakeTelemetry:
    def __init__(self, *args, **kwargs):
        self.gpu_total_dedicated_bytes = 16 * (2 ** 30)

    def start(self):
        pass

    def sample_now(self):
        return {}

    def stop(self):
        pass

    def write_csv(self, path):
        Path(path).write_text("", encoding="utf-8")

    def summary(self):
        return {}


class CachePrecisionForwardingTests(unittest.TestCase):
    def _write_full_config(self, path, lease_file):
        path.write_text("\n".join([
            "[execution]", "allow_local_inference = true", "allow_backend_probes = true",
            "[runtime]", "python = '/fake/python'", "sdk = '/fake/sdk'",
            "torch_lib = '/fake/torch'", "hsa_preload = '/fake/preload.so'",
            "source_dir = '/fake/src'", "extension_dir = '/fake/ext'",
            "candidate = '/fake/candidate'", "distribution = 'fake'",
            "extension_sha256 = 'abc'", f"lease_file = '{lease_file}'",
            "[limits]", "max_capture_seconds = 60",
        ]) + "\n", encoding="utf-8")

    def _run_and_capture_argv(self, mode, argv_extra):
        from types import ModuleType
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "local.toml"
            out = tmp / "run-out"
            self._write_full_config(cfg, (tmp / "lock").as_posix())
            argv = ["launch", mode, "--config", str(cfg), "--output", str(out), *argv_extra]
            child = MagicMock()
            child.poll.return_value = 0
            child.wait.return_value = 0
            telemetry_module = ModuleType("quantlab.telemetry")
            telemetry_module.TelemetryCollector = _FakeTelemetry
            quantlab_package = ModuleType("quantlab")
            quantlab_package.__path__ = []
            with patch.object(sys, "argv", argv), \
                    patch("subprocess.Popen", return_value=child), \
                    patch.object(launch, "worker", return_value=0), \
                    patch.dict(sys.modules, {"quantlab": quantlab_package,
                                             "quantlab.telemetry": telemetry_module}), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    launch.main()
            self.assertEqual(caught.exception.code, 0)
            request = json.loads((out / "request.json").read_text(encoding="utf-8"))
            return request["argv"]

    def _assert_pair(self, argv, key, value):
        position = argv.index("--cache-type-k")
        self.assertEqual(argv[position + 1:position + 3], [key, "--cache-type-v"])
        self.assertEqual(argv[position + 3], value)

    def test_no_flag_forwards_f16_pair_to_evaluator(self):
        argv = self._run_and_capture_argv("speed", [])
        self.assertIn("evaluate_exl3_candidate.py", argv[1])
        self._assert_pair(argv, "f16", "f16")

    def test_shorthand_forwards_pair(self):
        argv = self._run_and_capture_argv("speed", ["--cache-type", "q8"])
        self._assert_pair(argv, "q8", "q8")

    def test_mixed_pair_forwards(self):
        argv = self._run_and_capture_argv("speed", ["--cache-type-k", "q8", "--cache-type-v", "q4"])
        self._assert_pair(argv, "q8", "q4")

    def test_short_syntax_forwards(self):
        argv = self._run_and_capture_argv("speed", ["-ctk", "q4", "-ctv", "q4"])
        self._assert_pair(argv, "q4", "q4")

    def test_serve_mode_forwards_pair(self):
        argv = self._run_and_capture_argv("serve", ["--cache-type", "q8"])
        self.assertIn("serve_exl3.py", argv[1])
        self._assert_pair(argv, "q8", "q8")

    def test_prefix_cache_forwarded_only_to_serve(self):
        argv = self._run_and_capture_argv("serve", ["--prefix-cache", "on"])
        self.assertEqual(argv[argv.index("--prefix-cache") + 1], "on")
        argv = self._run_and_capture_argv("speed", [])
        self.assertNotIn("--prefix-cache", argv)


    def test_prefill_defaults_preserve_benchmark_protocol(self):
        for mode, extra, expected in (("serve", [], "1024"), ("speed", [], "256"),
                                      ("serve", ["-b", "512"], "512")):
            with self.subTest(mode=mode, extra=extra):
                argv = self._run_and_capture_argv(mode, extra)
                self.assertEqual(argv[argv.index("--prefill-chunk") + 1], expected)

    def test_prefill_gemm_forwarded_to_server_and_evaluator(self):
        for mode in ('serve', 'speed'):
            with self.subTest(mode=mode):
                argv = self._run_and_capture_argv(mode, ['--prefill-gemm', 'wmma'])
                self.assertEqual(argv[argv.index('--prefill-gemm') + 1], 'wmma')

    def test_f16_keeps_native_attention_available(self):
        argv = self._run_and_capture_argv("speed", ["--native-attention"])
        self.assertIn("--native-attention", argv)
        self._assert_pair(argv, "f16", "f16")

    def _assert_profile(self, argv, value):
        position = argv.index("--attention-profile")
        self.assertEqual(argv[position + 1], value)

    def test_default_profile_forwards_to_evaluator(self):
        argv = self._run_and_capture_argv("speed", [])
        self.assertIn("evaluate_exl3_candidate.py", argv[1])
        self._assert_profile(argv, "default")

    def test_long_profile_forwards_to_evaluator(self):
        argv = self._run_and_capture_argv("speed", ["--attention-profile", "long"])
        self.assertIn("evaluate_exl3_candidate.py", argv[1])
        self._assert_profile(argv, "long")

    def test_default_profile_forwards_to_server(self):
        argv = self._run_and_capture_argv("serve", [])
        self.assertIn("serve_exl3.py", argv[1])
        self._assert_profile(argv, "default")

    def test_long_profile_forwards_to_server(self):
        argv = self._run_and_capture_argv("serve", ["--attention-profile", "long"])
        self.assertIn("serve_exl3.py", argv[1])
        self._assert_profile(argv, "long")


def _write_metadata(path, lease_file, *, candidate=None, manifest=None, rows=None,
                     distribution='test-dist'):
    lines = ["[execution]", "allow_local_inference = true", "allow_backend_probes = true",
             "[limits]", "max_capture_seconds = 60",
             "[runtime]", f"distribution = '{distribution}'", "python = '/fake/python'",
             "sdk = '/fake/sdk'", "torch_lib = '/fake/torch'", "hsa_preload = '/fake/preload.so'",
             "source_dir = '/fake/src'", "extension_dir = '/fake/ext'",
             "extension_sha256 = '" + "ab" * 32 + "'", "gpu_arch = 'gfx1101'", f"lease_file = '{lease_file}'"]
    if candidate is not None:
        lines.append(f"candidate = '{candidate}'")
    if manifest is not None:
        lines.append(f"candidate_manifest = '{manifest}'")
    if rows is not None:
        lines.append(f"native_smallm_max_rows = {rows}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class MetadataResolutionTests(unittest.TestCase):
    def _run(self, root, args):
        from types import ModuleType
        out = root / "run-out"
        child = MagicMock()
        child.poll.return_value = 0
        child.wait.return_value = 0
        telemetry_module = ModuleType("quantlab.telemetry")
        telemetry_module.TelemetryCollector = _FakeTelemetry
        quantlab_package = ModuleType("quantlab")
        quantlab_package.__path__ = []
        with patch.object(launch, "ROOT", root), \
                patch.object(sys, "argv", ["launch", *args]), \
                patch("subprocess.Popen", return_value=child), \
                patch.object(launch, "worker", return_value=0), \
                patch.dict(sys.modules, {"quantlab": quantlab_package,
                                         "quantlab.telemetry": telemetry_module}), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                launch.main()
        self.assertEqual(caught.exception.code, 0)
        return json.loads((out / "request.json").read_text(encoding="utf-8"))

    def _assert_rejects(self, root, args):
        out = root / "run-out"
        with patch.object(launch, "ROOT", root), \
                patch.object(sys, "argv", ["launch", *args]), \
                patch("subprocess.Popen") as mock_popen, \
                patch.object(launch, "worker") as mock_worker, \
                patch.object(launch, "write_json") as mock_write, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                launch.main()
        self.assertEqual(caught.exception.code, 2)
        mock_popen.assert_not_called()
        mock_worker.assert_not_called()
        mock_write.assert_not_called()
        self.assertFalse(out.exists(), "rejected run must not create output")

    def _model(self, root):
        model = root / "model"
        model.mkdir()
        return model

    def test_installation_preferred_over_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install = root / ".runtime" / "installation.toml"
            _write_metadata(install, (root / "lock").as_posix(), rows=9,
                            distribution="install-dist")
            _write_metadata(root / "configs" / "local.toml", (root / "lock").as_posix(),
                            candidate="/fake/legacy", distribution="legacy-dist")
            model = self._model(root)
            request = self._run(root, ["speed", "--output", str(root / "run-out"),
                                      "-m", str(model), "--mtp", "4", "--warps", "8"])
        self.assertEqual(request["runtime"]["distribution"], "install-dist")
        self.assertEqual(request["runtime"]["candidate"], launch.linux_path(model.resolve()))
        argv = request["argv"]
        self.assertEqual(argv[argv.index("--config") + 1], launch.linux_path(install.resolve()))
        self.assertEqual(argv[argv.index("--native-smallm-max-rows") + 1], "9")
        self.assertEqual(argv[argv.index("--draft-tokens") + 1], "4")
        self.assertEqual(argv[argv.index("--gemv-splitk-warps") + 1], "8")
        self.assertIn("evaluate_exl3_candidate.py", argv[1])

    def test_legacy_used_when_no_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "configs" / "local.toml"
            manifest = root / "manifest.json"
            manifest.write_text("[]", encoding="utf-8")
            _write_metadata(legacy, (root / "lock").as_posix(), candidate="/fake/legacy",
                            manifest=manifest.as_posix())
            request = self._run(root, ["speed", "--output", str(root / "run-out")])
        self.assertEqual(request["runtime"]["candidate"], "/fake/legacy")
        argv = request["argv"]
        self.assertEqual(argv[argv.index("--config") + 1], launch.linux_path(legacy.resolve()))
        self.assertEqual(argv[argv.index("--candidate-manifest") + 1],
                         launch.linux_path(manifest))

    def test_installation_requires_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root / ".runtime" / "installation.toml",
                            (root / "lock").as_posix())
            self._assert_rejects(root, ["speed", "--output", str(root / "run-out")])

    def test_linux_model_path_survives_windows_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root / ".runtime" / "installation.toml",
                            (root / "lock").as_posix(), rows=9)
            request = self._run(root, ["serve", "-m", "/mnt/d/EXL3 Models/example",
                "--output", str(root / "run-out"), "--cache-type", "q8",
                "--attention-profile", "long", "--spec-type", "draft-mtp", "--mtp", "6"])
        self.assertEqual(request["runtime"]["candidate"], "/mnt/d/EXL3 Models/example")
        self.assertEqual(request["argv"][request["argv"].index("--cache-type-k")+1], "q8")
        self.assertEqual(request["argv"][request["argv"].index("--attention-profile")+1], "long")
        self.assertEqual(request["argv"][request["argv"].index("--draft-tokens")+1], "6")

    def test_invalid_installation_never_uses_valid_legacy(self):
        for before, after in (("gpu_arch = 'gfx1101'", "gpu_arch = 1"),
                              ("ab"*32, "bad-hash"),
                              ("native_smallm_max_rows = 9", "native_smallm_max_rows = true")):
            with self.subTest(after=after), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                installation = root / ".runtime" / "installation.toml"
                _write_metadata(installation, (root / "lock").as_posix(), rows=9)
                installation.write_text(installation.read_text().replace(before, after))
                _write_metadata(root / "configs" / "local.toml",
                                (root / "lock").as_posix(), candidate="/fake/legacy")
                self._assert_rejects(root, ["speed", "-m", str(root / "model"),
                                          "--output", str(root / "run-out")])

    def test_missing_metadata_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = self._model(root)
            self._assert_rejects(root, ["speed", "--output", str(root / "run-out"),
                                       "-m", str(model)])

    def test_malformed_installation_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            install = root / ".runtime" / "installation.toml"
            install.parent.mkdir(parents=True, exist_ok=True)
            install.write_text("[runtime\n", encoding="utf-8")
            model = self._model(root)
            self._assert_rejects(root, ["speed", "--output", str(root / "run-out"),
                                       "-m", str(model)])

    def test_explicit_missing_config_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root / ".runtime" / "installation.toml",
                            (root / "lock").as_posix())
            model = self._model(root)
            self._assert_rejects(root, ["speed", "--config", str(root / "missing.toml"),
                                       "--output", str(root / "run-out"),
                                       "-m", str(model)])

    def test_explicit_legacy_config_overrides_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "elsewhere.toml"
            _write_metadata(legacy, (root / "lock").as_posix(), candidate="/fake/legacy")
            model = self._model(root)
            request = self._run(root, ["speed", "--config", str(legacy),
                                      "--output", str(root / "run-out"),
                                      "-m", str(model)])
        self.assertEqual(request["runtime"]["candidate"], launch.linux_path(model.resolve()))
        self.assertNotIn("--candidate-manifest", request["argv"])

    def test_installation_ignores_configured_model_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root / ".runtime" / "installation.toml",
                            (root / "lock").as_posix(), candidate="/fake/stray",
                            manifest="/fake/stray-manifest.json")
            model = self._model(root)
            self._assert_rejects(root, ["speed", "--output", str(root / "run-out")])
            request = self._run(root, ["speed", "--output", str(root / "run-out"),
                                      "-m", str(model)])
        self.assertEqual(request["runtime"]["candidate"], launch.linux_path(model.resolve()))
        self.assertNotIn("--candidate-manifest", request["argv"])

    def test_model_manifest_flag_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root / ".runtime" / "installation.toml",
                            (root / "lock").as_posix())
            model = self._model(root)
            manifest = root / "manifest.json"
            manifest.write_text("[]", encoding="utf-8")
            request = self._run(root, ["speed", "--output", str(root / "run-out"),
                                      "-m", str(model), "--model-manifest", str(manifest)])
        argv = request["argv"]
        self.assertEqual(argv[argv.index("--candidate-manifest") + 1],
                         launch.linux_path(manifest.resolve()))

class ServeLifetimeTests(unittest.TestCase):
    def _worker_run(self, timeout, *, elapsed, rss=1024, mem_gib=8, disk_gib=200,
                    iters=1, pre_stop=False, mode=None, total_gib=24, limits=None,
                    sleep_error=None, shutdown_state=None, child_exit=0,
                    external_reason=None):
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
        if external_reason:
            (run / "external-stop.json").write_text(json.dumps({"reason": external_reason}))
        child = MagicMock()
        child.pid = 4321
        child.poll.side_effect = [None, None] if sleep_error else [None] * iters + [child_exit, child_exit]
        child.wait.return_value = child_exit
        child.returncode = child_exit
        def finish_shutdown(_child):
            if shutdown_state is not None:
                (run / "result/shutdown.json").write_text(json.dumps(shutdown_state))
        meminfo = (f"MemTotal: {total_gib * 1024 * 1024} kB\n"
                   f"MemAvailable: {mem_gib * 1024 * 1024} kB\n")
        real_read = Path.read_text
        def fake_read(self, *a, **k):
            if "meminfo" in str(self):
                return meminfo
            return real_read(self, *a, **k)
        disk = MagicMock()
        disk.free = disk_gib * launch.GIB
        monotonic = [1000.0] + [1000.0 + elapsed] * (iters + 2)
        with patch("subprocess.Popen", return_value=child), \
                patch.object(launch, "group_rss", return_value=rss), \
                patch.object(launch, "runtime_environment", return_value={}), \
                patch.object(launch.time, "monotonic", side_effect=monotonic), \
                patch.object(launch.time, "sleep", side_effect=sleep_error), \
                patch("shutil.disk_usage", return_value=disk), \
                patch.object(Path, "read_text", fake_read), \
                patch.object(launch, "terminate_group", side_effect=finish_shutdown) as term:
            code = launch.worker(str(req))
        monitor = json.loads((run / "monitor.json").read_text(encoding="utf-8"))
        return code, monitor, term

    def test_worker_serve_none_survives_elapsed(self):
        _, monitor, term = self._worker_run(None, elapsed=901)
        term.assert_not_called()
        self.assertIsNone(monitor["stop_reason"])
        self.assertIsNone(monitor["error"])

    def test_worker_serve_none_still_enforces_resources(self):
        _, monitor, term = self._worker_run(None, elapsed=901, rss=13 * launch.GIB)
        term.assert_called_once()
        self.assertEqual(monitor["stop_reason"], "group_rss")

    def test_worker_serve_none_honors_explicit_stop(self):
        _, monitor, term = self._worker_run(None, elapsed=901, pre_stop=True)
        term.assert_called_once()
        self.assertEqual(monitor["stop_reason"], "external_stop")

    def test_worker_numeric_timeout_still_fires(self):
        _, monitor, term = self._worker_run(60, elapsed=901)
        term.assert_called_once()
        self.assertEqual(monitor["stop_reason"], "timeout")

    def test_worker_ctrl_c_with_confirmed_server_shutdown_succeeds(self):
        state = dict(ready=False, active=False, failed=0, model_load_count=1)
        for child_exit in (0, -15):
            with self.subTest(child_exit=child_exit):
                code, monitor, term = self._worker_run(
                    None, elapsed=1, sleep_error=KeyboardInterrupt(),
                    shutdown_state=state, child_exit=child_exit)
                term.assert_called_once()
                self.assertEqual(code, 0)
                self.assertEqual(monitor["exit_code"], child_exit)
                self.assertEqual(monitor["stop_reason"], "user_stop")
                self.assertTrue(monitor["clean_server_shutdown"])
                self.assertIsNone(monitor["error"])

    def test_worker_ctrl_c_does_not_hide_failed_or_incomplete_shutdown(self):
        state = dict(ready=False, active=False, failed=0, model_load_count=1)
        cases = [dict(shutdown_state=None),
                 dict(shutdown_state=dict(state, failed=1)),
                 dict(shutdown_state=dict(state, active=True)),
                 dict(shutdown_state=state, external_reason="adapter_dedicated"),
                 dict(shutdown_state=state, mode="speed")]
        for case in cases:
            with self.subTest(case=case):
                code, monitor, term = self._worker_run(
                    None, elapsed=1, sleep_error=KeyboardInterrupt(), child_exit=-15, **case)
                term.assert_called_once()
                self.assertNotEqual(code, 0)
                self.assertFalse(monitor["clean_server_shutdown"])

    def test_worker_real_monitor_error_still_fails_after_clean_shutdown(self):
        state = dict(ready=False, active=False, failed=0, model_load_count=1)
        code, monitor, term = self._worker_run(
            None, elapsed=1, sleep_error=RuntimeError("monitor failed"),
            shutdown_state=state, child_exit=-15)
        term.assert_called_once()
        self.assertNotEqual(code, 0)
        self.assertEqual(monitor["stop_reason"], "monitor_error")
        self.assertEqual(monitor["error"], "RuntimeError: monitor failed")
        self.assertFalse(monitor["clean_server_shutdown"])

    def _windows_run(self, mode, argv_extra, *, elapsed, sample, iters=1, gpu_total_gib=16, disk_gib=200):
        if os.name != "nt":
            self.skipTest("Windows monitor requires Windows; worker tests run on both platforms")
        from types import ModuleType
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
        telemetry = _FakeTelemetry()
        telemetry.gpu_total_dedicated_bytes = gpu_total_gib * launch.GIB
        telemetry.sample_now = lambda: dict(sample)
        telemetry_module = ModuleType("quantlab.telemetry")
        telemetry_module.TelemetryCollector = lambda *a, **k: telemetry
        quantlab_package = ModuleType("quantlab")
        quantlab_package.__path__ = []
        disk = MagicMock()
        disk.free = disk_gib * launch.GIB
        monotonic = [1000.0] + [1000.0 + elapsed] * (iters + 2)
        stdout = io.StringIO()
        argv = ["launch", mode, "--config", str(cfg), "--output", str(out), *argv_extra]
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

    def test_serve_request_null_survives_windows_elapsed(self):
        request, out, stdout = self._windows_run("serve", [], elapsed=901, sample={})
        self.assertIsNone(request["timeout"])
        self.assertIsNone(json.loads((out / "request.json").read_text())["timeout"])
        self.assertFalse((out / "stop").exists())
        self.assertFalse((out / "external-stop.json").exists())
        self.assertIn("runs until stopped", stdout)
        self.assertNotIn("session limit", stdout)

    def test_serve_windows_still_enforces_resources(self):
        request, out, _ = self._windows_run("serve", [], elapsed=901,
                                            sample={"host_available_bytes": 0})
        self.assertIsNone(request["timeout"])
        self.assertTrue((out / "stop").exists())
        stop = json.loads((out / "external-stop.json").read_text(encoding="utf-8"))
        self.assertEqual(stop["reason"], "windows_available_ram")

    def test_benchmark_request_numeric_and_windows_timeout_fires(self):
        request, out, _ = self._windows_run("speed", [], elapsed=901, sample={})
        self.assertEqual(request["timeout"], 60)
        self.assertTrue((out / "stop").exists())
        stop = json.loads((out / "external-stop.json").read_text(encoding="utf-8"))
        self.assertEqual(stop["reason"], "windows_timeout")

    def test_generate_request_keeps_numeric_cap(self):
        request, _, _ = self._windows_run("generate", ["--prompt", "hi"], elapsed=0,
                                          sample={}, iters=0)
        self.assertEqual(request["timeout"], 60)



if __name__ == "__main__":
    unittest.main()
