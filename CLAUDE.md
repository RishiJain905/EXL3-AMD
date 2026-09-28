# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project rules (engineering standards, runtime boundaries, model-storage handling, validation reporting) live in AGENTS.md and apply here:

@AGENTS.md

## Commands

Unit suite (CPU-only, ~15 s, no model or GPU needed; ~46 tests skip when inference/HTTP deps are absent):

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python -m unittest tests.test_server -v                       # one module
python -m unittest tests.test_server.SomeCase.test_name -v    # one test
```

There is no linter, formatter or package build configured. `python run.py --help` checks CLI wiring without loading a model.

GPU-dependent checks (need a registered installation with execution permissions, a built extension and a free GPU lease):

- `python scripts/check_exl3_kernels.py --checks <names> --output artifacts/<new-dir>`: supervised native operator checks (`kernels/exl3/check_*.py`) against CPU references. See `docs/GPU-COMPATIBILITY.md`.
- `scripts/validate_server.py`, `scripts/validate_server_tools.py`: real HTTP/tool-call validation against a running server.
- `python run.py speed|quality|context|context-speed -m MODEL ...`: model-level measurement modes.

The native extension is built from `vendor/rocm-exl3` with `setup.py build_ext` inside WSL/Linux ROCm (`docs/BUILD.md`). A new build changes the SHA-256, which must be re-registered before use.

## Architecture

### Launch path (two processes, two OSes)

1. `run.py` → `scripts/launch_runtime.py:main` validates every CLI flag and cross-flag constraint, then reads `.runtime/installation.toml` (written once by `scripts/register_runtime.py` from `configs/local.toml`). It refuses to run unless `[execution] allow_local_inference` and `allow_backend_probes` are both true.
2. It translates the Windows paths to WSL paths, builds an argv for one of three backend entry scripts, and writes it as `request.json` inside a new private run directory `artifacts/RUN-<utc>-<id>/`:
   - `serve` → `scripts/serve_exl3.py`
   - `generate` with `--mmproj on` → `scripts/generate_multimodal.py`
   - everything else → `scripts/evaluate_exl3_candidate.py` (with `configs/evaluation.json` as the suite)
3. It takes a file lease on the GPU (`lease_file`). On Windows it starts `wsl -d <distro> -- python3 run.py _worker request.json` and samples host/GPU telemetry (`src/quantlab/telemetry.py`) while the child runs. On Linux it calls `worker()` in-process.
4. `worker()` (the `_worker` path in the same file) runs the backend interpreter from the installation with the ROCm environment, and enforces RAM, disk and time guards (`scripts/exl3_resources.py`).
5. Both sides stop the backend through a `stop` file in the run directory, not by signals. Results come back as `result/result.json`, `process.log`, `monitor.json`, `launch.json` and telemetry files.

A new CLI flag therefore has to be threaded through several places: the parser and its validation in `launch_runtime.py`, the argv construction there, and the argparse of the backend entry script(s). Update `README.md` and `docs/MODEL-CLI.md` too.

### Backend scripts

The backend entry scripts run under the WSL ROCm interpreter. They put `--source-dir` (the vendored `exllamav3`) on `sys.path`, verify the extension's SHA-256, force offline Hugging Face loading, and import sibling scripts as flat modules (`from exl3_timing import ...`, `from run_exl3_model_smoke import ...`). Tests load scripts with `importlib.util.spec_from_file_location`, not as a package.

### `src/quantlab` (project code)

- `server.py`: `create_app(engine, ...)` builds the FastAPI OpenAI-compatible app. It handles strict request validation, SSE streaming, disconnects, and a `_Gate` that allows one active GPU request plus a bounded queue. It never touches the GPU directly. `serve_exl3.py`'s `Engine` class supplies generation. This split lets `tests/test_server*.py` cover HTTP behavior with fake engines.
- `tool_calls.py`, `reasoning.py`, `sampling.py`, `images.py`: parsing and validation for the Qwen XML/Hermes JSON tool protocols, thinking output, sampling parameters and image input.
- `methods/exl3/`: runtime modifications to the vendored backend, installed by `Engine`/the evaluator after import:
  - `compat.install()` / `prepare_loaded_module()`: monkeypatch dispatch with the WSL fixes and the native small-M/packed kernel eligibility checks.
  - `optimizations.configure_native()`: probes the built `.so` for `quantlab_exl3_*_abi` symbols and enables only the capabilities the binary actually exports.
  - The other modules (`packed_head`, `packed_mlp`, `shortlist`, `draft_graph`, `cache_precision`, `cache_policy`, `prefix_cache`, `vision`, ...) each install one optimization or feature with its own eligibility guard.

### Native kernels and the vendored backend

- `vendor/rocm-exl3/` is the active build source. It is a modified copy of CarouselAether's rocm_exl3, pinned in `docs/UPSTREAMS.md`. The project's HIP kernels live under `exllamav3/exllamav3_ext/rocm/quant/rdna-*.hip.h` and export ABI-version symbols that the Python side checks.
- `kernels/exl3/` holds operator validators (`check_*.py`), CPU reference implementations and research scripts. Its `*.patch` files are historical references; do not reapply them.
- New kernel capabilities follow the same pattern: an exported ABI symbol, a Python capability probe with fallback to the inherited path, a `--no-<feature>` diagnostic override, and a `check_*.py` validator. Once validated, the feature is on by default (see AGENTS.md).

### Docs as the record

`docs/VALIDATION.md` is a dated log of what was actually validated, on which GPU/model/config. The feature docs (`*-PERFORMANCE.md`, `KV-CACHE.md`, `MIMO-*.md`) record measurements and dispatch policies that the code's eligibility guards implement. Read the relevant one before changing a threshold or shape guard, and add a new dated section when validating on hardware.
