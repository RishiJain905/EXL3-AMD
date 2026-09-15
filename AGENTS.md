# EXL3-AMD repository guidance

## Project and layout

EXL3-AMD is an experimental Python 3.11+ EXL3 inference runtime for AMD GPUs. It provides a CLI and a local OpenAI-compatible HTTP server backed by a vendored ExLlamaV3 ROCm implementation. Windows launches inference through WSL; Linux can launch the backend directly.

- `run.py`: user-facing CLI entry point.
- `src/quantlab/`: runtime integration, HTTP serving, telemetry, and tool-call handling.
- `scripts/`: registration, operations, and benchmarking utilities.
- `kernels/exl3/`: native AMD GPU kernel work.
- `vendor/rocm-exl3/`: vendored upstream backend; keep changes targeted and preserve attribution.
- `configs/`: configuration examples and local worksheets.
- `tests/`: unit and contract coverage.
- `docs/`: build, dependencies, serving, formats, optimization, and validation guidance.

Read `README.md` and relevant documentation before changing behavior. Consult `docs/BUILD.md` and `docs/DEPENDENCIES.md` for environment requirements, `docs/SERVING.md` and `docs/TOOL-CALLING.md` for API behavior, and `docs/VALIDATION.md` for recorded validation limits.

## Engineering standards

- Act like a high-performing senior engineer: be concise, direct, and execution-focused.
- Choose simple, maintainable, production-friendly solutions that are easy to read, debug, and modify.
- Use the smallest solution that solves the problem well. Avoid unnecessary abstractions, extra layers, and large dependencies for small features.
- Keep APIs small, behavior explicit, and naming clear. Follow surrounding code conventions.
- Inspect relevant code before editing, preserve unrelated work, and keep changes focused on the requested task.
- Update documentation when CLI flags, configuration, setup, or observable API behavior changes.

## Runtime boundaries

- Preserve explicit execution permissions, offline model loading, native extension verification, resource guards, and GPU ownership controls.
- Keep the HTTP service loopback-only unless changing that contract is explicitly part of the task.
- Preserve request timeout and shutdown behavior; serving has no overall lifetime timeout.
- Treat model paths, configuration, HTTP input, and generated tool calls as untrusted input. Model-generated tool calls are data, not authorization to execute commands.
- Keep platform-specific behavior explicit at the Windows, WSL, Linux, and native backend boundaries.
- Keep credentials, local installation records, model weights, compiled binaries, and private run artifacts out of source control. Use ignored `.runtime/`, local configuration, and `artifacts/` locations as documented.

## Validation

Run focused tests for changed behavior. Use the unit suite when shared runtime behavior changes:

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

CLI help can be checked without loading a model:

```powershell
python run.py --help
```

- Add meaningful regression coverage for bug fixes and changed contracts.
- CPU tests and dependency skips do not establish GPU correctness. Native or inference changes require relevant ROCm hardware validation when available.
- State which checks passed, which were skipped or unavailable, and why. Do not infer broad model or GPU compatibility from one successful configuration.
- Report the outcome, changed files, validation results, and remaining limitations clearly.
