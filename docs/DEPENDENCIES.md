# Dependencies and resource controls

Python orchestrates inference; C++/HIP and Triton execute GPU operations. Windows needs Python 3.11+ for the launcher and a compatible Linux/WSL backend. There is no native Windows inference binary.

## Inference packages

Use a compatible ROCm PyTorch distribution, AMD-compatible Triton and [the inherited requirements](../vendor/rocm-exl3/requirements_rocm.txt). The wrapper also requires Transformers. Those broad ranges are not a lockfile; inspect the resolver's plan in a separate environment.

| Packages | Purpose |
| --- | --- |
| PyTorch ROCm, Triton, ROCm SDK | Tensor execution, GPU kernels and extension compilation |
| Flash Linear Attention / fla-core | Gated DeltaNet |
| Transformers, Tokenizers, Safetensors | Tokenizer/templates and tensor containers |
| NumPy, Rich, typing-extensions | Runtime utilities |
| Ninja, setuptools, C++/HIP compiler | Native build |
| Pillow, PyYAML, marisa-trie, Pydantic, llguidance | Inherited dependencies, including optional upstream paths |

Existing-environment evidence uses Python 3.12, PyTorch 2.13.0+rocm7.2, ROCm SDK 7.2.4, Triton 3.7.1, FLA 0.5.2 and Transformers 5.16.1. These are recorded versions, not a verified clean-install recipe. GPU validation is scoped to gfx1101.

HTTP pins are in [requirements-server.txt](../requirements-server.txt); they are not a complete inference environment. See [SERVING.md](SERVING.md).

## Installation fields

Use [BUILD.md](BUILD.md) and [local.example.toml](../configs/local.example.toml).

| Field | Meaning |
| --- | --- |
| `distribution`, `python` | WSL distribution and inference interpreter |
| `sdk`, `torch_lib`, `hsa_preload` | Compatible SDK and shared-library paths |
| `gpu_arch` | Actual GPU compilation target |
| `source_dir` | This checkout's patched vendored source |
| `extension_dir`, `extension_sha256` | Matching compiled extension and hash |
| `native_smallm_max_rows` | 9 for current full build; use actual limit for older binaries |
| `server_deps` | Optional HTTP package overlay |
| `lease_file` | Shared exclusive GPU lease for cooperating launchers |

Both execution permissions must be true for inference. Models are selected separately with `-m`. Registration and launch install nothing. Automatic native-extension JIT rebuilding is disabled; Triton compilation remains part of execution.

## Resource controls

| CLI flag | Default | Scope |
| --- | --- | --- |
| `--gpu-memory-fraction` | 0.90 | Torch allocator fraction of detected GPU memory |
| `--max-gpu-memory-fraction` | 0.95 | Windows sampled dedicated adapter memory threshold |
| `--max-host-memory-fraction` | 0.50 | Linux process-group RSS relative to Linux-visible total RAM |
| `--min-free-ram-gib` | 1.0 | Minimum available RAM checked by monitors |
| `--min-free-disk-gib` | 1.0 | Minimum free disk space checked by monitors |

Fractions must be finite and strictly between zero and one; GiB values must be finite and positive. Linux-visible RAM under WSL may differ from Windows RAM. Missing Windows GPU capacity is not replaced with a guessed size.

Windows telemetry auto-selects only a unique physical AMD adapter with more than 1 GiB dedicated memory. Use `--telemetry-gpu` with a unique adapter-name substring on an ambiguous system. This selects monitoring only; it does not change Torch's inference device. Confirm the recorded adapter matches the device used for inference.

Monitors are sampled safeguards, not OS-enforced caps. Torch's allowance does not cover every non-Torch allocation. Leave room for caches, workspaces, the display and other processes. Model disk size alone cannot predict residency.

Non-server runs retain the bounded `limits.max_capture_seconds` duration. Serving has no overall duration cap; request deadlines, explicit shutdown and resource stops remain active.
