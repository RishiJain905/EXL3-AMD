# Build and register

Use a compatible Linux/WSL ROCm environment. Binaries depend on Python, Torch, ROCm and GPU target and are not bundled. See [dependencies](DEPENDENCIES.md) and the [pinned AMD upstream](https://github.com/CarouselAether/rocm_exl3/tree/550dcfed786ad7bffa08b7a6b2a216fc474cbbb5).

## Native extension

Choose a new build directory and substitute actual installation paths. Set `GPU_TARGET` to your GPU (gfx1101 is the validated RDNA3 path).

RDNA4 targets are `gfx1200` and `gfx1201`, with experimental hardware status.
See [GPU/codebook compatibility](GPU-COMPATIBILITY.md) for multi-target builds
and model-free acceptance checks. Header or target changes require a fresh build.

```bash
EXL3_REPO=/path/to/EXL3-AMD
EXL3_PYTHON=/path/to/rocm-venv/bin/python
EXL3_SDK=/path/to/rocm-sdk
EXL3_TORCH_LIB=/path/to/rocm-venv/lib/python3.12/site-packages/torch/lib
EXL3_HSA=/path/to/compatible/libhsa-runtime64.so.1
EXL3_BUILD=/path/to/new-build
GPU_TARGET=gfx1101

export ROCM_PATH="$EXL3_SDK" ROCM_HOME="$EXL3_SDK" HIP_PATH="$EXL3_SDK"
export PATH="$EXL3_SDK/bin:$EXL3_SDK/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$EXL3_TORCH_LIB:$EXL3_SDK/lib"
export LD_PRELOAD="$EXL3_HSA"
export EXL3_BACKEND=rocm PYTORCH_ROCM_ARCH="$GPU_TARGET" GPU_ARCHS="$GPU_TARGET"
export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=$GPU_TARGET"
export HIPCC_LINK_FLAGS_APPEND="--offload-arch=$GPU_TARGET"
export MAX_JOBS=2
cd "$EXL3_REPO/vendor/rocm-exl3"
"$EXL3_PYTHON" setup.py build_ext --build-temp "$EXL3_BUILD/temp" --build-lib "$EXL3_BUILD/lib"
sha256sum "$EXL3_BUILD/lib/"exllamav3_ext*.so
```

Choose an HSA library compatible with your driver stack and limit build parallelism to available RAM. A fresh build does not use `EXL3_REUSE_MANIFEST`; that optional development hook requires matching source/object/compiler dependencies. Private manifests are excluded.

## Register

Copy `configs/local.example.toml` to ignored `configs/local.toml` if it does not already exist. Fill in paths, GPU target and extension SHA256. Set execution permissions true when ready to authorize inference.

Use `native_smallm_max_rows = 9` with the current full build; never declare a larger envelope than the binary implements.

Current builds automatically use packed kernels for eligible 10–64-row
projections when the verified binary exposes packed-mid ABI 1. No enabling
flag is needed. Older binaries keep reconstruction fallback; `--no-packed-mid`
also selects that fallback for diagnosis. Keep the installation's
small-M/native-graph limit at 3, 5 or 9.
[Packed projection implementation and checks](PACKED-MLP-PERFORMANCE.md).

Current builds expose both cb0 and mul1 small-M capabilities. The runtime probes
the verified extension for that capability; older binaries retain their
cb0-only fused eligibility. Rebuilding and registering the new hash is required
to enable fused mul1 on an existing installation.

Current builds also expose packed-prefill and paired-MLP ABI 1. Eligible
shapes use them automatically, with dense fallback beyond the measured
crossover. `--no-packed-prefill` and `--no-mlp-pair` disable them for diagnosis.
The default `--prefill-gemm auto` selects verified dense WMMA when available
and BLAS otherwise. Explicit `wmma` requires prefill GEMM ABI 1. Rebuild and
register the actual new SHA-256 to use capabilities absent from an older
installation. [Shape policy and validation](PREFILL-MLP-FUSION.md).

Repacked-head ABI 3 enables the compressed vocabulary-head view automatically
on gfx1101. Older binaries and prototype ABIs 1/2 retain the inherited head.
FP16 and FP32 outputs preserve their original arithmetic and output dtype.
The adapter allocates at most 1 GiB of packed storage per distinct head and
leaves a 1 GiB workspace reserve within available VRAM and the allocator
fraction. Shared target/MTP heads reuse one view. If memory, shape or format
checks fail, the original projection remains available. No model-file
conversion or enabling flag is required. [Head policy and measurements](HEAD-ATTENTION-PERFORMANCE.md).

The latest sources also include specialized K5/K6 unpacking, format-qualified
projection-loop unrolling, staged packed prefill and compact multi-token GDN
recurrence. They are selected automatically within their supported geometry;
single-token GDN and unsupported shapes keep the previous implementation.
Rebuild and register the resulting binary to receive these changes. Native
binaries and benchmark artifacts are not source dependencies or release assets.

Narrow-GEMM ABI 1 (`quantlab_exl3_narrow_gemm_abi`) adds a deterministic dense
kernel for decode-sized projections: FP16 GatedDeltaNet `in_proj_a`/`in_proj_b`
through `hgemm` and the BF16 MTP draft projections. It is selected automatically
from the verified binary; `--no-narrow-gemm` restores BLAS for diagnosis and older
binaries keep BLAS. Validate a build with `--checks narrow-gemm`.
[Profile and measurements](DECODE-PROFILING.md).
See [native kernel qualification](NATIVE-KERNEL-PERFORMANCE.md) for the measured
9B/27B gains, rejected experiments and hardware limits.

```powershell
python scripts/register_runtime.py configs/local.toml
python run.py --help
python run.py -m "MODEL_DIRECTORY" -c 4096 -n 32 -p "Say hello."
```

Use `python3` on Linux. Registration writes ignored `.runtime/installation.toml`, selects no model and refuses overwrite. The launcher verifies the extension hash and permission gates before execution.

Conversion utilities need original high-precision weights and explicit calibration inputs. Bundled upstream calibration/reference corpora are excluded. Pass `--cal_data` a Safetensors file containing `input_ids` of shape `(rows, columns)`, produced with the same tokenizer; use `--cal_rows` and `--cal_cols` within that file's dimensions. Check the vendored converter's `--help` for other options. Cloning supplies neither a model nor calibration data.

Conversion error reporting processes complete token rows in bounded chunks,
avoiding full-size FP32 copies of large vocabulary outputs. This changes only
diagnostic reduction order, not calibration or quantized weights. The guarded
conversion wrapper retains fatal Python traces; its external supervisor owns
timeouts. Periodic background traceback dumps are disabled after a crash in
CPython's thread-dump routine was observed on WSL/Python 3.12.

An existing-environment smoke test does not establish a clean build or package installation on another machine. See [VALIDATION.md](VALIDATION.md).
