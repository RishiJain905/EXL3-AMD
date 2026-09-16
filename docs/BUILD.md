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

Current builds expose both cb0 and mul1 small-M capabilities. The runtime probes
the verified extension for that capability; older binaries retain their
cb0-only fused eligibility. Rebuilding and registering the new hash is required
to enable fused mul1 on an existing installation.

The current ROCm build also includes prefill GEMM ABI 1 for the optional
`--prefill-gemm wmma` path. Older registered binaries continue to support the
default `blas` setting; selecting WMMA requires rebuilding and registering
the resulting binary with its actual SHA-256.

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
