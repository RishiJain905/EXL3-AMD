# GPU and codebook compatibility

The active native source is `vendor/rocm-exl3`. Rebuild the extension and register
its actual SHA-256 to use these changes. Historical kernel copies and patches in
`kernels/exl3` are references, not alternate build inputs.

## mul1 small-batch projections

`mul1` is an EXL3 codebook: a multiply-based mapping from packed codes to
reconstructed weight values. Bits per weight are configured separately; this
runtime's fused mul1 path handles 2-, 3- and 4-bit packed projections.
See [real-model measurements](MUL1-VALIDATION.md) for the completed 9.402 GB
conversion, repeated inference, occupied 120K tests and observed quality limits.

The fused small-M path accepts the default codebook (`cb0`) and `mul1` (`cb2`)
for 2-, 3- and 4-bit packed projections, 2–9 token rows, and input/output widths
divisible by 128. This covers dot, WMMA and register-B WMMA implementations,
FP16/FP32 outputs, split-K and native graph replay. Single-row GEMV and the
reconstruction path already supported mul1 and remain available.
Codebook selection comes from the model's layer metadata; no new CLI flag is
needed. Existing `--smallm-kernel` choices select the implementation.

For a local quantized model, the CPU storage audit reports codebook marker
counts, packed bit widths and inconsistent marker metadata. An optional byte
ceiling includes the entire model directory, including MTP and metadata:

```bash
python3 scripts/audit_exl3_candidate.py --candidate MODEL_DIRECTORY \
  --output .runtime/model-audit.json --max-total-bytes 9500000000
```

The output file must be new and outside the model directory. Exceeding the
explicit ceiling writes the report and returns a nonzero exit code. The audit
does not establish model completeness, numerical correctness or GPU memory fit.

The extension exports `quantlab_exl3_smallm_codebooks()`: bit 0 is cb0 and bit 2
is mul1, so the current value is 5. The launcher discovers this capability from
the hash-verified binary. Older binaries without the export retain cb0-only
small-M eligibility; mul1 uses the existing fallback. Decode fusion retains its
single-row path for these layers and falls back for larger batches. Multirow
native graphs are admitted only when every packed projection in the module is
supported by the verified binary. Optimization ABI 2 stays
unchanged. `mcg` (`cb1`), other bit widths and unsupported shapes retain their
existing fallback; this change does not expand their fused small-M envelope.

## RDNA4 implementation

`gfx1200` and `gfx1201` use RDNA4 WMMA intrinsics in the shared
`rocm/rdna_wmma.hip.h` header. Its adapters preserve the external fragment
layout expected by packed GEMM, small-M, attention, MoE and prefill callers.
They select the native eight-element operands and exchange accumulator elements
between lanes. FP16/BF16 inputs, FP32/FP16 accumulation and INT8 signedness/clamp
variants are covered. FP16 accumulation preserves the unselected half slots.
This compatibility adapter adds shuffle overhead; it is not RDNA4 performance
tuning and makes no throughput claim.

The opt-in `--prefill-gemm wmma` dispatcher admits gfx1200/gfx1201 as well as
gfx1101. Its existing shape, storage and fallback checks still apply. The
`--attention-profile long` tuning remains scoped to gfx1101; RDNA4 retains the
default attention scheduling. GPU-specific compiler paths, Torch/Triton and
driver support are separate requirements.

The layout follows AMD's runnable
[RDNA4 WMMA example](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/)
and [matrix core guide](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/),
with intrinsic signatures checked against
[Clang's AMDGPU reference](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html).
AMD's matrix calculator currently gives a conflicting FP16 input mapping; this
implementation follows the GPUOpen example. Real RDNA4 numerical validation is
still required to settle that discrepancy for this runtime/toolchain.

## Build targets and validation

| Target | Status |
| --- | --- |
| gfx1101 (RX 7800 XT) | Hardware available; operator and model regressions recorded in [VALIDATION.md](VALIDATION.md) |
| gfx1200, gfx1201 (RDNA4) | Full extension and primitive probes cross-compiled; experimental until hardware numerical validation |
| Other listed RDNA3/3.5 targets | Existing build targets; not newly hardware-tested by this change |

For a single GPU, use [BUILD.md](BUILD.md) with `GPU_TARGET=gfx1200` or
`GPU_TARGET=gfx1201`. A compiler targeting RDNA4 does not need an RDNA4 GPU.
For a multi-target build, replace the single-target environment settings with:

```bash
export PYTORCH_ROCM_ARCH='gfx1101,gfx1200,gfx1201'
export GPU_ARCHS="$PYTORCH_ROCM_ARCH"
export HIPCC_COMPILE_FLAGS_APPEND='--offload-arch=gfx1101 --offload-arch=gfx1200 --offload-arch=gfx1201'
export HIPCC_LINK_FLAGS_APPEND="$HIPCC_COMPILE_FLAGS_APPEND"
```

Use a fresh build directory whenever headers or target lists change. Do not use
architecture spoofing to run a gfx11 code object on RDNA4. The registered
`gpu_arch` should describe the actual GPU; a matching multi-target binary is
acceptable.

## Model-free acceptance on a tester's GPU

Run inside the registered Linux/WSL environment, from the repository root:

```bash
python3 scripts/check_exl3_kernels.py --output .runtime/checks/my-gpu
```

Both local execution permissions must already be true in the registration. The
command verifies the extension hash, acquires the configured GPU lease, and
runs under a 900-second maximum, host RAM/disk guards and a 50% Torch allocator
limit. It downloads nothing and forbids native JIT builds. Results include the
device, architecture, Torch/HIP versions, extension hash and individual checks.
It exercises mul1/cb0 decoding, all three small-M kernels, graph replay and
prefill GEMM. It is a correctness test, not a throughput benchmark.

To test an isolated build without changing registration, provide both
`--extension-dir /path/to/build/lib` and
`--expected-extension-sha256 YOUR_ACTUAL_SHA256`. Use a new output directory
for every run; a partial result file does not establish success—check the
command exit and `monitor.json` too.

The standalone primitive probe needs only HIP:

```bash
hipcc -std=c++17 -O2 --offload-arch=gfx1201 \
  kernels/exl3/check_wmma.hip -o .runtime/check-wmma-gfx1201
sha256sum .runtime/check-wmma-gfx1201
python3 scripts/check_exl3_kernels.py --output .runtime/checks/my-gpu-with-primitives \
  --wmma-probe .runtime/check-wmma-gfx1201 --expected-wmma-sha256 YOUR_PROBE_SHA256
```

Compile it for the actual device. The validator runs the hash-verified probe
under the same GPU lease with a 120-second probe timeout. It checks asymmetric scalar-CPU references, both FP16
accumulator half slots, all INT8 signedness combinations and saturation. Its
2,304 layout checks execute the actual RDNA4 adapters on either GPU generation;
running those adapters on RDNA3 does not execute RDNA4 WMMA instructions.

Before promoting RDNA4 from experimental, run both suites on each RDNA4 target,
then a real mul1 model with prefill, decode, MTP where available, and graph replay.
Include default and optimized kernels, FP16/Q8/Q4 caches, supported model
families, clean installation and sustained serving in the hardware acceptance
matrix. Report exact configuration and output comparisons; do not infer all
model, OS or GPU compatibility from one passing configuration.
