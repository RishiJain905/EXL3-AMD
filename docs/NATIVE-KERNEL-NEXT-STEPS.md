# Native kernel opportunities before a megakernel

Source audit, September 26, 2026, against primary runtime commit `23b8d9b`.
The original audit below records candidates; its implementation follow-up
is now complete and measured in the linked report. The
current target is RX 7800 XT / gfx1101, MiMo 9B K5/H6 mul1 and Qwenseek 27B
default-codebook K2 body/K4 head. A bounded OMP audit covered GDN source;
the coordinator inspected projection, dequantization and prefill kernels.

Implementation follow-up, September 27: the user authorized all three stages
in the order below. All three stages are qualified and primary. After the
requested pause and local model-storage relocation, stage three qualified a
compact recurrent kernel for supported multi-token ROCm calls. Broader
single-token dispatch, cross-token state retention and convolution specialization
failed the model performance gates and were rejected. Measured changes,
rejected experiments and current status are in
[NATIVE-KERNEL-PERFORMANCE.md](NATIVE-KERNEL-PERFORMANCE.md).

## What is already native

The packed 10–64-row path, tiled packed prefill, FP32-output dense WMMA,
paired gate/up plus SwiGLU, compressed vocabulary head, and attention kernels
are real GPU implementations. Python selects supported paths and integrates
them with the model. The most recent residual/graph investigation reused the
existing binary; it did not add new projection kernels. Its results are in
[FUSION-DECODE-PERFORMANCE.md](FUSION-DECODE-PERFORMANCE.md).

## First priority: packed projection inner loops

The [dequantization dispatcher](../vendor/rocm-exl3/exllamav3/exllamav3_ext/rocm/quant/exl3_dq_rdna.hip.h)
already has aligned specialized 2-bit and 4-bit unpackers. K5 and K6 instead
call generic `dq4` twice for each eight-value fragment. This is a concrete
MiMo candidate: specialize the bit extraction, share loaded words where
possible, and inspect the emitted instructions before assuming the compiler
has left redundant work. Preserve the trellis's circular boundary semantics
and exact codebook values; this does not change quantization or model weights.

The [single-row dot loop](../vendor/rocm-exl3/exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_kernel_rdna.hip.h),
[small-row dot loop](../vendor/rocm-exl3/exllamav3/exllamav3_ext/rocm/quant/rdna-smallm.hip.h)
and [paired MLP loop](../vendor/rocm-exl3/exllamav3/exllamav3_ext/rocm/quant/rdna-mlp-pair.hip.h)
load/decode each packed tile and consume it with dot instructions. Test modest
unrolling or explicit next-tile staging to overlap memory, decode and dot work,
while retaining the accumulation order. This is deeper kernel work than another
warp-count sweep, which [already failed the model speed gate](PACKED-MLP-PERFORMANCE.md).
The 27B's K2 unpacker is already specialized; measure its pipeline separately.

Benchmark actual gate/up/down and QKV/output shapes at rows 1/2/3/5, with warm
and cache-evicted weights, then verify complete-model speed. Check register use
and scratch spills: more in-flight tiles can also slow the kernel. Head K-loop
unrolling already showed that tradeoff at five rows; do not blindly extend it.

## Second priority: prefill matrix kernels

The [packed-prefill kernel](../vendor/rocm-exl3/exllamav3/exllamav3_ext/rocm/quant/rdna-packed-prefill.hip.h)
uses one 128-by-64-by-32 tile, 256 threads and a repeated load/decode, barrier,
WMMA, barrier sequence. It has no explicit next-tile pipeline. Test staged
prefetch or double buffering of packed weights and activations, and alternative
tile sizes that reuse decoded weights without excessive register/LDS use.

The important gate is whether a revised kernel beats the current dense path
at larger rows. The previous wider packed policy lost: automatic selection is
currently limited to FP16 output through 128 rows and FP32 output through 512
rows. Merely enabling the existing kernel for all rows is not an optimization.
See [the existing crossover and numerical results](PREFILL-MLP-FUSION.md).

The [dense WMMA alternative](../vendor/rocm-exl3/exllamav3/exllamav3_ext/rocm/hgemm_wmma.hip)
already prefetches at 512 or more rows. It selects between two tile shapes by
row count alone. Test additional shapes using K, N and row count, particularly
4096↔12288 and 5120↔17408, and different thread counts/register budgets. The
last recorded prefetch build used 169 vector registers with no scratch spills;
that is a reason to measure resource tradeoffs, not proof of an occupancy
bottleneck. Earlier 128-by-128 prefetch and transposed-LDS trials lost, as
recorded in [RUNTIME-PERFORMANCE.md](RUNTIME-PERFORMANCE.md).

## Third priority: GDN recurrent core

The original native [GDN rule](../vendor/rocm-exl3/exllamav3/exllamav3_ext/gdn.cu)
used four block barriers and two shared-memory atomic reduction phases per
token in its 128-dimensional specialization. It read recurrent state for
the first dot product and again for the update/output pass. Candidates are
an explicit reduction layout in place of atomics, retaining selected state
fragments in registers, and reducing repeated per-head scalar work. All
trade synchronization or memory traffic against register pressure. The
follow-up measured these tradeoffs and promoted the narrower result below.

The qualified GDN implementation is a compact ROCm specialization for 128×128
heads with `V_SPLIT=4`, batch one, at most 64 value heads and more than one
token. The original 512-thread launch had only 128 threads performing
recurrent-state work. The new kernel uses 128 threads, distributes
Q/K normalization elements uniquely across them, retains each thread's 32 state
values between the two passes, and replaces the atomics with explicit four-way
partial sums. Separate partial buffers avoid a producer overwriting data that
another wave still needs. Existing single-token, CUDA, other-head-dimension
and unsplit kernels remain the fallback. There is no new enabling option.

Qualification compared state and outputs against an independent reference,
checked saved history and rewind, exercised graph replay with changed inputs
and slots, and measured both actual model head counts before full 9B/27B
comparisons. All 135 final native checks and 13,824 measured output tokens
passed, including exact model MTP windows. Qualifying MTP decode cases improved
0.86–1.32% on MiMo and 2.68–2.89% on Qwenseek against the faster controls;
other cases and limitations are retained in the performance report.

Keep MTP history slots, rewind, state layout and BF16 rounding correct.
The existing graph is already grouping native calls, so another Python
wrapper was not the change. A secondary experiment specialized native
convolution on width four to remove runtime predicates and make its short
window explicit. It was reverted after model regressions despite favorable
isolated timings; the final convolution kernels are unchanged.

This targets decode. Large prefills normally use the separate FLA chunked
rule when available and history is not requested; optimizing the recurrent
fallback cannot be credited as a normal prefill improvement. The models have
24/32 and 48/64 GDN blocks, respectively, but block counts are not time shares.

## Measurement and promotion

The latest Kineto probe did not expose GPU kernel events or hardware counters.
Inclusive module intervals point toward block/projection work but do not
identify bandwidth saturation, instruction stalls or kernel self-time. Collect
usable traces/counters where supported; independently time native operators
and inspect compiled instructions/resource metadata when counters are absent.
Do not describe weight bytes divided by wall time as measured DRAM traffic.

The implementation order is packed projection inner loops, prefill matrix
kernels, then GDN. All three stages are complete; their qualified improvements
are primary automatically. Isolated timing screened candidates, and full
model measurements determined promotion.
Use shape, format, dtype and GPU metadata for selection, allowing future
models with supported geometry to benefit. Preserve guarded fallbacks and
promote only repeated full-model improvements with correctness validation.

Lower token latency is the objective. Maximum bandwidth or occupancy is not
automatically maximum performance; AMD explains the register, cache and
latency tradeoffs in its [occupancy guide](https://gpuopen.com/learn/occupancy-explained/)
and [HIP performance guidance](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html).
A megakernel should follow evidence of substantial remaining launch or
intermediate-transfer cost, not a bandwidth percentage assumed from model size.

No native build, GPU benchmark or runtime change was made during the original
September 26 source audit. The subsequent implementation and validation are
documented separately above. The audit's OMP job exited zero; its package,
findings and completion record are
retained privately under `.codex/.omp-jobs/kernel-audit-20260926/`.
