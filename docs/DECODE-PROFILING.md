# Decode kernel profiling and MiMo decode tuning

September 28, 2026. Implements idea 5 of [OPTIMIZATION-PLAN.md](OPTIMIZATION-PLAN.md):
get per-kernel GPU self-time for a warm MTP decode round, then tune the
dominant kernels on MiMo's real shapes. RX 7800 XT (`gfx1101`), WSL distro
`EXL3-Runtime`, Torch 2.13.0+rocm7.2, MiMo 9B K5/H6 mul1 with BF16 MTP depth 2,
Q6/Q6 KV, 131072 context, chunk 1024, greedy decoding.

**Result:** warm MTP decode is **16% faster** on the serving configuration
(77.6–78.6 → 90.7 tokens/s short, 64.7–65.6 → 76.1 tokens/s at 16K occupied;
29.8 → 25.7 ms and 33.1 → 28.8 ms per round). Two changes are primary:

- a **narrow dense GEMM** for decode-sized projections: GatedDeltaNet
  `in_proj_a`/`in_proj_b` (4096 → 32, FP16 in, FP32 out), which BLAS ran at
  ~85 µs per call (48 calls, 13% of a round), and the BF16 MTP draft
  projections;
- **four-way K5/K6 loop unrolling** in the paired-MLP and small-M dot kernels
  at ≤ 3 rows (~1%, bit-identical).

`--no-narrow-gemm` is the diagnostic override. rocprofv3 cannot trace kernels
on this WSL stack; the event-based profiler below replaces it. The GPU is busy
for about 94% of a round, so launch/synchronization gaps (the megakernel
target, idea 7) are about 1.5–2 ms per round.

## 1. rocprofv3 on WSL

`rocprofv3` (rocprofiler-sdk 1.1.0, ROCm 7.2.0 in `/opt/rocm`) was run around a
small Torch workload in the backend environment.

1. Plain `rocprofv3 --kernel-trace --stats -- python …` aborts:
   `api registration failed with error code 16: Configuration request occurred
   outside of valid rocprofiler configuration period`. The Torch wheel bundles
   its own `librocprofiler-register.so` and `librocprofiler-sdk.so` (no SONAME),
   so HIP registers with a second SDK instance that never ran the tool's
   configuration. `ROCPROFILER_REGISTER_LIBRARY=/opt/rocm/lib/librocprofiler-sdk.so.1`
   does not change this.
2. Forcing one SDK instance works for the host API: a directory containing
   `librocprofiler-sdk.so → /opt/rocm/lib/librocprofiler-sdk.so.1`, placed first
   in `LD_LIBRARY_PATH`, plus `rocprofv3 --preload librocprofiler-sdk.so` (bare
   name, so the loader matches Torch's later `dlopen`). HIP API tracing then
   produces `hip_api_trace.csv`/`hip_api_stats.csv`.
3. Kernel, memory-copy and HSA traces stay empty. The tool logs
   `sysfs nodes path '/sys/class/kfd/kfd/topology/nodes' does not exist` (there
   is no `/dev/kfd`), and the WSL HSA runtime (`/opt/rocm/lib/libhsa-runtime64.so.1`,
   linked against `libdxcore.so`) does not link `librocprofiler-register`, so
   it never registers its API table. Without HSA interception there are no
   agents and no dispatch records. Kineto fails for the same reason.

Conclusion: device-side kernel tracing and counters are unavailable on this
installation; host HIP API tracing is possible with the workaround in step 2.
A native Linux boot or a ROCm stack whose WSL HSA registers with
rocprofiler-sdk would be needed for hardware traces.

## 2. Event-based per-kernel profiler

`scripts/profile_decode_round.py` builds the real serving engine
(`serve_exl3.Engine`, including startup warm-up) with the given serve flags,
under the installation's GPU lease and resource monitor, and decodes with fresh
greedy generators. Each decode round (one `Generator.iterate()`: two draft steps
plus a three-row verification) is measured in three passes:

| Pass | What it measures |
| --- | --- |
| `normal` | Uninstrumented wall time per round and emitted tokens. |
| `segments` | GPU busy time. After every host synchronization (scalar reads, blocking or pageable-memory copies, `synchronize()`) a spin kernel holds the GPU while the host enqueues the next segment, so each segment then runs back to back; one event pair per segment. |
| `kernels` | Same pre-queued rounds with an event after every GPU operation: native extension calls, whole EXL3 projections, Triton launches and Torch ops, labelled by phase, module and shape. |

`normal − segments` is the time the GPU waits for the host. Each segment
reports its host enqueue time against its spin; a segment whose host needed
longer than the spin was starved and is counted (zero in the runs below).
Calls that block the host for a spin-length time are listed so undetected
synchronizations can be found; this is how the composite `aten.to` copies were
found and added.

```bash
# inside EXL3-Runtime, from the repository root
python3 scripts/profile_decode_round.py -m /mnt/d/LOCAL-MODELS/MiMo/EXL3/MiMo-K5-H6-mul1-MTP-BF16 \
  --output artifacts/<new-dir> --occupied 0 16384 --tokens 96 \
  --passes normal segments kernels normal --wait 1800
```

Defaults are the MiMo serving flags with `--temperature 0`; `-- FLAGS…` replaces
them, `--serve-flag=…` appends one, `--config` selects a candidate installation
record and `--env EXL3_…=…` sets a diagnostic dispatch variable. Results are in
`result.json` (per-label ms per round, segment details) and `rounds-*.json`.

Limits: an event marker costs a few microseconds of GPU time, so the `kernels`
sum exceeds the `segments` busy time (33.3 vs 28.5 ms) and small operations are
overstated; compare shares, and use `segments` for totals. The first operation
after each spin reads 15–55 µs high (measured separately), which inflates busy
time by roughly 0.5 ms per round, so the true host gap is a little larger than
reported. A native call made through a reference captured before the hooks are
installed is attributed to the next operation. Sampling (temperature > 0) is
not profiled; greedy rounds have the same kernels except the sampler.

## 3. Warm round profile (baseline `gdn-core-v4`)

Run `artifacts/decode-profiling-20260928/p7`. 96 tokens per case, first two
decode rounds excluded; ~522 profiled operations (about 860 kernel launches)
per round. A back-to-back trivial kernel costs 2.0–2.3 µs of GPU time.

| Occupied context | Round wall (`normal`) | GPU busy (`segments`) | GPU waits for host | Tokens per round |
| --- | ---: | ---: | ---: | ---: |
| Short (76-token prompt) | 29.36 ms | 28.51 ms | 0.85 ms (2.9%) | 2.22 |
| 16,384 | 32.68 ms | 31.57 ms | 1.11 ms (3.4%) | 2.14 |

Ranked GPU time per round by category (`kernels` pass, short prompt; each
value includes one event marker per operation):

| Category | Ops/round | ms/round | Share |
| --- | ---: | ---: | ---: |
| MLP gate/up (paired K5, 4096→2×12288, 3 rows) | 32 | 5.82 | 17.5% |
| Vocabulary head (packed K6; 2 draft rows + 1 three-row verify) | 3 | 4.45 | 13.4% |
| GDN `in_proj_a`/`in_proj_b` (dense FP16 4096→32, BLAS) | 48 | 4.26 | 12.8% |
| GDN projections qkv/z/out (K5) | 72 | 4.02 | 12.1% |
| MLP down (K5, 12288→4096) | 32 | 3.17 | 9.5% |
| MTP draft BF16 projections (BLAS) | 16 | 3.02 | 9.1% |
| GDN core (conv, recurrence, gates, gated norm) | 96 | 1.88 | 5.7% |
| Norms and residual adds | 97 | 1.66 | 5.0% |
| Attention projections q/k/v/o (K5) | 32 | 1.50 | 4.5% |
| Attention core (paged Q6 decode, cache write, RoPE) | 48 | 1.20 | 3.6% |
| MTP draft other (norms, attention, concat, argmax) | 28 | 0.98 | 2.9% |
| MTP draft KV update after verification | 11 | 0.78 | 2.3% |
| Sampling, rewind and bookkeeping | 8 | 0.56 | 1.7% |

At 16K the paged attention grows to 3.36 ms (9.2%); everything else is
unchanged. Individual projections, in-round: paired gate/up 182 µs (63 MB),
down 99 µs (31.5 MB), qkv 69 µs (21 MB), out/z 56/42 µs (10.5 MB), draft head
1.44 ms and target head 1.57 ms (763 MB each, ~500 GB/s). The two tiny GDN gate
projections took 88–90 µs each, as long as a 21 MB packed projection.

**Between-kernel gaps (for idea 7):** the host-induced idle is 0.85–1.1 ms per
round before tuning and 1.5–1.8 ms after it (the GPU work shrank; see §5),
about 3–6% of a round, plus roughly 2 µs of GPU-side dispatch per kernel
(~1.8 ms for ~860 launches) that a persistent kernel could also remove. A
megakernel could therefore save at most roughly 10% of a warm round; the
remaining 90% is kernel execution.

## 4. Tuning on MiMo shapes

Operator timings use back-to-back calls over every layer's distinct weights
behind a spin (`seq_bench.py` in the evidence directory), which reproduces the
in-round projection times. Full-model comparisons use the profiler's `normal`
pass: baseline, candidate, baseline in separate processes, 128 tokens at short
and 16K occupied contexts, three measured passes each.

### Narrow dense GEMM — primary

The GDN gate projections are `LinearFP16` with FP32 output, so they go through
`extension.hgemm` → `hipblasGemmEx`. For M=3, N=32, K=4096 BLAS takes ~81 µs
back to back (~107 µs single). The new kernel (`rocm/narrow_gemm.hip`) gives
each block 32 output columns and spreads K over 1,024 threads, with a fixed
shuffle/LDS reduction order: **7 µs** per call. Relative error against an FP64
reference matches BLAS (max abs 9.1e-6 vs 1.1e-5 on the test shape).

The same kernel with 8-column vectors and a block size chosen so the grid fits
the device in one pass serves the BF16 MTP projections (`narrow_gemm_bf16`),
replacing `torch.matmul`:

| BF16 projection (1 row) | BLAS µs | Narrow µs | Narrow GB/s |
| --- | ---: | ---: | ---: |
| 4096 → 8192 (q) | 137 | 133 | 507 |
| 4096 → 1024 (k, v) | 45 | 21 | 407 |
| 4096 → 12288 (gate, up) | 246 | 193 | 521 |
| 12288 → 4096 (down) | 248 | 184 | 547 |
| 8192 → 4096 (fc) | 172 | 134 | 500 |

A first version with 1,024-thread blocks for every width lost on 4096→8192 and
12288→4096 (a nearly empty second wave of blocks) and was revised before model
testing. hipBLASLt instead of rocBLAS (`preferred_blas_library`) did not help.

Envelope: FP16 1–8 rows, 8–128 output columns (multiples of 8), FP32 or FP16
output, through `hgemm` (other shapes keep BLAS); BF16 1–4 rows, N a multiple of
8, contiguous 16-byte-aligned weights. Capability: exported
`quantlab_exl3_narrow_gemm_abi() == 1`, probed by `configure_native`, which
sets `EXL3_NARROW_GEMM`. Older binaries and `--no-narrow-gemm` keep BLAS.

### K5/K6 four-way unroll — primary

The paired-MLP and small-M dot loops overlapped two tiles for K4–K6 at ≤ 3
rows. Four tiles for K5/K6 (K4 keeps two) gave 1–4% on the MiMo projections
(down 93.3 → 90.1 µs, qkv 63.7 → 62.4 µs, pair 166 → 164.5 µs) with
byte-identical outputs, and +0.8% decode in the full model (below).

### Rejected

| Change | Result |
| --- | --- |
| mul1 hash with two 24-bit multiplies instead of `v_mul_lo_u32` | Bit-identical, no speedup (down 88.7 vs 90.1 µs, within noise). |
| Split-K warps 4 or 16 instead of the heuristic (`EXL3_GEMV_SPLITK_WARPS`) | No gain (down 96.8 / 87.7 vs 88.6 µs, pair 161 / 168 vs 166 µs) and changes reduction order. |
| Hypothesis that the K5 projections are DRAM-bound (motivating more loads in flight) | Cache-resident weights (same layer repeated) run as fast as streamed ones (down 94 vs 89 µs): the small-M kernels are limited by decode/issue work at ~280–390 GB/s logical, not by DRAM. |

## 5. Full-model results

Medians of three passes; tokens per round from the first pass. Runs in
`artifacts/decode-profiling-20260928/ab/`.

| Configuration (run) | Short ms/round | Short tok/s | 16K ms/round | 16K tok/s |
| --- | ---: | ---: | ---: | ---: |
| Baseline (`nv2-base1`) | 30.00 | 77.63 | 33.16 | 64.92 |
| Narrow GEMM + unroll (`nv2-cand`) | **25.68** | **90.71** | **28.81** | **76.10** |
| Unroll only, `--no-narrow-gemm` (`nv2-cand-nonarrow`) | 29.57 | 78.78 | 32.67 | 65.88 |
| Baseline (`nv2-base2`) | 29.76 | 78.12 | 32.99 | 65.34 |
| FP16 narrow GEMM only, earlier build (`hg1-cand`, bracket `hg1-base1/2`: 77.85–77.96 / 64.92–65.18) | 27.15 | 85.93 | 30.00 | 71.88 |

Against the faster control: +16.1% short and +16.5% at 16K for the promoted
binary; the unroll alone +0.8% in both; FP16 narrow alone +10.2% / +10.3%, so
the BF16 draft projections add about 1.1 ms per round.

After tuning (`p8-narrow-v2`), a short round is 25.7 ms with 24.2 ms GPU busy
(16K: 28.8 / 27.0 ms). The draft/head share is now larger: heads 15.6%, paired
MLP 20.4%, GDN projections 13.9%, GDN gate projections 2.8% (0.81 ms, marker
dominated).

Qwenseek 27B (K2, quantized MTP, Q6, GDN decode fusions; runs `q27-base1`,
`q27-cand`, `q27-base2`, two passes each): controls 40.65–40.94 tokens/s short
and 38.97–39.23 at 4K occupied, candidate 41.00–41.05 and 39.26–39.31, with
identical outputs. The gain (+0.2% against the faster control) is within drift:
its fused GDN path spends little time in these GEMMs. This is a no-regression
check, not a 27B speed claim.

### Output equivalence

Unroll changes are bit-identical. The narrow GEMM changes accumulation order,
so greedy outputs can diverge at near-ties:

- 128-token runs: all short cases identical to baseline; at 16K the FP16
  kernel alone was identical, the full candidate diverged.
- 256-token runs (`diag-*`): identical at short, 4K and 32K occupied. At 16K
  the candidate diverges from baseline at generated token 114 without MTP and
  at token 40 with MTP; both continuations are equivalent code/docstring text
  (for example "or -1 when target is absent" vs "not present").
- Native checks: FP32 results within 2e-6 relative RMS of an FP64 reference
  (BLAS meets the same bound); BF16 results within a few BF16 ulps of BLAS.

Draft changes alter MTP windows, which also changes the verification batch
alignment; acceptance was unchanged on the short case (2.321 tokens/round).

## 6. Validation

- Native checks on the promoted build (`narrow-v2`, SHA-256
  `f3c131266a3f8461bdd0b7f9bec36c0c986225b9e072fb1eeeb19923d9370047`):
  narrow-gemm 119, hgemm 56, smallm 408, highbit-smallm 60, mlp-pair 143,
  packed-mid 929, packed-prefill 232, head-tiled 185, gdn-recurrent 75,
  attention-schedule 116 — all passed (`scripts/check_exl3_kernels.py`, new
  suite `narrow-gemm`).
- Serving smoke with the candidate record: two sampled requests (temperature
  0.6, speculative sampling) at 101 and 87 tokens/s decode and two identical
  greedy 300-token requests at 98 tokens/s; clean shutdown
  (`RUN-20260928T205830Z-0726dd0b`).
- Unit suite: Windows 863 tests OK (66 skips), Linux runtime 863 OK (12 skips).
  New tests: `tests/test_profile_decode_round.py`, narrow-GEMM capability
  tests in `tests/test_exl3_optimizations.py`, `--narrow-gemm` forwarding in
  `tests/test_launcher.py`, BF16 dispatch in `tests/test_mtp_precision.py`.
- The rebuilt binary reuses 107 hash-verified objects from `gdn-core-v4`;
  `hgemm.cu`, `bindings.cpp`, `rocm/quant/exl3_gemv_rdna.hip` and the new
  `rocm/narrow_gemm.hip` were compiled. Its recorded source state matches the
  checked-in native sources.

## 7. Limitations and follow-ups

- Measured on one GPU and MiMo 9B; the 27B check is no-regression only. Other
  models with small dense FP16/BF16 projections use the kernel by shape.
- Greedy outputs are not bit-identical to BLAS; see §5.
- Sampled serving (speculative sampling) uses the same draft projections. It
  was measured afterwards in the combined run on this binary: 84.3 tokens/s
  with the serving sampler and 87.4 greedy (`RUN-20260928T210253Z-0ac321a6`,
  [VALIDATION.md](VALIDATION.md)).
- The K5 small-M kernels run at 280–390 GB/s and are not DRAM-bound; the next
  gain there needs a different inner loop (for example fusing the input
  Hadamard, or fewer VMEM operations per tile), not warp or unroll tuning.
- Host/launch gaps are now ~6% of a round plus ~2 µs per kernel; idea 7 has at
  most about a 10% ceiling on this configuration.
