# Residual fusion and MTP graph investigation

This round starts from main commit `23b8d9b` and the primary packed MLP,
prefill, vocabulary-head, and occupied-context attention optimizations. The
requested scope is additional fusion and larger decode/MTP graph regions.
The 9B and 27B profiles were refreshed before implementing candidates.

Completed September 26, 2026. No candidate was promoted. Residual fusion and
batched token readback did not establish a repeatable useful gain. The wider
MTP graph gives a small warmed 9B decode improvement, but adds first-token
latency when captured and does not consistently reduce total request time;
the 27B results are neutral or slower. The experimental runtime hooks were
removed after preserving their exact source and measurements privately.
The current primary inference implementation and native binary are unchanged;
no new opt-in optimization flags were added.

## Fresh profiling baseline — September 26, 2026

Hardware: RX 7800 XT (`gfx1101`), Ubuntu under WSL, Torch
`2.13.0+rocm7.2`, HIP `7.2.53211`. The verified native extension remains
`519a2343da6aad14b8ec7c8cd6c76266a6fb54c7c5860a665a6eed3449957576`.
These candidates reuse the existing native binary.

| Model | Draft and cache configuration | Short decode | 4,096 occupied | 16,384 occupied |
|---|---|---:|---:|---:|
| MiMo 9B K5 body/K6 head | BF16 MTP2, rowwise verifier, F16 KV, decode fusions off | 90.85 tok/s | 83.20 tok/s | 60.49 tok/s |
| Qwenseek 27B | Quantized MTP2, Q6 KV, default GDN fusions | 43.80 tok/s | 38.76 tok/s | 34.81 tok/s |

The short number is the median of three uninstrumented 96-token runs of
the `first_index` task. Occupied-context numbers are individual uninstrumented
96-token runs; they are starting measurements, not repeated promotion gates.
Both runs use a 16,896-token cache capacity and 1,024-token prefill chunks.
Timing version 2 excludes the entire first emitting iteration from decode.
First-use occupied prefills include shape compilation and must not be reported
as warm prefill throughput. Later A/B measurements discard each exact
prompt/shape/variant before comparing it.

Both profiles confirmed actual packed-head and paired-MLP calls. Profiling
wraps outer modules and selected boundaries without replacing individual
projection forwards, which would disable some optimized dispatch paths.

The local Kineto capability probe returned CPU/runtime events but no GPU
kernel events or device self-time. CUDA-event intervals are inclusive and
contain host submission gaps. Nested intervals cannot be summed as disjoint
work; time blocked in token readback can represent earlier queued GPU work.
These records establish neither GPU idle percentage nor measured DRAM
bandwidth utilization.

On 27B, quantized cache updates occupied 6.87 ms of 901.62 ms of measured
draft-plus-verification intervals at 4K, and 6.74 ms of 991.25 ms at 16K.
These are event-interval ratios, not hardware kernel self-time. Cache writes
are therefore a smaller immediate target than the block/projection path.
Q/K normalization plus RoPE and the intra-block attention residual plus MLP
normalization were already fused. This round first tests the remaining MLP
residual plus next input norm and a larger MTP graph that retains CPU
embeddings. It does not introduce a persistent scheduler or a new native
projection/cache-write kernel.

## Native residual/norm qualification

The existing `rms_norm_res_in` entry updates the residual and computes its
normalization together. The new cross-block use must preserve the result of
the original add followed by normalization.

- FP32 normalized output is unsupported by this binary; the candidate must
  require FP16 normalized output.
- A 144-case envelope sweep found 24 differences, all for FP16 residuals
  combined with FP32 updates at five or more rows. That mixed combination
  is excluded at every row count, including smaller cases that happened to
  agree. The existing intra-block implementation is outside this change.
- The remaining qualified combinations were exact. An expanded sweep passed
  324 bit-exact comparisons over widths 128, 256, 768, 1,536, 3,072, 4,096,
  5,120, 8,192, and 16,384; rows 1, 3, 5, 10, 64, and 256; FP16/BF16 norm
  weights; and FP16/FP32 residual/update combinations except the excluded
  mixed case.

These operator checks do not establish full-model correctness or throughput.
Model comparisons must also preserve generated tokens and MTP draft windows,
and must show actual fusion calls and graph replays.

## Candidate measurement protocol

Candidates use identical weights, cache precision, prompt tokens, MTP depth,
attention policy, prefill chunks and allocator settings within each comparison.
The comparison process loads each model once, discards a 128-token warm run
for every exact variant/prompt/context, then measures four repeats with the
variant order reversed on alternate repeats. Each continuation emits exactly
128 tokens. Reported rates are medians; the retained records also include
minimum/maximum rates, prefill times, token IDs and draft acceptance windows.

The short prompts are the existing SQL parameterization and `first_index`
fixtures. Occupied-context trials prepend the same audit-record filler to
the `first_index` prompt, making the total exactly 4,096 or 16,384 tokens.
Each measured candidate must match its control's input hash, output token IDs
and complete draft-window records. These are equivalence checks, not broad
answer-quality or sampling validation. Fixed continuations may pass EOS;
their rates must not be compared directly with the 96-token profiling baseline.

The isolated candidates are:

1. Cross-block MLP residual plus the next attention input RMSNorm (or the
   final RMSNorm), using the qualified existing native operator.
2. An MTP graph covering transformer blocks, final normalization, the shared
   compressed vocabulary head and greedy argmax. CPU embedding lookup and
   the input projection remain eager.
3. The existing batched greedy verifier/readback path, retested separately.
4. A wider MTP graph that also captures both input norms and the input
   projection. Only the CPU embedding lookup remains eager; the embedding
   table stays on CPU. This is a separate experiment, not a change to the
   first graph's recorded implementation.

Both new graph variants use owned input/metadata buffers, return owned output
copies, and limit their graph caches and capture attempts. Capture checks
compare finite hidden states and exact sampled IDs against eager execution.
The wider graph checks against the original complete draft forward method.
Unsupported shapes, calibration, or layouts retain eager execution.

Two initial 9B harness trials exposed guards that rejected actual runtime
objects: module device strings versus tensor device objects, and normal bound
head methods restored after warmup observation. Those trials were excluded;
subsequent comparisons require nonzero fusion counters and actual graph
replays. An inert candidate cannot qualify as an optimization.

## Short-context comparisons

Four measured repeats per cell; each candidate percentage is relative to
the baseline in that same row. These are separate runs from the initial
96-token profiling baseline.

| Model / mode / task | Baseline tok/s | Residual + norm | MTP tail graph | Batched greedy |
|---|---:|---:|---:|---:|
| 9B MTP2, SQL | 75.328 | +0.11% | +0.37% | -0.59% |
| 9B MTP2, first_index | 82.268 | +0.28% | +0.35% | +0.72% |
| 27B target-only, SQL | 22.298 | +1.10% | — | — |
| 27B target-only, first_index | 22.249 | +1.26% | — | — |
| 27B MTP2, SQL | 40.831 | +0.20% | +0.25% | +0.17% |
| 27B MTP2, first_index | 45.353 | +0.12% | -0.76% | -0.57% |

All 80 measured continuations in these two comparisons matched control tokens
and draft windows. Graph capture checks reported zero relative hidden-state
error on both models. The MTP gains are within overlapping timing ranges and
do not establish a general speed improvement. The larger target-only residual
result motivated the follow-up below.

## 27B occupied-context and single-row follow-up

The wider graph includes the two input norms and the input projection. This
table uses four repeats per cell and fresh, warmed prefills for each request.

| Occupied prompt | Baseline decode tok/s | Residual + norm | Wider MTP graph | Baseline prefill tok/s | Residual prefill | Graph prefill |
|---|---:|---:|---:|---:|---:|---:|
| Short, 97 tokens | 45.281 | +0.26% | +0.26% | 251.03 | +0.05% | +0.08% |
| 4,096 | 39.951 | +0.36% | +0.20% | 566.21 | +0.14% | +0.17% |
| 16,384 | 40.069 | -0.03% | -0.68% | 512.87 | -0.01% | +0.12% |

All 36 measured continuations matched control tokens and draft windows. The
wider graph captured all three context shapes, reported zero relative
hidden-state error, and used actual replays without eager fallback. It did
not yield a useful full-model improvement. The residual 4K measurements also
included a 38.06 tok/s outlier; it is retained in the records rather than
discarded to improve the result.

A separate eight-repeat follow-up restricted residual fusion to single-row
calls, leaving prefill and multirow target verification eager. Its target-only
median was 22.368 → 22.407 tok/s (+0.17%), with individual paired changes from
-2.97% to +3.47%. MTP was 45.217 → 45.308 tok/s (+0.20%). All 32 measured
continuations matched. This narrower policy did not establish the earlier
roughly 1% target-only result as a repeatable gain; it is a distinct policy
trial, not an exact replication of the all-row prototype.

## 9B follow-up

The all-row and single-row residual policies were compared in the same process
alongside the wider graph. Each cell is the median of four measurements.

| Mode / occupied prompt | Baseline decode tok/s | All-row residual | Single-row residual | Wider MTP graph |
|---|---:|---:|---:|---:|
| Target-only, short | 50.197 | +0.15% | +0.21% | — |
| MTP2, short | 82.519 | +0.08% | +0.01% | +0.79% |
| MTP2, 4,096 | 82.528 | +0.11% | -0.58% | +0.48% |
| MTP2, 16,384 | 78.951 | +0.21% | -0.14% | +0.41% |

Baseline MTP prefill rates were 735.65, 1,848.55 and 1,760.49 tok/s for the
short, 4K and 16K prompts. At 4K/16K the all-row residual prefill changes were
+0.28%/+0.05%, and the graph changes were +0.14%/+0.10%. These small changes
do not establish a prefill improvement. All 60 measured continuations matched
control tokens and draft windows. All three graph capture checks had zero
relative hidden-state error and exact IDs, with actual replays and no eager
fallbacks.

The wider graph's warmed short/4K request medians improved by 0.71%/0.28%.
At 16K, total request latency was effectively unchanged (+0.03%), despite
the small median decode-rate increase. Medians of individual timing components
need not add to the median total.

### Capture and bookkeeping cost

The first captures in the context sweep added tens of milliseconds to the
observed time to first tokens relative to warmed graph runs. Those individual
first-use observations also contain normal timing variation; they are not
isolated capture-kernel measurements.

A separate controlled follow-up warmed the same prompts and ordinary kernels,
then created a fresh graph object for every graph trial. Four alternating
repeats per prompt compared identical 128-token continuations. This measures
the effect of needing a graph capture, not repeated server reuse of an already
captured shape.

| 9B MTP2 task | Decode change | Baseline total | Fresh-graph total | Total latency change | Baseline first tokens | Fresh-graph first tokens |
|---|---:|---:|---:|---:|---:|---:|
| SQL | +0.78% | 1.839417 s | 1.838080 s | -0.07% | 154.45 ms | 166.56 ms |
| first_index | +0.48% | 1.694156 s | 1.700733 s | +0.39% | 156.90 ms | 170.60 ms |

All 16 measured continuations matched. Capture setup, validation, staging and
owned output copies consume the small replay saving in these requests. The
prototype also keys graphs by cache identity, stream and occupied page-table
shape; new shapes require new captures, with bounded eager fallback after its
cache/attempt limits. A general default therefore needs more than a warmed
single-shape rate improvement. A policy that reliably amortizes preparation
in repeated BF16-MTP serving remains possible, but is not qualified here.

## Outcome and remaining work

- Cross-block residual/RMSNorm fusion: operator correctness established within
  the stated envelope; full-model gains too small and variable to promote.
- Larger MTP graphs and batched greedy readback: no general latency win. The
  wider graph's small 9B warm-decode gain is retained in the evidence, together
  with its first-token cost and neutral/slower 27B results.
- QKV/cache-write fusion: deferred after profiling. Quantized cache writes are
  under 1% of the measured 27B draft/verify intervals; the existing Q/K norm,
  RoPE, paired MLP and compressed-head paths already remove substantial work.
- Persistent scheduling: not implemented. The measurements still point toward
  target block/projection work, but usable GPU kernel timelines/counters are
  needed to separate packed-weight decode, arithmetic, memory traffic and
  launch gaps before choosing a larger scheduler or another projection kernel.

This completes the requested investigation without adding unqualified runtime
complexity. Broader graph deployment checks, including concurrent streams,
multi-job serving, sampling and reload/unload lifetime, were not pursued after
the performance decision. No additional model family or GPU is certified.

## Validation

The six completed comparison suites contain **224 measured continuations and
28,672 output tokens**, all identical to their within-case controls, including
MTP draft-window records. Discarded warmups and the two inert-dispatch trials
are excluded. The independent native residual/norm sweep passed 324 exact
comparisons. The final focused CPU suite passed 36 tests in the Linux inference
environment; it covers guard/scope behavior and existing MTP bookkeeping, not
GPU mathematics. Both OMP audit and implementation jobs completed successfully
using the configured default model.

After removing the unpromoted hooks, the unchanged primary runtime passed
the full unit suite on Windows (804 tests, 46 dependency/platform skips) and
Linux (804 tests, 10 Windows-only skips). CLI help and diff whitespace checks
passed. All six comparison launchers exited zero and released the GPU lease.
No native rebuild, installation change, or new serving default was made.

## Evidence

Private records are under ignored `artifacts/fusion-decode-20260926/`:
`profile-9b`, `profile-27b`, the retained baseline profiling scripts, the
failed wider-envelope probes, and `norm-qualified-wide`. Qualified comparison
directories are `compare-9b-v3`, `compare-27b-first`, `compare-27b-long`,
`compare-27b-target-confirm`, `compare-9b-long`, and `compare-9b-capture-cost`.
Each retains plans, model identity, exact tokens/windows, timings, counters,
and launcher/monitor results. `evaluated-prototypes/` contains source copies,
the tracked-file patch and a SHA-256 manifest. Private comparison scripts and
their JSON plans remain alongside the results.
OMP audit and implementation job metadata, logs, exit status, and handoff
records are retained under ignored `.codex/.omp-jobs/`.

Execution uses the registered GPU lease, verified extension, offline model
loading, and existing resource/time monitors. No model weights, native
binaries, or private run artifacts are included in source control.
