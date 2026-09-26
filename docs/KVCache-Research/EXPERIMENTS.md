# AsterKV experiment record

Status: stage 0 controls and stage 1 prototype in progress. No candidate has
passed the joint quality, memory and throughput acceptance gate yet.

## Protocol and controls

The first campaign uses one Qwen-family 27B EXL3 model on an RX 7800 XT
(gfx1101), with capacity 120064, WMMA prefill, chunk 1024, MTP depth 6,
confidence 0.6, prefix reuse enabled and `long` attention requested. The long
profile's existing Q8-only guard is preserved. Capacity is not occupied context.
Models, local paths and raw request/run data remain in ignored local storage.

Measurements used working-tree research changes over base commit
`08853cb88ec70f317729ae37d0a53e85788ea793`, with per-trial source snapshots.
The verified native extension SHA-256 was
`df35ba80b0f6a0a61bb0fe31f4cac6ebfa90dca0fea2f62c38a584f041cb455e`.
No new native binary was built for this campaign; the new codec and attention
work use Triton with that existing extension.

The initial synthetic pilot froze 12 code-execution questions, 12 arithmetic
questions, 12 revision-tracking questions and two retrieval questions per
context band before testing. The first screen uses 4K and 16K retrieval
bands, greedy decoding, thinking disabled and a 96-token answer budget.
Frozen case SHA-256:
`c776f7fd7cbd0e4ed19784364d447e0644af23e9ba8917a9a66c94af05ada06f`.

Q8 exposed a scorer problem: correct objects were marked wrong when the prompt
requested arrays. Before scoring other caches, scoring v2 separated exact
answer values from JSON format compliance. It accepts only the specified named
fields in a known order; arbitrary text, extra fields and Boolean numeric
substitutions fail. Raw Q8 answers and original strict scores were retained.

Q8, Q4, Q5 and Q6 each scored 16/40 semantically: all revision/retrieval
items passed, while every code/arithmetic item failed. Q8/Q4 scored 17/40.
This floor/ceiling
effect makes the screen unsuitable for establishing a small quality margin.
The shared failures are retained, not attributed to reduced cache precision.

A follow-up protocol enables thinking for six fixed code/arithmetic questions
(1536-token budget). Timing requests use three fresh 256-token code-generation
requests and one identical request for explicit prefix reuse per band. Q8 and
Aster receive identical frozen requests. Follow-up case SHA-256:
`701325079ada1e8788306a475aaa48d5eafb7a7c01c300673fbe0c25a476510f`.
This remains a pilot, not a statistically powered equivalence study.
During the Q8 thinking pilot, correct arrays also appeared inside single-field
JSON objects. Before Aster evaluation, scoring v3 added that unambiguous wrapper
and the exact named pair `number_of_full_boxes` / `units_left_outside_boxes`
to semantic scoring while retaining format failures. The same scorer is applied
to both caches; original responses and earlier scores are preserved.

## Stage 1: calibrated five-bit grid

The prototype retains the inherited normalized H32 rotation, group size 32,
five-bit bitplane layout and one FP16 absolute-maximum scale per group. A
symmetric 32-centroid Lloyd-Max grid replaces uniform bins. Endpoints are
pinned at -1 and +1; assignments use the unrounded scale, nearest midpoint
thresholds and a lower-index tie rule. The scale is rounded to FP16 on storage.
K and V share the same grid. Attention decodes indices through that grid in
the rotated domain, without allocating a full FP16 cache.

This combines published scalar-quantization ideas with the inherited EXL3
cache. It is not a direct TurboQuant implementation: it uses group absmax
scales and the existing H32 transform, without TurboQuant's QJL correction.
It does not establish algorithmic novelty. See the research plan's references.

Calibration captured at most 4096 groups per K/V side in each of 16 global
attention layers. Calibration prompts used public `scripts/run_exl3_model_smoke.py`
and `docs/EXL3.md`; held-out prompts used `docs/TOOL-CALLING.md` and
`src/quantlab/server.py`. Each corpus supplied an 8192-token prefill. No model
weight tensors were exported. Calibration timings are instrumented and are
not performance measurements.

Mean held-out relative reconstruction MSE, averaged over layer/KV datasets:

| Format | Relative MSE |
| --- | ---: |
| Uniform Q4 | 0.007624 |
| Uniform Q5 | 0.001906 |
| Uniform 32-centroid grid, exact endpoints | 0.001852 |
| AsterKV-5, pinned grid | 0.001614 |
| Uniform Q6 | 0.000477 |
| Uniform Q8 | 0.0000299 |

AsterKV-5 reduces this error by 15.3% versus Q5 at the same packed payload
size. That does **not** establish better task answers or Q8-level quality.
Unpinned centroids and least-squares scale refinement produced MSE 0.001592
and 0.001537 respectively; they are recorded ablations, not changes selected
using the held-out scores. The first GPU candidate kept the preselected
pinned grid without scale refinement.

For 4 KV heads of dimension 256, K+V occupy 1408 bytes per token versus
2176 for Q8, before a 128-byte immutable codebook per layer. This is a 35.3%
attention-cache saving. Weights, recurrent state, MTP state and workspaces
are separate; total VRAM savings must be measured rather than inferred.

## First serving candidate: indexed Lloyd grid

Both services completed 22 requests (including warmup), with zero failed or
cancelled requests and a clean shutdown. These runs used the frozen follow-up
protocol above. Each timing cell is the median of the second and third fresh
requests; the first request is retained separately because compilation can
dominate it. Throughputs include MTP and therefore are not isolated attention
kernel timings.

| Occupied input | Q8 decode tok/s | Aster LUT decode tok/s | Q8 prefill tok/s | Aster LUT prefill tok/s |
| --- | ---: | ---: | ---: | ---: |
| 4,016 | 46.36 | 43.70 | 594.39 | 533.57 |
| 16,305 | 40.06 | 34.25 | 548.95 | 408.54 |
| 32,689 | 40.63 | 28.86 | 482.09 | 304.87 |

The long Q8 profile applied 6,061 times; it did not apply to Aster. The LUT
path required narrower, single-stage attention tiles on this GPU to stay
within its LDS limit. MTP accepted/rejected counts were similar on the warmed
timing requests; the profiler below separates kernel cost from drafting.

Measured target plus draft attention tensors occupy **4.136 GiB with Q8** and
**2.676 GiB with Aster** (including codebooks): **1.461 GiB saved, or 35.3%**.
Peak PyTorch allocated memory was 12.266 versus 10.805 GiB, a smaller **11.9%**
whole-process allocation reduction. These allocator peaks exclude memory
outside PyTorch and are not a measurement of the entire adapter's memory.

Semantic pilot results were Q8 3/3 code, 2/3 arithmetic and 3/3 retrieval;
Aster 3/3 in each family. Most answers failed the requested array formatting.
These tiny samples do not establish a two-percentage-point quality margin.
Exact reused-prefix answer text matched the corresponding fresh request for
both caches at all three bands. Reused prefix lengths were 3,840, 16,128 and
32,512 tokens. This does not resolve all earlier prefix-reuse limitations.

**Decision: reject this LUT schedule for adoption.** Memory savings meet the
target, but decode and prefill slow down. Quality remains inconclusive. The
64K and 110K cases are frozen but have not been run for this rejected variant.

## Stage 1 refinement: analytic reconstruction

The next candidate fits the calibration-derived grid with an endpoint-normalized
cubic, `p(x) = x * (a + b*x*x)`, where `x = (2*index - 31)/31`,
`a = 0.8307890996269471` and `b = 0.16921090037305286`.
The grid remains symmetric and monotone. The encoder selects nearest centroids
using midpoint thresholds, while attention evaluates the cubic directly. This
removes indexed grid reads inside attention. Packing and memory usage are
unchanged; the small codebook remains for encoding and fallback decoding.

Coefficients were fitted to the calibration grid, not held-out KV values.
Held-out relative reconstruction MSE is **0.0016154**, versus 0.0016140 for
the original grid. The cubic was chosen for its low evaluation cost; higher
polynomial degrees are recorded ablations. This is a stage 1 implementation
refinement, with no claim of a new quantization algorithm or quality equivalence.
The original grid remains an explicit profiler control.

For completeness, fifth- and seventh-degree fits measured relative MSE
0.00161441 and 0.00161397 respectively. Neither replaced the cheaper cubic.

A fixed equally spaced 32-centroid grid with exact endpoints measures MSE
0.0018515. The cubic reduces error by another 12.8% against that control,
so endpoint normalization alone does not explain the entire improvement.
This ablation did not change the selected grid or coefficients.

The same cubic family already appears in the inherited
[`lmq.cuh` compander](../../vendor/rocm-exl3/exllamav3/exllamav3_ext/cache/lmq.cuh).
A NumPy mathematical control of its midpoint grid and inverse-uniform
encoder measured MSE 0.0019582 with its fixed default coefficient 0.65 and
0.0018086 with the already selected calibration coefficient 0.8307891.
The latter is a coefficient-controlled comparison, not a held-out parameter
search. These FP64 reference calculations are not bit-exact GPU validation
of the inherited compander. Aster's differences here are its endpoint grid,
nearest-centroid encoding, calibration and consistent online attention
decoding; the cubic formula itself is inherited, not a claimed invention.

GPU correctness and serving performance must pass before this refinement can
be adopted. Stage 2 precision allocation is not started merely because the
first stage 1 schedule was slow.

Model-free profiling compares each schedule against its own codec's default
output, with relative L2 below 0.005. At 32K occupied context, the selected
cubic attention timings were 0.384 ms for one query, 0.993 ms for seven queries
and 45.83 ms for a 1024-query prefill. Q8 with its long decode profile measured
0.466 / 0.844 ms; Q8's default staged prefill measured 49.82 ms. Thus the cubic
candidate is not uniformly faster even in the isolated attention test.

The profiler uses two warmups and event means over ten decode or three prefill
calls. It includes host launch gaps and temporary allocations; it does not
replace warmed serving measurements. Shorter-context probes also favored the
128-row / 16-column, eight-warp, single-stage prefill tile. The `long` option
now applies this Aster schedule only to the documented gfx1101 geometry and
query/context bounds. Q8 dispatch is unchanged.

The diagnostic profiler used upstream Q8 prefill staging. The serving CLI
disables full-cache staging for all packed formats, including Q8. Treat those
microbenchmarks as separate measurements; the following CLI comparison uses
the actual serving configuration for both caches.

## Cubic candidate: paired serving results

The paired run used the same frozen follow-up cases, serving flags and warmed
median protocol as the first candidate. Aster ran first, then Q8. Both completed
22 requests, with no request failures or cancellations and clean shutdown.
The decode-profile counter recorded 6,061 applications for Q8 and 29,576 for
Aster. Aster's separate guard starts at a smaller page-table bound; these
counts report dispatch, not equivalent units of work or a speedup factor.

| Occupied input | Q8 decode tok/s | Cubic Aster decode tok/s | Q8 prefill tok/s | Cubic Aster prefill tok/s |
| --- | ---: | ---: | ---: | ---: |
| 4,016 | 45.49 | 48.75 | 594.79 | 593.15 |
| 16,305 | 39.22 | 44.20 | 544.04 | 578.23 |
| 32,689 | 41.08 | 42.47 | 478.23 | 519.24 |

Decode was 7.1%, 12.7% and 3.4% faster respectively in this pilot. Prefill was
approximately equal at 4K and 6.3% / 8.6% faster at 16K / 32K. These are two
warmed repetitions per cell, not confidence bounds or general speed guarantees.
Both caches use MTP; accepted/rejected counts and generated text can differ.

Attention tensors remain 4.136 versus 2.676 GiB, including the draft cache and
codebooks. Peak PyTorch allocated memory remains 12.266 versus 10.805 GiB.
The runtime's Windows monitor sampled whole-adapter dedicated-memory peaks
of **13.893 GiB for Q8** and **12.432 GiB for Aster**, about 10.5% lower. Adapter
figures include other applications and are sampled, not exact allocation totals.
After shutdown, dedicated usage returned to approximately 0.83 / 0.81 GiB.

Both scored 3/3 code, 2/3 arithmetic and 3/3 retrieval semantically. Requested
array-format compliance was 1/9 for Q8 and 0/9 for Aster; correct values in an
object do not count as compliant formatting. This is still insufficient to
establish the planned two-percentage-point quality margin. Exact prefix-reused
answer text matched the corresponding fresh answer for both caches at all
three context bands.

**Decision: retain as experimental and continue targeted stage 1 validation.**
The early memory and speed screens pass; broad quality equivalence remains
inconclusive. The next screen occupies approximately 64K and 110K context,
using a retrieval request first, one fresh timing request and exact prefix
reuse at each band. Single fresh timings there are descriptive, not repeated
throughput estimates. Its cases were selected and frozen before any long run:
SHA-256 `754450dc531d6487346c7b0e99f238cceab00112b0a423339a0f04a48502d009`.

## Cubic candidate: occupied long-context screen

Q8 then Aster each completed seven requests including warmup, with zero failed
or cancelled requests and clean shutdown. Each band has one fresh timing
request after its retrieval probe; these are not repeated speed estimates.

| Occupied input | Q8 decode tok/s | Cubic Aster decode tok/s | Q8 prefill tok/s | Cubic Aster prefill tok/s |
| --- | ---: | ---: | ---: | ---: |
| 65,457 | 34.08 | 32.56 | 384.52 | 431.61 |
| 109,922 | 35.75 | 29.74 | 307.06 | 355.70 |

Prefill improved 12.2% and 15.8%; decode declined 4.5% and 16.8%. The 110K
result misses the planned 95%-of-Q8 decode screen. MTP accepted/rejected
counts for that fresh request were 199/90 for Q8 and 180/114 for Aster;
these are end-to-end measurements, not a pure attention-kernel comparison.

Both caches answered the two long retrieval questions correctly, but failed
their requested array formatting. Exact fresh/reused text matched for Q8 in
both bands and Aster at 64K. **Aster failed exact text reproduction at 110K**:
the answers shared their first 998 characters, then differed in a generated
code docstring and continuation. This is not classified as a task-quality
failure or dismissed as harmless numerical noise; its cause is unresolved.
It is separate from the earlier documented Q8 30K reproduction limitation.

At 110K, prefix reuse skipped 109,824 input tokens; server-reported first-token
latency was 1.02 seconds for Q8 and 1.06 seconds for Aster. This latency saving does
not turn the failed exact-reproduction check into a pass. Sampled adapter
memory peaks were 14.010 GiB for Q8 and 12.577 GiB for Aster; cache tensor
sizes remain those reported above.

**Decision: do not adopt the current schedule as a Q8 replacement.** Retain
the experimental prototype and investigate long-context decode scheduling
and reproduction before proceeding to another quantization idea. Quality
equivalence remains unestablished; stages 2–5 remain unstarted.

### Prefix reproduction without MTP

A targeted diagnostic disabled MTP and repeated the exact same frozen 109,922-
token fresh/reuse prompt with Aster, keeping capacity, WMMA, chunk size and
prefix reuse unchanged. Both 256-token answer texts and token ID sequences
matched exactly. They
also matched the earlier fresh MTP answer; the earlier reused MTP answer was
the differing result. All three requests including warmup completed, with
zero failures/cancellations and clean shutdown.

This narrows the observed discrepancy to execution with MTP and prefix reuse
for this prompt. It does not prove a floating-point cause or establish general
cache correctness. In the earlier MTP trace, the first 230 token IDs matched;
verification groupings first differed after output token 125 (one round
finished at 129 versus 130). Shape/grouping differences precede the text
difference, but this does not establish a causal explanation. No checkpoint
policy was changed to make the test pass.
Target-only decode was 16.04 / 15.77 tok/s for fresh/reused requests, so
disabling MTP is a diagnostic, not the selected speed configuration.

### Long-context scheduling diagnosis

A model-free follow-up used capacity 120064, occupied lengths 65,536 / 110,080,
query lengths 1 / 7, graph-replay timing and `EXL3_QC_STAGING=0`, matching
serving's packed-cache staging policy. Every measured schedule stayed below
0.000479 relative L2 against its codec's default attention output.

| Occupied | Query length | Q8 long ms | Initial Aster ms | Best measured Aster ms |
| --- | ---: | ---: | ---: | ---: |
| 65,536 | 1 | 0.929 | 0.825 | 0.759 |
| 65,536 | 7 | 1.772 | 1.992 | 1.850 |
| 110,080 | 1 | 1.528 | 1.354 | 1.231 |
| 110,080 | 7 | 2.843 | 3.278 | 2.956 |

For one query, the candidate increases splits to 128 with a 16-column tile.
For seven queries, four-head/four-warp tiles with 64 or 128 splits improve on
the initial eight-head/eight-warp tile. These are diagnostic results, not yet
an end-to-end speed claim. The best seven-query result still trails Q8.
Ten graph replays form each timing mean; no confidence interval is inferred.
One-/seven-token encoding timings were approximately 0.037 / 0.041 ms for Q8
and 0.038 / 0.040 ms for Aster, including graph-replay launch costs. This probe
does not justify attributing the serving slowdown to the encoder alone.

A second graph probe checked query lengths 5, 6 and 8 at both occupancies.
The 128-split, four-head/four-warp schedule reduced their attention time by
5.8–10.8%, with maximum relative L2 0.000481. The profile now selects this
schedule only at a page-table/physical-cache bound of at least 65,536 tokens
for queries 5–8; query 1 uses 128 splits there. Queries 2–4 and all shorter
contexts retain their previous schedules. Q8 dispatch is unchanged. A normal
CLI rerun at 110K checks the change in the full model, as recorded below.

### Refined schedule: limited CLI rerun

The final source reran the frozen 110K retrieval, fresh generation and exact
prefix repeat, with the normal MTP/WMMA/chunk/prefix flags and capacity 120064.
All four requests including warmup completed; the retrieval answer was correct
and shutdown returned zero. The source snapshot was unchanged during the run.
This rerun did not repeat the earlier 64K ramp or collect warmed repetitions;
unisolated first-use costs and MTP grouping limit the speed comparison.

| 109,922-token input | Earlier Q8 | Initial Aster | Refined Aster |
| --- | ---: | ---: | ---: |
| Fresh prefill tok/s | 307.06 | 355.70 | 356.12 |
| Fresh decode tok/s | 35.75 | 29.74 | 25.18 |
| Reused decode tok/s | 34.93 | 29.21 | 31.88 |
| Fresh/reused token IDs identical | yes | no | yes |

The refined schedule improved the measured reused decode rate by 9.1% versus
initial Aster, but fresh decode was worse. Neither Aster schedule establishes
the 95%-of-Q8 decode target at 110K. The final exact-reuse pass does not erase
the earlier failure or prove that its cause was fixed. Fresh/reused MTP counts
were 180/122 and 186/103 accepted/rejected respectively, so these are not
identical verification workloads.

Attention storage remains **2.676 GiB**, versus **4.136 GiB** for Q8. Peak
PyTorch allocated memory was **10.802 GiB** and sampled whole-adapter peak
was **12.509 GiB**; the larger split count did not erase the measured memory
saving in this run. Adapter usage returned to about 0.906 GiB after shutdown.

**Decision: retain the opt-in prototype for research; no accepted replacement.**
The measured memory goal passes, and prefill improves. Broad task-quality
equivalence, repeated long-context decode performance and MTP/reuse consistency
remain open. The next gate is a matched, repeated long-context comparison and
a discriminating held-out quality set, before changing precision allocation.
Stages 2–5 remain unstarted.

## Correctness and remaining gates

The uniform packed-cache checker passed Q4/Q5/Q6/Q8 and Q8/Q4, Q8/Q6 on
gfx1101, including paged updates and online attention against an independent
FP32 oracle. Q8 also passed nondefault-stream and changed-input HIP graph
replay checks. These are kernel checks, not task-quality claims.

The expanded hardware gate passed **21 uniform-cache checks and 22 Aster
checks** on gfx1101. It includes the uniform native BC compile/load bridge,
independent unpacking and FP32 attention oracles, zero/tiny/constant/outlier
groups, shuffled and partial pages, physical-input updates, incompatible page
copy rejection, MTP query lengths 1–8, decode/prefill windowing, the original
Lloyd lookup grid, actual serving dispatch, mixed causal/non-causal spans, and
graph replay with changed inputs and lengths. Maximum attention relative L2
must stay below 0.004; lossy quantization error is measured separately.

The final CPU suite completed **483 tests: 458 passed and 25 dependency/platform skips**. Those
skips do not establish GPU compatibility. The model-free GPU processes exit
successfully but the backend reports a `SharedSignalPool` teardown warning;
that warning is retained in private logs. CLI shutdown results are reported
separately. Cancellation is covered by the existing CPU lifecycle suite, not
by a new Aster-specific GPU cancellation test.

One optional `/health` probe with a three-second client timeout failed during
the final 110K prefill. Post-request health and shutdown succeeded. This probe
does not identify a cause; responsiveness under heavy GPU load is not
established by the completed inference requests or the CPU lifecycle tests.

The results above describe the Stage 1 checkpoint. Stage 2 has since been
explicitly authorized; its implementation, separate calibration and expanded
research gate are recorded in [STAGE2.md](STAGE2.md). Stages 3–5 remain unstarted.
