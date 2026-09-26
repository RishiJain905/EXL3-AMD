# AsterKV: KV cache research

**Status: active research.** Stage 0 controls and an experimental stage 1
implementation are under evaluation. Stage 2's offline layer policy is now
implemented and entering its research gate. `--cache-type aster5` selects the prototype;
it is not a recommended Q8 replacement. [Experiments](EXPERIMENTS.md).

The first prototype meets the measured attention-storage target. Long-context
decode, MTP/prefix consistency and broad task-quality equivalence remain open;
the memory result alone does not satisfy the joint stopping rule.

AsterKV is the working name for a cache developed within EXL3-AMD: lower KV
memory than Q8, with coding, reasoning and long-context recall close to Q8.
The aim is a distinct implementation built on published research, with clear
credit for inherited ideas. Renaming an existing format or tuning a Q6 preset
does not establish algorithmic novelty.

This directory is the research home inside the existing repository. The list
below is a priority order, **not a commitment to implement every idea**. Stop
at the first candidate that meets the acceptance criteria. Later ideas remain
unstarted unless a measured shortfall justifies them. Novelty is a research
objective; it does not justify sacrificing the practical result.

## Ordered ideas

Stages 0–2 are in progress; Stage 2 was explicitly authorized for implementation
and comparison despite Stage 1's inconclusive joint gate. Stages 3–5 remain
unstarted. Stage 0 establishes
controls; stages 1 onward prioritize our own design rather than a direct
TurboQuant port.

| Order | Idea | Question to answer | When to try it |
| --- | --- | --- | --- |
| 0 | Freeze Q4/Q8 baselines and test plain Q5/Q6 controls | How much quality, memory and speed does each precision actually cost? | First, as the comparison foundation. |
| 1 | **AsterKV-5: rotated, nonuniform five-bit quantization** | Can a better quantization grid preserve more useful information at five bits? | The controls do not already provide an acceptable middle ground. |
| 2 | **AsterKV with calibrated precision per layer and K/V side** | Can extra bits in sensitive layers recover quality without paying for Q8 everywhere? | Stage 1 misses the target and layer/K/V sensitivity suggests a better bit allocation. |
| 3 | **AsterKV-4 with selective higher precision** | Can tolerant layers use the four-bit codec while sensitive layers retain five, six or eight bits? | Earlier candidates still need more compression and measurements identify tolerant layers. |
| 4 | **Bounded outlier or residual protection** | Is a small set of poorly represented values responsible for the remaining failures? | Earlier candidates fail and error measurements justify the extra metadata and compute. |
| 5 | **A bounded high-precision recent-token window** | Are remaining failures concentrated in recently generated state? | Earlier candidates fail and experiments show that temporal precision is relevant. |

Stage 1 uses `aster5`; Stage 2 adds an explicit
[`--cache-policy` profile](STAGE2.md).
Native integration and decode measurements belong to **each** candidate; they
are not deferred until the whole list is finished.

### 0. Establish the controls

Use identical model weights and prompts for Q8/Q8, Q8/Q4 and Q4/Q4. Validate
plain Q6/Q6, Q5/Q5 and Q8/Q6 as additional controls. Where memory permits,
include an FP16 KV reference using the same EXL3 weights; this does not provide
a full-precision model-weight reference.

The vendored cache class and native packing support widths from two to eight
bits. The runtime exposes `f16`, `q8`, `q6`, `q5` and `q4` as controls, with
`aster5` as an experimental nonuniform format. The `long` profile retains its
Q8/Q8 guard and has a separate bounded schedule for the cubic Aster candidate;
record the actual attention dispatch for every comparison.

With the existing FP16 scale per 32 values, calculated storage is:

| K / V precision | KV tensor storage relative to Q8/Q8 |
| --- | ---: |
| Q8 / Q8 | 100% |
| Q8 / Q6 | 88% |
| Q6 / Q6 or Q8 / Q4 | 76% |
| Q5 / Q5 | 65% |
| Q4 / Q4 | 53% |

These ratios assume equal K/V geometry and describe the inherited format.
They exclude recurrent state, weights and workspaces. New codebooks, scales,
corrections and calibration metadata must be counted separately. If a plain
control is sufficient, retain its accurate Q5/Q6 name and stop; it is not a new
quantization algorithm.

### 1. Build the smallest AsterKV codec

Start with the existing Hadamard rotation and five-bit packing. Compare a
nonuniform grid based on the rotated distribution against uniform Q5. Use
TurboQuant's rotation and distortion-minimization ideas as references, without
assuming every implementation called "Turbo4" uses the same algorithm.

Measure reconstruction error, attention-score/output error and downstream task
quality. Compare distribution-derived centroids with a small offline calibrated
grid; use separate evaluation prompts. The inherited cubic compander is another
control, not a new invention. Its encoder and every attention/dequantization
path must agree; changing only a constructor option is insufficient.

Initially keep group size and rotation fixed so the grid's contribution is
measurable. Only investigate another rotation or normalization if the error
analysis identifies a reason. Avoid adding residual correction by default.

### 2. Allocate precision where it helps

Following KVTuner's sensitivity analysis, measure K and V separately in each
global-attention layer. Search a small, static set of precision assignments
under a byte budget. Begin with whole layers and K/V sides, which are simpler
to schedule than token-by-token or head-by-head decisions.

Combine this policy with the best codec from stage 1, and compare against the
same policy using ordinary quantization. This separates the benefit of the
codec from the benefit of allocating bits. Store a model-specific profile
offline; inference should not run a policy search for each generated token.

### 3. Lower the bulk precision to four bits

Try the nonuniform four-bit codec in the least sensitive layers and retain
higher precision elsewhere. Compare at equal measured cache bytes against
uniform Q5/Q6 and a clearly specified TurboQuant reference. A smaller average
bit width is useful only if task quality and runtime performance survive.

### 4. Protect exceptional values only when necessary

Use KVQuant's outlier-separation work as a reference. Investigate a bounded
number of higher-precision exceptions or a small residual representation only
if a few values dominate the remaining error. Include indexes, scales and
correction computation in the cost. Test each mechanism separately before
combining it with the chosen codec and precision policy.

### 5. Consider temporal precision last

Use KIVI's treatment of recent, unquantized state as a reference for a small
high-precision window. Its size must be bounded. Validate window transitions,
MTP rollback, page copying and prefix reuse before considering adoption.
Recent tokens are not necessarily the important tokens; this mechanism must
also preserve retrieval from the distant context.

## Acceptance and stopping rules

The preferred target is **25-40% less attention KV storage than Q8/Q8**, close
to Q8 task quality, and no material slowdown with the normal serving CLI.
Report total peak VRAM as well: a cache-only saving is not the same percentage
reduction in total GPU memory. A simple Q6 result near the memory target remains
a useful alternative, with its actual tradeoff reported.

Before evaluating candidates, freeze prompts, scorers, comparison margins and
repeat counts. Initial proposed margins are at most **two percentage points**
of quality loss in each task family/context band, and at least **95% of Q8's
decode throughput**. Use paired confidence intervals to establish the quality
margin; a small sample or a nonsignificant difference alone does not prove
equivalence. These are planning targets, not measured achievements.

- Evaluate coding correctness, reasoning, multi-turn instructions and retrieval
  with distractors on held-out prompts. Report each family and context band;
  an aggregate must not conceal a long-context collapse.
- Use capacity **120064 tokens**, with occupied prompts around 4K, 16K, 32K,
  64K and 110K, leaving room for templates and output. Allocation at 120K is
  not validation of a nearly full context. Record any band that cannot run.
- Isolate quantization quality with fresh requests and fixed sampling; use
  target-only runs to diagnose MTP effects. Final comparisons use the ordinary
  CLI configuration: long attention requested, WMMA prefill, chunk 1024, MTP
  depth 6/confidence 0.6 and prefix reuse enabled. Record actual dispatch,
  draft precision, acceptance rate and both fresh and reused performance.
- Measure decode first, then prefill, first-token latency and complete request
  time. Use warmed, repeated comparisons. Count only computed input tokens in
  prefill throughput, and distinguish allocated cache from occupied cache.
- Count all cache metadata and scratch buffers, target/draft caches and peak
  process/device memory. Verify real packed storage; a simulation is not a
  demonstrated VRAM or GPU-speed improvement.
- Validate native numerical behavior, partial pages, graph replay, cancellation,
  shutdown, MTP rollback and prefix reuse. Preserve the existing documented
  30K fresh/reused exact-output discrepancy as a separate baseline limitation;
  token mismatch alone is not a task-quality score.

After each stage, record **accept and stop**, **reject**, or **inconclusive**.
An accepted candidate must pass quality, memory, runtime and correctness checks
together. An inconclusive result calls for targeted measurement, not automatic
promotion or automatic implementation of the next idea. Keep unsuccessful
results and explain the specific shortfall that motivates any later stage.

## Evidence and attribution

For each experiment, record the hypothesis, inherited method, our modification,
source/binary revision, configuration, effective bits including metadata,
quality scores, memory, timings, limitations and the stopping decision. Ablate
one addition at a time. Claim a new contribution only when comparison supports
what differs from the referenced work; a new name is not that evidence.

Publish portable summaries and relative repository links. Model weights, local
profiles, raw tensors, prompts/responses containing private data, installation
records and machine-specific paths belong in ignored local artifacts. This
research plan adds no launch scripts and changes no runtime defaults.

Primary references:

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
  — rotation, scalar quantizers and reconstruction/inner-product distortion.
- [KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization](https://arxiv.org/abs/2502.04420)
  — offline layer-wise K/V precision allocation.
- [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079)
  — nonuniform quantization, outlier separation and pre-RoPE key quantization.
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
  — K/V distribution differences and a recent unquantized residual window.

Repository context: [current cache support](../KV-CACHE.md),
[measurement guidance](../OPTIMIZATION.md),
[full CLI results and limitations](../RUNTIME-PERFORMANCE.md),
[validation scope](../VALIDATION.md), and [upstream attribution](../UPSTREAMS.md).
