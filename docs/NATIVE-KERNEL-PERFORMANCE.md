# Native kernel optimization qualification

All three stages qualified, September 27, 2026. Baseline: `23b8d9b`,
RX 7800 XT (`gfx1101`). Packed projection inner loops, prefill matrix kernels
and the GDN recurrent core were investigated in that order. The winning
implementations are automatic primary paths after independent correctness
checks and repeated complete-model comparisons. Unsupported shapes retain
the existing fallback. The current local binary is `gdn-core-v4`, including
all three stages; the source and scope of each promotion are recorded below.

## Source and local evidence

Runtime implementations belong in the tracked vendored backend, with targeted
checks and standalone kernel sources under `kernels/exl3/`. Python integration
belongs in `src/quantlab/` when required. Local binaries, benchmark outputs and
supervisor logs stay in ignored `.runtime/`, `artifacts/` and `.codex/.omp-jobs/`.
Those directories are not runtime source dependencies for a fresh checkout.

GitHub users need the published source revision and a fresh extension build,
followed by registration of that build's actual hash. Pulling source alone does
not replace an already registered native binary. See [BUILD.md](BUILD.md).
The promotion descriptions refer to qualified source changes; a compatible
extension build is still required on each installation.

## Method

- MiMo 9B: K5 body/K6 vocabulary head, mul1 codebook, BF16 MTP at depth two,
  rowwise verification, F16 KV and the previously qualified unfused decode path.
- Qwenseek 27B: K2 body/K4 head, default codebook, quantized MTP at depth two,
  Q6 KV and the previously qualified default decode fusions.
- Complete-model comparisons use 1,024-token prefill chunks, 16,896-token cache
  capacity, short prompts and occupied contexts of 4,096 and 16,384 tokens.
  Target-only and MTP short-prompt cases are both included. Each case discards
  a warmup and records three fixed 128-token continuations.
- Compare the primary, candidate and primary again in separate processes with
  verified extension hashes and the exclusive GPU lease. Compare input hashes,
  generated token IDs and speculative draft windows, as well as prefill,
  decode, first-token and total time.
- Operator comparisons use actual packed layer-zero matrices, rows 1/2/3/5,
  FP16/FP32 outputs, 25 HIP-event samples and both warm and cache-evicted
  measurements. A 256 MiB eviction buffer is not a DRAM bandwidth measurement.
  Reconstruction slices are checked against an independent CPU bitstream
  oracle. Record input/output byte hashes for cross-binary comparisons.

No hardware-counter claim is made. Earlier Kineto profiling did not expose
usable GPU kernel events on this installation. Register/scratch metadata and
operator timings supplement complete-model measurements; neither establishes
memory-bandwidth saturation.

An early binary-inspection command rewrote the input ELF container. The
expected-hash check rejected it before recorded performance tests. Relinking
the untouched original objects restored both affected binaries byte-for-byte
to their original hashes; expected hashes were not relaxed. Inspection now
operates on a copy and verifies that the original remains unchanged.

## 1. Packed projection inner loops

The first candidate shares packed-word loads in specialized eight-value K5/K6
unpackers and tests two-iteration unrolling in direct GEMV and narrow small-M
and paired-MLP dot loops. The accumulation order is preserved. Five-row paths
keep conservative loop scheduling.

Correctness evidence:

- 19,600 CPU comparisons matched the original pre-codebook words exactly.
- Fresh native build completed. The seven GPU suites passed 2,013 checks:
  highbit small-M 60, small-M 408, packed-mid 929, packed-prefill 232,
  paired MLP 143, repacked head 185 and dense HGEMM 56.
- The highbit check now includes zeros, ones and every individual packed bit
  position for K5/K6 with both supported codebooks, plus output canaries.
- Windows unit suite: 804 tests, 46 dependency/platform skips.

Compiled metadata showed no private-memory spills among 1,552 selected kernel
specializations in either binary. Some narrow-row variants used more vector
registers; this is a tradeoff to measure, not a speedup result.

MiMo's primary/initial-candidate/primary comparison completed. All 54 measured
continuations (6,912 generated tokens) matched their corresponding input hashes,
output token IDs and draft windows exactly. Rates below are the medians of
three measured repetitions per process. The change compares the candidate with
the mean of the two control medians.

| MiMo case | Control before / after, tokens/s | Candidate, tokens/s | Decode change | Total-time change |
| --- | ---: | ---: | ---: | ---: |
| Target only, task 0 | 48.717 / 47.061 | 52.745 | +10.14% | -8.90% |
| Target only, task 4 | 48.709 / 48.373 | 52.033 | +7.19% | -6.25% |
| MTP, task 0 | 72.649 / 69.412 | 75.742 | +6.63% | -5.96% |
| MTP, task 4 | 79.694 / 76.857 | 84.450 | +7.89% | -6.83% |
| MTP, occupied 4K | 80.840 / 78.536 | 82.431 | +3.44% | -0.68% |
| MTP, occupied 16K | 77.429 / 74.725 | 80.473 | +5.78% | -0.92% |

Control decode rates drifted by up to about 4.7%; the candidate exceeded both
controls in every case. Prefill changed by -1.07% at 4K and +0.39% at 16K.
Short-prompt prefill ranged from -4.03% to +1.66%; task 4 target-only first-token
latency increased about 3 ms, while its complete continuation finished faster.
This is a decode improvement, not a new prefill-speed claim.

The layer-zero operator experiment had 48 byte-identical cases per binary/run.
Its cache-evicted medians were mixed: MLP down improved about 3%, while several
GDN projections regressed. Warm operator timings were noisier. The complete
model result, which includes the changed K6 vocabulary head, therefore matters
more than extrapolating a layer-zero microbenchmark to whole-model throughput.

The initial candidate is not eligible for general promotion: the 27B's
cache-evicted K2 body projection groups regressed about 6–10% against bracketed
controls. All 104 operator outputs per run were byte-identical. Its K4 MTP
gate/up projections improved about 5%, so the refinement restricts two-tile
unrolling to K4–K6 and preserves rolled loops for lower-bit formats. The
selection is a compile-time format check, with no new user switch.

The initial candidate's 27B full-model run was deliberately omitted after its
operator regression. The refinement below completed qualification on both models.

The refinement rebuilt the two affected projection translation units and
verified the hashes of 108 unchanged objects. All 2,013 native checks passed
again. Its binary SHA-256 is
`8e9120b06a67d095b59851f066b3094a7409eea40c0c07678aec34ab9283cab9`.
The sampled M=3, K2, cb0, FP16, eight-warp kernel returned from 49 to 34 vector
registers, and its complete 330-instruction sequence matches the primary.
The corresponding sampled K5 kernel's instructions match the first candidate.
None of the 1,552 inspected variants has a private-memory spill allocation.

The revised 27B comparison completed with exact input, token and draft-window
matches across all 54 measured continuations (6,912 tokens). An attempted
repeated control was stopped before measured generation after runtime
initialization alone took 338 seconds; it is excluded and was restarted with
the same prompts, repetition count and 900-second limit.

| Revised 27B case | Control before / after, tokens/s | Candidate, tokens/s | Decode change against control midpoint |
| --- | ---: | ---: | ---: |
| Target only, task 0 | 20.964 / 22.527 | 21.210 | -2.46% |
| Target only, task 4 | 21.286 / 22.404 | 22.164 | +1.46% |
| MTP, task 0 | 37.841 / 40.810 | 40.111 | +2.00% |
| MTP, task 4 | 43.448 / 45.209 | 44.508 | +0.41% |
| MTP, occupied 4K | 37.792 / 40.002 | 39.371 | +1.22% |
| MTP, occupied 16K | 38.065 / 40.033 | 39.445 | +1.01% |

Every candidate decode rate lies between its two controls, whose drift reaches
about 7.8%. These results do not establish a whole-model 27B speedup. The K2
regression from the initial candidate has been removed from the sampled
machine code; K4 operator improvements are not presented as a proven model
gain.

The revised MiMo comparison also completed with all 54 measured continuations
(6,912 tokens) matching input hashes, output IDs and draft windows exactly.

| Revised MiMo case | Control before / after, tokens/s | Candidate, tokens/s | Decode change against control midpoint |
| --- | ---: | ---: | ---: |
| Target only, task 0 | 47.061 / 51.664 | 54.993 | +11.41% |
| Target only, task 4 | 48.373 / 50.431 | 54.878 | +11.08% |
| MTP, task 0 | 69.412 / 75.075 | 79.137 | +9.54% |
| MTP, task 4 | 76.857 / 82.680 | 88.272 | +10.66% |
| MTP, occupied 4K | 78.536 / 82.941 | 86.963 | +7.71% |
| MTP, occupied 16K | 74.725 / 79.356 | 84.124 | +9.20% |

Control drift is substantial: the more conservative comparison against the
faster control gives **4.8–8.8% faster decode**. The candidate exceeds both
controls in every case. Long-context prefill changes against the control
midpoint were +0.56% and +1.38%; these are not a separate prefill-speed claim.
Short-case total time fell 8.16–10.07%, and occupied-context total time fell
2.25–3.48%. All runs exited successfully without resource stops. One MiMo
candidate process took 615 seconds to finish runtime/model initialization;
inference timings exclude that startup delay.

**Promoted:** the K5/K6 unpackers and format-qualified K4–K6 narrow-loop
unrolling are now the automatic source implementation. The validated
`decode-inner-v2` binary was registered as the local primary before stage two.
K2/K3 retain rolled
loops. Selection uses format and row count, without model-name checks or a
new user option. Validation is limited to the stated GPU and configurations;
future models with supported formats use the same dispatch automatically.

## 2. Prefill matrix kernels

Stage one is the baseline. A bounded OMP job implemented two separately
measurable candidates in the native source: next-tile register staging in
packed prefill and a 128×64×32 / 256-thread dense WMMA prefetch tile in place
of the existing 256×64×32 / 512-thread tile. The smaller-row serial dense path,
WMMA accumulation order, fold interval, codebooks and dispatch bounds are
unchanged. The worker stopped before coordinator validation.

This stage's operator comparisons use six actual layer-zero projection
matrices per model at 65/96/128/256/512/1024 rows, FP16 and FP32 output, and
automatic, dense and packed selection: 216 cases per model/run. They retain
the same 25-sample warm/cache-evicted timing and independent reconstruction
checks described above.

The candidate build passed all 232 packed-prefill and 56 dense-HGEMM GPU
checks, covering independent references, canaries, tails, long reductions,
non-default streams and graph replay. Its SHA-256 is
`0b8f5c1eedff1106cc97c8a38ef227f7aa2d057c610c4998b0343256ea61eaee`.
Two translation units were rebuilt and 108 unchanged object hashes verified.

Compiled packed K5 kernels use 173 vector registers versus 130 before;
K2 uses 166 versus 149. Packed LDS remains 16,896 bytes. The new dense tile
uses 152 vector registers versus 169, and 16,896 LDS bytes versus 29,184.
None of these inspected variants spills to private memory. These resource
changes are tradeoffs, not evidence of a speedup.

The first bracketed operator comparisons completed with 216 byte-identical
cases per model/run. Dense geometry regressed: the 1,024-row FP32 path's
cache-evicted median group change was -4.39% on 9B and -3.45% on 27B.
That geometry change was reverted. The existing dense kernel remains primary.

Packed staging is more promising but shape-dependent. At 65/96/128 rows,
median cache-evicted FP32 operator gains were 9.17–14.11% on 9B and
13.56–19.40% on 27B. These group medians conceal small-row QKV regressions:
up to 19.8% on 9B and 4.3% on 27B against the faster control. K5 at 256 rows
also regressed on several projections. The initial universal replacement
was therefore rejected despite its average gains.

The refinement keeps both serial and staged native specializations, choosing
by bitwidth, codebook, rows and output width. K2/default can stage above 128
rows; its smaller rows and K5/mul1's smaller rows stage outside the measured
QKV width band. Larger K5 staging is limited to at least 512 rows and 8,192
output columns. Other regions retain serial execution. This does not expand
the existing packed-versus-dense crossover or add a user option.

The refined build passed all 288 native checks and all 216 real-weight cases
per model remained byte-identical across candidate and both controls. Its
SHA-256 is `fbd68ead6e528aa14316375ccceba5104d0c392b320d5425bb04461960d0747f`.
None of 1,568 inspected specializations has a private-memory spill allocation.
With automatic selection, median cache-evicted FP32 operator gains against
the faster control were 10.07–13.56% at 65–128 rows on 9B, and 11.83–17.95%
on 27B. At 256/512 rows they were 0.80%/4.15% on 9B and 14.24%/18.70% on
27B. The unchanged 1,024-row dense path was within about 0.2%.

Complete-model comparisons are complete. In addition to the six stage-one
cases, this stage includes 256- and 512-token occupied prompts to exercise
the medium-row paths. Each case still discards a warmup and measures three
fixed 128-token continuations.

The 9B comparison completed with 72 measured continuations / 9,216 generated
tokens, all matching inputs, output IDs and draft windows exactly.

| MiMo prefill case | Control before / after, tokens/s | Candidate, tokens/s | Prefill change against control midpoint |
| --- | ---: | ---: | ---: |
| Target only, task 0 | 661.524 / 659.457 | 737.696 | +11.69% |
| Target only, task 4 | 779.865 / 785.439 | 859.420 | +9.81% |
| MTP, task 0 | 627.090 / 626.150 | 693.508 | +10.67% |
| MTP, task 4 | 745.295 / 747.246 | 821.772 | +10.12% |
| MTP, occupied 256 | 1508.296 / 1507.147 | 1505.897 | -0.12% |
| MTP, occupied 512 | 1284.033 / 1297.524 | 1306.014 | +1.18% |
| MTP, occupied 4K | 1858.393 / 1861.211 | 1862.905 | +0.17% |
| MTP, occupied 16K | 1769.784 / 1768.896 | 1769.530 | +0.01% |

Short-prompt first-token latency fell 7.25–8.91%. Decode changed -0.90% to
+0.55%; the target-only task-4 continuation took 0.38% longer overall despite
its faster prefill. Other total-time changes were -0.90% to +0.11%. These are
short-prompt prefill gains, not a broad long-prefill or decode-speed claim.
The 27B comparison matched input hashes, output tokens and draft windows in
all seven cases below: 63 measured continuations / 8,064 tokens. The original
256-token case had a baseline-repeat discrepancy described separately below.

| Qwenseek prefill case | Control before / after, tokens/s | Candidate, tokens/s | Prefill change against control midpoint |
| --- | ---: | ---: | ---: |
| Target only, task 0 | 226.695 / 222.829 | 249.764 | +11.12% |
| Target only, task 4 | 265.667 / 261.723 | 292.116 | +10.78% |
| MTP, task 0 | 213.189 / 213.867 | 239.750 | +12.28% |
| MTP, task 4 | 255.788 / 252.440 | 282.123 | +11.02% |
| MTP, occupied 512 | 413.063 / 402.797 | 414.104 | +1.51% |
| MTP, occupied 4K | 572.693 / 570.627 | 573.494 | +0.32% |
| MTP, occupied 16K | 520.491 / 516.190 | 521.453 | +0.60% |

Short-prompt first-token latency fell 8.30–9.56%. Decode changed -1.02% to
+0.48% across these cases. Target-only total time increased 0.38% and 0.16%;
the other complete continuations finished 0.44–1.61% sooner. Long-prompt
prefill remains essentially unchanged. Against the faster control rather than
the midpoint, short-prompt prefill improved **9.4–11.5% on 9B** and
**10.0–12.1% on 27B**.

In the original 27B 256-token case, all three candidate repetitions matched
the first control exactly, but all three repeated-control outputs diverged
at generated token 84. The two processes using the unchanged baseline thus
disagreed with each other. This original case is excluded from speed claims;
it is not counted as an exact full-model comparison. A separate fresh-process
baseline/candidate/baseline diagnostic using the same 256-token input produced
identical inputs, 1,152 output tokens and draft windows across all nine measured
continuations. The original cross-process reproducibility discrepancy remains
a validation limitation; its underlying cause has not been established.
No GDN implementation was changed as part of this diagnosis.

All completed model processes exited successfully without resource stops.
The first 27B control took 247 seconds to load; startup is excluded from the
inference measurements. OMP implemented the bounded native candidates and
stopped before coordinator builds, numerical checks and performance validation.

Final source cleanup restored the packed header's original LF line endings.
The final build rebuilt one translation unit and verified/reused 109 unchanged
objects. Its file hash differs because internal symbol identifiers changed;
the GPU instruction and constant sections are byte-identical to the measured
candidate. GPU metadata matches after normalizing those identifiers, as do
host registration strings; host instructions and relocations are identical.
The final build independently passed all 288 native GPU checks and has no
private-memory spill allocation among the 1,568 inspected specializations.
Both builds' checker processes logged ROCm WSL topology and SharedSignalPool
shutdown warnings; they exited zero with no resource-monitor stop.

**Promoted:** packed prefill now automatically selects the qualified staged
kernel by format and geometry, retaining serial execution elsewhere. The
existing dense WMMA path and packed/dense crossover remain unchanged. There
is no new user option or model-name check. The stage-two primary was
`prefill-pipeline-v3`, containing the stage-one changes as well, with SHA-256
`b286d3fbe4752425868a7c363893e19777e90d40d7560236fe913a065444e6c2`.
Source fingerprints match the final build. These results qualify the stated
GPU and model configurations, not every future model or GPU.

## 3. GDN recurrent core

**Promoted:** the compact 128-thread recurrent kernel is now selected
automatically on ROCm for batch one, 128-by-128 heads, at most 64 value heads
and more than one token. Single-token calls retain the previous implementation.
Selection uses geometry, not model names or a new option. The implementation
is in the tracked vendored `gdn.cu`; the local registration uses the verified
`gdn-core-v4` binary. New users receive this behavior when they build and
register the published source revision.

Work resumed after the user-authorized storage relocation. A bounded OMP job
implemented the first compact specialization; the coordinator refined its
dispatch, isolated regressions and performed qualification. Independent state,
output, history/rewind and graph checks preceded operator timing and repeated
9B/27B model comparisons. The experiment sequence below explains why broader
recurrent dispatch, cross-token state retention and convolution specialization
were rejected.

The compact implementation is confined to the vendored `gdn.cu`: 128 threads in
the selected 128-by-128, four-value-split ROCm case, unique Q/K normalization
loads, 32 retained state floats per thread and explicit four-way partial
reductions. CUDA, unsplit and other-dimension paths retain the existing kernels.
The explicit reduction tree changes floating-point reduction order from the
old shared atomics; it does not change the state or output precision.

The coordinator's new `gdn-recurrent` suite passed all 67 checks against both
the existing primary and candidate. It uses an independent float64 NumPy
recurrence, checks every written/untouched history slot, native rewind plus
suffix replay, grouped heads, generic fallbacks, canaries, non-default streams,
and graph replay with changed input/state contents and slot values. An
analytical case distinguishes BF16 truncation from round-to-nearest. Maximum
state absolute error was about `6.25e-8` in each binary. Graph pointer-patching
and model-level MTP behavior remain part of complete-model qualification.

The candidate rebuilt `gdn.cu` and verified/reused 109 unchanged objects. Its
SHA-256 is `e259bd262d1f6b152e47c0f33374b21cd79ce040065b5d3dbf719c63dd83f123`.
The no-history/history compact kernels use 85/86 vector registers versus
79/63 for the previous four-value-split kernels. Shared memory stays at
2,080 bytes; neither candidate specialization spills to private memory.

A primary/candidate/primary operator comparison covered 32/48 value heads,
sequence lengths 1/2/3/5/9, and both history modes. Each mode retained 31 HIP
event samples. Warm graphs contain 32 recurrent launches per sample to reduce
host-gap effects; evicted graphs contain one launch after touching 256 MiB.
Eager intervals include dispatch gaps. These timings are not bandwidth counters.
Inputs and initial-state hashes match across all three processes. Warm graph
gains were substantial, while some evicted 9B cases regressed. Full-model
latency, output and MTP comparisons determine promotion.

The first complete 9B comparison matched all input hashes, output token IDs
and MTP draft windows across 54 measured continuations / 6,912 tokens. Each
case discarded a warmup and measured three fixed 128-token continuations in
primary/candidate/primary processes. Decode changes against the control
midpoint ranged from -0.56% to +0.45%; against the faster control they ranged
from -0.87% to +0.38%. These measurements do not establish a useful 9B model
speedup. The complete 27B comparison also matched all 54 measured
continuations / 6,912 tokens and draft windows. Against the faster control,
short-context MTP decode improved 3.94% and 4.62%, but target-only decode
regressed 3.35% and 1.81%. Occupied 4K/16K decode changed +1.33%/-1.46%.
The first candidate was not promoted. Model loading from D: is excluded from
all prefill/decode measurements.

The refinement retains the previous recurrent kernel for single-token calls.
For the compact multi-token path, each thread keeps its state slice in
registers across the sequence, writing every saved history checkpoint or
only the final state when history is disabled. It also specializes the
native convolution on width four, removing runtime width predicates and
dynamic window indexing while preserving FMA order. Both choices use
geometry, not model names or new user options. Other convolution widths
retained the generic implementation. This second candidate was subsequently
rejected by complete-model measurements.

The refined binary rebuilt one translation unit and reused 109 verified
objects; its SHA-256 is
`eebd68650ab2d83c759a4a92eaf3fdd973d6a29719390da86443a6bcb265e74d`.
Both primary and candidate passed the expanded 75-record recurrence suite
and 60 convolution cases. Added recurrence cases cover 17-token saved
history, a 65-token sequence and multi-token BF16 truncation. The convolution
oracle builds CPU windows and independently rounds each FP32 multiply-add;
checks include bias, activation, widths 1/2/3/4/5/8/16, slot selection,
partial blocks, graph replay and rewind/suffix replay. All convolution output
and state hashes matched between binaries.

The compact no-history/history kernels use 133/92 vector registers and no
private-memory spill allocation. Width-four convolution uses 27 registers
with activation and 23 without, versus 79/64 and 75/60 respectively for the
generic no-history/history variants. None of the 1,578 inspected selected
specializations spills. The larger recurrent register allocation is a
tradeoff for fewer state reads/writes, not a performance claim by itself.

A second primary/candidate/primary operator bracket measured recurrence and
convolution with identical inputs and initial states. For 2–9-token calls,
warm-graph recurrent throughput improved 61–203% against the faster control;
width-four convolution improved 21–49% over the tested 1–9-token shapes.
Single-token recurrence deliberately retains the old kernel. Evicted
measurements include regressions in individual cases, so these operator
numbers cannot stand in for model latency.

The refined 27B comparison matched all 54 measured continuations / 6,912
tokens and draft windows, but failed the performance gate. At occupied 16K,
decode was 40.046 tokens/s versus controls of 43.298 and 43.341, a 7.56%
regression against the midpoint. Short-context results were mixed despite
an 8.14% gain on one MTP workload and a 3.17% gain at 4K against the faster
controls. The candidate was rejected. Its unused 9B baseline was stopped
during loading through the supervisor's scoped stop file; this was an
intentional stop, not a resource failure, and no 9B candidate run started.

The isolation build restores the first candidate's state handling between
tokens while keeping the single-token fallback and width-four convolution.
It passed all 135 native checks, including exact convolution output/state
hashes, and has SHA-256
`55a2e33a0ba28044df678c7b2acf87eefdeaa194f42cded7ac050e84381f9907`.
A separate 16K diagnostic recovered decode to 42.706 tokens/s with identical
input, output and MTP-window records. Those controls preceded this diagnostic;
it is evidence for dropping cross-token state retention, not a fresh
bracketed speed claim.

Its fresh 27B bracket again matched all 54 measured continuations / 6,912
tokens and draft windows. MTP decode improved 3.55% and 5.46% on the short
workloads and 5.61% at 4K against the faster control. However, target-only
task 0 regressed 3.32% against the midpoint (4.45% against the faster control).
The 16K candidate median of 42.029 tokens/s fell between controls of 43.316
and 41.923; that case cannot establish a stable gain or a regression beyond
the observed control drift. The target-only regression was sufficient to
reject this combination. The queue stopped before starting a 9B process.

The final isolation candidate restored the original convolution completely
while keeping the same compact recurrent implementation and dispatch. It
used the most recent completed primary process as the preceding control,
then ran a full candidate and a fresh primary with the same six cases.

This recurrent-only build has SHA-256
`0ad3db65fcda9a59629a81d3a93b85f2af7dd40013207ced7088b19a6b0c132a`.
It passed all 135 native checks. All 14 preserved convolution, recurrent
fallback and surrounding GDN helper kernel instruction bodies are
byte-identical to the primary; the compact multi-token kernel is the only
newly selected GPU implementation. Its no-history/history variants use
85/86 vector registers, with 2,080 bytes of shared memory and no scratch
spill allocation. None of the 1,574 selected inspected specializations spills.

The final 27B bracket matched all 54 measured continuations / 6,912 tokens,
input hashes and MTP windows. The preceding control was the last completed
primary run from the prior bracket; a new primary run followed this candidate.

| Qwenseek decode case | Control before / after, tokens/s | Candidate, tokens/s | Change against control midpoint |
| --- | ---: | ---: | ---: |
| Target only, task 0 | 23.394 / 23.207 | 23.668 | +1.58% |
| Target only, task 4 | 23.250 / 23.075 | 23.660 | +2.15% |
| MTP, task 0 | 42.649 / 44.009 | 43.310 | -0.05% |
| MTP, task 4 | 47.003 / 46.596 | 48.361 | +3.34% |
| MTP, occupied 4K | 41.857 / 41.941 | 43.067 | +2.79% |
| MTP, occupied 16K | 41.923 / 43.318 | 42.602 | -0.04% |

The repeatable claim is **2.89% faster short-task-4 MTP decode and 2.68%
faster 4K decode against the faster controls**. The other MTP case and 16K
fall between their controls; they do not establish a gain. The unchanged
single-token GPU kernels' timings are retained for transparency, not credited
as new single-token kernel speedups. Total continuation time at 4K/16K was
0.42%/0.20% longer against the midpoint despite the 4K decode improvement;
this stage does not claim a general prefill improvement.

The final MiMo bracket used fresh primary/candidate/primary processes. All
54 measured continuations / 6,912 tokens, input hashes and MTP windows matched.

| MiMo decode case | Control before / after, tokens/s | Candidate, tokens/s | Change against control midpoint |
| --- | ---: | ---: | ---: |
| Target only, task 0 | 57.749 / 57.610 | 57.785 | +0.18% |
| Target only, task 4 | 57.628 / 57.628 | 57.539 | -0.15% |
| MTP, task 0 | 84.105 / 83.919 | 84.178 | +0.20% |
| MTP, task 4 | 92.157 / 92.065 | 92.997 | +0.96% |
| MTP, occupied 4K | 91.544 / 91.682 | 92.896 | +1.40% |
| MTP, occupied 16K | 88.205 / 87.780 | 88.964 | +1.10% |

MiMo's task-4 MTP decode improved **0.91%, 1.32% and 0.86%** at short, 4K
and 16K contexts against the faster controls. Other cases were effectively
unchanged. Task-4 total continuation time improved 0.82%, 0.59% and 0.32%
against the control midpoint. These are modest workload-specific gains,
not evidence of a universal GDN or prefill speedup.

Final qualification preserved **108 measured continuations / 13,824 tokens**
and exact MTP windows across both model comparisons. This total includes
the reused preceding 27B control. The 75 recurrent and 60 convolution native
checks passed; all convolution output/state hashes matched the previous
primary. After promotion, the public validator used the ordinary installation
record and passed all 75 recurrent checks again with the promoted hash.
All completed qualification processes exited zero without a resource stop.
ROCm WSL topology and SharedSignalPool shutdown warnings also occurred with
the controls; they did not prevent successful completion.

Four focused validator CPU tests passed, as did CLI help, checker help and
diff whitespace checks. The whole unit suite was subsequently repeated on
Windows and Linux for [integration into main](VALIDATION.md#native-kernel-integration-into-main-2026-09-27).
All 109
other native objects were verified and reused, and build source fingerprints
match the final source. The registration change updated only the extension
directory and its SHA-256, preserving permissions, offline loading, GPU
ownership and resource controls.

The final primary SHA-256 is
`0ad3db65fcda9a59629a81d3a93b85f2af7dd40013207ced7088b19a6b0c132a`.
Private evidence is retained in `artifacts/gdn-core-20260927/`: model summaries,
native results, build/resource and preserved-instruction comparisons,
`promotion-audit.json`, and the pre-promotion installation backup. The OMP job
record is under `.codex/.omp-jobs/gdn-core-20260927/`.

This completes the three authorized native-kernel stages. The measured scope
is the two stated models on RX 7800 XT / gfx1101. CUDA and unsupported shapes
keep their previous kernels; other ROCm GPUs and future models need their own
performance validation. Large prefills normally use the separate FLA chunked
rule, and no memory-bandwidth saturation or megakernel claim follows from
these measurements.
