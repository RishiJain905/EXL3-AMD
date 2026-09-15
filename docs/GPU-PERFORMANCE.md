# Native prefill measurements

This records the initial native-prefill study. The subsequent decode and
prefill campaign using the full 120064-context serving CLI is recorded in
[RUNTIME-PERFORMANCE.md](RUNTIME-PERFORMANCE.md), including the revised kernel,
flag comparisons and a newly observed exact prefix-reuse limitation.

These measurements cover one dense 27B Qwen-family EXL3 model on a Radeon
RX 7800 XT (gfx1101), using the existing WSL ROCm environment, Q8 K/V and
the long-attention profile. They do not establish performance on other models
or GPUs. Model files and raw benchmark artifacts are private and untracked.

## What the profile showed

The warm 4096-token workload spent most of its measured native-entry GPU
intervals in reconstructed-weight GEMMs, particularly projections with FP32
output. Python submission time for those calls was much smaller. The installed
profiler exposed CPU events but no usable HIP kernel self-time, so CUDA events
around native entry points supplied the comparison. Those intervals include
scheduling gaps and must not be read as exclusive kernel times.

Substituting ATen for half-output GEMMs did not improve the complete workload.
The retained change is an optional HIP WMMA path for FP16 inputs with FP32
accumulation and output. It does not round the output through FP16.

`--prefill-gemm wmma` requires a verified extension exporting prefill GEMM ABI 1.
The default remains `blas`. The specialized path admits gfx1101, at least 64
rows, aligned contiguous inputs and K/N dimensions divisible by 128; unsupported
geometry falls back to BLAS. Shape, device, capacity, stride, alias and integer
bounds are checked before dispatch. The kernel uses the caller's stream and
allocates no persistent buffers.

This initial study selected a 128 by 128 by 64 tile with 16 wave32 groups. Long uninterrupted
WMMA reductions initially exceeded the existing numerical threshold. Reducing
in groups of 256 K elements and combining partial results with FP32 additions
lowered the measured relative RMS error at K=17408 from approximately 5.4e-5
to 2.5e-6. The acceptance threshold stayed at 5e-5. Exact half-subnormal cases
passed; the experiment does not attribute the earlier drift to a specific
undocumented hardware behavior.

## Controlled comparison

Both runs used the same final binary, changing only `--prefill-gemm`. The
context capacity was 32768, chunk 1024, MTP depth 6 with confidence 0.6, greedy
sampling and prefix reuse disabled. Fixed tokenized source-code prompts had
4096 or 16384 tokens, with 128 output tokens. Each ran twice after a separate
warmup; the table reports the second request. First-use costs are excluded.

| Occupied prompt | BLAS prefill | WMMA prefill | Gain | First token, BLAS / WMMA |
| --- | ---: | ---: | ---: | ---: |
| 4096 tokens | 290.37 tok/s | 540.40 tok/s | 1.86x | 14.30 / 7.78 s |
| 16384 tokens | 275.16 tok/s | 492.55 tok/s | 1.79x | 59.76 / 33.50 s |

All 11 input hashes and generated token sequences matched between these two
runs, including three short Python, SQL and explanatory prompts. Warm decode
was effectively unchanged: approximately 47-48 tok/s for the code
continuations and 35-39 tok/s for the short prompts. Small timing differences
are not treated as decode improvements.

The native operator suite passed 39 GPU checks on the final binary. It covers
M tails, large K/N, strided-output canaries, exact subnormals, FP16 and shape
fallbacks, invalid arguments, output alias rejection, non-default streams and
CUDA graph replay with changed inputs. These are bounded numerical and memory
boundary checks, not a proof for arbitrary tensors or every ROCm release.

## Runtime flag sweep

The sweep completed 112 requests across 12 configurations, each in a separate
process with the GPU lease and unchanged resource guards. All used capacity
32768, Q8 K/V, long attention and prefix reuse off. Except where shown, chunk
size was 1024 and MTP depth/confidence were 6 / 0.6. Each configuration used
warmup, two 4096-token code continuations and two rounds of three short chat
prompts; the two native-control runs also included two 16384-token requests.
Non-warmup output budgets were 128 tokens. Thinking was disabled for chat prompts.

Rates below use the second request for each prompt. The short-prompt column
divides the combined decode-token count by the combined decode time for Python,
SQL and prose; it excludes each first emitting MTP iteration. Agreement counts
include warmup and compare all shared input hashes and output token sequences
with the BLAS control. Differences in a candidate do not establish poor answer
quality, but they prevent claiming exact output equivalence for that candidate.

| Configuration | 4K prefill, tok/s | 4K decode, tok/s | Short decode, tok/s | Exact agreement |
| --- | ---: | ---: | ---: | ---: |
| BLAS, chunk 1024, MTP 6 / 0.6 | 290.4 | 48.0 | 36.8 | 11/11 |
| WMMA, chunk 1024, MTP 6 / 0.6 | 540.4 | 46.7 | 36.9 | 11/11 |
| WMMA, chunk 512 | 509.3 | 47.8 | 37.1 | 9/9 |
| WMMA, chunk 2048 | 552.8 | 44.6 | 36.9 | 7/9 |
| WMMA, chunk 4096 | 519.3 | 46.0 | 36.9 | 9/9 |
| WMMA, MTP 4 / 0.6 | 541.2 | 47.3 | 38.6 | 9/9 |
| WMMA, MTP 8 / 0.6 | 541.2 | 39.3 | 36.3 | 7/9 |
| WMMA, MTP 6 / 0.8 | 541.5 | 46.1 | 36.9 | 9/9 |
| WMMA, MTP 4 / 0.6, gdn-mlp | 540.6 | 47.2 | 38.6 | 9/9 |
| WMMA, MTP 4 / 0.6, wmma-register | 541.1 | 36.7 | 30.7 | 7/9 |
| WMMA, fixed MTP 4 | 540.3 | 47.4 | 38.7 | 9/9 |
| WMMA, fixed MTP 4, GPU embedding/draft/metadata | 542.8 | 48.2 | 39.5 | 9/9 |

Chunk 1024 is the selected compromise. Chunk 2048 gained only about 2% prefill
and changed both longer continuations; 512 and 4096 were slower. The alternate
small-row kernel and depth 8 regressed decode. Adding the MLP fusion or raising
confidence supplied no useful aggregate gain.

Fixed MTP depth 4 and adaptive depth 4 / 0.6 were effectively tied on the
shorter workloads and preserved the tested outputs. GPU embedding
plus draft bookkeeping added about 2.37 GiB of reserved VRAM for approximately
2% more short-prompt decode than fixed depth alone. That result is limited to
32768 capacity and is not the selected larger-context configuration.

Jinja is already enabled; `--jinja` changes no load or compute behavior. CPU MoE
offload does not apply to this dense model. All reported runs retained Q8 precision.

## Larger-context selection

Three further configurations used capacity 120064, Q8 K/V, long attention,
WMMA prefill and chunk 1024. Prefix reuse was disabled. All 33 requests passed
and matched the BLAS control's input hashes and generated token sequences.
The table uses warmed second requests and the same decode accounting as above.

| Drafting | 4K prefill | 16K prefill | 4K decode | 16K decode | Short decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed MTP 4 | 540.4 | 492.8 | 47.3 | 44.8 | 38.6 |
| MTP 6, confidence 0.6 | 547.6 | 506.2 | 47.2 | 49.6 | 37.8 |
| MTP 6, confidence 0.9 | 553.7 | 504.1 | 46.7 | 49.1 | 36.7 |

All rates are tok/s. These separate runs include normal clock and system
variation; differences of a few percent are not treated as kernel improvements.
Depth 6 with confidence 0.6 is selected for the larger-context command: it
retains the stronger long-continuation decode result with modest short-prompt
cost. Fixed depth 4 remains a reasonable short-prompt alternative. Confidence
0.9 regressed the combined short-prompt rate and is not selected.

The selected run peaked at approximately 12.7 GiB reserved GPU memory. Resource
limits were unchanged. This validates allocation at capacity 120064 and occupied
prompts through 16384 tokens, not a fully occupied 120K prompt or long-context
answer quality.

## Final HTTP acceptance

The normal runtime CLI loaded the registered final binary at capacity 120064
with Q8 K/V, long attention, WMMA prefill, chunk 1024, MTP depth 6/confidence 0.6
and prefix reuse enabled. All 11 reasoning, streaming, tool-call, validation and
cancellation checks passed. Non-streaming and streamed reasoning/content agreed;
model-generated tools stayed data and no generated command was executed.

Nine code-workload requests then used the saved chat template, greedy sampling,
thinking disabled and 32 output tokens. This process was already warm from API
acceptance. Each input size used a different system prefix for its initial cache
miss. All four repeated outputs matched. First-token times below are the engine's
reported timing, excluding client transport and queueing.

| Prompt tokens | Fresh compute, tok/s | Fresh first token | Repeated first token | Reused tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1037 | 418.32 | 2.57 s | 0.52 s | 1024 |
| 3800 | 495.73 | 7.90 s | 0.67 s | 3584 |
| 7074 | 525.17 | 13.71 s | 0.69 s | 6912 |
| 15011 | 506.87 | 29.87 s | 0.76 s | 14848 |

The appended turn reused 6912 tokens and reached its first token in 0.75 seconds.
A separate 2831-token prompt with temperature, top-k/top-p/min-p, penalties and a
fixed seed produced identical 128-token output on the fresh and repeated request;
the repeat reused 2816 tokens. These cache benefits reduce latency without being
counted as additional computed prefill tokens. The 32-token HTTP responses are
functional/latency checks; use the 128-token matrix for decode comparisons.

Foreign Host, foreign Origin and non-JSON browser POST probes returned 400, 403
and 415 before GPU admission. Observed process-owned sockets were a loopback
listener and the incoming loopback test connection during acceptance, then only
the loopback listener at idle. No outbound connection was observed in those
snapshots. This is a bounded observation, not a guarantee against every leak.

Final health recorded 18 completed requests, one deliberate cancellation and
zero engine failures. Live recurrent checkpoints occupied 886.5 MiB, below the
1 GiB bound; peak reserved GPU memory was approximately 12.7 GiB. The server
was explicitly stopped and the supervisor confirmed clean shutdown. Native
benchmark processes still printed the pre-existing SharedSignalPool warning;
the final HTTP-server log did not contain it.

## Reproduction and limits

Use a separately built and registered compatible extension as described in
[BUILD.md](BUILD.md). Compare identical model bytes, input tokens, precision,
sampling and output lengths; keep cached-token credit separate from raw
prefill. [OPTIMIZATION.md](OPTIMIZATION.md) describes the controls and timing
accounting. [PREFILL-PERFORMANCE.md](PREFILL-PERFORMANCE.md) records the earlier
storage and prefix-reuse study before this native change.

Native benchmark processes print a `SharedSignalPool` signal-leak warning during
process teardown with both BLAS and WMMA. This work does not resolve that
pre-existing native resource-lifetime warning. Passing the measured requests
does not establish prolonged serving, full-context quality or BF16 fidelity.
