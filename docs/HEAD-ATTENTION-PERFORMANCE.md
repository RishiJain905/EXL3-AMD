# Vocabulary head and occupied-context attention

Work dated September 26, 2026, on RX 7800 XT / gfx1101. This report covers
roadmap items 5 and 6 only. Measurements are against the previous primary
packed-prefill/paired-MLP implementation, not the older BLAS baseline.
The validated head and attention paths are primary automatically. The local
registration now points to the qualified ABI-3 build; no enabling flags are
needed for normal generation or serving.

## Scope and method

The 9B model is the local MiMo K5/H6 mul1 quantization, with 4096 hidden
features, 248320 vocabulary entries and 16 query / 4 KV heads of dimension
256. The 27B model is Qwenseek-3.8-27B-CyberLite-EXL3: default-codebook K2
target / K4 head, hidden size 5120, and 24 query / 4 KV heads of dimension
256. Model identity is never used for dispatch.

Attention controls and candidates use identical model weights, tokenized
prompts, K/V precision, cache capacity, prefill chunk and generation settings.
Each treatment is bracketed by controls in the same process. At 16K, each
treatment has a warmup and three measured 128-token continuations. The 32K
runs use two measured continuations per treatment, after the initial model
warmup, to stay within the existing 900-second run limit. The final 32K Q6/MTP
repeat also warms the candidate at that full context before measuring it.
The reported control
is the mean of the before/after medians. These are target-only decode tests
unless explicitly labeled MTP. Prefill kernels and cache encoders are unchanged.

MiMo BF16-MTP checks use the frozen sibling Step4 evaluator (commit b502d37)
with this checkout's current projection compatibility and paired-MLP modules.
The sibling checkout was not edited. This is distinct from the normal public
main-checkout CLI, whose MiMo validation is target-only.

All GPU work uses the existing exclusive lease, verified extension hash,
offline model loading, allocator cap and process monitor. Operator timings
include launch/submission overhead. A 256 MiB buffer update puts pressure on
cache before the head's evicted samples; it does not prove a measured DRAM
hit rate. No bandwidth-saturation percentage is inferred.

## Attention implementation

A pure metadata policy chooses tile size, split count, grouped-head tile and
parallel split reduction by GPU, query/KV geometry, query rows, K/V formats
and the occupied page-table bound. It neither changes cache precision nor
reads a device length back to the CPU. Device lengths remain authoritative
for causal masking on every graph replay.

Short contexts retain inherited scheduling after the 4K model trial showed
no reliable improvement. Unknown GPUs, unsupported geometries, batched jobs,
local/noncausal attention, sinks, softcaps and nonuniform research codecs
retain their established paths. This is automatic reuse for supported future
model geometries, not a claim that every model or GPU has been tuned.

The automatic policy currently admits these geometry envelopes. Context bounds
are page-table bounds capped by physical pages, so padding can round occupied
length upward; they are not the configured maximum context.

| Query / KV heads / dimension | K/V formats | Query rows | Bound |
| --- | --- | --- | --- |
| 16/4/256 or 24/4/256 | F16, Q5, Q6 or Q8 | 1–8 | 16384–131072 |
| 16/4/256 or 24/4/256 | Q4 | 1–4 | 16384–131072 |
| 16/4/256 or 24/4/256 | Q8/Q4, Q6/Q4, Q8/Q6, Q5/Q8 | 1–4 | 16384–131072 |
| 32/8/128, 16/2/128, 32/4/128, 32/8/256 | F16 or Q8 | 1–4 | 16384–65536 |

All paths require one sequence, causal global attention and FP16 queries.
Unsupported cases fall back automatically. The extra geometries have operator
coverage; full-model coverage is limited to the two models named above.

The benchmark now accepts per-case query heads, KV heads and head dimensions,
as well as independent integer K/V precision. The initial and edge sweeps
passed 716 independent FP32-oracle comparisons across F16/Q4/Q5/Q6/Q8,
mixed formats, 1/3/5/7 query rows, roughly 1K to 120K occupied positions,
and additional 128-dimensional and eight-KV-head geometries. These operator
results alone do not establish full-model throughput or broad model quality.
Maximum relative L2 against the independent FP32 references was 0.000506
(rounded upward). The final policy also passed 116 nondefault-stream and
changed-query/changed-length graph checks, with maximum relative L2 0.000409
against inherited attention. Queries, masks and K/V formats stay identical
within each comparison.

Selected 32K operator timings illustrate why format-specific schedules matter:

| Geometry / query rows | K/V format | Inherited ms | Tuned ms |
| --- | --- | ---: | ---: |
| 16/4/256, 1 | Q8 | 0.736 | 0.419 |
| 24/4/256, 3 | Q8 | 1.296 | 0.570 |
| 24/4/256, 3 | Q5 | 2.942 | 0.634 |
| 24/4/256, 3 | Q6 | 2.035 | 0.623 |
| 24/4/256, 7 | Q5 | 4.094 | 1.188 |
| 24/4/256, 7 | Q6 | 2.825 | 1.133 |

These are individual attention calls on seeded caches, not model token rates.

## Model attention results

Measured target-only results:

| Model | K/V cache | Occupied input tokens | Control decode tok/s | Automatic decode tok/s | Gain | Exact token agreement |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MiMo 9B | F16 | 16000 | 53.43 | 54.69 | 2.36% | Yes |
| 27B default codebook | F16 | 16000 | 21.01 | 21.42 | 1.93% | Yes |
| MiMo 9B | Q8 | 16000 | 49.06 | 50.45 | 2.84% | Yes |
| 27B default codebook | Q8 | 16000 | 19.69 | 20.54 | 4.34% | Yes |
| MiMo 9B | Q8 | 32000 | 43.38 | 48.24 | 11.20% | No, diverges after 84 shared tokens |
| 27B default codebook | Q8 | 32000 | 17.71 | 19.62 | 10.80% | Yes |

With integrated MTP depth two:

| Model / cache | Occupied input tokens | Control decode tok/s | Automatic decode tok/s | Gain | Exact tokens and draft windows |
| --- | ---: | ---: | ---: | ---: | --- |
| 27B / Q8 | 16000 | 39.13 | 41.78 | 6.78% | Yes, 1152 measured tokens |
| 27B / Q6 | 32000 | 28.93 | 37.56 | 29.82% | Yes, 768 measured tokens |

The warmed Q6 candidate samples were 37.78 and 37.34 tok/s; the four control
samples ranged from 28.69 to 29.06 tok/s. Its automatic schedule recorded
2613 applied calls across warmup and two measured continuations, versus zero
in both controls. Prefill remained 442.71 versus 442.67 tok/s. The initial
Q6 attempt included a compilation stall in its first candidate decode
(27.66 tok/s), followed by 37.51 tok/s on the next continuation. That attempt
is retained as cold-start evidence; the table uses the separate warmed repeat.

The 9B 32K continuation changes when the split schedule changes floating-point
reduction order. Both inspected generated binary-search functions passed
10801 sorted-list/target examples, including duplicates, empty inputs and
missing targets. Timing only the 84 identical prefix tokens retains an
11.24% gain (43.38 to 48.25 tok/s). This demonstrates that changed wording did
not produce the timing gain. It is a narrow functional check, not proof of
unchanged model quality or universal greedy-token agreement. The inherited
profile remains available when exact reproduction of that schedule matters.

## Vocabulary head implementation and MTP

The ABI-3 head keeps a second compressed view in
`[N/128,K/16,8,16*bits]` order. Eight wave32 groups share a 128-column output
block, reuse decoded weight fragments across verification rows and fuse the
output Hadamard. The projection uses two launches instead of three. Repacked
rows 1/2/3 use four-way K-loop unrolling; row 5 retains the shorter loop.

The dot accumulation order is unchanged. For FP16 logits, the kernel rounds
the dot result to FP16 before applying the inherited FP16 Hadamard and
post-scale. FP32 logits keep the original FP32 output transform. Both tested
models actually emit FP16 logits. Changing only the final cast would not
preserve their arithmetic.

The explicit entry validates shape, dtype, devices, alignment and writable
buffer overlap before launching on the current stream. The adapter retains
the original trellis and allocates at most 1 GiB per distinct head, leaving
a 1 GiB workspace reserve within both free memory and the allocator fraction.
A failed memory check blocks repeated allocation attempts until explicit
reinstallation. Shared target/MTP heads use one copy; disabling and unloading
release it. MiMo's view occupies 762839040 bytes (about 0.71 GiB), and the
27B view occupies 635699200 bytes (about 0.59 GiB).

The final native build passed 185 head checks. Coverage includes independent
reconstructed weights, exact inherited outputs, malformed buffers, canaries,
nondefault streams, changed-input graph replay, shared-copy reuse, source
replacement, allocation budgets, cold capture and unload. There are 52 FP16
numerical cases; maximum relative L2 against the independent weight oracle
across all head numerical cases was 0.000385 (rounded upward).

Actual-head operator measurements compare the same packed weights, inputs
and output dtype. Multi-row FP16 calls improved by approximately 5–15% in the
initial cache-pressure sweep. Single-row timing is measured again with 64
warmup pairs and 96 alternating control/candidate pairs to reduce order and
clock effects. The initial noninterleaved 9B single-row sweep was nearly
flat (1.8231 to 1.8453 ms); the better-warmed alternating comparison below
resolved that uncertainty. Every repacked output matched the control exactly.

| Actual head / rows | Inherited FP16 output ms | Repacked FP16 output ms | Operator speed gain |
| --- | ---: | ---: | ---: |
| MiMo 9B / 1 | 1.9199 | 1.8187 | 5.56% |
| MiMo 9B / 2 | 2.1250 | 1.9098 | 11.27% |
| MiMo 9B / 3 | 2.2166 | 1.9439 | 14.03% |
| MiMo 9B / 5 | 2.4375 | 2.3125 | 5.40% |
| 27B / 1 | 2.1446 | 1.9970 | 7.39% |
| 27B / 2 | 2.4020 | 2.1372 | 12.39% |
| 27B / 3 | 2.6074 | 2.2658 | 15.08% |
| 27B / 5 | 2.8714 | 2.7454 | 4.59% |

One-row entries use the 96-pair alternating protocol. Other rows use the
21-sample cache-pressure sweep. These are eager operator medians, not model
token rates. Separate FP32-output operator sweeps also matched exactly;
FP32 is not substituted for the models' FP16 logits.

Full-model head comparisons use MTP depth two, F16 KV, context capacity 4096
and 256-token prefill chunks. Each of the three tasks has a warmup and three
128-token continuations per treatment, with the candidate bracketed by
controls. Every candidate must record a positive packed-call delta; controls
must record zero. These comparisons isolate the head and retain inherited
attention scheduling.

| Model | Task | Control decode tok/s | Packed-head decode tok/s | Gain |
| --- | --- | ---: | ---: | ---: |
| MiMo 9B, BF16 MTP | Binary search | 78.95 | 80.06 | 1.40% |
| MiMo 9B, BF16 MTP | Short response | 76.55 | 76.89 | 0.44% |
| MiMo 9B, BF16 MTP | SQL parameters | 71.88 | 73.27 | 1.94% |
| 27B, quantized MTP | Binary search | 43.86 | 44.28 | 0.96% |
| 27B, quantized MTP | Short response | 42.57 | 43.41 | 1.96% |
| 27B, quantized MTP | SQL parameters | 39.45 | 40.22 | 1.97% |

MiMo matched all 3456 measured output tokens and every draft acceptance
window. The candidate made 1932 packed-head calls, versus zero in both
controls. The 27B model matched another 3456 tokens and every acceptance
window, with 1776 candidate packed-head calls and zero in controls. Both
released their extra views when disabled. These are modest decode gains;
MiMo's short-response change overlaps individual-sample variation. These
head results and the attention table isolate separate changes and should
not be added together as a measured combined gain.

A separate 32-token MiMo decode profile with the packed head disabled recorded
58.59 ms for the shared head, 271.88 ms for target blocks and 41.88 ms for the
draft block. The head is about 15% of the 391.39 ms total across outer modules.
These are inclusive CUDA-event intervals, including submission gaps, rather
than kernel self-time or hardware bandwidth counters. Parent and child module
timings are not added together.

### Rejected prototypes and corrected measurements

- The raw-layout prototype matched FP32 operators but had no reliable
  single-row gain. It remains an explicit experiment, outside automatic
  selection. The split-K boundary at vocabulary width 32768 is excluded
  because it uses a different inherited reduction order.
- Four-way unrolling for every row count regressed the five-row repacked
  operator from 2.2878 to 3.8438 ms. Unrolling is limited to rows 1/2/3.
- V1, V2 and V4 full-model head trials are invalid optimization evidence:
  their FP32-only kernels never ran for the actual FP16 heads. The initial
  adapter also rejected FP16-default heads at installation. Removing that
  guard alone still left zero packed calls, and the added dispatch assertion
  correctly rejected the trial. Previously reported 0.27–2.11% V2 model
  differences are withdrawn. The FP32 operator results and all attention
  comparisons remain valid. The initial 27B V4 trial was stopped through its
  monitor. ABI 3 adds the required FP16 arithmetic and is the only repacked
  version admitted for automatic installation.

## Reuse and fallback contract

The head adapter supports FP16 inputs and FP16 or FP32 logits for 1, 2, 3 or 5 rows,
hidden width up to 65536, and vocabulary width above 32768 and at most
1048576, with both widths divisible by 128. Default-codebook K2/K3/K4 and
mul1 K2 through K6 have independent weight-oracle checks. Bias, alternate
codebooks, other devices, unsupported rows, reconstruction, overrides and
native pointer-patching captures retain the inherited path. A prepared view
can participate in external graph capture; a cold capture cannot allocate it.

Packing happens once on the first eligible call after model/cache allocation,
so measured warm decode excludes that one-time allocation and transpose.
Weights and cache bytes are not requantized. The original trellis stays intact;
replacement invalidates the extra view. Geometry and tensor properties select
the kernel, with no model-name allowlist. Future compatible models benefit
automatically inside the tested envelope; unsupported cases remain functional.

For diagnosis, `--head-warps 1` selects the inherited wide-head path and
`--attention-profile default` selects inherited attention scheduling. Normal
generation and serving need neither enabling flag. The server's `/health`
contains head call counts, packed bytes and memory-fallback status; the CLI
records the same head statistics in its private result artifact.

With a registered matching runtime, run the public model-free acceptance from
Linux/WSL using a new private output directory:

```bash
python scripts/check_exl3_kernels.py --checks head-tiled attention-schedule --output artifacts/head-attention-checks
```

For new model geometry, `scripts/benchmark_exl3_attention.py` accepts per-case
`query_heads`, `kv_heads`, `head_dim`, `cache_type` or independent `k_bits` and
`v_bits`. It checks candidate output against an independent FP32 reference.
Keep operator gains separate from warmed full-model comparisons with identical
weights, prompts, precision and MTP settings. The runtime does no online
autotuning in the token loop.

## Normal CLI and HTTP acceptance

After promotion, ordinary launches used the local registration with no
`--config` or enabling flags. MiMo target-only generation on a held-out
14372-token prompt matched all 64 output token IDs against the diagnostic
control (`--head-warps 1 --attention-profile default`). The primary launch
recorded head ABI 3, 96 packed-head calls including warmup, zero head
fallbacks, one 0.71 GiB view and 512 automatic attention calls. Both CLI
launchers exited zero.

The normal 27B MTP2 HTTP server used Q8 KV and context capacity 20480. Its
14349-token greedy request produced a 64-token response whose content and
reasoning matched the streamed response exactly; the stream emitted its
completion marker. A seeded temperature-0.7/top-p-0.9 request then completed
with nine generated tokens and finite timing metadata. Final health reported
three completions, zero failures, zero cancellations, 175 packed-head calls,
748 automatic attention calls, one 0.59 GiB shared view and no memory fallback.

The supervisor stopped this owned server through its stop file. Uvicorn
completed application shutdown, the monitor recorded `clean_server_shutdown`
with expected SIGTERM exit -15, and the launcher returned zero. This is a
bounded three-request interface check, not a prolonged-serving test. MiMo
BF16-MTP validation remains the frozen Step4 evaluator described above;
that separate feature was not backported into the main CLI in this work.

## Implementation references

| Area | Source |
| --- | --- |
| Compressed head kernel and native buffer contract | [rdna-head-tiled.hip.h](../vendor/rocm-exl3/exllamav3/exllamav3_ext/rocm/quant/rdna-head-tiled.hip.h), with native declarations/bindings and the existing GEMV dispatch |
| Shared head view, memory budget and lifecycle | [packed_head.py](../src/quantlab/methods/exl3/packed_head.py); capability selection in [optimizations.py](../src/quantlab/methods/exl3/optimizations.py) |
| Attention policy and metadata dispatch | [schedule.py](../vendor/rocm-exl3/exllamav3/modules/attention_fn/schedule.py) and [triton_paged.py](../vendor/rocm-exl3/exllamav3/modules/attention_fn/triton_paged.py) |
| Automatic CLI/server integration and telemetry | [evaluate_exl3_candidate.py](../scripts/evaluate_exl3_candidate.py), [serve_exl3.py](../scripts/serve_exl3.py), [cache_precision.py](../src/quantlab/methods/exl3/cache_precision.py) |
| GPU regression checks | [check_head_tiled.py](../kernels/exl3/check_head_tiled.py), [check_attention_schedule.py](../kernels/exl3/check_attention_schedule.py), through [check_exl3_kernels.py](../scripts/check_exl3_kernels.py) |
| New-geometry operator sweeps | [benchmark_exl3_attention.py](../scripts/benchmark_exl3_attention.py) |

## Reproducibility

Private evidence, frozen baseline harnesses, input/model tensor hashes,
source snapshots, raw samples and monitor records are under
`artifacts/head-attention-20260926/`. The pre-turn primary native SHA-256 is
`a00edbf45f5e70431510c3a28db70447e85a47574de26f4f86f7004c9f89fed9`.
The first candidate is
`43136eef403f5377fb7d686ebccf8edecba18ac0321e7f6ba89fa8da494b26c0`;
the second fresh 110-unit build is
`51cde437155205f9ee6f8d569655e192a2e199f00a448a35843982a4983ff4db`.
All 332 native/build source fingerprints matched the second build.
The third candidate is
`22505a76c467b1eafa3dcf4029d460f0420712efab4119c2b23420b76ada9abd`.
The fourth fresh 110-unit build took 662.65 seconds and produced
`47835c108bc68e6e7ee4b21853017f291fe6086534c206fadb8795506e20dab3`;
all 332 source fingerprints matched. It passed 2044 GPU checks: 112 head,
116 attention-schedule, and 1816 existing projection/prefill/paired-MLP
regression checks.

The fifth fresh 110-unit build took 682.72 seconds and produced
`519a2343da6aad14b8ec7c8cd6c76266a6fb54c7c5860a665a6eed3449957576`.
All 332 native/build fingerprints matched. It passed 2117 GPU checks: 185
head checks covering FP16/FP32, 116 attention-schedule checks and the same
1816 projection/prefill/paired-MLP regression checks. The first validation
attempt stopped on a stale expected call count after adding an eligible
FP16 call; correcting that test counter required no native change, and the
complete rerun passed. Windows and Linux each ran 567 unit tests successfully,
with 25 Windows dependency/platform skips and 10 Linux Windows-only skips.

Promotion changed only `runtime.extension_dir` and
`runtime.extension_sha256` in the ignored local installation record. Its
previous contents are retained in the private evidence directory. The
qualification checked the actual binary hash, all native source fingerprints,
the runtime Python source hashes, model dispatch counts, exact tokens and
MTP windows before changing that registration.

The inherited SharedSignalPool cleanup warning appeared in GPU processes
that otherwise completed and exited zero. It remains a recorded runtime
limitation; these tests do not establish prolonged-serving behavior.

OMP 18.1.16, using the configured `@default` model, implemented bounded
benchmark and native-kernel packages. The coordinator reviewed the changes,
corrected guards, integrated dispatch and owns final validation. Metadata,
actual process IDs, logs and zero exit statuses are retained in
`.codex/.omp-jobs/head-attention-bench-20260926/`,
`.codex/.omp-jobs/head-kernel-20260926/`,
`.codex/.omp-jobs/head-repacked-20260926/` and
`.codex/.omp-jobs/head-fp16-20260926/`.
