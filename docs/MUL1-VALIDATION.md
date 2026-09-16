# mul1 model validation

Date: 2026-09-16. Hardware: RX 7800 XT, gfx1101, Windows launcher with Ubuntu
WSL inference. The environment and native build are recorded in
[VALIDATION.md](VALIDATION.md). No RDNA4 device was available.

## Conversion and comparison scope

The candidate was converted directly from the owner's local dense Qwen-family
27B BF16 checkpoint. Text and integrated MTP weights are included; vision is
excluded. The original BF16 and EXL3 directories were preserved.

| Setting | Existing EXL3 | New candidate |
| --- | --- | --- |
| Codebook | Default / 3inst (cb0) | mul1 (cb2) |
| Main packed projections | 2 bits | 2 bits |
| Output head | 4 bits | 3 bits |
| MTP | 4 bits | 4 bits |
| Calibration | 4 rows x 512 tokens | Same token file |

The output-head reduction is necessary for this candidate's strict
9,500,000,000-byte complete-directory target. This comparison changes both
codebook and head precision. It does not isolate mul1's effect on model quality,
and the small calibration set does not establish fidelity to BF16.

Conversion retained resource guards throughout. Large output-head error metrics
now use bounded row chunks with the original relative Frobenius, cosine-error
and mean per-row SQNR formulas. CPU tests compare these metrics with the prior
full-tensor calculation. Fatal Python diagnostics remain enabled; periodic
thread dumps were removed after a crash in CPython's thread-dump routine.
The output head completed with the bounded metrics, and its packed Safetensors
file matched the saved pre-fix quantized head byte for byte (SHA-256 verified).

## Protocol

Both models use the same verified multi-target native extension, Q8 K/V,
WMMA prefill, register-B WMMA small-M, 1024-token prefill chunks, greedy sampling,
and integrated MTP depth 6 with confidence 0.6. The target-only control disables
MTP. The native-MLP-graph control selects `--decode-fusions gdn-mlp`.

At 4096 context capacity, two fresh speed processes per model each discard a
32-token initial warmup and a 128-token warmup for each exact measured prompt.
Each process then repeats the SQL and binary-search prompts three times with
128 output tokens. Decode timing excludes the first emitting GPU iteration.
A separate run uses the same mul1 candidate with the older verified binary to
compare its fallback with the new fused implementation.

Quality uses the six public tasks and rubrics in `configs/evaluation.json`,
with at most 384 output tokens per task. This is a small smoke suite, not a
general capability benchmark. Code answers are reviewed before any execution.

Long-context tests use capacity 120064 with exactly 119808 occupied input tokens,
Q8 K/V and the gfx1101 long-attention profile. Retrieval inserts three exact
values near 10%, 50% and 90% of the synthetic record stream. A separate occupied
coding run measures a fixed 128-token continuation. Allocation and throughput
alone do not establish useful retrieval quality.

## Results

The complete model directory is 9,401,715,980 bytes (9.402 decimal GB), including integrated MTP,
tokenizer and metadata. All 409 packed projections contain verified mul1
markers: 400 at 2 bits, one at 3 bits and eight at 4 bits. No malformed,
conflicting or orphan codebook markers were found.

### Warm decode throughput

Rates are tokens/second, shown as median (minimum–maximum). The baseline and
new fused candidate each have six measured continuations per prompt across two
fresh processes. The fallback control has three continuations per prompt.

| Model / implementation | SQL | Binary search |
| --- | ---: | ---: |
| Existing 3inst, new binary | 29.26 (28.82–29.34) | 51.88 (51.83–51.93) |
| mul1, new fused binary | 31.90 (31.77–31.96) | 58.15 (58.03–58.27) |
| Same mul1 weights, old fallback binary | 12.37 (12.36–12.37) | 17.72 (17.72–17.74) |

The same-weight fused/fallback comparison is approximately 2.58x / 3.28x for
these two prompts. Every compared input and output token matched, including
warmups (nine cases, 1,056 output tokens per process). The old binary admitted
only single-row decode fusion; the new binary admitted rows 1–9. Both models
also reproduced all output tokens between their two fresh speed processes.

The candidate's 9% / 12% gain over the existing 3inst model also reflects
different quantized weights, output-head precision, generated continuations
and MTP acceptance. It is not an isolated codebook throughput measurement or
a general speedup claim.

Fixed-length speed runs deliberately continue for 128 tokens even past a
normal stop token. Their continuations can include text after the answer;
they measure throughput rather than complete-answer quality.

### Quality smoke results

The candidate is **not quality-equivalent** on this small suite. Reviewed code
answers were checked with the same seeded, bounded examples, including empty,
duplicate, boundary and input-mutation cases.

| Task | Existing EXL3 | mul1 candidate |
| --- | --- | --- |
| Parameterized SQL | Correct placeholder and parameter tuple | Correct placeholder and parameter tuple |
| JWT verification | Core checks covered; algorithm pinning not explicit and unnecessary `jti` uniqueness requirement | Core checks partly covered; trusted-key/algorithm policy not explicit and unstated static-secret assumption |
| Leaked credential | Revoke first, rotate/audit/cleanup; recurrence prevention omitted | Revoke first, audit/cleanup; audit follows cleanup and recurrence prevention omitted |
| Stable unique values | 104/104 examples | 104/104 examples |
| First matching index | 4,680/4,680 examples | 3,984/4,680 examples; wrong duplicate handling |
| Merge intervals | 105/105 examples | 105/105 examples |

For `first_index([1, 1, 1], 1)`, the candidate returns 1 instead of 0. This
same answer occurs with MTP disabled and in the old-binary speed control.
The tested new fused path reproduces the old binary's output, so this evidence
does not identify a new fusion defect. It also does not isolate whether
codebook choice, head precision or calibration causes the model-quality loss.
Both models put requested code-only answers inside Markdown fences.

All six mul1 answers matched exactly when enabling native MLP graphs. With
MTP disabled, five of six matched exactly; the leaked-key answer differed only
in bold formatting after 235 common output tokens. Exact MTP/target-only token
equivalence is therefore not claimed for every prompt.

### Occupied context

The candidate recovered all three values at 32K capacity (32,512 occupied
input tokens). Both models also recovered all three exact values in the same
order at 120K capacity, with 119,808 actual input tokens and identical 30-token
answers. Baseline/candidate prefill rates were 270.29 / 271.31 tokens/second.
This establishes this synthetic retrieval case on gfx1101, not general
comprehension or accuracy across every 120K prompt.

The separate fixed 128-token coding runs used the same 119,808-token occupancy:

| Measurement | Existing EXL3 | mul1 candidate |
| --- | ---: | ---: |
| Coding prefill, tokens/s | 270.92 | 271.23 |
| Coding decode, tokens/s | 25.51 | 31.56 |
| Peak Torch reserved memory across 120K runs, GiB | 12.73 | 12.58 |
| Peak Windows dedicated adapter memory across 120K runs, GiB | 14.25 | 14.10 |

The adapter measurement includes the display and other processes. Each 120K
coding measurement is one continuation, not a distribution of repeated timings.
The baseline answer is truncated at 128 tokens; the candidate reaches its stop
token earlier and continues because this is the fixed-length speed protocol.
The roughly 24% decode difference is specific to these continuations and their
MTP acceptance; it does not establish equivalent coding quality.

All 14 fresh benchmark processes completed with successful runtime/monitor
exits and no timeout or resource stop. The final model audit stayed below
9,500,000,000 bytes after adding a local validation summary. Its weight-file
hashes were rechecked before delivery; the original model directories were
preserved.

## Remaining limits

RDNA4 gfx1200/gfx1201 builds and primitive probes cross-compile successfully.
The shared adapter is exercised on gfx1101, but RDNA4 WMMA instructions have
not executed on physical RDNA4 hardware. The model-free validator and then real
model inference must pass there before claiming RDNA4 numerical compatibility.

This validation does not cover clean installation, all model families, MoE,
FP16/Q4 caches, sustained serving, or broad BF16 fidelity. The pre-existing
ROCm `SharedSignalPool` teardown warning also occurs with the baseline.
