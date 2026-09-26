# AsterKV Stage 2: offline precision allocation

Stage 2 adds experimental `--cache-policy POLICY.json` support to the launcher,
server and evaluator. A profile assigns precision to every global-attention
layer in the target and, when enabled, draft model. Recurrent state is unchanged.
Inference loads the assignment once; no search runs during generation.

This layer allocation experiment draws on
[KVTuner](https://arxiv.org/abs/2502.04420) and the Stage 1 codec. It is not a
reproduction of KVTuner's full search or a claim of a new quantization algorithm.
Local attention error does not establish downstream quality or speed.

## Profile contract

Version 1 supports `q4`, `q5`, `q6`, `q8` and `aster5`. Uniform layers may have
different K/V widths. Aster layers require `aster5` on both sides; other layers
may use uniform formats. FP16 and tensor-parallel policies are outside this
version's tested scope.

```json
{
  "version": 1,
  "config_sha256": "SHA256_OF_EXACT_CONFIG_JSON_BYTES",
  "components": {
    "target": [
      {"layer_idx": 3, "kv_heads": 4, "head_dim": 256, "k": "aster5", "v": "aster5"},
      {"layer_idx": 7, "kv_heads": 4, "head_dim": 256, "k": "q8", "v": "q6"}
    ],
    "draft": [
      {"layer_idx": 0, "kv_heads": 4, "head_dim": 256, "k": "q8", "v": "q8"}
    ]
  }
}
```

This abbreviated schema example is not runnable. Supply the real lowercase
64-character hash and every actual attention layer. `layer_idx` means the model
layer index, not its ordinal among attention layers. MTP requires an explicit
draft section; target-only runs may ignore a supplied draft section.

The parser bounds input to 1 MiB and rejects duplicate keys, unknown fields,
invalid pairs, duplicate layers and nonfinite values. Configuration hash,
complete layer coverage and geometry are checked before weights load.
The hash binds configuration **not weights**: recalibrate for different weights,
even if their configuration files match.

```text
python run.py serve -m MODEL_DIRECTORY -c 120064 --cache-type aster5 --cache-policy POLICY.json --attention-profile long --prefill-gemm wmma -b 1024 --spec-type draft-mtp --spec-draft-n-max 6 --draft-confidence 0.6 --prefix-cache on --request-timeout 900
```

A quantized base type is required. The profile overrides it for every covered
layer. Existing native-attention and draft-graph restrictions remain. Loaded
events report assignments, a profile hash and measured per-layer tensor bytes.
Health reports the profile without its filename. Model-specific profiles and
raw research data belong in ignored local storage.
The longer request timeout accommodates fresh long-context prefill; it does
not improve throughput or set a server lifetime limit.

## Frozen research gate

Protocol `asterkv-stage2-gate-v1` compares Q8, Stage 1, Stage 2, and the same
Stage 2 allocation with uniform Q5 replacing remaining Aster layers. All use
the same weights and 120064 capacity.

| Variant | Target attention layers | Draft attention |
| --- | --- | --- |
| Q8 | All Q8/Q8 | Q8/Q8 |
| Stage 1 | All Aster5/Aster5 | Aster5/Aster5 |
| Stage 2 | Five Aster5/Aster5, eleven Q6/Q6 | Q8/Q8 |
| Uniform allocation control | Five Q5/Q5, eleven Q6/Q6 | Q8/Q8 |

Stage 2 versus Stage 1 changes both target allocation and draft precision.
The uniform allocation control matches Stage 2's draft and layer assignment,
isolating the codec choice in the five lowest-precision target layers.

- Calibration: two public-text corpora, each 8192 input tokens; 32 actual
  post-RoPE queries per global-attention layer. Measure attention-output error
  against FP32 attention using unquantized K/V from a Q8 trajectory. Validate
  the chosen assignment on two separate corpora. This local proxy does not
  measure downstream error accumulation.
- Short quality: 24 Python program traces, 24 arithmetic problems, 24
  multi-turn state/instruction tasks. Thinking enabled, greedy sampling,
  1024-token output limit. Tracing tests code comprehension, not synthesis;
  generated code is never executed.
- Long quality: four independently seeded documents per approximate occupied
  band (4K, 16K, 32K, 64K, 110K), each with four direct recalls, four linked
  records and four revision-aware values spread through distractors. Thinking
  disabled, 512-token output limit. One exact repeat per band tests reuse.
- Runtime: long attention, WMMA, chunk 1024, MTP depth 6/confidence 0.6, prefix
  reuse enabled, 900-second request timeout. Record actual dispatch, target/draft bytes, computed and cached
  prefill, decode, first-token time and peak memory. Report end-to-end HTTP
  latency separately from generator timings; preparation/transport overhead
  is not silently attributed to GPU prefill.
- Speed: four fresh requests per band, excluding the first from warmed medians;
  reuse reported separately. Retrieval answers are real workloads, not fixed
  output-length microbenchmarks; output lengths and MTP acceptance affect rates.
- Execution order is Q8, Stage 2, Stage 1, then the uniform allocation control,
  with one loaded process per variant. This saves repeated loading, but leaves
  thermal/background-load order effects as a limitation. Each variant receives
  the same ordered prompts, warmup and repeat count.
- Scoring: exact semantic answers and JSON compliance reported separately.
  Repeats do not count as independent quality observations. Compare paired
  scores by family/band and bootstrap by document, not by treating correlated
  answers as independent. Retain truncations and failed answers.

Joint provisional targets: at least 25% less attention storage, at least 95%
of Q8 decode throughput, and at most two percentage points of quality loss per
family/band. Four documents per band cannot establish tight population-level
equivalence. A perfect observed tie still requires that limitation; bootstrap
intervals alone must not convert a small sample into a quality guarantee.

Frozen prompt fixture SHA-256:
`53fd35ac99e7328f0df3b199589bff1905d0d9b58ab419129b92dfa0edfc7e4f`.

The portable fixture builder creates the same synthetic documents, questions
and expected answers with the same tokenizer. It does not launch inference:

```text
python scripts/build_kv_research_cases.py --tokenizer MODEL_DIRECTORY/tokenizer.json --output artifacts/kv-research/cases.json
```

An independent fixture audit checked all 240 retrieval answer keys by parsing
the actual archive records and following aliases/revisions. It also checked
all 72 short-task keys: arithmetic by conservation/remainder identities,
register state from the user updates, and program traces by executing only
the reviewed, hash-bound fixture programs. No generated model response was
executed. All answer keys passed; this validates the test data, not the model.

Scoring was revised to `stage2-score-v2` during the Q8 control, before any
compressed candidate ran. Q8 returned a correct arithmetic answer using named
JSON fields instead of the requested array. The revised
[`score_cache_research.py`](../../kernels/exl3/score_cache_research.py) accepts
explicitly equivalent named arithmetic fields, alpha/beta/gamma mappings and
single-field code output/result wrappers for semantic scoring. It still checks
the requested JSON shape separately. Duplicate/malformed answers and booleans
masquerading as numbers are rejected. Original v1 scores and raw responses
remain saved; every variant receives the same v2 re-score, and score changes
are enumerated. No model or quantizer was changed for this grader correction.

Q8 also exhausted the 1024-token output limit on a code task. Before running
compressed candidates, a supplemental pass was therefore declared: all 24 code
tasks at a 2048-token output limit for every variant, with otherwise identical
CLI settings and prompts. Report both budgets; do not silently replace the
original scores or rerun only favorable candidate failures.

A later review of the completed Q8 baseline also found a truncated arithmetic
answer. While the first candidate was still in its long-context pass, before
any candidate arithmetic results, a separate supplemental pass was declared:
all 24 arithmetic tasks at 2048 tokens for all four variants. It supplements
the frozen primary gate; both output budgets remain reported.

## First calibrated allocation

The discrete search minimizes summed per-layer attention-output normalized MSE
over the two calibration corpora, under a measured-format byte budget. Candidate
pairs are Aster5/Aster5, Q6/Q6, Q8/Q6, Q6/Q8 and Q8/Q8. The draft is fixed to Q8
as a conservative control; its sensitivity was not calibrated. No held-out task
answer was used to choose the allocation.

The portable offline allocator,
[`allocate_cache_precision.py`](../../kernels/exl3/allocate_cache_precision.py),
accepts a JSON object containing `budget_bytes` and `layers`. Each layer has
`layer_idx` and `options`; each option has `k`, `v`, integer `bytes` and finite
nonnegative calibration `loss`. It minimizes the summed loss within the exact
byte budget, including caller-supplied scale/codebook costs. Reserve fixed
components such as the draft cache before allocating the target budget.

```text
python kernels/exl3/allocate_cache_precision.py calibration-options.json --output allocation.json
```

The helper uses a bounded Pareto search and fails explicitly if the retained
frontier exceeds 4096 states. It does not load a model or produce a runtime
profile automatically: bind the resulting assignment to the checked model
configuration and geometry using the profile contract above. Its 12 focused
CPU tests include exhaustive comparison on 20 small generated problems. An
independent run with exact tensor byte costs reproduced the frozen allocation
below without changing the profile under evaluation.

For this model's sixteen target global-attention layers, the selected assignment
keeps model indices 3, 23, 27, 59 and 63 at Aster5, and promotes the other eleven
to Q6/Q6. The live CLI confirms **3,304,161,920 bytes (3.077 GiB)** of target-plus-
draft attention storage, including five 128-byte codebooks. Q8 measures
**4,441,407,488 bytes (4.136 GiB)**: the measured saving is **25.61%**. This is
attention storage, not the percentage reduction in total GPU use.

| Variant | Calibration mean output NMSE | Held-out mean output NMSE |
| --- | ---: | ---: |
| Q8 | 0.00001869 | 0.00001972 |
| Stage 1 | 0.00097033 | 0.00102372 |
| Stage 2 | 0.00043361 | 0.00045426 |
| Same allocation, uniform Q5 bulk | 0.00048441 | 0.00049982 |

The Stage 2 held-out local proxy is 55.6% below Stage 1, and 9.1% below the
uniform allocation control. These are attention reconstruction measurements,
not task accuracy or proof of Q8 equivalence. The calibration contains software
text, only two documents, and queries from the final prefill chunk; broader
domain/decode calibration remains a limitation.

## Interpreting the allocation

The search optimizes a local reconstruction proxy and a storage budget. It
does not price the latency of each format. In the tested serving path, the
`long` decode schedule has separate guards for Q8/Q8 and Aster5. The selected
Q6/Q6 layers retain the inherited schedule. An assignment can therefore have
lower reconstruction error while running more slowly than Stage 1. Identical
CLI flags do not imply identical kernel schedules for different formats.

This is a concrete dispatch difference, not a complete GPU attribution of
the measured slowdown. Aggregate draft acceptance, actual dispatch, output
length and end-to-end timings must also accompany the comparison. Tuning Q6
or restricting a future search to measured fast formats would require a new
frozen evaluation; neither changes the candidate in this gate.

Full model results and the stopping decision will follow the measurements.

## Implementation checks before the task matrix

On gfx1101, the GPU gate passed 21 uniform-cache and 25 Aster/policy checks.
The three added policy checks construct one cache containing Aster5, Q8/Q6 and
Q6/Q8 layers, then execute attention through actual layer lookup against
independently reconstructed FP32 attention. Relative L2 errors were 0.000816,
0.001063 and 0.000614. Coverage also includes existing packing, partial pages,
changed-input graph replay and attention dispatch tests.

The model-free checker exited successfully but emitted a `SharedSignalPool`
teardown warning (269 signals). The instrumented calibration emitted the same
kind of warning (731 signals); these are retained rather than described as
clean shutdowns. Ordinary HTTP server shutdown is checked separately.

Before freezing the runtime source, the CPU suite ran 489 tests, with 464
passes and 25 dependency/platform skips. Hardware results apply to this tested
GPU/configuration; skipped CPU tests are not GPU evidence.
