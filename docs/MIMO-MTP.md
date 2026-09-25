# MiMo MTP precision and verification

This experimental path targets the MiMo 5.0/H6 mul1 package with the separately
packaged, one-layer Qwen3.5-9B MTP donor. Model files are unchanged. It does not
establish compatibility with other MTP architectures or improve the target's
quantization quality. Vision/mmproj work is separate.

## BF16 means weights and draft projections

`--mtp-dtype bf16` reloads the eight original dense draft matrices directly from
the BF16 checkpoint and performs their matrix multiplication with BF16 operands
and BF16 results. Seven normalization weights already retain BF16. The normal
loader's FP16 copies are not converted back to reconstruct the original values.

| Component | Arithmetic or storage with this option |
| --- | --- |
| Eight dense donor matrices | Original BF16 weights; BF16 matrix operands/result |
| Seven donor normalization weights | Existing BF16 weights |
| Projection interfaces | Convert results to the caller's FP16 or FP32 interface |
| Draft attention, KV and other operators | Existing mixed precision paths |
| Shared target embedding and EXL3 output head | Existing target paths |

The adapter accepts the supported one-layer, eight-matrix/seven-norm inventory,
unsliced BF16 sources and at most 512 MiB of projection payload. It rejects packed
or structurally different drafts. Native projection fusion handles are disabled
before replacement. No source tensor or packed model file is rewritten.

The default remains `fp16`. BF16 requires MTP, `--decode-fusions off`, no native
attention, no draft-step graph and no MTP projection cache. These restrictions
are checked before model loading in the public launcher and both consumers.

## Matching ordinary greedy verification

`--verify-attention rowwise` submits each short cached target attention query
separately, with its matching causal prefix. This preserves the ordinary
one-token attention shape while allowing the packed projections to remain
batched. It affects cached windows of 2–9 rows; longer prefill uses its existing
path. Only this target model's attention instances receive the override.

The option requires F16 KV, disabled decode fusions and no native attention.
Noncausal spans and misaligned Q/K/V are rejected before cache mutation. This
is a measured correction for the MiMo selection fixture, not a guarantee of
bitwise equivalence for arbitrary models, kernels or prompts.

On RX 7800 XT, the original failure first appeared in full attention at target
layer 7, position 74. The post-RoPE Q/K/V and active cache prefix matched exactly;
two of 4096 attention outputs differed by at most 0.0001220703125. Those small
differences propagated to a later greedy tie. The rowwise policy restored all
662 selection IDs/stops at FP16 depth 1 and BF16 depth 2.

## High-bit packed projection extension

The candidate contains 200 K5 body matrices and one K6 head. The earlier shared
small-M kernel admitted K2/K3/K4, so it could not reuse the decoded weights for
these matrices. The new optional native ABI extends the existing mul1 dot core:

| Property | Bounded extension |
| --- | --- |
| Codebook | mul1 (internal codebook 2) |
| Bits | K5 and K6 |
| Query rows | 2, 3 or 5; other widths retain the ordinary fallback |
| Output | FP16 or FP32 |
| Warp choices | Existing 1/4/8/16 policy |
| Dispatch | Requires `quantlab_exl3_smallm_highbit_abi() == 1` and dot mode |
| New WMMA/internal graph support | None claimed |

An older verified extension keeps the previous fallback. The capability must
come from the loaded binary; a Python flag alone cannot add kernel support.
The implementation reuses the inherited EXL3 decoder, Hadamard transforms and
dot accumulation; it introduces no new quantized representation.

On RX 7800 XT with this target and F16 context 4096, shared BF16 depth 2
passes all 662 selection IDs/stops, 790 unused-family confirmation IDs/stops
and 795 thinking-mode confirmation IDs/stops. Its confirmed warmed request
speed ratio is 1.5043×; peak adapter allocation is 7.263 GiB versus 6.320 GiB
for MTP off. Depth 4 tied on selection speed, so depth 2 was locked before
confirmation. These are measured short-context, greedy CLI-evaluator results,
not HTTP or broad model-quality qualification. [Research report and evidence](https://github.com/RishiJain905/QuantizationResearch/blob/codex/mimo-mtpopt-20260925/reports/QEXP-002/QEXP-002-20260925T004303Z-agent-mimo-mtpopt-b0627d23/execution-report.md). Do not infer a speedup from draft acceptance, kernel availability or
the BF16 precision choice alone.

## Controls

The supported precision/verification combination is:

```text
--mtp 2 --mtp-dtype bf16 --verify-attention rowwise --decode-fusions off --cache-type f16
```

These are public `run.py` arguments. Its launch worksheet must select the
qualified extension and set `runtime.native_smallm_max_rows = 5`; the public
launcher enables native small-M dispatch automatically. The direct evaluator
and server instead use `--mtp --draft-tokens 2` and
`--native-smallm --native-smallm-max-rows 5`. Use dot mode, an explicitly selected
local model and the verified launch worksheet.
No global installation or default model was changed by the experiment.
