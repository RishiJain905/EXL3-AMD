# Measure and tune

Compare identical model bytes, prompt tokens, cache precision, output budget, sampler and timing protocol. Occupied context, MTP acceptance, memory bandwidth and host dispatch affect throughput; model size alone does not imply a token rate.

```powershell
python run.py speed -m "MODEL_DIRECTORY" --cache-type q8 -c 4096
python run.py speed -m "MODEL_DIRECTORY" --cache-type q8 --spec-type draft-mtp --spec-draft-n-max 4 -c 4096
python run.py speed -m "MODEL_DIRECTORY" --cache-type q8 --spec-type draft-mtp --spec-draft-n-max 6 --draft-confidence 0.6 -c 4096
```

Speed mode uses fixed short prompts, warmup and three 128-token continuations. Inspect per-task output agreement and timings. A coding gain may regress SQL. `quality` runs a small smoke suite; `context-speed` measures occupied context. Neither establishes BF16 fidelity or broad security quality.

Timing version 2 groups events by GPU iteration and excludes the whole first emitting iteration. Older first-event MTP rates could inflate the numerator and are not republished as release performance promises. HTTP latency additionally includes queueing, prefill and delivery.

## Optional controls

Begin with defaults. Compatible builds automatically use packed kernels for
eligible 10–64-row projections in generation and serving. These specialized
controls are not uniformly faster.

| Flag | Effect / restriction |
| --- | --- |
| `--decode-fusions off\|gdn\|gdn-mlp` | Decode fusions; default gdn |
| `--draft-confidence` | Adaptive drafting; requires MTP, conflicts with GPU drafting/shortlists |
| `--gpu-embedding` | GPU embedding table, with additional VRAM use |
| `--gpu-draft`, `--gpu-draft-metadata` | GPU bookkeeping; require GPU embedding/drafting respectively |
| `--batch-greedy` | Batched independent pure-greedy target sampling |
| `--smallm-kernel dot\|wmma\|wmma-register` | Projection implementation; optional variants need matching ABI |
| `--no-packed-mid` | Use reconstruction for 10–64-row projections instead of the automatic packed path; diagnostic control |
| `--smallm-mlp-warps 4\|8\|16` | Experimental scheduling override; leave unset for the measured default |
| `--prefill-gemm auto\|blas\|wmma` | Default auto selects verified FP32-output WMMA; unsupported shapes/devices and legacy binaries retain BLAS |
| `--no-packed-prefill` | Disable automatic tiled packed prefill for diagnosis; primary policy admits FP16 rows 65–128 and FP32 rows 65–512, output width at most 32768 |
| `--no-mlp-pair` | Disable automatic paired gate/up and fused output-Hadamard/SwiGLU work for diagnosis |
| `--no-narrow-gemm` | Use BLAS instead of the automatic narrow dense GEMM for 1–8-row FP16 projections up to 128 outputs and 1–4-row BF16 MTP projections; diagnostic control |
| `--warps`, `--head-warps` | Split-K scheduling overrides |
| `--cache-mtp off\|fc\|attention\|mlp\|all` | Reconstructed draft projections with extra cache bound/checks |
| `--shortlist-groups`, `--shortlist-mode` | Compact draft vocabulary; target still verifies full vocabulary |
| `--native-attention` | Experimental native attention with build/geometry guards |
| `--draft-step-graph` | Whole draft graph; requires GPU metadata and fixed drafting |
| `--attention-profile auto\|default\|long` | Automatic measured scheduling by default; inherited and legacy long controls |

Use new output directories and retain failures. Optional native paths require their matching extension; raising a declared row/ABI limit cannot upgrade an older binary. [Long-context scope](LONG-CONTEXT.md).

The [MiMo prefill/MLP follow-up](PREFILL-MLP-FUSION.md) records the primary
shape selection, full-model comparisons, rejected wide policy and small
decode gains. No enabling flags are required for its validated paths.

The [head and attention follow-up](HEAD-ATTENTION-PERFORMANCE.md) records the
next two items. Attention scheduling is primary automatically for supported
cache formats and occupied-context bounds; it does not change precision.
The verified ABI-3 compressed head is also automatic for supported rows,
formats and memory budgets. `--head-warps 1` retains the inherited wide-head
path for a diagnostic comparison. Head timings must include both the target
verification and repeated draft-head calls; an isolated projection gain is
not an end-to-end MTP gain.

[GPU-PERFORMANCE.md](GPU-PERFORMANCE.md) records the controlled native-prefill
comparison and runtime flag sweep. The results are specific to the tested
model, GPU, occupied prompts and environment; use them as starting points for
measurement rather than universal defaults.

[RUNTIME-PERFORMANCE.md](RUNTIME-PERFORMANCE.md) adds measurements through the
full 120064-context CLI with prefix reuse enabled, revised WMMA scheduling,
decode flag comparisons and exact-output validation limits.

[The inference optimization roadmap](INFERENCE-OPTIMIZATION-ROADMAP.md) examines
remaining packed-kernel, prefill, fusion, and persistent-kernel opportunities
on gfx1101, with separate measurement requirements for the 9B and 27B models.

The scoped MiMo baseline, packed MLP scheduling and 10–64-row implementation
are recorded in [PACKED-MLP-PERFORMANCE.md](PACKED-MLP-PERFORMANCE.md).
