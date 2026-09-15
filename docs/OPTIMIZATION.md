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

Begin with defaults. These specialized controls are not uniformly faster.

| Flag | Effect / restriction |
| --- | --- |
| `--decode-fusions off\|gdn\|gdn-mlp` | Decode fusions; default gdn |
| `--draft-confidence` | Adaptive drafting; requires MTP, conflicts with GPU drafting/shortlists |
| `--gpu-embedding` | GPU embedding table, with additional VRAM use |
| `--gpu-draft`, `--gpu-draft-metadata` | GPU bookkeeping; require GPU embedding/drafting respectively |
| `--batch-greedy` | Batched independent pure-greedy target sampling |
| `--smallm-kernel dot\|wmma\|wmma-register` | Projection implementation; optional variants need matching ABI |
| `--prefill-gemm blas\|wmma` | FP32-output prefill projections; default BLAS. WMMA requires the matching extension and uses BLAS for unsupported shapes/devices |
| `--warps`, `--head-warps` | Split-K scheduling overrides |
| `--cache-mtp off\|fc\|attention\|mlp\|all` | Reconstructed draft projections with extra cache bound/checks |
| `--shortlist-groups`, `--shortlist-mode` | Compact draft vocabulary; target still verifies full vocabulary |
| `--native-attention` | Experimental native attention with build/geometry guards |
| `--draft-step-graph` | Whole draft graph; requires GPU metadata and fixed drafting |
| `--attention-profile long` | Guarded Q8 long-context scheduling |

Use new output directories and retain failures. Optional native paths require their matching extension; raising a declared row/ABI limit cannot upgrade an older binary. [Long-context scope](LONG-CONTEXT.md).

[GPU-PERFORMANCE.md](GPU-PERFORMANCE.md) records the controlled native-prefill
comparison and runtime flag sweep. The results are specific to the tested
model, GPU, occupied prompts and environment; use them as starting points for
measurement rather than universal defaults.

[RUNTIME-PERFORMANCE.md](RUNTIME-PERFORMANCE.md) adds measurements through the
full 120064-context CLI with prefix reuse enabled, revised WMMA scheduling,
decode flag comparisons and exact-output validation limits.
