# Cold start: autotune cache, warm-up and NVMe runtime

September 27, 2026. Implements ideas 1–3 of [OPTIMIZATION-PLAN.md](OPTIMIZATION-PLAN.md).
Baseline evidence is the MiMo 9B 131K serving run described there: ready
after 76.3 s, then a first 8K prefill at 315 t/s against 1848 t/s warm, and
Triton compiles stalling later requests as occupied context grew.

All three changes are primary. Nothing needs to be enabled.

## 1. Persistent Triton autotune results

`runtime_environment()` in `scripts/launch_runtime.py` sets
`TRITON_CACHE_AUTOTUNING=1` for every backend process. Triton 3.7.1 then
stores each `@triton.autotune` winner (FLA chunked GDN prefill has about 70
such sites) next to the compiled kernels in its cache directory. Later
processes load the winner instead of benchmarking every configuration.
A new autotune key (for example a new shape) still tunes once and is then
cached. Kernel numerics are unchanged: the same configuration space and
selection rule apply; only repeated benchmarking is skipped.

Regression coverage: `tests/test_launcher.py`
`test_persists_triton_autotune_results`.

## 2. Startup warm-up

`scripts/serve_exl3.py` runs `Engine._warmup()` before `ready=1`, backed by
`src/quantlab/methods/exl3/warmup.py`. It is on by default. `--warmup off`
(serve only) is a diagnostic override. Usage is in
[SERVING.md](SERVING.md#startup-warm-up).

### What Triton specializes

Triton 3.7.1 compiles one binary per combination of constexpr arguments,
`num_warps`/`num_stages`, and, for plain integer and pointer arguments,
whether the value is divisible by 16. On AMD, tensors also specialize on
storage under 2 GiB. There is no `== 1` specialization in this version. For
the Q6 paged attention in `triton_paged.py` this means:

- **Decode kernels (≤ 16 query rows):** `q_len` and `kv_append_len` are
  constexpr, so each row count 1..16 compiles separately. MTP verification,
  prefill tails and prefix-hit remainders of up to 16 rows use them.
  - The gfx1101 decode schedule (`block_n` 16, 64 splits, parallel combine,
    `head_block` 4 for 5–8 rows) applies at a block-table bound of ≥ 64 pages
    (16384 tokens).
  - The parallel combine kernel takes `num_splits` as a constexpr.
  - `split_len`, `num_pages_per_seq` and `num_splits` are runtime integers
    with %16 specialization.
- **Prefill kernels:** `IS_SPLIT` depends on a key/value bound of ≥ 8192 plus
  a cost model. `num_stages` 1 or 2 comes from an in-process timing pick.
  Rows, append length and page count specialize on %16.
- **Table widths:** target decode tables are padded to 16 pages. Draft (MTP)
  and prefill tables are not, so every width class occurs.

For MiMo 9B Q6 at 131K this gives 100 attention variants: decode split 64,
decode combine 16, parallel combine 8, prefill 10, prefill combine 2. With
FLA, conv1d and the rest, 115 Triton specializations. In the baseline run,
13 of these compiled mid-serving.

### How warm-up covers it

1. **Real generation** on a throwaway generator:
   - one full prefill chunk followed by 16 decode tokens (MTP rounds, serve
     sampler);
   - prompts of 47, 48, 95, 96, 191, 192 and 257 rows, one of them greedy.

   These cover the conv1d Triton row buckets (64/128/256 and the split
   kernels above 256, in both %16 classes), FLA's chunked rule and its
   autotuning, the WMMA/packed GEMMs, the packed head, the BF16 MTP draft
   and the real prefill `num_stages` pick. The generator is then cleared and
   released, so the first request's prefix lookup cannot find warm-up pages
   or recurrent checkpoints.
2. **`warm_attention()`** calls the real `attn_dispatch` once per distinct
   attention layer signature (target and draft).
   - Metadata is synthetic: `cache_seqlens = 0`.
   - Block-table widths: {1..16} ∪ {16k−1, 16k, 16k+1} ∪ {max}, which is
     about 110 widths for 512 pages.
   - Query rows: {1..16, 17, 32k, 32k+1 ≤ chunk}.
   - Launches of the module's `*_kernel` functions go through Triton's
     compile-only path (`run(warmup=True)` plus module load). They compile
     or load the binary but produce no attention output.
   - Kernel choice comes from the real dispatch code; warm-up copies no
     thresholds.
   - Stage picks first made on compile-only launches are discarded, and the
     attention-profile counters are restored, so `applied_calls` stays
     request-only.

A Triton post-compile hook counts specializations. Any first used after ready
is logged as `stage=jit after_ready=1` and counted in `jit_after_ready`.

### Measurements

MiMo K5/H6 mul1, BF16 MTP 2, Q6/Q6, 131072 context, chunk 1024, prefix
cache on, RX 7800 XT.

| Launch | Warm-up | Of which generation / attention |
| --- | ---: | ---: |
| First with new code, Triton cache missing ~100 variants | 107.3 s | 28.4 / 78.9 s |
| Second, timing pick chose the other `num_stages` (7 prefill variants compiled) | 53.0 s | — |
| Fully cached | 4.58 s | 2.95 / 1.63 s |

Once cached, the attention enumeration costs about 0.06 ms per call plus
0.7–1.2 s to load 101 kernels.

Requests after ready: a fresh 8.5K prompt, fresh ~900-token prompts with
sampling and with greedy, then prefix hits growing to 17K, 25.6K, 34K and
42.7K occupied. 64 output tokens each. Cached Triton disk cache in both
columns:

| Measure | Warm-up on | Warm-up off |
| --- | ---: | ---: |
| Request 1 prefill (fresh 8.5K) | 1734 t/s | 1588 t/s |
| Request 1 first token | 5.02 s | 5.50 s |
| Request 1 decode | 75.6 t/s | 69.2 t/s |
| Requests 2–7 | identical | identical |
| `jit_after_ready` | 0 | 33 |

- **The 33 late specializations with warm-up off:**
  - request 1: FLA ×6, conv1d ×4, decode ×7, prefill ×5;
  - request 2: prefill ×2 and conv1d;
  - request 4, crossing 25K occupied: decode split ×4 and parallel
    combine ×3;
  - request 5: conv1d.

  Here they were disk-cache loads; with a cold or partial cache each is a
  3–8 s compile, which is the stall pattern of the baseline run.
- **First launch:** with warm-up on it served up to 72K occupied context
  (fresh 14K prefill at 1839 t/s) with `jit_after_ready = 0` and no new
  Triton cache files after ready. Compare the baseline's first 8K request
  at 315 t/s.

**Greedy output:** two warm-up-on processes agree on every greedy request,
and so do two warm-up-off processes. Between on and off, the first request
of an off process can diverge in a near-tie. An off process that first
serves a short request matches the on outputs exactly. The likely cause is
the vendored prefill `num_stages` timing pick: it runs once per process, and
1 and 2 stages differ by up to 6.1e-5 absolute on a Q6 1024-row prefill.
With warm-up on, the first request matches steady-state behavior.

Runs: `RUN-20260928T023039Z-c49d998c`, `RUN-20260928T031519Z-436a6111`,
`RUN-20260928T032454Z-f147cee8`, `RUN-20260928T033952Z-79f63623` (ignored
`artifacts/`). Unit coverage: `tests/test_warmup.py` (widths, rows,
segments, compile-only routing, JIT monitor) and the `--warmup`
forwarding/rejection tests in `tests/test_launcher.py`.

### Limitations and follow-ups

- Not warmed beyond the real generation: vision (`--mmproj on`),
  native-attention/BC graphs and other experimental paths. Quantized layers
  outside quant-direct dispatch (compand, `EXL3_QC_STAGING=2`) skip the
  attention enumeration. `jit_after_ready` reveals any such gap.
- The first launch after a new extension, Triton, context or cache format
  pays all compiles during warm-up. A launch whose timing pick flips
  `num_stages` compiles about 7 prefill variants once.
- `/health` optimization counters (`packed_head`, `mlp_pair`) include
  warm-up calls.
- Warm-up calls scale with table widths × chunk/32. Chunk 8192 needs about
  4× the calls; that is still cheap once cached.
- `evaluate_exl3_candidate.py` is unchanged. Its suite already has warm-up
  rows, and its modes deliberately report first-iteration timing.
- Follow-up: marking the runtime integers `do_not_specialize` (as upstream
  `mla_triton` does) would cut the variant count; it needs a speed check.
- Follow-up: persisting or fixing the prefill `num_stages` pick would make
  greedy output independent of per-process timing.

## 3. Runtime on NVMe

The inference runtime previously ran in the WSL distro `Ubuntu`. Its 219 GB
virtual disk sits on a USB HDD with about 35 ms read latency, so startup's
many small-file reads (Python packages, ROCm libraries, the Triton cache)
were seek-bound. The guest page cache is dropped whenever the WSL VM idles,
so most launches were cold.

The whole distro does not fit on the NVMe drive. A slim copy with identical
paths was made instead: system files, `/opt/rocm-7.2.0`, `~/.local` including
the quantlab runtime, `~/.triton` and `~/.cache`. It was imported as the WSL
distro `EXL3-Runtime` on F: (45 GB `ext4.vhdx`), with `systemd=false`. Only
`distribution` changed in the installation record. Model weights stay on the
HDD and are read through `/mnt/d` as before. The procedure is in
[DEPENDENCIES.md](DEPENDENCIES.md#runtime-storage-placement). The original
distro is unchanged.

### Measurements

**Import chain** (Torch, Transformers, Triton, FLA), page cache dropped
before each cold sample:

| Distro | Cold | Warm |
| --- | ---: | ---: |
| `Ubuntu` (USB HDD) | 58.6 s (torch 24.4, transformers 28.9, fla 5.3) | 3.5 s |
| `EXL3-Runtime` (NVMe) | 6.0 s | 3.3 s |

**Full serve startup**, measured from launching `python run.py serve` to
`ready=1`. The WSL VM was shut down before each launch, which matches a
first launch after WSL idles. Same MiMo 131K Q6 command, warm-up on, Triton
cache complete on both distros, runs alternated HDD/NVMe.

| Stage (seconds) | `Ubuntu` HDD, 2 runs | `EXL3-Runtime` NVMe, 2 runs |
| --- | ---: | ---: |
| Launch → engine start (VM/distro boot, worker) | 40.0 / 41.6 | 5.2 / 4.6 |
| Engine: extension imported | 25.4 / 24.6 | 3.9 / 4.0 |
| Engine: imports and device ready | 57.9 / 58.4 | 11.4 / 11.5 |
| Draft + target weight load (model still on HDD) | 16.5 / 15.6 | 14.7 / 13.1 |
| Warm-up | 24.5 / 25.6 | 4.9 / 4.7 |
| Engine ready (`load_seconds`) | 103.9 / 104.9 | 34.6 / 33.0 |
| **Launch → ready (wall)** | **143.9 / 146.5** | **39.8 / 37.6** |

The HDD distro also slows warm-up (24–26 s against 4.7–4.9 s), because the
cached kernel binaries themselves are read from disk. The baseline run
reported `load_seconds` 76.3 without any warm-up, on a partly warm page
cache; the NVMe runtime is ready sooner while also completing warm-up.

Runs: `RUN-20260928T034554Z-ef44d251`, `RUN-20260928T034931Z-6ae15281`
(HDD) and `RUN-20260928T034836Z-d72dea3d`, `RUN-20260928T035222Z-0b05e7f8`
(NVMe). All four stopped cleanly through the stop file.

### Operational notes

- Runtime changes (packages, SDK, rebuilt extensions) must now be made in
  `EXL3-Runtime`. Kernel caches in the old distro are no longer used.
- Weight loading (about 15 s for MiMo 9B) is now the largest remaining
  startup stage and depends on the model disk. It was intentionally kept on
  the HDD.
- Remaining cost: about 11 s of imports and device initialization, and 4–5 s
  of cached warm-up.

## Result

For MiMo 9B at 131K on the RX 7800 XT:

- **Startup:** launch to ready went from about 145 s to about 39 s, including
  warm-up, on a cold VM with the model still on the HDD.
- **First request:** the first 8K request now runs at warm speed. Warm-up
  measured 1734–1839 t/s prefill on the first request, against 315 t/s in the
  baseline.
- **No late stalls:** no Triton compile or cache load happens while serving
  (`jit_after_ready = 0` up to 72K occupied context).
- **Output:** with warm-up on, greedy outputs match steady-state behavior.
