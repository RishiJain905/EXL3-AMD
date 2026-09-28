# Long-context prefill: staged quantized KV and RDNA3 tiles

September 28, 2026. Implements idea 6 of [OPTIMIZATION-PLAN.md](OPTIMIZATION-PLAN.md).
Both changes are primary. At 122K occupied context, prefill on MiMo 9B with a
Q6 cache is 2.2× faster (463 → 1008 tokens/s).

## Finding

The plan assumed that Q6 prefill dequantizes each span into an FP16 scratch.
It did not: `serve_exl3.py` and `evaluate_exl3_candidate.py` set
`EXL3_QC_STAGING=0` for every quantized cache. Every Q6 prefill chunk ran the
direct path, which re-expands each packed K/V tile inside the attention kernel
once per query block and sibling head.

The FP16 prefill kernel had its own problem on gfx1101. At head_dim 256 the
vendored tile `(block_m 64, block_n 32, 8 warps, 2 stages)` puts 8 query rows
on each warp. The file's own RDNA note says to keep 16 rows per warp; off-ratio
tiles measured up to 4× slower. It ran at about 15 TFLOP/s.

## Changes

1. **Staged prefill for quantized caches** (`--prefill-staging on`, the
   default). Chunks of at least 256 rows dequantize the referenced window once
   (`dequant_cache_paged_window`) into a shared FP16 scratch and run the FP16
   kernel.
   - The scratch holds one layer's K and V for the whole cache pool:
     2 × pool tokens × KV heads × head dim × 2 bytes. That is 512 MiB for
     MiMo 9B at 131072 tokens.
   - It is shared by every layer and by the MTP draft cache, and is allocated
     at the first staged prefill, which warm-up performs before ready.
   - Shorter chunks and prefix-hit remainders keep the direct path, as the
     vendored threshold intends.
   - The decision lives in `cache_precision.qc_staging_env`.
   - `--prefill-staging off` restores in-kernel dequantization (diagnostic);
     it is accepted by every mode.
2. **Measured gfx1101 tiles for the FP16 prefill kernel** at head_dim 129–256
   (`schedule.prefill_tiles`, used by `triton_paged.paged_attn_triton_prefill`):
   - `(64, 64, 4, 1)` for up to 128 query rows;
   - `(128, 64, 8, 1)` above that.
   - Both keep 16 rows per warp.
   - They apply to FP16 caches and to staged quantized caches.
   - Other GPUs, head sizes, the direct quantized path and codebook caches keep
     the inherited tiles.
   - `EXL3_PREFILL_TILES=inherited` restores the old tile for direct backend
     runs (the launcher does not forward `EXL3_*` variables).

## Kernel measurements

`scripts/benchmark_exl3_prefill_attention.py`: MiMo geometry (16 Q / 4 KV
heads, head_dim 256), causal, FP16 paged cache, RX 7800 XT. Every candidate is
checked against the default configuration, and optionally against an FP32
attention reference.

| Context | Query rows | Inherited tile | New tile | Speed-up |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 17 | 0.839 ms | 0.402 ms | 2.09× |
| 8192 | 64 | 0.841 ms | 0.413 ms | 2.04× |
| 8192 | 256 | 2.551 ms | 1.205 ms | 2.12× |
| 8192 | 1024 | 9.648 ms | 4.257 ms | 2.27× |
| 32768 | 1024 | 36.64 ms | 15.41 ms | 2.38× |
| 65536 | 17 | 5.803 ms | 2.569 ms | 2.26× |
| 65536 | 1024 | 72.03 ms | 30.10 ms | 2.39× |
| 128000 | 1024 | 140.3 ms | 59.8 ms | 2.35× |
| 128000 | 2048 | 292.3 ms | 117.3 ms | 2.49× |

- **Throughput.** The inherited tile ran at about 15 TFLOP/s; the new tile
  reaches about 36.
- **Search space.** 42 configurations for 1024/2048 rows and 8 for 17–1024
  rows. Two-stage variants and 8-rows-per-warp tiles were always slower.
- **Accuracy.** Relative L2 against the FP32 reference is
  2.79–2.96 × 10⁻⁴ for both the inherited and new tiles. That is the FP16
  output rounding floor: no accuracy change.

Results are in `artifacts/prefill-attention-20260928/tiles-1` and `tiles-2`.

## Full-model measurements

`scripts/benchmark_prefill_context.py` against the serving config: MiMo K5/H6
mul1, BF16 MTP 2, Q6/Q6, 131072 context, prefix cache on. It sends growing
prefixes of one document, so each request computes about 17K new tokens on
top of the occupied context. The table shows the compute rate of that span in
tokens/s.

| Occupied (end) | Direct, chunk 1024 (before) | Direct, 2048 | Staged, old tiles, 1024 | **Staged, new tiles, 1024** | Staged, new tiles, 2048 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8.8K | 1578 | 1594 | 1666 | **1742** | 1768 |
| 17.7K | 1414 | 1435 | 1551 | **1746** | 1752 |
| 34.9K | 1126 | 1151 | 1307 | **1614** | 1622 |
| 52.1K | 880 | 903 | 1068 | **1439** | 1452 |
| 69.6K | 722 | 744 | 903 | **1299** | 1312 |
| 86.8K | 610 | 631 | 783 | **1198** | 1212 |
| 105.3K | 526 | 546 | 685 | **1093** | 1109 |
| 122.5K | 463 | 483 | 612 | **1008** | 1027 |

- **Tiles alone change nothing on the direct path.** Direct runs with and
  without the new tiles matched within 0.3%.
- **Staging alone** gives +5% to +32%.
- **Together** they give +10% at 8K rising to +118% at 122K.
- **Memory.** Reserved memory rose from 8658 to 9170 MiB, the scratch.
- **Chunk re-sweep.** Chunk 2048 is 1–2% faster than 1024 in both paths but
  reserves another 248 MiB, and coarsens cancellation and first-token
  granularity. Chunk 1024 stays the serve default.

**Greedy output:**

- The first token after each of the nine prefixes (8.8K–128K) was identical
  between the direct path and staged + new tiles.
- Short greedy chat prompts matched token for token in every configuration.
  They stay below the 256-row staging threshold, so they check the rest of
  the pipeline, not staging.
- **64-token greedy continuations.** Staged + new tiles against direct:
  - ~77K prompt: identical.
  - ~26K prompt: identical for about 45 tokens, then a wording near-tie
    diverges ("uniform layer_types (full or sliding…" against "layer_types
    (uniform full/sliding…"); both continuations are coherent.
  - This is the FP16 accumulation-order sensitivity that already made
    direct-path greedy output depend on each process's `num_stages` pick.
  - Prefill on these requests: 1776 against 1432 tokens/s at 26K, and 1392
    against 800 at 77K. Runs: `RUN-20260928T204139Z-d0ee33b3` (staged) and
    `RUN-20260928T205211Z-9ec0951b` (direct).

**Side effect:** the vendored per-process `num_stages` timing pick belongs to
the direct quantized path at 256 or more rows. Staging removes that path from
default serving, so greedy output no longer depends on which stage count a
process picked (a follow-up from [COLD-START.md](COLD-START.md)).

Runs (ignored `artifacts/`):

- `RUN-20260928T200522Z-cd866336`: direct, inherited tiles.
- `RUN-20260928T201050Z-910d5b8d`: direct, new tiles.
- `RUN-20260928T201436Z-52358f63`: direct, chunk 2048.
- `RUN-20260928T201829Z-9fb46f40`: staged, new tiles.
- `RUN-20260928T202554Z-9dd48d7a`: staged, inherited tiles.
- `RUN-20260928T202929Z-978535cd`: staged, chunk 2048.

## Limitations

- **Hardware and geometry.** Measured on gfx1101 at head_dim 256 (MiMo 9B;
  Qwenseek 27B shares the head size but was not run). Other GPUs keep the
  inherited tile.
- **Memory.** The scratch costs about 4 KiB per cached token for MiMo 9B's 4
  KV heads (2 × 4 × 256 × 2 bytes). A context that only just fit with Q6 may
  need a smaller `-c` or `--prefill-staging off`. Warm-up allocates the
  scratch, so a shortfall fails at startup, not mid-request.
- **Direct quantized path untuned.** Queries under 256 rows and non-causal
  spans still use the inherited direct tile.
- **Decode attention** is unchanged here (see
  [HEAD-ATTENTION-PERFORMANCE.md](HEAD-ATTENTION-PERFORMANCE.md)).
