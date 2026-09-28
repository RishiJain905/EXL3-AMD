# Cold start and next optimization plan

Written September 27, 2026 from the MiMo 9B 131K serving run
`artifacts/RUN-20260928T005508Z-2dd374a8` (K5/H6 mul1, BF16 MTP depth 2, Q6/Q6
KV, chunk 1024, prefix cache on, temperature 0.6 / top-p 0.95 / top-k 20) and
a source/cache inspection of this checkout. It records seven ranked ideas so
later work can start from the plan instead of re-deriving it. Ideas 1–3 were
authorized for implementation on September 27; they are complete and recorded in
[COLD-START.md](COLD-START.md): launch-to-ready went from about 145 s to about
39 s with warm-up included, and the first request now runs at warm speed.
Ideas 4–6 were completed on September 28 (see the table and the dated
sections below). Idea 7 stays deferred. The next plan, toward 150–200 tok/s
MiMo decode and 100+ tok/s Qwenseek decode with DFlash drafters, is
[SPEED-TARGETS-PLAN.md](SPEED-TARGETS-PLAN.md).

Combined result on the promoted binary (`narrow-v2`), serving config, one
process with every default (`RUN-20260928T210253Z-0ac321a6`):

- **Decode:** 84.3 tokens/s with the serving sampler and 87.4 greedy, against
  about 73–77 before ideas 4–6.
- **Prefill:** 1757 tokens/s at 8.8K occupied context and 1088 tokens/s over
  69K → 128K, against 1578 and about 500.
- **Correctness:** no Triton compile after ready, and no speculative-sampling
  fallbacks.

The user decision behind this plan: a megakernel is **not** the first step.
Model weights stay on the D: HDD (they load in about 15 s); runtime files may
move to F: (NVMe).

## Evidence from the run

### Cold start is first-use compilation and disk-bound imports

| Observation | Value |
| --- | --- |
| Verify → extension imported | 26.3 s |
| Extension → device query (imports, HIP init) | 31.0 s |
| Draft + target weight loads | 4.6 + 10.3 s |
| Ready | 76.3 s |
| Request 1: 8033-token fresh prefill | 25.5 s (315 t/s) |
| Request 7: 8092-token fresh prefill, warm | 4.4 s (1848 t/s) |
| Request 2: 545-token fresh prefill, cold shape | 8.2 s |
| Request 6: 549-token fresh prefill, warm | 0.35 s |

- Requests 1 and 3 had **identical** MTP rounds (133 rounds, same
  accept histogram) but decoded at 38.2 vs 63.1 t/s. The difference is
  first-use cost, not kernel speed.
- 13 new Triton kernels compiled during the run despite a 2.8 GB persistent
  `~/.triton/cache`: `_paged_attn_prefill_kernel` (×5),
  `_paged_attn_prefill_combine_kernel` (×2), `_paged_attn_decode_split_kernel`
  (×4), `_paged_attn_decode_combine_kernel` and
  `_paged_attn_decode_combine_parallel_kernel`. Each took about 3–8 s. They
  coincide with requests 1–2 and with the 0.6–0.9 s prefill-to-first-token gaps
  of requests 8–9, when occupied context crossed new schedule thresholds.
- FLA 0.5.2 (chunked GDN prefill) has about 70 `@triton.autotune` sites.
  Triton 3.7.1 only persists autotune results with
  `TRITON_CACHE_AUTOTUNING=1`, which was unset, so every process re-benchmarks
  every configuration on first use.
- The WSL distro `Ubuntu` is a 219 GB `ext4.vhdx` on the USB Seagate HDD
  (`D:\WSL\Ubuntu`, ~35 ms read latency per the local `.wslconfig` note). The
  runtime venv (15 GB), ROCm SDK (2.2 GB), Triton cache (2.8 GB) and
  `~/.cache` (1.4 GB) are all inside it. The guest page cache is dropped when
  the WSL VM idles, so each fresh launch reads imports and cached kernels from
  the HDD again.

### Warm decode has kernel-internal headroom

- Warm MTP rounds take about 31–34 ms and emit about 2.2–2.5 tokens.
- A rough streaming floor is ~17 ms per round: 6.8 GB target weights plus
  ~1.5 GB of head/draft reads per round at the ~495 GB/s measured
  device copy rate. This is a diagnostic bound, not a promised rate.
- Earlier graph/fusion work ([FUSION-DECODE-PERFORMANCE.md](FUSION-DECODE-PERFORMANCE.md))
  measured launch/sync savings under 1%. The gap is inside kernels:
  packed projections reach about 100 GB/s logical packed bytes in
  [the roadmap table](INFERENCE-OPTIMIZATION-ROADMAP.md) (27B shapes).
- The long attention profile did apply (3750 applied calls), so that is
  not a gap in this run.

### MTP acceptance is low under sampling

Acceptance ranged 46–75% per request; 20–40% of rounds accepted zero drafts.
`Generator` (vendored `generator.py`, the `draft_tokens[j, i] != sampled_token`
check) samples a target token and accepts only on exact match.

### Long-context prefill slows as occupied context grows

Warm fresh prefill: 1848 t/s at 8K. Prefix-hit prefills that compute new
tokens on top of 22–38K occupied context ran at ~1000–1500 t/s. Attention's
share grows with context. (Correction found during idea 6: serve and the
evaluator disabled FP16 staging for quantized caches, so Q6 prefill
dequantized inside the attention kernel rather than into a scratch buffer.)

## Ranked ideas

| # | Idea | Targets | Effort | Status |
| --- | --- | --- | --- | --- |
| 1 | Persist Triton autotune results | Cold prefill on every restart | One line | Done, primary — [COLD-START.md](COLD-START.md) |
| 2 | Startup warm-up before `ready=1` | Cold prefill/decode, first-token latency | Small–medium | Done, primary — [COLD-START.md](COLD-START.md) |
| 3 | Move runtime (venv, SDK, Triton/kernel caches) to F: NVMe | Import/startup, cached-kernel reads | Operational | Done — [COLD-START.md](COLD-START.md) |
| 4 | Proper speculative sampling for temperature > 0 | MTP tokens per round | Medium | Done, primary — [SPECULATIVE-SAMPLING.md](SPECULATIVE-SAMPLING.md) |
| 5 | Kernel self-time profiling, then MiMo-shape packed decode tuning | Warm decode (largest gap) | Large | Done, primary — [DECODE-PROFILING.md](DECODE-PROFILING.md) |
| 6 | Long-context prefill tuning | Prefill with occupied context | Medium | Done, primary — [PREFILL-ATTENTION.md](PREFILL-ATTENTION.md) |
| 7 | Megakernel / persistent scheduler | Residual launch/sync overhead | Very large | Deferred: idea 5 bounds it at about 10% of a round |

### 1. Persist Triton autotune results

Set `TRITON_CACHE_AUTOTUNING=1` in the backend environment
(`runtime_environment()` in `scripts/launch_runtime.py`). Triton then stores
each autotune winner in its cache directory and later processes skip the
benchmark. Autotune keys are shape-derived, so a new key still tunes once.

### 2. Startup warm-up

Before reporting `ready=1`, run bounded synthetic work that triggers every
Triton specialization normal serving will hit: one full prefill chunk plus
representative tail lengths (FLA, conv1d, paged prefill, WMMA/packed GEMMs),
decode/verify with 1..depth+1 query rows, and the attention decode/prefill
schedule variants at each occupied-length threshold up to the configured
context. Threshold variants can be compiled by calling the attention kernels
directly on synthetic metadata; no long real prefill is needed because
specializations depend on constexpr/schedule values, not cache contents.
Discard all warm-up state (generator, pages, recurrent stash) so prefix reuse
and outputs are unaffected. Also check whether integer arguments Triton
specializes (`==1`, `%16==0`) multiply the variant count; `do_not_specialize`
or bucketing keeps the set finite.

### 3. Runtime on NVMe

Place the runtime venv, ROCm SDK, `TRITON_CACHE_DIR` and the other kernel
caches (`~/.cache/{comgr,miopen,torch}`) on F:. The whole distro does not fit
(219 GB vs 195 GB free on F:). Plain `/mnt/f` is a 9P mount and is slow for
Python's many small-file imports, so it needs an ext4 filesystem hosted on F:.
Model weights stay on D:.

### 4. Proper speculative sampling (done)

Done September 28, 2026; see [SPECULATIVE-SAMPLING.md](SPECULATIVE-SAMPLING.md).

- **What shipped.** The draft samples its top-K proposal (one `topk` plus one
  Triton kernel). Verification computes the target's exact fused-sampler
  distribution on its top-K candidates. A tie group at the cutoff hands over
  to exact matching per position.
- **Correctness.** Distribution tests against MTP-off sampling pass. They also
  caught, and the fix removed, a bias in a first version that fell back for
  whole rounds.
- **Gain.** Tokens per round +21% at T=1 / top-k 50. About neutral at the
  serving config (T=0.6, top-p 0.95, top-k 20), where the draft already
  matches the target's top token most of the time. Round overhead is gone
  (30.58 vs 30.53 ms).
- **Depth.** The re-sweep keeps depth 2. Depth 3's 4-row verify has no native
  kernels (74.6 ms/round); depth 4 costs 39 ms/round for 2.67 tokens.

The original plan follows.

Replace exact-match acceptance at temperature > 0 with the standard rule:
sample draft token x from the draft's own tempered/filtered distribution q,
accept with probability min(1, p(x)/q(x)) under the target's processed
distribution p, and on rejection resample from normalize(max(0, p − q)).
Output remains exactly distributed as the target sampler. If the draft
currently picks argmax, exact match is already equivalent to the rule with a
one-hot q, so the gain comes from sampling the draft.

- Keep draft probabilities on the GPU; apply identical temperature, top-k,
  top-p and penalties to both distributions.
- Greedy decoding is unchanged and must stay bit-identical.
- Preserve GDN rollback, recurrent checkpoints, EOS handling and
  `draft_stats` accounting.
- Then re-sweep MTP depth 2 vs 3 under the serving sampling config and Q6
  KV. Depth 2 was locked on a greedy-style F16/4K selection test
  ([MIMO-MTP.md](MIMO-MTP.md)).
- Validate with distribution tests (seeded, large-sample token-frequency
  comparison against non-speculative sampling) plus model runs.

### 5. Kernel self-time, then packed decode tuning (done)

Done September 28, 2026; see [DECODE-PROFILING.md](DECODE-PROFILING.md).

- **Profiling.** rocprofv3 cannot trace kernels on this WSL stack: the WSL HSA
  runtime never registers with rocprofiler-sdk, and there is no KFD. Only HIP
  API host tracing works. `scripts/profile_decode_round.py` measures per-op
  GPU time with events and pre-queued work instead.
- **Profile.** The warm short round was dominated by:
  - the K5 MLP and GDN projections;
  - the vocabulary head;
  - two tiny dense FP16 GDN gate projections at about 88 µs each, as slow as
    a 21 MB packed projection;
  - BF16 MTP draft projections.
- **Promoted.**
  - A narrow dense GEMM for decode-sized FP16/BF16 projections: 81 → 7 µs per
    gate projection. `--no-narrow-gemm` is the diagnostic override.
  - A four-way unroll for K5/K6 at 3 rows or fewer.
- **Result.** Warm MTP decode 77.6–78.1 → 90.7 tokens/s short, and
  64.9–65.3 → 76.1 at 16K. Greedy output diverges from the old build only at
  near-ties.
- **Remaining limit.** K5 small-M kernels run at about 280–390 GB/s whether
  the weights are cached or streamed. They are bound by decode and
  instruction issue, not memory; more speed needs a new inner loop.
- **Idea 7.** Time between kernels is about 6% of a round after tuning, plus
  about 2 µs of dispatch across ~860 launches. A megakernel's ceiling is
  therefore about 10%.

The original plan follows.

No usable per-kernel GPU timeline exists yet (Kineto shows no device events).
Either validate `rocprofv3` on this WSL stack or run one native-Linux
profiling session. Then continue the packed GEMV, small-M and paired-MLP work
on MiMo's real shapes (4096↔12288, K5 body / K6 head, rows 1–3), measured with
warm and cache-evicted weights. Promote only full-model wins, as before.

### 6. Long-context prefill (done)

Done September 28, 2026; see [PREFILL-ATTENTION.md](PREFILL-ATTENTION.md).

- **Staging.** Quantized caches now stage each prefill window to FP16 by
  default. Serve had disabled staging, so Q6 prefill dequantized inside the
  kernel.
- **Tiles.** The gfx1101 FP16 prefill kernel uses measured head_dim-256 tiles:
  2.0–2.5× faster, with the same error against FP32.
- **Result.** MiMo 9B Q6 prefill with occupied context: 1008 against 463
  tokens/s at 122K, 1614 against 1126 at 35K. The cost is a 512 MiB scratch
  at 131K.
- **Chunk re-sweep.** Chunk 2048 is only 1–2% faster and uses more memory;
  chunk 1024 stays the default.
- **Side effect.** The per-process `num_stages` pick is no longer reached in
  default serving.

The original plan follows.

Re-sweep chunk size 1024 vs 2048 on MiMo with WMMA (the old sweep used the
27B with BLAS). Tune the Q6 paged prefill attention schedule, scratch
dequantization and tiles at 16K–128K occupied context.

### Follow-ups found while doing 1–3

- The vendored Q6 prefill attention picks `num_stages` 1 or 2 by timing, once
  per process. The two differ by up to 6.1e-5, so greedy output can depend on
  which one a process picked. Persisting or fixing the pick would make output
  process-independent (and avoid compiling both variant sets).
- `do_not_specialize` on the paged-attention runtime integers (`split_len`,
  `num_pages_per_seq`, `num_splits`, prefill `q_len`/`kv_append_len`) would
  reduce the ~100 attention variants and shorten first-launch
  warm-up (~100 s). It needs a kernel speed check before promotion.
- Weight loading (~15 s from the D: HDD) is now the largest startup stage.
  The user chose to keep weights on the HDD.

### 7. Megakernel (deferred)

Only after idea 5 shows substantial time between kernels. WSL reports no
cooperative launch, so persistence needs a bounded device queue with proven
forward progress. It must preserve cancellation, GDN history/rollback and MTP
checkpoints. See [INFERENCE-OPTIMIZATION-ROADMAP.md](INFERENCE-OPTIMIZATION-ROADMAP.md).
