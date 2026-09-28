# Speed targets plan: DFlash drafting and multi-row verification

Written September 28, 2026, after ideas 1–6 of
[OPTIMIZATION-PLAN.md](OPTIMIZATION-PLAN.md). This is a plan only: nothing
below is implemented yet. It records the targets, the reasoning behind the
order, and each step's scope and exit criteria, so later work can start from
it.

## Targets

RX 7800 XT (`gfx1101`, 16 GB, 624 GB/s GDDR6). Current numbers are from the
combined run `RUN-20260928T210253Z-0ac321a6` and
[DECODE-PROFILING.md](DECODE-PROFILING.md).

| Metric | Now | Target |
| --- | ---: | ---: |
| MiMo 9B decode (K5/H6, BF16 MTP 2, Q6/Q6) | 84 sampled, 87–91 greedy tok/s | 150–200 tok/s |
| Qwenseek 27B decode (K2/K4, quantized MTP 2, Q6) | ~41 tok/s (23 without MTP) | 100+ tok/s |
| MiMo 9B prefill | 1757 tok/s at 8.8K, ~1000 at 122K | 2000+ tok/s |

The prefill target applies to MiMo at short-to-medium context (roughly up to
16–32K). At 2000 tok/s a 27B model needs about 108 TFLOP/s, above this
card's ~75 TFLOP/s FP16 peak, and at 128K MiMo spends about 0.48 s per
1024-token chunk in attention alone, plus about 0.55 s outside attention.

## Why drafting comes first

Decode speed is tokens accepted per round divided by round time.

- Every round reads the target weights at least once. MiMo's are 6.8 GB, which
  takes about 13.7 ms at the ~495 GB/s this card sustains. The "2,708 GB/s
  effective" figure in AMD's specifications is the 64 MB Infinity Cache; model
  weights do not fit in it and stream from GDDR6.
- At today's 2.2 tokens per round, 150 tok/s needs a 14.7 ms round, about 1 ms
  above that floor. 200 tok/s is impossible at 2.2 tokens per round.
- At 4–5 tokens per round, the same targets need a 20–33 ms round. The
  current round is 25.7 ms.

So the targets depend mainly on more accepted tokens per round, with the
condition that verifying 8–16 rows costs close to what 3 rows cost now.
Kernel speed-ups multiply with that, but cannot reach the targets alone.

Qwenseek is further from its memory limit: its target-only forward moves
about 165 GB/s, and the K2 kernels are limited by decode instructions, not
memory. Physics allows about 90 tok/s without speculation there, so 100+ with
good drafting is a kernel and drafting problem, not a hardware one.

## Published drafters

Both models are fine-tunes of Qwen base models with published DFlash
drafters. DFlash is a block-diffusion drafter: one forward pass proposes a
whole block of 8–16 tokens, conditioned on hidden states tapped from several
target layers. It drafts a single chain, not a tree, so GatedDeltaNet
verification needs no per-branch recurrent state. Output stays exact: greedy
matches the target, and speculative sampling preserves its distribution.

| | MiMo 9B | Qwenseek 27B |
| --- | --- | --- |
| Base model | Qwen3.5-9B (`Qwen3_5ForConditionalGeneration`; MTP already donated from the base, [MIMO-MTP-PACKAGING.md](MIMO-MTP-PACKAGING.md)) | Qwen3.8-27B ([HEAD-ATTENTION-PERFORMANCE.md](HEAD-ATTENTION-PERFORMANCE.md)) |
| Drafter | [`z-lab/Qwen3.5-9B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash), Apache-2.0 | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (mirror of `incoai/…`), Apache-2.0 |
| Architecture | `DFlashDraftModel` (DFlash 1): 6 layers, hidden 4096, 5 sliding-window (4096) + 1 full attention, block 16, 8 target taps | `DFlash2DraftModel`: 5 sliding-window (2048) layers, hidden 5120, block 8, 5 target taps, candidate selector (rank 256, top 16), two-tap dynamic convolutions |
| Size (BF16, no embedding or vocabulary head; the target's are used) | 2.58 GB (~1.3B parameters) | 3.85 GB (~1.9B parameters) |
| Published acceptance (tokens per round, base model, NVIDIA) | 4.6–5.8 at block 8, 5.7–7.9 at block 16; greedy, thinking on | 4.1–5.5 at block 8; T=1.0, top-p 0.95, top-k 20 |
| Built-in MTP for comparison | 3.2–3.5 at 3 steps, 4.6–5.5 at 7 steps | 3.7–5.0 at 7 steps |
| Vendored backend support | Yes: `architecture/dflash.py` and `Generator.iterate_draftmodel_dflash_gen`; never run on ROCm | No: DFlash 2 needs a port (check upstream ExLlamaV3 first) |

Published acceptance comes from the base models on math, code and MT-Bench.
Fine-tunes will accept less; MiMo's donated base MTP already reaching 2.2 per
round at depth 2 suggests the drafters transfer usefully. Chat prompts sit at
the low end (MT-Bench).

## Revised order

### 1. Verify-cost curve and Qwenseek baseline

Measure before building anything, because the answer decides how wide the
drafts can usefully be.

- **Row-cost curve.** For MiMo and Qwenseek, time one warm target forward at
  1, 2, 3, 5, 8, 16 and 32 query rows at short and 16K occupied context,
  broken down by category with the event profiler
  (`scripts/profile_decode_round.py`): quantized projections, vocabulary head,
  GatedDeltaNet multi-token core, attention, norms.
- **Qwenseek baseline** on the current binary at its serving sampler
  (T=1.0, top-p 0.95, top-k 20): decode with speculative sampling on and off,
  tokens per round, prefill with and without staging. Speculative sampling
  (idea 4) gave MiMo +10% at T=1.0 and has not been measured on the 27B.
- **First Qwenseek decode-round profile.** The 27B has never been profiled
  with the event profiler.

Exit: a recorded row-cost table per model, the Qwenseek baseline, and a
recommended block size per model for step 2.

### 2. DFlash bring-up for both models

#### 2a. Obtain the drafters

With explicit authorization, and after reading `.runtime/MODEL-STORAGE.md`,
download both drafters to the documented model location (weights stay out of
source control):

- `z-lab/Qwen3.5-9B-DFlash` (2.58 GB);
- `z-lab/Qwen3.8-27B-DFlash2` (3.85 GB).

Record the files' SHA-256 and check each drafter against its target:
`num_target_layers` equals the target's layer count, the vocabulary is
248320, the mask token exists, and hidden sizes match.

#### 2b. Drafter variants

Draft quality affects only acceptance, never output, so quantization is judged
by acceptance and speed alone.

| Variant | MiMo 9B | Qwenseek 27B |
| --- | --- | --- |
| BF16 drafter | Yes: first bring-up and reference acceptance | Only if memory allows, as the acceptance reference |
| Quantized (EXL3) drafter | Yes: compare acceptance and speed against BF16 | Yes: the expected serving form |

- Quantize with the vendored EXL3 converter at two or three bit widths (for
  example 4, 5 and 6 bits). At 4 bits the drafters are about 0.65 GB and
  0.95 GB.
- Open question: the drafters' inputs are target hidden states, so
  calibration has to feed them target activations rather than plain text
  embeddings. Check how the converter calibrates a DFlash input layer before
  quantizing.
- The narrow and small-M kernels only cover some shapes. Record which drafter
  projections fall back to slower paths.

#### 2c. Disable the built-in MTP when DFlash is active

When a DFlash drafter is loaded, the built-in MTP layer is not needed. Serve
already loads the MTP component and allocates its draft cache only with
`--mtp` (`scripts/serve_exl3.py`), so a DFlash configuration leaves both out
of memory without rewriting model files. Make that automatic when a drafter is
configured, and confirm the target load itself holds no MTP tensors. The
vocabulary head and embedding stay, because the target uses them and DFlash
borrows them.

Memory expectations, all to be measured:

| Item | MiMo 9B | Qwenseek 27B |
| --- | --- | --- |
| Freed: MTP layer weights | ~0.52 GB (BF16) | Quantized MTP layer (measure) |
| Freed: MTP draft KV cache | One layer of 4 KV heads × 256 at the configured context | One layer at the configured context |
| Added: drafter weights | 2.58 GB BF16, or ~0.65–1.0 GB quantized | ~0.95–1.4 GB quantized |
| Added: drafter KV cache | 6 layers × 8 KV heads × 128 | 5 layers × 8 KV heads × 128 |
| Added: tapped target hidden states | 8 × 4096 per drafted row | 5 × 5120 per drafted row |

- A quantized drafter roughly cancels out the removed MTP weights. The BF16
  MiMo drafter adds about 2 GB net.
- **Risk:** if the vendored draft cache sizes every drafter layer for the full
  context, the drafter's cache at 131K costs more than its weights (about
  3.2 GB in FP16 for MiMo's drafter). Sliding-window layers only need their
  window. Check the allocation and, if needed, give sliding layers a
  window-sized cache.
- Keep the MTP path as the fallback when no drafter is configured.

#### 2d. Qwenseek: DFlash 2 support

The vendored backend supports DFlash 1 only. Before porting, check whether
upstream ExLlamaV3 (or CarouselAether's ROCm fork, pinned in
[UPSTREAMS.md](UPSTREAMS.md)) has added `DFlash2DraftModel`. Otherwise port
its candidate selector and two-tap dynamic convolutions from the reference
implementation (`z-lab/dflash`), with CPU reference tests against the
published model's outputs.

#### 2e. Correctness

- **Greedy:** DFlash output must match the non-speculative greedy output,
  apart from the near-tie divergences that multi-row verification already
  causes (see [DECODE-PROFILING.md](DECODE-PROFILING.md)).
- **Sampling:** extend speculative sampling (idea 4,
  [SPECULATIVE-SAMPLING.md](SPECULATIVE-SAMPLING.md)) to DFlash drafts. Each
  position's draft is sampled from that position's drafter distribution, which
  serves as the proposal q. Repeat the seeded permutation χ² distribution tests
  against non-speculative sampling.
- **GatedDeltaNet rollback** over 8–16-token windows: check that partial
  acceptance restores recurrent state correctly and measure its cost.
- **Supported paths:** check that the drafter's attention runs on the ROCm
  paths (the vendored DFlash code requests `flash_attn` mode) and that nothing
  compiles after ready (extend warm-up).

#### 2f. Measurements and promotion

For each variant, at short, 16K and long occupied context, with the serving
sampler and greedy, on chat and code prompts:

- tokens per round, round time, decode tok/s;
- VRAM reserved, compared with the current MTP configuration;
- block size sweep (8 and 16 for MiMo; 8 for Qwenseek unless step 1 favors
  more).

Promote the fastest correct configuration per model as the primary default
when its drafter is present, with a diagnostic override back to MTP. Record
the results in a new dated feature document and in
[VALIDATION.md](VALIDATION.md).

### 3. Cheap 8–16-row verification

Guided by step 1's curve, make the target's multi-row forward cost close to
its 3-row cost:

- quantized projections: extend the native small-M, packed head and paired
  MLP kernels to the chosen row counts (today they cover 1, 2, 3 or 5 rows;
  4 rows falls back, [SPECULATIVE-SAMPLING.md](SPECULATIVE-SAMPLING.md)), or
  tune the packed 10–64-row path for small row counts;
- the GatedDeltaNet multi-token recurrent core at those lengths;
- paged attention at the chosen query length.

### 4. Qwenseek: deeper built-in MTP as the comparison

Qwen3.8's own MTP reaches 3.7–5.0 tokens per round at 7 steps on the base
model. Once extra rows are cheap (step 3), re-sweep Qwenseek's MTP depth and
compare against DFlash 2 at equal memory. Keep whichever is faster; the other
remains available.

### 5. Faster quantized decode kernels (K5 and K2)

MiMo's K5 small-M kernels run at 280–390 GB/s whether weights are cached or
streamed, so they are limited by decode work
([DECODE-PROFILING.md](DECODE-PROFILING.md) §4). Candidates: fewer memory
instructions per tile, and fusing the input transform into the kernel. For
Qwenseek, start from step 1's profile; its K2 projections are further from the
memory limit.

### 6. Prefill

Extend the event profiler to a prefill chunk. At 8K, about 0.55 s of each
1024-token chunk is outside attention (quantized matrix multiplies and the
GatedDeltaNet chunk path); that part decides the 2000 tok/s target. For long
context, a native matrix-core attention kernel could give 1.3–1.5× over the
current ~36 TFLOP/s Triton tiles
([PREFILL-ATTENTION.md](PREFILL-ATTENTION.md)).

### 7. Megakernel

Last. Bounded at about 10% of a round
([DECODE-PROFILING.md](DECODE-PROFILING.md) §3).

## Risks and open questions

- Acceptance on the fine-tunes is unknown until step 2 measures it.
- DFlash has never run on this ROCm stack; the vendored path may assume
  CUDA-only attention.
- Quantizing a drafter needs target-conditioned calibration (2b).
- Drafter KV cache allocation at 131K (2c).
- VRAM headroom on Qwenseek at long context with a drafter loaded.
- The estimates in this document come from profiles and published NVIDIA
  results, not measurements on this card; each step records measured results.

## Decisions needed before step 2

- Authorization to download both drafters (6.4 GB in total).
