# Speculative sampling for MTP drafts

September 28, 2026. Implements idea 4 of [OPTIMIZATION-PLAN.md](OPTIMIZATION-PLAN.md).
It is primary: sampled requests on an MTP server use it without any flag.
`--spec-sampling off` (serve) is a diagnostic override.

## What changed

The vendored generator drafted with argmax and accepted a draft position only
when the target's independently sampled token matched it. That rule is exact:
it is the speculative rule with a one-hot proposal. At temperature > 0 it
rejects whenever the target samples anything other than the draft's top token.

Now, for sampled requests:

1. **Draft:** each MTP step samples its token from the draft's own top-K
   proposal q. K is the request's `top_k`, or 64 without top-K. Temperature,
   min-P and top-P are applied on those candidates.
2. **Verify:** position i accepts draft xᵢ with probability min(1, p(xᵢ)/q(xᵢ)).
   The first rejection emits a sample of normalize(max(0, p − q)). If every
   draft is accepted, the bonus position emits a sample of p.

p is the target's sampling distribution exactly as the fused sampler
(`SS_Fused`) defines it:

- temperature;
- min-P as `l ≥ max(l) + minp_log`;
- top-K with ties at the cutoff kept;
- top-P over the top-K-truncated mass, dropping the tie group that crosses the
  target and always keeping the top token.

Emitted tokens then follow p exactly, whatever q is. Only acceptance changes.

### Fast path

The proposal only has to be the distribution its token was actually drawn from.
So the draft works on its top-K candidates (`torch.topk` plus one Triton kernel).

For 1 ≤ `top_k` ≤ 64, every token p keeps is among the target's top-K
candidates, with one exception: a tie group at the K-th value that extends
past them. Verification per round is:

- a `topk` of the k + 1 target rows;
- a dense tie count at the K-th value;
- one Triton kernel for exact p, acceptance and the residual sample;
- a single readback.

Other `top_k` settings use a dense reference path (`filtered_probs`).

### Tie handoff

A row whose kept set extends past its candidates is flagged. At the first
flagged position, the rest of the window goes to the inherited exact-match
verification, keeping the drafts accepted before it (`resolve`).

This must be decided per position. Row i's logits depend on drafts 0..i−1,
which are already emitted when position i is decided. They never depend on
draft i or later. Exact matching is correct for any fixed draft, but the
speculative rule is only correct averaged over x ~ q. So a switch that looked
at later rows would make earlier decisions depend on later drafts, which
introduces bias. The first implementation did exactly that; see
[Validation](#validation).

### Eligibility

Speculative sampling applies to MTP drafting without `--draft-confidence`.
Everything else keeps exact matching, which is correct but accepts less:

- greedy requests (unchanged, bit-identical);
- penalties;
- logit masks and filters, forced tokens, banned strings;
- `min_new_tokens` still pending;
- probability or top-token outputs.

In the HTTP API, only repetition/presence/frequency penalties opt a request
out. GDN rollback, recurrent checkpoints, EOS handling and `draft_stats`
accounting are unchanged: the generator's per-position loop consumes the
resolved tokens.

### Code

- `src/quantlab/methods/exl3/spec_sampling.py`: filter semantics, sparse
  kernels, `resolve`, the per-round state.
- Vendored `generator.py`: three hooks.
  - The MTP draft loop calls `SpecRound.draft_step`.
  - `iterate_gen` calls `SpecRound.verify` once per job.
  - Resolved tokens replace per-position sampling; exact matching continues
    after a handoff.
- `scripts/serve_exl3.py`: installs it, reports `spec_sampling` counters in
  the optimization record and `/health` (rounds, verified, fallbacks, tie
  handoffs), and warms both kernels.
- Warm-up: `warm_kernels` compiles every dtype × block × %16 class × filter
  combination during startup (96 launches). The first sampled request never
  compiles, even when the serve defaults are greedy.

## Validation

MiMo K5/H6 mul1, BF16 MTP, Q6/Q6 KV, 131072 context, chunk 1024, RX 7800 XT,
`validate_spec_sampling.py`.

### Unit coverage

`tests/test_spec_sampling.py`:

- **Filter semantics:** checked against a sort-based reference with forced ties.
- **Sparse target exactness and tie flags:** checked against the dense reference.
- **Joint distributions:**
  - dense and sparse rules, against the exact two-token law; a wrong
    acceptance rule scores TV 0.84;
  - tie-handoff: tie-heavy logits whose flags depend on the drafted token.
    The per-position rule scores TV 0.001; the whole-round fallback scores
    0.21, so this test catches that bug.
- **Integration:**
  - draft-loop behavior and eligibility;
  - the real vendored `ComboSampler`;
  - Triton kernels equal to the torch references given identical uniforms (GPU);
  - warm-up coverage.

The torch-dependent tests skip on the Windows host. Run them with the WSL
runtime interpreter; GPU tests need the lease.

### Distribution tests on the model

Seeded `/v1/completions` requests (seeds 0–1499) sent to separate servers.
Each comparison is a permutation χ² homogeneity test on output token IDs, per
position and per prefix. The high-entropy check uses a digits prompt at T=1.0 /
top-k 50 / top-p 1.0; at the serving config that prompt is almost
deterministic.

| Comparison (T=1.0, top-k 50, 6 tokens) | p-values, positions 1–6 |
| --- | --- |
| MTP off vs spec off | 1.000 everywhere (χ² ≈ 0.1: token-identical, see below) |
| MTP off vs first sparse version (whole-round tie fallback) | prefix@4 0.002, prefix@5 0.002 — **biased** |
| MTP off vs fixed version (per-position handoff) | 0.22–0.93 |

- **Exact matching reproduces non-speculative sampling token for token.** It
  consumes the same per-position random stream, so MTP-off and spec-off
  outputs are identical per seed. That makes MTP off the correct reference
  here.
- **How the bias was found.** The first sparse version fell back for the
  whole round whenever any row was flagged. The deficit sat on the bonus
  token: after `' ', '1', ' '`, the next digit `'5'` came out 9% against 18%.
- **Ruling out other causes.**
  - Sampling the real fused sampler and the sparse kernel on the same
    synthetic logits agreed within noise.
  - A fresh server without prefix cache reproduced the biased run exactly.
- **Handoff frequency.** 229 of 3525 rounds (6.5%) at this config. At the
  serving config (top-p 0.95 usually binds before the 20th token) flags are
  rare.

**Serving config** (T=0.6, top-p 0.95, top-k 20):

- Animals-list prompt, 4 tokens: spec off against the sparse version gives
  p = 0.82–0.98 at every position. No tie handoff occurred in any
  serving-config run, so that version behaves exactly like the fixed one
  there.
- Digits prompt: near-deterministic at this config (2 outcomes). MTP off
  against the final version gives p = 0.80.

### Speed

Final code, same session, spec on vs off. 4 chat prompts × 3 seeds, 512
output tokens. Decode rate is total tokens over total decode time.

| Sampling | Spec | ms/round | Tokens/round | Decode t/s |
| --- | --- | ---: | ---: | ---: |
| T=0.6, top-p 0.95, top-k 20 (serving) | off | 30.53 | 2.223 | 72.8 |
| T=0.6, top-p 0.95, top-k 20 (serving) | **on** | 30.58 | 2.234 | **73.1** |
| T=1.0, top-k 50 | off | 30.31 | 1.891 | 62.4 |
| T=1.0, top-k 50 | **on** | 30.79 | 2.115 | **68.7** (+10%) |
| Greedy (4 prompts) | off / on | 30.17 / 30.53 | 2.327 / 2.327 | identical tokens |

- **Serving config:** neutral within run-to-run variation (a separate
  session measured 75.8 against 73.5 t/s).
- **Higher-entropy sampling:** decode +10%.
- **Greedy:** output token-identical (1962 tokens across the 4 prompts).

Development measurements on the same config:

| Variant (depth 2 unless noted) | ms/round | Tokens/round | Decode t/s |
| --- | ---: | ---: | ---: |
| MTP off | — | — | 51.2 |
| Spec off (exact matching) | 30.09 | 2.210 | 73.5 |
| First version: dense filters (~30 torch ops per step) | 31.26 | 2.253 | 72.1 |
| Sparse Triton path | 29.74 | 2.254 | 75.8 |
| Depth 3, dense version (4 verify rows) | 74.64 | 2.559 | 34.9 |
| Depth 4, sparse (5 verify rows) | 39.05 | 2.673 | 69.3 |

- **Microbenchmark** (V = 248320):
  - draft step: 68 µs sparse, against 397 µs dense and 18 µs for the old argmax;
  - verify including readback: 251 µs. It replaces the per-position sampler
    calls and syncs of exact matching.
- **Depth re-sweep.** Depth 2 stays the default.
  - Depth 3 verifies 4 rows. The native high-bit small-M, packed-head and
    MLP-pair kernels support 1, 2, 3 or 5 rows, so 4 rows fall back to the
    slow path.
  - Depth 4 accepts more (2.67 tokens/round) but its 5-row round costs 39 ms.
  - Even with native 4-row kernels, depth 3 would need a round under about
    33 ms to beat depth 2.
- **Gain depends on sampling entropy.** At the serving config the MTP head
  already agrees with the target's top token most of the time, so tokens per
  round barely change. At T=1 / top-k 50 they rise 11.8% on chat prompts
  (1.891 → 2.115) and 21% on the digits prompt (2.111 → 2.553). Low-temperature
  use gains little; creative or high-temperature use gains most.

Runs are in ignored `artifacts/`. Result files are in
`artifacts/spec-sampling-20260928/`.

- `RUN-20260928T173807Z-bbbfd9dc`: spec off.
- `RUN-20260928T174736Z-4583e5fe`: dense.
- `RUN-20260928T175257Z-341dcadd`: depth 3.
- `RUN-20260928T181129Z-c0f5cdbb`: depth 4.
- `RUN-20260928T181451Z-5d2aa07a`: sparse, prefix cache.
- `RUN-20260928T185103Z-faeed32f`: MTP off.
- `RUN-20260928T190753Z-8c2b5cce`: sparse, no prefix cache.
- `RUN-20260928T191536Z-692b299e`: spec off, no prefix cache.
- `RUN-20260928T192931Z-5dc4872a`: fixed, no prefix cache.
- `RUN-20260928T193913Z-d24ba8a1` / `RUN-20260928T195033Z-10d423e8`: final
  on / off at the serving config.
- `RUN-20260928T203611Z-592a66b6` / `RUN-20260928T195415Z-23b38875`: final
  on / off at T=1.

## Limitations

- **Scope.** Only the integrated MTP draft path; not n-gram, DFlash or
  separate draft models. Not used with `--draft-confidence`, whose calibrator
  is built on argmax confidence.
- **Exact relative to the fused sampler's definition.** The fused kernel
  quantizes top-P mass to 2⁻⁴⁰ and bins logits more than 32 nats below the
  maximum. Differences are at the level of exp(−32).
- **Seeds.** A seed reproduces a sampled response only with the same
  speculative setting; spec on and off draw different random streams. Greedy
  output does not depend on the setting.
- **Batching.** One job per generator in serving; the code handles per-row
  settings, but batched sampled MTP is not measured.
