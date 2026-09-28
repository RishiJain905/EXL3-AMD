"""Speculative sampling for MTP drafts at temperature > 0.

The vendored generator drafts with argmax and accepts a draft position only when
the independently sampled target token matches it. That is already exact
(a one-hot proposal q), but at temperature > 0 it rejects whenever the target
samples anything other than its draft's top token.

With this installed, the draft instead samples x ~ q from its own logits under
the job's sampler settings, and verification uses the standard rule
(Leviathan et al. 2023, Chen et al. 2023): accept x with probability
min(1, p(x) / q(x)); on the first rejection emit a sample of
normalize(max(0, p - q)); if every draft is accepted, emit a sample of p at the
bonus position. Emitted tokens are distributed exactly as the target sampler's
p, for any q.

The target distribution p reproduces the fused sampler (``SS_Fused``) that the
serving ``ComboSampler`` collapses to: temperature, then min-P
(``l >= max(l) + minp_log``), top-K (ties at the cutoff kept) and top-P over
the top-K-truncated mass (the tie group crossing the target dropped, the top
token always kept). Greedy, penalties, filters, logit masks, forced tokens,
banned strings and probability outputs keep the inherited exact-match path,
which stays correct for a sampled draft because the draft is drawn
independently of the target's sample.

Fast path: the proposal q only has to be the distribution its token was drawn
from, so the draft uses its top-K candidates (K = top_k, or ``SPARSE_MAX_K``).
With 1 <= top_k <= ``SPARSE_MAX_K`` every token p keeps is among the target's
top-K candidates, except a tie group at the K-th value extending past them;
such a position is flagged and hands the rest of the window to exact matching.
Row i's flag depends only on the prefix already emitted before position i, never on
draft i or later, so the output law is unchanged (``resolve``). Each
step is one ``topk`` plus one Triton kernel; ``filtered_probs`` and
``speculative_verify`` are the dense reference and the path for other top-K
settings.
"""

from __future__ import annotations

from typing import NamedTuple

__all__ = ["FilterConfig", "filter_config", "filtered_probs", "speculative_verify",
           "draft_sparse", "verify_sparse", "begin_round", "reset_stats", "STATS"]

# Process-wide counters, reported in the serve optimization record.
STATS = dict(rounds=0, verified=0, fallback_verifies=0, tie_fallbacks=0)

SPARSE_MAX_K = 64


def reset_stats():
    """Zero the counters in place (the serve record holds this dict)."""
    STATS.update(rounds=0, verified=0, fallback_verifies=0, tie_fallbacks=0)


class FilterConfig(NamedTuple):
    inv_temp: float
    minp_log: float | None
    top_k: int
    top_p: float
    inv_temp_filter: float


def filter_config(sampler) -> FilterConfig | None:
    """Filter settings of a plain fused sampling stack, or None when ineligible."""
    from exllamav3.generator.sampler.custom import CustomSampler, SS_Fused
    # fused_only means no penalty, bias or ban step survived simplification. reqs_past_ids
    # is not checked: it is set before no-op steps (e.g. repetition penalty 1.0) are dropped.
    if (not isinstance(sampler, CustomSampler) or "forward" in vars(sampler)
            or not getattr(sampler, "fused_only", False)):
        return None
    steps = sampler.steps
    if len(steps) != 1 or type(steps[0]) is not SS_Fused:
        return None
    step = steps[0]
    if step.mode == SS_Fused.MODE_SAMPLE:
        return FilterConfig(step.inv_temp, None, 0, 1.0, step.inv_temp_filter)
    if step.mode == SS_Fused.MODE_SAMPLE_MINP:
        return FilterConfig(step.inv_temp, step.minp_log, 0, 1.0, step.inv_temp_filter)
    if step.mode == SS_Fused.MODE_SAMPLE_FILTERS:
        return FilterConfig(
            step.inv_temp,
            step.minp_log if step.filters & SS_Fused.F_MINP else None,
            step.top_k if step.filters & SS_Fused.F_TOPK else 0,
            step.top_p if step.filters & SS_Fused.F_TOPP else 1.0,
            step.inv_temp_filter,
        )
    return None


def sparse_k(cfg: FilterConfig) -> int:
    return cfg.top_k if 0 < cfg.top_k <= SPARSE_MAX_K else SPARSE_MAX_K


def exact_sparse(cfg: FilterConfig) -> bool:
    return 0 < cfg.top_k <= SPARSE_MAX_K


# Dense reference ------------------------------------------------------------------------------

def filtered_probs(logits, cfg: FilterConfig):
    """Normalized sampling distribution for each row of ``logits`` (rows, vocab)."""
    import torch
    l = logits.float()
    m = l.amax(dim=-1, keepdim=True)
    keep = l > float("-inf")
    if cfg.minp_log is not None:
        keep &= l >= m + cfg.minp_log
    vocab = l.shape[-1]
    if (cfg.top_k and cfg.top_k < vocab) or cfg.top_p < 1.0:
        masked = l.masked_fill(~keep, float("-inf"))
        top_k = cfg.top_k and cfg.top_k < vocab
        if top_k:
            # Every token above the K-th value is among the top K candidates, so
            # the kept set is exact including ties at the cutoff.
            cand = torch.topk(masked, cfg.top_k, dim=-1).values
            keep &= l >= cand[..., -1:]
        else:
            cand = torch.sort(masked, dim=-1, descending=True).values
        if cfg.top_p < 1.0:
            z = (torch.exp((l - m) * cfg.inv_temp_filter) * keep).sum(dim=-1, keepdim=True)
            cum = torch.exp((cand - m) * cfg.inv_temp_filter).cumsum(dim=-1)
            # Mass at or above each candidate value, including its whole tie group
            last = torch.searchsorted((-cand).contiguous(), (-cand).contiguous(), right=True) - 1
            mass_ge = cum.gather(-1, last)
            if top_k:
                # The cutoff tie group may extend past the candidates; it closes the set
                mass_ge = torch.where(cand == cand[..., -1:], z, mass_ge)
            allowed = mass_ge <= cfg.top_p * z
            index = (allowed.sum(dim=-1, keepdim=True) - 1).clamp_min(0)
            keep &= l >= cand.gather(-1, index)
    weights = torch.exp((l - m) * cfg.inv_temp) * keep
    return weights / weights.sum(dim=-1, keepdim=True)


def _race(probs, generator):
    """One categorical sample per row: argmax(p / E), E ~ Exp(1), without a sync."""
    import torch
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return torch.argmax(probs / noise, dim=-1)


def speculative_verify(p, q, draft, generator):
    """Accept flags (k,) and per-position samples (k + 1,) for one job.

    ``p`` (k + 1, V) holds target distributions, ``q`` (k, V) the draft
    distributions and ``draft`` (k,) the drafted ids. Sample ``i < k`` is from
    normalize(max(0, p_i - q_i)), the last from p_k.
    """
    import torch
    k = q.shape[0]
    rows = torch.arange(k, device=p.device)
    px = p[rows, draft]
    qx = q[rows, draft]
    u = torch.rand((k,), device=p.device, generator=generator)
    accept = u * qx < px
    residual = torch.cat(((p[:k] - q).clamp_min(0), p[k:]), dim=0)
    total = residual.sum(dim=-1, keepdim=True)
    # p == q leaves no residual mass, and then rejection has probability zero
    residual = torch.where(total > 0, residual, p)
    return accept, _race(residual, generator)


# Sparse torch reference (CPU, tests) ----------------------------------------------------------

def draft_sparse_torch(vals, ids, noise, cfg: FilterConfig):
    """Proposal q over sorted candidates (rows, K) and one sampled token (rows, 1)."""
    import torch
    v = vals.float()
    m = v[:, :1]
    keep = v > float("-inf")
    if cfg.minp_log is not None:
        keep &= v >= m + cfg.minp_log
    if cfg.top_p < 1.0:
        wf = torch.where(keep, torch.exp((v - m) * cfg.inv_temp_filter), 0.0)
        above = (wf[:, None, :] * (v[:, None, :] > v[:, :, None])).sum(-1)
        keep &= (above < cfg.top_p * wf.sum(-1, keepdim=True)) | (v == m)
    w = torch.where(keep, torch.exp((v - m) * cfg.inv_temp), 0.0)
    q = w / w.sum(-1, keepdim=True)
    score = torch.where(q > 0, q / -torch.log(noise.clamp_min(1e-20)), -1.0)
    return q, ids.gather(-1, score.argmax(-1, keepdim=True))


def target_sparse_torch(tv, ties, cfg: FilterConfig):
    """Exact p over sorted top-K target candidates, and a per-row tie fallback flag.

    ``ties`` counts tokens of the full row equal to the K-th candidate value.
    """
    import torch
    v = tv.float()
    m = v[:, :1]
    kth = v[:, -1:]
    keep = v > float("-inf")
    kth_kept = kth > float("-inf")
    if cfg.minp_log is not None:
        keep &= v >= m + cfg.minp_log
        kth_kept &= kth >= m + cfg.minp_log
    extra = torch.where(kth_kept, ties.view(-1, 1) - (v == kth).sum(-1, keepdim=True), 0)
    if cfg.top_p < 1.0:
        wf = torch.where(keep, torch.exp((v - m) * cfg.inv_temp_filter), 0.0)
        z = wf.sum(-1, keepdim=True) + extra * torch.exp((kth - m) * cfg.inv_temp_filter)
        mass_ge = (wf[:, None, :] * (v[:, None, :] >= v[:, :, None])).sum(-1)
        mass_ge = torch.where(v == kth, z, mass_ge)
        allowed = keep & (mass_ge <= cfg.top_p * z)
        vstar = torch.where(allowed, v, float("inf")).amin(-1, keepdim=True)
        vstar = torch.where(allowed.any(-1, keepdim=True), vstar, m)
        keep &= v >= vstar
        extra = torch.where(kth >= vstar, extra, 0)
    w = torch.where(keep, torch.exp((v - m) * cfg.inv_temp), 0.0)
    return w / w.sum(-1, keepdim=True), (extra > 0).view(-1)


def verify_sparse_torch(tv, ti, ties, qd, qi, draft, noise, cfg: FilterConfig):
    """(3, k + 1) long: accept (0-padded), samples, tie fallback flags.

    ``noise`` (k + 1, K + 1) holds uniforms: K for the residual race, the last for acceptance.
    """
    import torch
    p, flag = target_sparse_torch(tv, ties, cfg)
    k, big_k = qd.shape[0], tv.shape[1]
    px = (p[:k] * (ti[:k] == draft[:, None])).sum(-1)
    qx = (qd * (qi == draft[:, None])).sum(-1)
    accept = noise[:k, -1] * qx < px
    qc = (qd[:, None, :] * (ti[:k, :, None] == qi[:, None, :])).sum(-1)
    residual = torch.cat(((p[:k] - qc).clamp_min(0), p[k:]), dim=0)
    residual = torch.where(residual.sum(-1, keepdim=True) > 0, residual, p)
    score = torch.where(residual > 0, residual / -torch.log(noise[:, :big_k].clamp_min(1e-20)), -1.0)
    samples = ti.gather(-1, score.argmax(-1, keepdim=True)).view(-1)
    pad = torch.zeros((samples.shape[0] - k,), dtype=torch.long, device=samples.device)
    accept = torch.cat((accept.long(), pad))
    return torch.stack((accept, samples.long(), flag.long()))


# Triton kernels (GPU) -------------------------------------------------------------------------

_KERNELS = None


def _kernels():
    """Triton versions of the two sparse steps, one launch each, built on first use."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def spec_draft_kernel(vals, ids, noise, q_out, tok_out, K, inv_t, minp_log, top_p, inv_f,
                          HAS_MINP: tl.constexpr, HAS_TOPP: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < K
        v = tl.load(vals + row * K + offs, mask=mask, other=float("-inf")).to(tl.float32)
        t = tl.load(ids + row * K + offs, mask=mask, other=-1)
        m = tl.max(v, 0)
        keep = v > float("-inf")
        if HAS_MINP:
            keep = keep & (v >= m + minp_log)
        if HAS_TOPP:
            wf = tl.where(keep, tl.exp((v - m) * inv_f), 0.0)
            above = tl.sum(tl.where(v[None, :] > v[:, None], wf[None, :], 0.0), 1)
            keep = keep & ((above < top_p * tl.sum(wf, 0)) | (v == m))
        w = tl.where(keep, tl.exp((v - m) * inv_t), 0.0)
        q = w / tl.sum(w, 0)
        u = tl.load(noise + row * K + offs, mask=mask, other=0.5)
        score = tl.where(q > 0, q / -tl.log(tl.maximum(u, 1e-20)), -1.0)
        j = tl.argmax(score, 0)
        tl.store(q_out + row * K + offs, q, mask=mask)
        tl.store(tok_out + row, tl.sum(tl.where(offs == j, t, 0), 0))

    @triton.jit
    def spec_verify_kernel(tv, ti, ties, qd, qi, draft, noise, out, rows, K, KD, NDRAFT,
                           inv_t, minp_log, top_p, inv_f,
                           HAS_MINP: tl.constexpr, HAS_TOPP: tl.constexpr,
                           BLOCK: tl.constexpr, BLOCK_D: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < K
        v = tl.load(tv + row * K + offs, mask=mask, other=float("-inf")).to(tl.float32)
        t = tl.load(ti + row * K + offs, mask=mask, other=-1)
        m = tl.max(v, 0)
        kth = tl.min(tl.where(mask, v, float("inf")), 0)
        keep = v > float("-inf")
        kth_kept = kth > float("-inf")
        if HAS_MINP:
            keep = keep & (v >= m + minp_log)
            kth_kept = kth_kept & (kth >= m + minp_log)
        extra = tl.load(ties + row) - tl.sum((mask & (v == kth)).to(tl.int64), 0)
        extra = tl.where(kth_kept, extra, 0)
        if HAS_TOPP:
            wf = tl.where(keep, tl.exp((v - m) * inv_f), 0.0)
            z = tl.sum(wf, 0) + extra.to(tl.float32) * tl.exp((kth - m) * inv_f)
            mass_ge = tl.sum(tl.where(v[None, :] >= v[:, None], wf[None, :], 0.0), 1)
            mass_ge = tl.where(v == kth, z, mass_ge)
            allowed = keep & (mass_ge <= top_p * z)
            vstar = tl.min(tl.where(allowed, v, float("inf")), 0)
            vstar = tl.where(tl.sum(allowed.to(tl.int32), 0) > 0, vstar, m)
            keep = keep & (v >= vstar)
            extra = tl.where(kth >= vstar, extra, 0)
        w = tl.where(keep, tl.exp((v - m) * inv_t), 0.0)
        p = w / tl.sum(w, 0)
        is_draft = row < NDRAFT
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = (offs_d < KD) & is_draft
        dq = tl.load(qd + row * KD + offs_d, mask=mask_d, other=0.0)
        dt = tl.load(qi + row * KD + offs_d, mask=mask_d, other=-2)
        x = tl.load(draft + row, mask=is_draft, other=-3)
        px = tl.sum(tl.where(t == x, p, 0.0), 0)
        qx = tl.sum(tl.where(dt == x, dq, 0.0), 0)
        u0 = tl.load(noise + row * (K + 1) + K)
        qc = tl.sum(tl.where(t[:, None] == dt[None, :], dq[None, :], 0.0), 1)
        r = tl.maximum(p - qc, 0.0)
        r = tl.where(tl.sum(r, 0) > 0, r, p)
        u = tl.load(noise + row * (K + 1) + offs, mask=mask, other=0.5)
        score = tl.where(r > 0, r / -tl.log(tl.maximum(u, 1e-20)), -1.0)
        j = tl.argmax(score, 0)
        tl.store(out + row, (is_draft & (u0 * qx < px)).to(tl.int64))
        tl.store(out + rows + row, tl.sum(tl.where(offs == j, t, 0), 0))
        tl.store(out + 2 * rows + row, (extra > 0).to(tl.int64))

    _KERNELS = (spec_draft_kernel, spec_verify_kernel)
    return _KERNELS


def _block(k):
    return max(16, 1 << (k - 1).bit_length())


def _use_triton(tensor):
    if not tensor.is_cuda:
        return False
    try:
        _kernels()
    except ImportError:
        return False
    return True


def draft_sparse(vals, ids, noise, cfg: FilterConfig):
    """Proposal q (rows, K) and sampled token (rows, 1) from sorted top-K candidates."""
    if not _use_triton(vals):
        return draft_sparse_torch(vals, ids, noise, cfg)
    import torch
    kernel, _ = _kernels()
    rows, k = vals.shape
    q = torch.empty((rows, k), dtype=torch.float32, device=vals.device)
    tokens = torch.empty((rows, 1), dtype=torch.long, device=vals.device)
    kernel[(rows,)](vals.contiguous(), ids.contiguous(), noise.contiguous(), q, tokens, k,
                    cfg.inv_temp, cfg.minp_log or 0.0, cfg.top_p, cfg.inv_temp_filter,
                    HAS_MINP=cfg.minp_log is not None, HAS_TOPP=cfg.top_p < 1.0, BLOCK=_block(k))
    return q, tokens


def verify_sparse(tv, ti, ties, qd, qi, draft, noise, cfg: FilterConfig):
    """(3, k + 1) long rows: accept (0-padded), samples, tie fallback flags."""
    if not _use_triton(tv):
        return verify_sparse_torch(tv, ti, ties, qd, qi, draft, noise, cfg)
    import torch
    _, kernel = _kernels()
    rows, k = tv.shape
    out = torch.empty((3, rows), dtype=torch.long, device=tv.device)
    kernel[(rows,)](tv.contiguous(), ti.contiguous(), ties.contiguous(), qd.contiguous(), qi.contiguous(),
                    draft.contiguous(), noise.contiguous(), out, rows, k, qd.shape[1], qd.shape[0],
                    cfg.inv_temp, cfg.minp_log or 0.0, cfg.top_p, cfg.inv_temp_filter,
                    HAS_MINP=cfg.minp_log is not None, HAS_TOPP=cfg.top_p < 1.0,
                    BLOCK=_block(k), BLOCK_D=_block(qd.shape[1]))
    return out


def warm_kernels(device):
    """Compile both kernels for every logit dtype, block size, K divisibility class (Triton
    specializes integers on % 16) and filter flag set; returns the number of launches.

    Synthetic inputs only. Serve calls this during warm-up so a sampled request never
    compiles after ready, whatever the serve sampling defaults are.
    """
    import torch
    launches = 0
    for dtype in (torch.half, torch.float):
        for k in (12, 16, 20, 32, 50, 64):
            for minp_log in (None, -1.0):
                for top_p in (1.0, 0.9):
                    cfg = FilterConfig(1.0, minp_log, k, top_p, 1.0)
                    logits = torch.randn((3, 256), device=device).to(dtype)
                    vals, ids = torch.topk(logits[:1], k, dim=-1)
                    q, token = draft_sparse(vals, ids, torch.rand(vals.shape, device=device), cfg)
                    tv, ti = torch.topk(logits, k, dim=-1)
                    ties = torch.ones((3,), dtype=torch.long, device=device)
                    verify_sparse(tv, ti, ties, q.expand(2, k), ids.expand(2, k), token.view(1).expand(2),
                                  torch.rand((3, k + 1), device=device), cfg)
                    launches += 2
    return launches


# Generator integration ------------------------------------------------------------------------

def _job_eligible(job) -> bool:
    return (job.new_tokens >= job.min_new_tokens and job.new_tokens >= 0
            and len(job.sequences) == 1 and job.forced_ids is None
            and not job.filters and not job.banned_strings and not job.checkpoint
            and not job.return_probs and not job.return_top_tokens)


class SpecRound:
    """Sparse draft proposals for one MTP round; consumed by the verification step."""

    def __init__(self, jobs, configs, vocab):
        self.jobs = jobs
        self.configs = configs
        self.vocab = vocab
        self.q = [[] for _ in jobs]
        self.qi = [[] for _ in jobs]
        self.ids = [[] for _ in jobs]

    def _generator(self, job, device):
        import torch
        gen = getattr(job, "quantlab_spec_generator", None)
        if gen is None or gen.device != device:
            gen = torch.Generator(device=device)
            gen.manual_seed(job.rng.randint(0, (1 << 63) - 1))
            job.quantlab_spec_generator = gen
        return gen

    def draft_step(self, draft_model, state, params):
        """Sample one draft token per row from its top-K proposal."""
        import torch
        target = draft_model.attached_model()
        head = target.modules[target.logit_layer_idx]
        logits = head.forward(head.prepare_for_device(state, params), params)
        logits = logits[:, -1, :self.vocab]
        out = []
        for row, (job, cfg) in enumerate(zip(self.jobs, self.configs)):
            vals, ids = torch.topk(logits[row:row + 1], min(sparse_k(cfg), logits.shape[-1]), dim=-1)
            noise = torch.rand(vals.shape, device=vals.device, generator=self._generator(job, vals.device))
            q, token = draft_sparse(vals, ids, noise, cfg)
            self.q[row].append(q)
            self.qi[row].append(ids)
            self.ids[row].append(token.view(1))
            out.append(token.view(1))
        return torch.stack(out, dim=0)

    def verify(self, job, job_logits):
        """``(tokens, final)`` for ``job``, or None for inherited exact matching.

        ``tokens`` (1, m) are resolved positions. ``final`` means the last one ends the
        window (a rejection sample or the bonus token); otherwise all m are accepted
        drafts and exact matching continues from position m (see ``resolve``).
        """
        import torch
        try:
            row = next(i for i, j in enumerate(self.jobs) if j is job)
        except StopIteration:
            return None
        k = job_logits.shape[1] - 1
        if not 0 < k <= len(self.q[row]) or job.device_logit_mask is not None or not _job_eligible(job):
            STATS["fallback_verifies"] += 1
            return None
        cfg = self.configs[row]
        logits = job_logits[0, :, :self.vocab]
        q = torch.cat(self.q[row][:k], dim=0)
        qi = torch.cat(self.qi[row][:k], dim=0)
        draft = torch.cat(self.ids[row][:k], dim=0)
        gen = self._generator(job, logits.device)
        if exact_sparse(cfg):
            top_k = min(cfg.top_k, logits.shape[-1])
            tv, ti = torch.topk(logits, top_k, dim=-1)
            ties = (logits == tv[:, -1:]).sum(-1)
            noise = torch.rand((k + 1, top_k + 1), device=logits.device, generator=gen)
            host = verify_sparse(tv, ti, ties, q, qi, draft, noise, cfg).cpu()
            accept, samples, flags = host[0], host[1], host[2]
        else:
            p = filtered_probs(logits, cfg)
            dense = torch.zeros((k, logits.shape[-1]), dtype=torch.float32, device=logits.device)
            dense.scatter_(-1, qi, q)
            accept, samples = speculative_verify(p, dense, draft, gen)
            host = torch.cat((accept.long(), samples)).cpu()
            accept, samples, flags = host[:k], host[k:], torch.zeros(k + 1, dtype=torch.long)
        tokens, final = resolve(accept, samples, flags, draft.cpu())
        if not final:
            STATS["tie_fallbacks"] += 1
            if tokens.numel() == 0:
                return None
        STATS["verified"] += 1
        return tokens.view(1, -1), final


def resolve(accept, samples, flags, draft):
    """Walk the window in order: ``(tokens, final)``.

    Position i uses the speculative outcome unless row i is flagged (p not exact on the
    sparse candidates). A flag hands position i onward to exact matching, keeping the
    drafts accepted before it. The flag of row i depends only on the prefix emitted before
    position i (row i's logits see drafts 0..i-1, all accepted by then), never on draft i
    or later, so the switch cannot bias the speculative decisions made before it.
    Flags of rows after the ending position are ignored for the same reason.
    """
    import torch
    k = draft.shape[0]
    for i in range(k + 1):
        if flags[i]:
            return draft[:i], False
        if i == k or not accept[i]:
            return torch.cat((draft[:i], samples[i:i + 1])), True
    raise AssertionError("unreachable")


def begin_round(generator, jobs):
    """A SpecRound when every drafting job has a plain sampling stack, else None."""
    draft = generator.draft_model
    if (not jobs or draft is None or getattr(draft, "loaded_tp", False)
            or getattr(draft.sample_from_state, "__self__", None) is not draft):
        return None
    configs = []
    for job in jobs:
        cfg = filter_config(job.sampler)
        if cfg is None or not _job_eligible(job):
            return None
        configs.append(cfg)
    STATS["rounds"] += 1
    return SpecRound(jobs, configs, generator.tokenizer.actual_vocab_size)
