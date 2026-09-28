"""Speculative sampling: filter semantics, output distribution and draft-loop wiring."""
import ast
import importlib
import math
from pathlib import Path
import sys
import textwrap
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / 'vendor/rocm-exl3/exllamav3/generator/generator.py'


def _stub_sampler_module():
    """Stand-ins for the vendored CustomSampler/SS_Fused (which import the native extension)."""
    module = ModuleType('exllamav3.generator.sampler.custom')

    class CustomSampler:
        def __init__(self, steps, fused_only=True, reqs_past_ids=False):
            self.steps, self.fused_only, self.reqs_past_ids = steps, fused_only, reqs_past_ids

    class SS_Fused:
        MODE_GREEDY, MODE_SAMPLE, MODE_SAMPLE_MINP, MODE_SAMPLE_FILTERS = 0, 1, 2, 3
        F_TOPK, F_TOPP, F_MINP = 1, 2, 4

        def __init__(self, mode, temperature=1.0, minp_log=0.0, filters=0, top_k=0, top_p=1.0,
                     temp_first=False):
            self.mode, self.inv_temp, self.minp_log = mode, 1.0 / temperature, minp_log
            self.filters, self.top_k, self.top_p = filters, top_k, top_p
            self.inv_temp_filter = self.inv_temp if temp_first else 1.0

    module.CustomSampler, module.SS_Fused = CustomSampler, SS_Fused
    return module


def _serve_sampler(module, temperature=0.6, top_k=20, top_p=0.95):
    """What ComboSampler(temperature, top_k, top_p) collapses to."""
    fused = module.SS_Fused
    step = fused(fused.MODE_SAMPLE_FILTERS, temperature, 0.0, fused.F_TOPK | fused.F_TOPP,
                 top_k, top_p, temp_first=True)
    # The real ComboSampler sets reqs_past_ids from its no-op repetition penalty step
    return module.CustomSampler([step], reqs_past_ids=True)


def _reference_probs(row, cfg):
    """Sort-based eager rule, one row of Python floats."""
    m = max(row)
    keep = [x > -math.inf for x in row]
    if cfg.minp_log is not None:
        keep = [k and x >= m + cfg.minp_log for k, x in zip(keep, row)]
    kept = sorted((x for k, x in zip(keep, row) if k), reverse=True)
    if cfg.top_k and cfg.top_k < len(row):
        cutoff = kept[min(cfg.top_k, len(kept)) - 1]
        keep = [k and x >= cutoff for k, x in zip(keep, row)]
        kept = [x for x in kept if x >= cutoff]
    if cfg.top_p < 1.0:
        z = sum(math.exp((x - m) * cfg.inv_temp_filter) for x in kept)
        bound = kept[0]
        for value in sorted(set(kept), reverse=True):
            mass = sum(math.exp((x - m) * cfg.inv_temp_filter) for x in kept if x >= value)
            if mass > cfg.top_p * z:
                break
            bound = value
        keep = [k and x >= bound for k, x in zip(keep, row)]
    weights = [math.exp((x - m) * cfg.inv_temp) if k else 0.0 for k, x in zip(keep, row)]
    total = sum(weights)
    return [w / total for w in weights]


@unittest.skipIf(torch is None, 'CPU Torch required')
class FilterTests(unittest.TestCase):
    def setUp(self):
        self.spec = importlib.import_module('quantlab.methods.exl3.spec_sampling')

    def test_matches_sort_reference_including_ties(self):
        cfg = self.spec.FilterConfig
        configs = (cfg(1 / 0.6, None, 20, 0.95, 1 / 0.6), cfg(1.0, None, 0, 0.9, 1.0),
                   cfg(1 / 0.8, 0.8 * math.log(0.05), 0, 1.0, 1 / 0.8), cfg(1.0, None, 5, 1.0, 1.0),
                   cfg(1 / 1.3, math.log(0.1), 7, 0.5, 1 / 1.3), cfg(2.0, None, 0, 1.0, 2.0))
        gen = torch.Generator().manual_seed(3)
        for c in configs:
            for trial in range(6):
                # A coarse grid forces exact ties, including at the cutoffs
                logits = torch.round(torch.randn((3, 300), generator=gen) * 3 * 4) / 4
                logits[:, -7:] = -math.inf
                got = self.spec.filtered_probs(logits, c)
                for r in range(3):
                    want = torch.tensor(_reference_probs(logits[r].tolist(), c))
                    with self.subTest(cfg=c, trial=trial, row=r):
                        self.assertTrue(torch.equal(got[r] > 0, want > 0))
                        self.assertLess(float((got[r] - want).abs().max()), 1e-6)

    def test_sparse_target_matches_dense_or_flags_a_kept_tie_group(self):
        cfg = self.spec.FilterConfig
        configs = (cfg(1 / 0.6, None, 20, 0.95, 1 / 0.6), cfg(1.0, None, 5, 1.0, 1.0),
                   cfg(1 / 1.3, math.log(0.1), 7, 0.5, 1 / 1.3), cfg(1 / 0.8, -1.0, 12, 0.9, 1 / 0.8))
        gen = torch.Generator().manual_seed(9)
        flagged = exact = 0
        for c in configs:
            for trial in range(30):
                # Coarse grids make cutoff ties common; fine ones make them rare
                scale = 4 if trial % 2 else 256
                logits = torch.round(torch.randn((3, 300), generator=gen) * 3 * scale) / scale
                dense = self.spec.filtered_probs(logits, c)
                tv, ti = torch.topk(logits, c.top_k, dim=-1)
                ties = (logits == tv[:, -1:]).sum(-1)
                sparse, flag = self.spec.target_sparse_torch(tv, ties, c)
                for r in range(3):
                    outside = int((dense[r] > 0).sum() - (dense[r].gather(0, ti[r]) > 0).sum())
                    with self.subTest(cfg=c, trial=trial, row=r):
                        # Flag exactly when p keeps tokens outside the candidates
                        self.assertEqual(bool(flag[r]), outside > 0)
                        if not flag[r]:
                            self.assertLess(float((sparse[r] - dense[r].gather(0, ti[r])).abs().max()), 1e-6)
                    flagged += bool(flag[r])
                    exact += not flag[r]
        self.assertGreater(flagged, 0)
        self.assertGreater(exact, flagged)

    def test_top_token_group_is_always_kept(self):
        cfg = self.spec.FilterConfig(1.0, None, 0, 0.1, 1.0)
        probs = self.spec.filtered_probs(torch.tensor([[5.0, 5.0, 1.0, 0.0]]), cfg)
        self.assertEqual(probs.tolist(), [[0.5, 0.5, 0.0, 0.0]])

    def test_config_from_fused_stack_only(self):
        module = _stub_sampler_module()
        fused = module.SS_Fused
        with patch.dict(sys.modules, {'exllamav3.generator.sampler.custom': module}):
            cfg = self.spec.filter_config(_serve_sampler(module))
            self.assertEqual(cfg, (1 / 0.6, None, 20, 0.95, 1 / 0.6))
            minp = self.spec.filter_config(module.CustomSampler([fused(fused.MODE_SAMPLE_MINP, 0.7, -2.0)]))
            self.assertEqual((minp.minp_log, minp.top_k, minp.top_p), (-2.0, 0, 1.0))
            self.assertIsNone(self.spec.filter_config(module.CustomSampler([fused(fused.MODE_GREEDY)])))
            penalised = _serve_sampler(module)
            penalised.fused_only = False
            self.assertIsNone(self.spec.filter_config(penalised))
            self.assertIsNone(self.spec.filter_config(module.CustomSampler([fused(1), fused(1)])))
            self.assertIsNone(self.spec.filter_config(object()))


def _real_combo_sampler():
    try:
        from exllamav3.generator.sampler.presets import ComboSampler
    except Exception:
        return None
    return ComboSampler


@unittest.skipIf(torch is None or _real_combo_sampler() is None,
                 'needs the vendored backend and extension on sys.path')
class RealSamplerTests(unittest.TestCase):
    def test_serving_combo_sampler_is_eligible_and_penalties_are_not(self):
        spec = importlib.import_module('quantlab.methods.exl3.spec_sampling')
        combo = _real_combo_sampler()
        serving = combo(rep_p=1.0, freq_p=0.0, pres_p=0.0, temperature=0.6, min_p=0.0, top_k=20, top_p=0.95)
        self.assertTrue(serving.reqs_past_ids)  # set by the no-op penalty steps
        self.assertEqual(spec.filter_config(serving), (1 / 0.6, None, 20, 0.95, 1 / 0.6))
        self.assertIsNone(spec.filter_config(combo(rep_p=1.1, temperature=0.6, top_k=20)))
        self.assertIsNone(spec.filter_config(combo(temperature=0.0)))


@unittest.skipIf(torch is None, 'CPU Torch required')
class DistributionTests(unittest.TestCase):
    """Two emitted tokens, sampled round by round, must follow the target's joint law."""
    V = 4

    def setUp(self):
        self.spec = importlib.import_module('quantlab.methods.exl3.spec_sampling')
        gen = torch.Generator().manual_seed(11)
        dist = lambda n: torch.softmax(torch.randn((n, self.V), generator=gen) * 1.5, dim=-1)
        # Distributions conditioned on a prefix of up to two tokens: index 0 = empty prefix
        self.p = {(): dist(1)[0]}
        self.q = {(): dist(1)[0]}
        for a in range(self.V):
            self.p[(a,)], self.q[(a,)] = dist(1)[0], dist(1)[0]
            for b in range(self.V):
                self.p[(a, b)], self.q[(a, b)] = dist(1)[0], dist(1)[0]

    def sample_pair(self, verify, gen):
        out = []
        while len(out) < 2:
            prefix = tuple(out)
            drafts, qs = [], []
            for _ in range(2):
                q = self.q[(prefix + tuple(drafts))[:2]]
                drafts.append(int(torch.multinomial(q, 1, generator=gen)))
                qs.append(q)
            ps = [self.p[(prefix + tuple(drafts[:i]))[:2]] for i in range(3)]
            accept, samples = verify(torch.stack(ps), torch.stack(qs), torch.tensor(drafts), gen)
            n = 0
            while n < 2 and accept[n]:
                n += 1
            out += drafts[:n] + [int(samples[n])]
        return tuple(out[:2])

    def joint_tv(self, verify, trials=12000):
        gen = torch.Generator().manual_seed(5)
        counts = {}
        for _ in range(trials):
            pair = self.sample_pair(verify, gen)
            counts[pair] = counts.get(pair, 0) + 1
        tv = 0.0
        for a in range(self.V):
            for b in range(self.V):
                want = float(self.p[()][a] * self.p[(a,)][b])
                tv += abs(counts.get((a, b), 0) / trials - want)
        return tv / 2

    def sparse_rule(self, p, q, draft, gen):
        """The sparse path on the same distributions (logits log p / log q, no filters)."""
        k = q.shape[0]
        cfg = self.spec.FilterConfig(1.0, None, self.V, 1.0, 1.0)
        tv, ti = torch.topk(p.log(), self.V, dim=-1)
        ties = (p.log() == tv[:, -1:]).sum(-1)
        qv, qi = torch.topk(q.log(), self.V, dim=-1)
        qd, _ = self.spec.draft_sparse_torch(qv, qi, torch.rand(qv.shape, generator=gen), cfg)
        noise = torch.rand((k + 1, self.V + 1), generator=gen)
        out = self.spec.verify_sparse_torch(tv, ti, ties, qd, qi, draft, noise, cfg)
        self.assertFalse(out[2].any())
        return out[0][:k].bool(), out[1]

    def test_joint_distribution_matches_target(self):
        self.assertLess(self.joint_tv(self.spec.speculative_verify), 0.025)

    def test_sparse_path_joint_distribution_matches_target(self):
        self.assertLess(self.joint_tv(self.sparse_rule), 0.025)

    def test_detects_a_wrong_acceptance_rule(self):
        def always_accept(p, q, draft, gen):
            accept, samples = self.spec.speculative_verify(p, q, draft, gen)
            return torch.ones_like(accept), samples
        self.assertGreater(self.joint_tv(always_accept, trials=3000), 0.1)

    def test_rejection_sample_never_repeats_the_rejected_draft(self):
        p = torch.tensor([[0.1, 0.9, 0.0], [0.3, 0.3, 0.4]])
        q = torch.tensor([[1.0, 0.0, 0.0]])
        gen = torch.Generator().manual_seed(1)
        for _ in range(200):
            accept, samples = self.spec.speculative_verify(p, q, torch.tensor([0]), gen)
            if not accept[0]:
                self.assertEqual(int(samples[0]), 1)


@unittest.skipIf(torch is None, 'CPU Torch required')
class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.spec = importlib.import_module('quantlab.methods.exl3.spec_sampling')

    def resolve(self, accept, samples, flags, draft=(7, 8)):
        tokens, final = self.spec.resolve(torch.tensor(accept), torch.tensor(samples), torch.tensor(flags),
                                          torch.tensor(draft))
        return tokens.tolist(), final

    def test_window_outcomes(self):
        self.assertEqual(self.resolve([1, 1, 0], [1, 2, 3], [0, 0, 0]), ([7, 8, 3], True))
        self.assertEqual(self.resolve([1, 0, 0], [1, 2, 3], [0, 0, 0]), ([7, 2], True))
        self.assertEqual(self.resolve([0, 1, 0], [1, 2, 3], [0, 0, 0]), ([1], True))

    def test_flag_hands_over_at_its_position_and_later_flags_are_ignored(self):
        self.assertEqual(self.resolve([1, 1, 0], [1, 2, 3], [0, 1, 0]), ([7], False))
        self.assertEqual(self.resolve([1, 1, 0], [1, 2, 3], [0, 0, 1]), ([7, 8], False))
        self.assertEqual(self.resolve([1, 1, 0], [1, 2, 3], [1, 0, 0]), ([], False))
        # A rejection at 0 ends the window before the flagged rows
        self.assertEqual(self.resolve([0, 1, 0], [1, 2, 3], [0, 1, 1]), ([1], True))


@unittest.skipIf(torch is None, 'CPU Torch required')
class TieHandoffDistributionTests(unittest.TestCase):
    """Cutoff ties that depend on the drafted token must not bias the output."""
    V, K = 6, 3

    def setUp(self):
        self.spec = importlib.import_module('quantlab.methods.exl3.spec_sampling')
        self.cfg = self.spec.FilterConfig(1.0, None, self.K, 1.0, 1.0)
        gen = torch.Generator().manual_seed(21)
        distinct = lambda: torch.randperm(self.V, generator=gen).float() * 0.7
        # The draft reverses the target's order of the top three; tokens 0 and 2 are even
        self.target = {(): torch.tensor([2.0, 1.5, 1.0, -1.0, -1.5, -2.0])}
        self.draft = {(): torch.tensor([1.0, 1.5, 2.0, -1.0, -1.5, -2.0])}
        for a in range(self.V):
            # After an even token the third-ranked value is tied past the candidates: flagged
            self.target[(a,)] = (torch.tensor([2.0, 1.2, 0.5, 0.5, 0.5, -1.0])[torch.randperm(self.V, generator=gen)]
                                 if a % 2 == 0 else distinct())
            self.draft[(a,)] = distinct()

    def logits(self, table, prefix):
        return table[prefix[:1]] if prefix else table[()]

    def p(self, prefix):
        return self.spec.filtered_probs(self.logits(self.target, prefix).view(1, -1), self.cfg)[0]

    def first_tv(self, old_rule, trials=6000):
        gen = torch.Generator().manual_seed(8)
        counts = torch.zeros(self.V)
        for _ in range(trials):
            drafts, qs, qis = [], [], []
            for _ in range(2):
                vals, ids = torch.topk(self.logits(self.draft, tuple(drafts)).view(1, -1), self.K)
                q, tok = self.spec.draft_sparse_torch(vals, ids, torch.rand(vals.shape, generator=gen), self.cfg)
                drafts.append(int(tok)); qs.append(q); qis.append(ids)
            rows = torch.stack([self.logits(self.target, tuple(drafts[:i])) for i in range(3)])
            tv, ti = torch.topk(rows, self.K)
            ties = (rows == tv[:, -1:]).sum(-1)
            out = self.spec.verify_sparse_torch(tv, ti, ties, torch.cat(qs), torch.cat(qis), torch.tensor(drafts),
                                               torch.rand((3, self.K + 1), generator=gen), self.cfg)
            if old_rule and out[2].any():
                tokens, final = torch.tensor([], dtype=torch.long), False
            else:
                tokens, final = self.spec.resolve(out[0], out[1], out[2], torch.tensor(drafts))
            if tokens.numel():
                first = int(tokens[0])
            else:
                # Exact matching from position 0: the target sample is emitted
                first = int(torch.multinomial(self.p(()), 1, generator=gen))
            counts[first] += 1
        return float((counts / trials - self.p(())).abs().sum() / 2)

    def test_per_position_handoff_is_exact(self):
        self.assertLess(self.first_tv(old_rule=False), 0.02)

    def test_whole_round_fallback_would_be_biased(self):
        self.assertGreater(self.first_tv(old_rule=True), 0.04)


class FakeTarget:
    def __init__(self, logits):
        self.logits = logits
        head = SimpleNamespace(prepare_for_device=lambda state, params: state,
                               forward=lambda x, params: self.logits.expand(x.shape[0], 1, -1).clone())
        self.modules, self.logit_layer_idx = [head], 0


class FakeDraft:
    def __init__(self, target):
        self.target, self.inputs, self.loaded_tp = target, [], False

    def attached_model(self):
        return self.target

    def forward(self, ids, params):
        self.inputs.append(ids.clone())
        return torch.zeros((ids.shape[0], 1, 4))

    def sample_from_state(self, state, params):
        return torch.argmax(self.target.logits[:, -1], dim=-1).view(1, 1)


def _load_draft_loop():
    source = GENERATOR.read_text(encoding='utf-8')
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == 'iterate_draftmodel_mtp_gen')
    namespace = dict(torch=torch, time=time, PAGE_SIZE=256, cuda_sync_active=lambda: None)
    exec(compile(textwrap.dedent(ast.get_source_segment(source, method)), str(GENERATOR), 'exec'), namespace)
    return namespace[method.name]


@unittest.skipIf(torch is None, 'CPU Torch required')
class DraftLoopTests(unittest.TestCase):
    def setUp(self):
        self.spec = importlib.import_module('quantlab.methods.exl3.spec_sampling')
        self.module = _stub_sampler_module()
        self.patch = patch.dict(sys.modules, {'exllamav3.generator.sampler.custom': self.module})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.spec.reset_stats()

    def run_round(self, sampler, enabled=True, **job_fields):
        import random
        # Flat logits over three tokens: argmax is always token 0
        target = FakeTarget(torch.tensor([[[1.0, 1.0, 1.0, -math.inf]]]))
        draft = FakeDraft(target)
        seq = SimpleNamespace(block_index_tensor=torch.zeros((1, 1), dtype=torch.int32), kv_position=5)
        fields = dict(is_prefill_done=lambda: True, get_max_seq_len=lambda: 5, sequences=[seq],
                      mtp_last_hidden=torch.zeros((1, 1, 4)), time_first_token=1,
                      get_input_ids_list=lambda: [torch.tensor([[7]])], sampler=sampler,
                      rng=random.Random(0), new_tokens=3, min_new_tokens=0, forced_ids=None,
                      filters=[], banned_strings=[], checkpoint=None, return_probs=False,
                      return_top_tokens=0, device_logit_mask=None)
        fields.update(job_fields)
        job = SimpleNamespace(**fields)
        gen = SimpleNamespace(active_jobs=[job], num_draft_tokens=2, draft_calibrator=None,
                              draft_model=draft, draft_cache=None, model=target,
                              tokenizer=SimpleNamespace(actual_vocab_size=3),
                              draft_input_ids_pinned=torch.zeros((1, 1), dtype=torch.long),
                              draft_ids_pinned=torch.zeros((1, 2), dtype=torch.long),
                              quantlab_spec_sampling=enabled)
        drafts = [_load_draft_loop()(gen, []).clone() for _ in range(40)]
        return gen, job, drafts

    def test_sampled_requests_draft_from_the_top_k_proposal(self):
        gen, job, drafts = self.run_round(_serve_sampler(self.module, top_k=2, top_p=1.0))
        seen = {int(t) for d in drafts for t in d.view(-1)}
        # Two of the three tied tokens form the proposal; 3 is past the vocabulary
        self.assertEqual(len(seen), 2)
        self.assertTrue(seen <= {0, 1, 2})
        self.assertEqual(self.spec.STATS['rounds'], 40)
        round_ = gen.quantlab_spec_round
        self.assertEqual(len(round_.q[0]), 2)
        # p equals the proposal on the same two tokens, so every draft is accepted
        target = torch.full((1, 3, 3), -5.0)
        for token in seen:
            target[..., token] = 1.0
        tokens, final = round_.verify(job, target)
        self.assertTrue(final)
        self.assertEqual(tokens.shape, (1, 3))
        self.assertTrue(torch.equal(tokens[0, :2], drafts[-1][0]))
        self.assertIn(int(tokens[0, 2]), seen)

    def test_verify_falls_back_when_a_kept_tie_group_passes_the_candidates(self):
        gen, job, _ = self.run_round(_serve_sampler(self.module, top_k=2, top_p=1.0))
        self.assertIsNone(gen.quantlab_spec_round.verify(job, torch.zeros((1, 3, 3))))
        self.assertEqual(self.spec.STATS['tie_fallbacks'], 1)

    def test_greedy_and_ineligible_jobs_keep_argmax_drafts(self):
        fused = self.module.SS_Fused
        cases = (dict(sampler=self.module.CustomSampler([fused(fused.MODE_GREEDY)])),
                 dict(sampler=_serve_sampler(self.module), filters=[object()]),
                 dict(sampler=_serve_sampler(self.module), return_probs=True),
                 dict(sampler=_serve_sampler(self.module), min_new_tokens=9),
                 dict(sampler=_serve_sampler(self.module), enabled=False))
        for fields in cases:
            with self.subTest(fields=sorted(fields)):
                gen, _, drafts = self.run_round(**fields)
                self.assertIsNone(gen.quantlab_spec_round)
                self.assertTrue(all(d.view(-1).tolist() == [0, 0] for d in drafts))

    def test_verify_falls_back_when_a_mask_appears(self):
        gen, job, _ = self.run_round(_serve_sampler(self.module))
        job.device_logit_mask = torch.zeros((1, 3))
        self.assertIsNone(gen.quantlab_spec_round.verify(job, torch.zeros((1, 3, 3))))
        self.assertEqual(self.spec.STATS['fallback_verifies'], 1)


def _gpu_triton():
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_gpu_triton(), 'GPU and Triton required')
class TritonKernelTests(unittest.TestCase):
    """The Triton kernels must reproduce the torch references given the same uniforms."""

    def test_kernels_match_references(self):
        spec = importlib.import_module('quantlab.methods.exl3.spec_sampling')
        cfg = spec.FilterConfig
        configs = (cfg(1 / 0.6, None, 20, 0.95, 1 / 0.6), cfg(1.0, None, 5, 1.0, 1.0),
                   cfg(1 / 1.3, math.log(0.1), 7, 0.5, 1 / 1.3), cfg(1 / 0.8, -1.0, 64, 0.9, 1 / 0.8))
        gen = torch.Generator(device='cuda').manual_seed(4)
        for c in configs:
            for trial in range(20):
                scale = 4 if trial % 2 else 256
                target = (torch.round(torch.randn((3, 4096), device='cuda', generator=gen) * 3 * scale)
                          / scale).half()
                draft_logits = (target[:2].float() + torch.randn((2, 4096), device='cuda', generator=gen)).half()
                dv, di = torch.topk(draft_logits, spec.sparse_k(c), dim=-1)
                dnoise = torch.rand(dv.shape, device='cuda', generator=gen)
                q, tok = spec.draft_sparse(dv, di, dnoise, c)
                q_ref, tok_ref = spec.draft_sparse_torch(dv, di, dnoise, c)
                tv, ti = torch.topk(target, c.top_k, dim=-1)
                ties = (target == tv[:, -1:]).sum(-1)
                noise = torch.rand((3, c.top_k + 1), device='cuda', generator=gen)
                out = spec.verify_sparse(tv, ti, ties, q, di, tok.view(-1), noise, c)
                ref = spec.verify_sparse_torch(tv, ti, ties, q_ref, di, tok_ref.view(-1), noise, c)
                with self.subTest(cfg=c, trial=trial):
                    self.assertLess(float((q - q_ref).abs().max()), 1e-5)
                    self.assertTrue(torch.equal(tok, tok_ref))
                    self.assertTrue(torch.equal(out, ref), (out, ref))


@unittest.skipIf(torch is None, 'CPU Torch required')
class WarmKernelTests(unittest.TestCase):
    def test_warm_up_enumerates_every_variant_class(self):
        spec = importlib.import_module('quantlab.methods.exl3.spec_sampling')
        device = 'cuda' if _gpu_triton() else 'cpu'
        # 2 dtypes x 6 K (3 blocks x both %16 classes) x 2 min-P x 2 top-P, two kernels each
        self.assertEqual(spec.warm_kernels(torch.device(device)), 96)


class GeneratorHookTests(unittest.TestCase):
    def test_verify_window_ends_at_the_speculative_sample(self):
        source = GENERATOR.read_text(encoding='utf-8')
        self.assertIn('spec = spec_round.verify(job, job_logits)', source)
        self.assertIn('if spec_tokens is not None and i < spec_tokens.shape[-1]:', source)
        self.assertIn('spec_end = spec_final and i == spec_tokens.shape[-1] - 1', source)
        self.assertIn('or cp_boundary or spec_end', source)
        # The round is consumed exactly once, before any early return in iterate_gen
        body = source[source.index('def iterate_gen('):]
        self.assertLess(body.index('self.quantlab_spec_round = None'), body.index('if batch_size == 0'))


if __name__ == '__main__':
    unittest.main()
