"""Bounded single-step MTP graph experiment with stable input buffers."""


class DraftStepGraph:
    def __init__(self, draft):
        self.draft = draft
        self.signature = None
        self.graph = None
        self.stats = dict(captures=0, replays=0, eager_calls=0, checks=[])

    def _eager(self, ids, params):
        state = self.draft.forward(ids, params)
        tokens = self.draft.sample_from_state(state, params)
        return state, tokens

    def __call__(self, ids, params):
        import torch
        # Warmup diagnostics replace the sampler with an observing closure.
        # Let those CPU checks see the eager head, outside graph capture.
        eligible = (getattr(self.draft.sample_from_state, '__self__', None) is self.draft
                    and tuple(ids.shape) == (1, 1) and ids.is_cuda and ids.is_contiguous()
                    and set(params) == {'target_hidden', 'attn_mode', 'block_table', 'cache', 'cache_seqlens'}
                    and params['attn_mode'] == 'flash_attn'
                    and all(params[k].is_cuda for k in ('target_hidden', 'block_table', 'cache_seqlens')))
        if not eligible:
            self.stats['eager_calls'] += 1
            return self._eager(ids, params)
        keys = ('target_hidden', 'block_table', 'cache_seqlens')
        signature = (id(params['cache']), tuple(ids.shape), str(ids.device),
                     tuple((k, tuple(params[k].shape), params[k].dtype) for k in keys))
        if signature != self.signature:
            if self.stats['captures'] >= 32:
                raise RuntimeError('Draft graph capture budget exceeded')
            self.graph = None
            self.static_ids = ids.clone()
            self.static_params = dict(params)
            for key in keys:
                self.static_params[key] = params[key].clone()
            # Rewriting the same cache position with the same inputs is idempotent.
            # Keep an independent eager reference before capture reuses scratch.
            ref_state, ref_ids = self._eager(self.static_ids, dict(self.static_params))
            ref_state, ref_ids = ref_state.clone(), ref_ids.clone()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    self.state, self.tokens = self._eager(self.static_ids, dict(self.static_params))
            except Exception as error:
                raise RuntimeError('Whole MTP step capture failed: ' + str(error)) from error
            graph.replay()
            torch.cuda.synchronize()
            relative_l2 = float((self.state.float()-ref_state.float()).norm()) / max(float(ref_state.float().norm()), 1e-12)
            exact_ids = bool(torch.equal(self.tokens, ref_ids))
            finite = bool(torch.isfinite(self.state).all() and torch.isfinite(ref_state).all())
            self.stats['checks'].append(dict(relative_l2=relative_l2, exact_ids=exact_ids, finite=finite))
            if not finite or relative_l2 > 1e-3 or not exact_ids:
                raise RuntimeError('Whole MTP step graph differs from eager reference')
            self.graph, self.signature = graph, signature
            self.stats['captures'] += 1
        else:
            self.static_ids.copy_(ids)
            for key in keys:
                self.static_params[key].copy_(params[key])
            self.graph.replay()
        self.stats['replays'] += 1
        # The generator retains each draft ID until it assembles the window.
        # Replay writes the same graph buffers again, so return owned tensors
        # just like the eager path. Otherwise every saved ID becomes the last
        # replay's token, even though the single-step capture check passes.
        return self.state.clone(), self.tokens.clone()
