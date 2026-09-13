"""Compact packed draft vocabulary shortlist for the Qwen3.5 MTP draft path.

Builds a narrow packed EXL3 head that holds a top-K subset of the target
lm_head's 128-wide output (Hadamard) groups, refreshed from each full target
verification step. The draft sampler then scores only the shortlisted columns
and maps the local argmax back to the global token id.

Owns no model state: the original head weights and forward are never touched,
only ``draft.sample_from_state`` is wrapped (with full-head fallback), and a
``reset``/``update`` handle is attached at ``model.quantlab_shortlist``.
"""

from __future__ import annotations

__all__ = ["install_draft_shortlist"]

_MIN_GROUPS = 8
_MAX_GROUPS = 128
_GROUP_WIDTH = 128
_TILES_PER_GROUP = 8
_TILE_WIDTH = 16
_MAX_EXTRA_BYTES = 256 * 1024 * 1024
_MAX_UPDATE_ROWS = 9
_PROBE_ROWS = (1, 5, 9)
_PROBE_SEED = 905
_REL_L2_TOL = 1e-3
_MODES = ("packed", "dense")


def install_draft_shortlist(model, draft, groups: int, mode: str) -> dict:
    """Install a packed draft vocabulary shortlist; return JSON-safe stats.

    ``model`` is the single-device target, ``draft`` its exact
    ``Qwen3_5MTPModel``. ``groups`` (8..128) selects how many 128-wide output
    groups the compact head holds; ``mode`` is ``"packed"`` (native packed
    GEMV on the compact trellis) or ``"dense"`` (one FP16 reconstruct of the
    compact trellis per update, reused for every draft step).
    """
    if model is None or draft is None:
        raise ValueError("install_draft_shortlist: model and draft must not be None")
    if not isinstance(groups, int) or isinstance(groups, bool):
        raise ValueError(f"install_draft_shortlist: groups must be an integer, got {groups!r}")
    if not (_MIN_GROUPS <= groups <= _MAX_GROUPS):
        raise ValueError(
            f"install_draft_shortlist: groups must be {_MIN_GROUPS}..{_MAX_GROUPS}, got {groups!r}"
        )
    if mode not in _MODES:
        raise ValueError(f"install_draft_shortlist: invalid mode {mode!r}, expected one of {list(_MODES)}")

    import torch
    from exllamav3.architecture.qwen3_5_mtp import Qwen3_5MTPModel
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.linear import Linear
    from exllamav3.modules.quant.exl3 import LinearEXL3

    if type(draft) is not Qwen3_5MTPModel:
        raise ValueError(
            f"install_draft_shortlist: draft must be exactly Qwen3_5MTPModel, got {type(draft).__name__}"
        )
    if getattr(model, "loaded_tp", False) or getattr(draft, "loaded_tp", False):
        raise ValueError("install_draft_shortlist: single-device target only (TP is loaded)")
    attached = getattr(draft, "attached_model", None)
    if attached is not None and attached() is not model:
        raise ValueError("install_draft_shortlist: draft is not attached to this model")
    if hasattr(model, "quantlab_shortlist"):
        raise RuntimeError("install_draft_shortlist: shortlist already installed on this model")
    if getattr(draft.sample_from_state, "_quantlab_shortlist", False):
        raise RuntimeError("install_draft_shortlist: draft sampler is already wrapped")

    head = model.modules[model.logit_layer_idx]
    if not isinstance(head, Linear):
        raise ValueError(
            f"install_draft_shortlist: target head must be a Linear, got {type(head).__name__}"
        )
    inner = head.inner
    if inner is None or head.quant_type != "exl3" or not isinstance(inner, LinearEXL3):
        raise ValueError("install_draft_shortlist: target head must be a plain loaded LinearEXL3")
    if getattr(head, "is_sliced", False):
        raise ValueError("install_draft_shortlist: sharded (TP) target head is not supported")
    if getattr(head, "lora_a_tensors", None) or getattr(head, "lora_b_tensors", None):
        raise ValueError("install_draft_shortlist: LoRA adapters on the target head are not supported")
    if head.softcap:
        raise ValueError("install_draft_shortlist: softcapped target head is not supported")
    if head.pre_scale != 1.0 or head.post_scale != 1.0:
        raise ValueError("install_draft_shortlist: non-unit pre/post scales are not supported")
    if inner.mcg or inner.mul1:
        raise ValueError("install_draft_shortlist: mcg/mul1 special codebooks are not supported")
    if inner.K not in (2, 3, 4):
        raise ValueError(f"install_draft_shortlist: unsupported K={inner.K}, expected 2/3/4")

    in_features = inner.in_features
    full_width = inner.out_features
    if in_features % _GROUP_WIDTH or full_width % _GROUP_WIDTH:
        raise ValueError("install_draft_shortlist: head dims must be 128-divisible")
    trellis = inner.trellis
    if trellis.shape[1] != full_width // _TILE_WIDTH:
        raise ValueError("install_draft_shortlist: trellis/output width mismatch on target head")
    if not trellis.is_cuda:
        raise ValueError("install_draft_shortlist: target head must live on CUDA")
    device = trellis.device
    has_bias = inner.bias is not None

    width = groups * _GROUP_WIDTH
    if width >= full_width:
        raise ValueError(
            f"install_draft_shortlist: shortlist width {width} must be < head width {full_width}"
        )
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is None:
        vocab_size = head.out_features_unpadded
    vocab_size = int(vocab_size)
    if vocab_size < 1:
        raise ValueError("install_draft_shortlist: vocabulary size is unknown")
    if vocab_size > full_width:
        vocab_size = full_width

    # Fixed budget, checked before any allocation.
    compact_tiles = width // _TILE_WIDTH
    trellis_bytes = trellis.shape[0] * compact_tiles * trellis.shape[2] * 2
    svh_bytes = width * 2
    bias_bytes = width * int(inner.bias.element_size()) if has_bias else 0
    dense_bytes = in_features * width * 2 if mode == "dense" else 0
    index_bytes = width * 8 + width * 4 + (_GROUP_WIDTH + _TILES_PER_GROUP) * 8
    extra_bytes = trellis_bytes + svh_bytes + bias_bytes + dense_bytes + index_bytes
    if extra_bytes > _MAX_EXTRA_BYTES:
        raise RuntimeError(
            f"install_draft_shortlist: {extra_bytes} extra bytes exceed "
            f"{_MAX_EXTRA_BYTES} budget (groups={groups}, mode={mode!r})"
        )

    compact_trellis = torch.empty(
        (trellis.shape[0], compact_tiles, trellis.shape[2]), dtype=torch.int16, device=device
    )
    compact_svh = torch.empty((width,), dtype=torch.half, device=device)
    compact_bias = (
        torch.empty((width,), dtype=inner.bias.dtype, device=device) if has_bias else None
    )
    compact = LinearEXL3(
        config=inner.config,
        in_features=in_features,
        out_features=width,
        trellis=compact_trellis,
        suh=inner.suh,
        svh=compact_svh,
        bias=compact_bias,
        out_dtype=inner.default_out_dtype,
        key="mtp.shortlist_head",
    )
    dense_w = torch.empty((in_features, width), dtype=torch.half, device=device) if mode == "dense" else None
    arange_group = torch.arange(_GROUP_WIDTH, dtype=torch.long, device=device)
    arange_tiles = torch.arange(_TILES_PER_GROUP, dtype=torch.long, device=device)
    pad_mask = torch.zeros((width,), dtype=torch.float32, device=device)

    stats: dict = {
        "mode": mode,
        "groups": groups,
        "vocabulary_width": width,
        "extra_bytes": extra_bytes,
        "updates": 0,
        "shortlisted_draft_calls": 0,
        "full_head_fallback_calls": 0,
        "resets": 0,
        "checks": {},
    }

    def _refill(sel):
        """Gather one 128-group's tiles/signs/bias into the compact head (GPU only)."""
        token_ids = (sel.unsqueeze(1) * _GROUP_WIDTH + arange_group).reshape(-1)
        tile_ids = (sel.unsqueeze(1) * _TILES_PER_GROUP + arange_tiles).reshape(-1)
        torch.index_select(trellis, 1, tile_ids, out=compact_trellis)
        torch.index_select(inner.svh, 0, token_ids, out=compact_svh)
        if has_bias:
            torch.index_select(inner.bias, 0, token_ids, out=compact_bias)
        pad_mask.zero_()
        pad_mask.masked_fill_(token_ids >= vocab_size, float("-inf"))
        if dense_w is not None:
            ext.reconstruct(dense_w, compact_trellis, inner.K, False, False)
        return token_ids

    def _dense_forward(x):
        shape = x.shape
        rows = x.numel() // shape[-1]
        xflat = x.view(rows, in_features)
        xh = torch.empty_like(xflat)
        ext.had_r_128(xflat, xh, inner.suh, None, 1.0)
        y = torch.empty(shape[:-1] + (width,), dtype=torch.float32, device=x.device)
        yview = y.view(rows, width)
        ext.hgemm(xh, dense_w, yview)
        ext.had_r_128(yview, yview, None, compact_svh, 1.0)
        if compact_bias is not None:
            y += compact_bias
        return y

    class _DraftShortlist:
        def __init__(self):
            self._ready = False
            self._token_ids = None
            self.stats = stats

        def reset(self):
            self._ready = False
            stats["resets"] += 1

        def update(self, full_target_logits):
            if full_target_logits.dim() == 3:
                batch, rows, last = full_target_logits.shape
                if batch != 1:
                    raise ValueError(
                        f"shortlist update: expected batch 1, got shape {tuple(full_target_logits.shape)}"
                    )
            elif full_target_logits.dim() == 2:
                rows, last = full_target_logits.shape
            else:
                raise ValueError(
                    "shortlist update: expected (1, rows, vocab) or (rows, vocab), "
                    f"got shape {tuple(full_target_logits.shape)}"
                )
            if not (1 <= rows <= _MAX_UPDATE_ROWS):
                raise ValueError(f"shortlist update: expected 1..{_MAX_UPDATE_ROWS} rows, got {rows}")
            if last != full_width:
                raise ValueError(
                    f"shortlist update: expected FULL TARGET width {full_width}, got {last}"
                )
            if full_target_logits.device != device:
                raise ValueError("shortlist update: logits are not on the head device")
            flat = full_target_logits.reshape(-1, full_width)
            peak = flat.amax(dim=0)
            if vocab_size < full_width:
                peak[vocab_size:] = float("-inf")
            group_max = peak.view(full_width // _GROUP_WIDTH, _GROUP_WIDTH).amax(dim=1)
            top = torch.topk(group_max, groups).indices
            sel, _ = torch.sort(top)
            self._token_ids = _refill(sel)
            stats["updates"] += 1
            self._ready = True

    shortlist = _DraftShortlist()

    # Numerical setup gate, before the sampler is replaced: deterministic spaced
    # groups within the vocabulary, probed against the full head's columns.
    num_vocab_groups = (vocab_size + _GROUP_WIDTH - 1) // _GROUP_WIDTH
    if groups > num_vocab_groups:
        raise ValueError(
            f"install_draft_shortlist: cannot place {groups} spaced groups "
            f"within {num_vocab_groups} vocabulary groups"
        )
    setup_sel = (torch.arange(groups, device=device) * num_vocab_groups) // groups
    setup_token_ids = _refill(setup_sel)
    shortlist._token_ids = setup_token_ids
    with torch.no_grad():
        for rows in _PROBE_ROWS:
            gen = torch.Generator().manual_seed(_PROBE_SEED)
            x = torch.randn(rows, in_features, generator=gen).to(device=device, dtype=torch.float16)
            ref = inner.forward(x, {}, torch.float32).index_select(-1, setup_token_ids)
            got = compact.forward(x, {}, torch.float32) if mode == "packed" else _dense_forward(x)
            if tuple(got.shape) != tuple(ref.shape):
                raise RuntimeError(
                    f"install_draft_shortlist: rows={rows} shape mismatch "
                    f"{tuple(got.shape)} vs {tuple(ref.shape)}"
                )
            if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(got).all()):
                raise RuntimeError(f"install_draft_shortlist: rows={rows} non-finite output")
            diff = got.float() - ref.float()
            rel_l2 = float(diff.norm()) / max(float(ref.float().norm()), 1e-12)
            max_abs = float(diff.abs().max())
            stats["checks"][str(rows)] = {"relative_l2": rel_l2, "max_abs": max_abs}
            if rel_l2 > _REL_L2_TOL:
                raise RuntimeError(
                    f"install_draft_shortlist: rows={rows} relative L2 {rel_l2:.3e} "
                    f"exceeds {_REL_L2_TOL}"
                )
    shortlist._ready = False

    original_sample = draft.sample_from_state

    def _sample_from_state(state, params):
        if (
            not shortlist._ready
            or params.get("export_draft_conf")
            or "ovr" in params
            or params.get("reconstruct")
        ):
            stats["full_head_fallback_calls"] += 1
            return original_sample(state, params)
        if (
            state.dim() not in (2, 3)
            or state.shape[-1] != in_features
            or (state.dim() == 3 and state.shape[0] != 1)
            or not state.is_contiguous()
        ):
            stats["full_head_fallback_calls"] += 1
            return original_sample(state, params)
        x = head.prepare_for_device(state, params)
        if not x.is_contiguous() or x.device != device:
            stats["full_head_fallback_calls"] += 1
            return original_sample(state, params)
        if mode == "packed":
            local = compact.forward(x, {}, torch.float32)
        else:
            local = _dense_forward(x)
        local += pad_mask
        global_ids = shortlist._token_ids[torch.argmax(local, dim=-1)]
        stats["shortlisted_draft_calls"] += 1
        return global_ids

    _sample_from_state._quantlab_shortlist = True
    draft.sample_from_state = _sample_from_state
    model.quantlab_shortlist = shortlist
    return stats
