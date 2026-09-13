"""Cache reconstructed MTP projection weights for a bounded runtime experiment.

Reconstructs each selected packed weight once via ``get_inner_weight_tensor``
(transformed basis, FP16, stays on GPU) and patches that inner's ``forward``
so short-decode rows reuse the cached matrix instead of reconstructing per
call. The original trellis is never modified; no files are touched.
"""

from __future__ import annotations

__all__ = ["cache_mtp_projections"]

_VALID_MODES = ("fc", "attention", "mlp", "all")
_MAX_CACHE_BYTES = 1536 * 1024 * 1024  # 1.5 GiB budget, checked before allocation
_MAX_OUT_FEATURES = 32768  # cached path has no output slicing
_MAX_CACHED_ROWS = 9
_PROBE_ROWS = (1, 5, 9)
_PROBE_SEED = 905
_REL_L2_TOL = 1e-3


def _matches(key, mode):
    if mode == "all":
        return True
    if key is None:
        return False
    if mode == "fc":
        return key == "mtp.fc"
    if mode == "attention":
        return ".self_attn." in key
    if mode == "mlp":
        return ".mlp." in key
    return False


def cache_mtp_projections(draft, mode: str) -> dict:
    """Cache selected MTP projection weights; return JSON-safe stats.

    Attaches a cached forward to each selected ``LinearEXL3`` inner. The
    cached path serves contiguous inputs with 1..9 rows; explicit
    ``ovr``/``reconstruct`` params, non-contiguous input, and other shapes
    fall back to the original forward. Per-projection ``cached_calls`` /
    ``fallback_calls`` counters are mutated by the attached closures.
    """
    if draft is None:
        raise ValueError("cache_mtp_projections: draft must not be None")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"cache_mtp_projections: invalid mode {mode!r}, expected one of {list(_VALID_MODES)}"
        )

    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3 import LinearEXL3

    try:
        descendants = list(draft)
    except TypeError:
        descendants = [draft]

    selected = {}
    for desc in descendants:
        inner = getattr(desc, "inner", None)
        if inner is None and isinstance(desc, LinearEXL3):
            inner = desc
        if not isinstance(inner, LinearEXL3):
            continue
        key = getattr(inner, "key", None) or getattr(desc, "key", None)
        if _matches(key, mode) and id(inner) not in selected:
            selected[id(inner)] = (inner, key)
    if not selected:
        raise ValueError(f"cache_mtp_projections: mode {mode!r} matched no LinearEXL3 projections")

    inners = list(selected.values())
    for inner, key in inners:
        if inner.out_features > _MAX_OUT_FEATURES:
            raise ValueError(
                f"cache_mtp_projections: {key!r} out_features {inner.out_features} "
                f"exceeds {_MAX_OUT_FEATURES}"
            )

    sizes = [inner.in_features * inner.out_features * 2 for inner, _ in inners]
    total_cache_bytes = sum(sizes)
    if total_cache_bytes > _MAX_CACHE_BYTES:
        raise RuntimeError(
            f"cache_mtp_projections: {total_cache_bytes} cached weight bytes exceed "
            f"{_MAX_CACHE_BYTES} byte budget; refusing to allocate"
        )

    def _cached_compute(inner, w, x, out_dtype):
        shape = x.shape
        rows = x.numel() // shape[-1]
        xflat = x.view(rows, inner.in_features)
        xh = torch.empty_like(xflat)
        ext.had_r_128(xflat, xh, inner.suh, None, 1.0)
        y = torch.empty(
            shape[:-1] + (inner.out_features,),
            dtype=out_dtype or inner.default_out_dtype,
            device=x.device,
        )
        yview = y.view(rows, inner.out_features)
        ext.hgemm(xh, w, yview)
        ext.had_r_128(yview, yview, None, inner.svh, 1.0)
        if inner.bias is not None:
            y += inner.bias
        return y

    def _make_cached_forward(inner, w, record, original):
        def cached_forward(x, params, out_dtype=None):
            if "ovr" in params or params.get("reconstruct"):
                record["fallback_calls"] += 1
                return original(x, params, out_dtype)
            if not x.is_contiguous():
                record["fallback_calls"] += 1
                return original(x, params, out_dtype)
            rows = x.numel() // x.shape[-1]
            if rows < 1 or rows > _MAX_CACHED_ROWS:
                record["fallback_calls"] += 1
                return original(x, params, out_dtype)
            record["cached_calls"] += 1
            return _cached_compute(inner, w, x, out_dtype)

        return cached_forward

    records = []
    with torch.inference_mode():
        staged = []
        for (inner, key), nbytes in zip(inners, sizes):
            w = inner.get_inner_weight_tensor()
            record = {
                "key": key,
                "bytes": nbytes,
                "cached_calls": 0,
                "fallback_calls": 0,
                "checks": {},
            }
            original = inner.forward
            for rows in _PROBE_ROWS:
                gen = torch.Generator().manual_seed(_PROBE_SEED)
                x = torch.randn(rows, inner.in_features, generator=gen).to(
                    device=w.device, dtype=torch.float16
                )
                ref = original(x, {}, torch.float32)
                got = _cached_compute(inner, w, x, torch.float32)
                if tuple(got.shape) != tuple(ref.shape):
                    raise RuntimeError(
                        f"cache_mtp_projections: {key!r} rows={rows} shape mismatch "
                        f"{tuple(got.shape)} vs {tuple(ref.shape)}"
                    )
                if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(got).all()):
                    raise RuntimeError(
                        f"cache_mtp_projections: {key!r} rows={rows} non-finite output"
                    )
                diff = got.float() - ref.float()
                rel_l2 = float(diff.norm()) / max(float(ref.float().norm()), 1e-12)
                max_abs = float(diff.abs().max())
                record["checks"][str(rows)] = {"relative_l2": rel_l2, "max_abs": max_abs}
                if rel_l2 > _REL_L2_TOL:
                    raise RuntimeError(
                        f"cache_mtp_projections: {key!r} rows={rows} relative L2 "
                        f"{rel_l2:.3e} exceeds {_REL_L2_TOL}"
                    )
            staged.append((inner, w, record, original))
        for inner, w, record, original in staged:
            inner.forward = _make_cached_forward(inner, w, record, original)
            records.append(record)

    return {"mode": mode, "total_cache_bytes": total_cache_bytes, "projections": records}
