"""Model-free AsterKV scheduling profiler; caller verifies the extension and owns the GPU.

run_profile(torch, extension, ...) fills one packed cache per (codec, occupancy)
with seeded synthetic fp16 rows, times decode/prefill scheduling variants with
CUDA events, and checks every variant against its codec's default output. No
models, network, installation, CLI, or environment changes.
"""
from __future__ import annotations

__all__ = ["run_profile"]

_SEED = 20260919
_BSZ = 1
_N_Q = 24
_N_KV = 4
_HEAD_DIM = 256
_CAPACITY = 32768
_Q_LENS = (1, 7, 1024)
_WARMUPS = 2
_REPEATS_DECODE = 10
_REPEATS_PREFILL = 3
_GATE_REL_L2 = 5e-3
_DECODE_MAX_Q = 16
_CODECS = ("q8", "q5", "aster5", "aster5-lloyd")
_COMBINE = {"parallel_combine": True, "num_stages": 1}
_DECODE_Q1 = (
    ("split32-block32", {"num_splits": 32, "block_n": 32}),
    ("split64-block32", {"num_splits": 64, "block_n": 32}),
    ("split32-block16", {"num_splits": 32, "block_n": 16}),
)
_DECODE_Q7 = (
    ("split32-block32", {"num_splits": 32, "block_n": 32}),
    ("split64-block32", {"num_splits": 64, "block_n": 32}),
    ("split32-block16-head8-warps8",
     {"num_splits": 32, "block_n": 16, "head_block": 8, "num_warps": 8}),
    ("split64-block16-head8-warps8",
     {"num_splits": 64, "block_n": 16, "head_block": 8, "num_warps": 8}),
    ("split32-block16-head8-warps4",
     {"num_splits": 32, "block_n": 16, "head_block": 8, "num_warps": 4}),
    ("split32-block32-head8-warps8",
     {"num_splits": 32, "block_n": 32, "head_block": 8, "num_warps": 8}),
)
_DECODE_Q7_POLY = (
    ("split32-block32-head8-warps8-stages2",
     {"num_splits": 32, "block_n": 32, "head_block": 8,
      "num_warps": 8, "num_stages": 2}),
    ("split64-block16-head4-warps4",
     {"num_splits": 64, "block_n": 16, "head_block": 4, "num_warps": 4}),
    ("split128-block16-head4-warps4",
     {"num_splits": 128, "block_n": 16, "head_block": 4, "num_warps": 4}),
    ("split128-block16-head8-warps8",
     {"num_splits": 128, "block_n": 16, "head_block": 8, "num_warps": 8}),
    ("split64-block32-head8-warps8",
     {"num_splits": 64, "block_n": 32, "head_block": 8, "num_warps": 8}),
    ("split64-block32-head8-warps8-stages2",
     {"num_splits": 64, "block_n": 32, "head_block": 8,
      "num_warps": 8, "num_stages": 2}),
)
_PREFILL_TILES = ((32, 4, 16), (32, 4, 32), (64, 4, 16), (64, 4, 32),
                 (64, 8, 16), (64, 8, 32), (128, 8, 16))
_PREFILL_POLY = ((32, 4, 32, 2), (64, 4, 32, 2), (64, 8, 64, 2))


def _decode_configs(q_len, poly):
    tuned = _DECODE_Q1 if q_len == 1 else _DECODE_Q7
    rows = [(n, {**_COMBINE, **kw}) for n, kw in tuned]
    if poly and q_len > 1:
        rows += [(n, {**_COMBINE, **kw}) for n, kw in _DECODE_Q7_POLY]
    elif poly:
        rows.append(("split128-block16",
                     {**_COMBINE, "num_splits": 128, "block_n": 16}))
    block_m = 1 << (q_len - 1).bit_length()
    return [(name, kw) for name, kw in rows
            if "head_block" not in kw or 16 <= kw["head_block"] * block_m <= 64]


def _prefill_configs(poly):
    rows = [(f"blockm{m}-warps{w}-blockn{n}",
             {"block_m": m, "num_warps": w, "block_n": n, "num_stages": 1})
            for m, w, n in _PREFILL_TILES]
    if poly:
        rows += [(f"blockm{m}-warps{w}-blockn{n}-stages{s}",
                  {"block_m": m, "num_warps": w, "block_n": n,
                   "num_stages": s}) for m, w, n, s in _PREFILL_POLY]
    return rows


def run_profile(torch, extension, *, occupancies=(32768,),
                codecs=("q8", "q5", "aster5", "aster5-lloyd"),
                include_half=False, on_row=None, capacity=_CAPACITY,
                q_lens=_Q_LENS, graph_timing=False):
    import inspect
    from types import SimpleNamespace

    from exllamav3.cache import CacheLayer_quant
    from exllamav3.cache.aster import CacheLayer_aster
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.modules.attention_fn import triton_paged as attn
    from exllamav3.modules.attention_fn.common import AttnArgs
    from triton.runtime.errors import OutOfResources

    assert callable(getattr(extension, "quant_cache_paged", None)), \
        "native extension lacks quant_cache_paged"
    if (not isinstance(capacity, int) or isinstance(capacity, bool)
            or capacity <= 0 or capacity % PAGE_SIZE):
        raise ValueError("capacity must be a positive PAGE_SIZE multiple")
    if not q_lens or any(not isinstance(q, int) or isinstance(q, bool)
                         or not 1 <= q <= 1024 for q in q_lens):
        raise ValueError("q_lens must contain integer query lengths in [1, 1024]")
    if not codecs or any(c not in _CODECS for c in codecs):
        raise ValueError(f"codecs must be one or more of {_CODECS}, "
                         f"got {tuple(codecs)}")
    if not occupancies or any(not isinstance(o, int)
                              or isinstance(o, bool) or o <= 0
                              or o > capacity or o % PAGE_SIZE != 0
                              or o < max(q_lens) for o in occupancies):
        raise ValueError(f"occupancies must be one or more PAGE_SIZE "
                         f"multiples in [{max(q_lens)}, {capacity}], "
                         f"got {tuple(occupancies)}")
    poly_ok = all("polynomial" in inspect.signature(f).parameters for f in
                  (attn.paged_attn_triton_decode, attn.paged_attn_triton_prefill))
    if "aster5" in codecs and not poly_ok:
        raise AssertionError("backend paged_attn lacks the polynomial "
                             "parameter required for codec 'aster5'")

    device = torch.device("cuda", torch.cuda.current_device())
    geometry = SimpleNamespace(num_kv_heads=_N_KV, head_dim=_HEAD_DIM)
    rows = []

    def rel(a, b):
        num = torch.linalg.vector_norm((a.float() - b.float()).flatten()).item()
        den = torch.linalg.vector_norm(b.float().flatten()).item()
        return num / den if den else (0.0 if num == 0 else float("inf"))

    def timed_call(fn, repeats):
        for _ in range(_WARMUPS):
            fn()
        graph = None
        if graph_timing:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            fn = graph.replay
            for _ in range(_WARMUPS):
                fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        ms = float(start.elapsed_time(end)) / repeats
        del graph
        return ms

    def cache_bytes(layer):
        total = sum(t.numel() * t.element_size() for t in layer.get_tensors())
        meta = getattr(layer, "get_metadata_tensors", None)
        if callable(meta):
            total += sum(t.numel() * t.element_size() for t in meta())
        return int(total)

    def check_finite(out, where):
        if not bool(torch.isfinite(out).all()):
            raise AssertionError(f"non-finite output at {where}")

    def check_gate(err, where):
        if not err < _GATE_REL_L2:
            raise AssertionError(f"scheduling error above gate at {where}: {err}")

    def emit(codec, occupied, q_len, name, tuning, ms, err, nbytes, note=None):
        row = {"codec": codec, "occupied": int(occupied),
               "q_len": int(q_len), "config": {"name": name, **tuning},
               "timing_ms": ms, "relative_error": err,
               "actualcachebytes": nbytes}
        if note is not None:
            row["note"] = note
        rows.append(row)
        if on_row is not None:
            on_row(row)

    def make_layer(codec):
        if codec == "q8":
            return CacheLayer_quant(None, geometry, 0, capacity, 8, 8), 8, 8
        if codec == "q5":
            return CacheLayer_quant(None, geometry, 0, capacity, 5, 5), 5, 5
        if codec == "aster5-lloyd":
            from exllamav3.cache.aster_codebook import ASTER5_LLOYD_CENTROIDS
            layer = CacheLayer_aster(None, geometry, 0, capacity, 5, 5, 0.0,
                                     centroids=ASTER5_LLOYD_CENTROIDS)
            return layer, 5, 5
        return CacheLayer_aster(None, geometry, 0, capacity, 5, 5), 5, 5

    with torch.inference_mode(), torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(_SEED)
        gen = torch.Generator().manual_seed(_SEED)
        full = max(occupancies)
        base_k = torch.randn((full, _N_KV, _HEAD_DIM),
                             generator=gen, dtype=torch.float32).half()
        base_v = torch.randn((full, _N_KV, _HEAD_DIM),
                             generator=gen, dtype=torch.float32).half()
        queries = {q: (torch.randn((_BSZ, q, _N_Q, _HEAD_DIM),
                                   generator=gen, dtype=torch.float32) * 0.125).half()
                   for q in q_lens}
        assert torch.isfinite(base_k).all() and torch.isfinite(base_v).all()

        for codec in codecs:
            poly_path = codec == "aster5"
            for occupied in occupancies:
                pages = occupied // PAGE_SIZE
                layer, kb, vb = make_layer(codec)
                poly = getattr(layer, "polynomial", None)
                if poly_path:
                    assert isinstance(poly, (tuple, list)) and len(poly) == 2, \
                        f"aster5 default must carry polynomial (a, b), got {poly!r}"
                elif codec == "aster5-lloyd":
                    assert poly is None, \
                        f"aster5-lloyd must decode via LUT, got {poly!r}"
                pak = {"polynomial": tuple(poly) if poly is not None else None} \
                    if poly_ok else {}
                layer.alloc(device)
                table = torch.arange(pages, dtype=torch.int32,
                                     device=device).unsqueeze(0)
                k_in = base_k[:occupied].unsqueeze(0).to(device)
                v_in = base_v[:occupied].unsqueeze(0).to(device)
                layer.update_kv_direct(
                    torch.zeros((_BSZ,), dtype=torch.int32, device=device),
                    table, k_in, v_in, occupied)
                torch.cuda.synchronize()
                del k_in, v_in
                nbytes = cache_bytes(layer)
                qc = (layer.sk, layer.sv, kb, vb)
                codebook = getattr(layer, "codebook_tensor", None)

                for q_len in q_lens:
                    is_decode = q_len <= _DECODE_MAX_Q
                    fn = attn.paged_attn_triton_decode if is_decode \
                        else attn.paged_attn_triton_prefill
                    reps = _REPEATS_DECODE if is_decode else _REPEATS_PREFILL
                    q = queries[q_len].to(device)
                    seqlens = torch.full((_BSZ,), occupied - q_len,
                                         dtype=torch.int32, device=device)
                    out = torch.empty_like(q)
                    where = f"{codec}/{occupied}/q{q_len}"

                    def launch(tuning, cb, _fn=fn):
                        return _fn(q, None, None, layer.qk, layer.qv,
                                   table, seqlens, causal=True, qc=qc,
                                   codebook=cb, pre_appended_len=q_len,
                                   n_kv_heads_override=_N_KV, out=out,
                                   **tuning, **pak)

                    def measure(name, tuning, ref, label, cb, nb):
                        try:
                            ms = timed_call(lambda: launch(tuning, cb), reps)
                        except OutOfResources:
                            emit(label, occupied, q_len, name, tuning, None,
                                 None, nb, note="skipped: Triton OutOfResources")
                            return
                        check_finite(out, f"{where}/{name}")
                        err = float(rel(out, ref))
                        check_gate(err, f"{where}/{name}")
                        emit(label, occupied, q_len, name, tuning, ms, err, nb)

                    ms = timed_call(lambda: launch({}, codebook), reps)
                    check_finite(out, where + "/baseline")
                    ref = out.clone()
                    emit(codec, occupied, q_len, "baseline", {}, ms, 0.0, nbytes)

                    if codec == "q8" and occupied >= 32768 and is_decode:
                        args = AttnArgs(
                            bsz=_BSZ, q_len=q_len, num_q_heads=_N_Q,
                            dim=_HEAD_DIM, kv_len=occupied,
                            num_kv_heads=_N_KV, q=q, k=None, v=None,
                            k_cache=None, v_cache=None, causal=True,
                            sm_scale=_HEAD_DIM ** -0.5, cu_seqlens=None,
                            max_seqlen=None, window_size=None, softcap=0.0,
                            block_table=table, cache_seqlens=seqlens,
                            q_cache=(layer.qk, layer.sk, layer.qv,
                                     layer.sv, 8, 8), sinks=None)
                        opts = dict(attn._long_qc_decode_options(args, 8, 8))
                        if opts:
                            measure("q8-long", opts, ref, codec, codebook, nbytes)
                        else:
                            emit(codec, occupied, q_len, "q8-long", {}, None,
                                 None, nbytes, note="q8-long unavailable: "
                                 "_long_qc_decode_options returned {} "
                                 "(long profile inactive)")

                    if codec in ("aster5", "aster5-lloyd"):
                        configs = _decode_configs(q_len, poly_path) if is_decode \
                            else _prefill_configs(poly_path)
                        for name, tuning in configs:
                            measure(name, tuning, ref, codec, codebook, nbytes)
                        if include_half and codec == "aster5-lloyd":
                            cb16 = codebook.half()
                            label = "aster5-lloyd-fp16lookup"
                            measure("baseline", {}, ref, label, cb16, nbytes + 64)
                            for name, tuning in configs:
                                measure(name, tuning, ref, label, cb16, nbytes + 64)
                    del q, seqlens, out

                if occupied == max(occupancies):
                    zero = torch.zeros((_BSZ,), dtype=torch.int32, device=device)
                    for q_len in q_lens:
                        k_enc = base_k[:q_len].unsqueeze(0).to(device)
                        v_enc = base_v[:q_len].unsqueeze(0).to(device)
                        ms = timed_call(
                            lambda: layer.update_kv_direct(
                                zero, table, k_enc, v_enc, q_len),
                            _REPEATS_DECODE if q_len <= _DECODE_MAX_Q
                            else _REPEATS_PREFILL)
                        del k_enc, v_enc
                        emit(codec, occupied, q_len, "encode", {}, ms, None,
                             nbytes, note="encode-only: update_kv_direct, "
                             "same positions repeated")
                    del zero
                layer.free()
                del layer, table, qc

    return {"rows": rows,
            "meta": {"seed": _SEED, "bsz": _BSZ, "num_q_heads": _N_Q,
                     "num_kv_heads": _N_KV, "head_dim": _HEAD_DIM,
                     "capacity": capacity, "occupancies": list(occupancies),
                     "codecs": list(codecs), "include_half": bool(include_half),
                     "q_lens": list(q_lens), "warmups": _WARMUPS,
                     "graph_timing": bool(graph_timing),
                     "qc_staging": attn._qc_staging,
                     "decode_profile": attn._qc_decode_profile,
                     "repeats_decode": _REPEATS_DECODE,
                     "repeats_prefill": _REPEATS_PREFILL,
                     "gate_rel_l2": _GATE_REL_L2,
                     "decode_max_q": _DECODE_MAX_Q}}
