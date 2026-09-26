"""Packed KV baseline GPU checks for an explicitly loaded native extension.

Call run_checks(torch, extension) from an environment that has already verified
the binary, imported the vendor source, and acquired the GPU. This module runs
model-free GPU operations; it never loads models or builds native extensions.
"""
from __future__ import annotations

_PAIRS = ((8, 8), (6, 6), (5, 5), (4, 4), (8, 4), (8, 6))
_N_Q, _N_KV, _HEAD_DIM = 24, 4, 256
_CAPACITY = 1024  # 4 pages; live rows stay <= 1024, device use stays in the MBs
_L1, _L2, _EXTRA = 250, 20, 7  # second append crosses the page boundary at row 256
_SEED = 9177
_KERNEL_REL, _KERNEL_ABS = 5e-3, 5e-2
_DECODE_REL = 2e-2
_SENT_I32 = 0x6B6B6B6B


def run_checks(torch, extension):
    from types import SimpleNamespace
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode

    for _name in ("quant_cache_paged", "dequant_cache_paged"):
        assert callable(getattr(extension, _name, None)), f"native extension lacks {_name}"
    results = []
    device = torch.device("cuda", torch.cuda.current_device())
    n_pages = _CAPACITY // PAGE_SIZE
    total, full = _L1 + _L2, _L1 + _L2 + _EXTRA

    def h32():
        h = torch.ones((1, 1), dtype=torch.float32, device=device)
        while h.shape[0] < 32:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        return h / (32.0 ** 0.5)

    def ref_roundtrip(x, bits, hs):
        # Independent FP32 reference: normalized H32 rotation, group absmax,
        # midpoint grid; scales rounded through fp16 like the native store.
        rows = x.shape[0]
        y = x.float().reshape(-1, 32) @ hs.T
        s = y.abs().amax(1, keepdim=True) + 1e-10
        m = float(1 << (bits - 1))
        q = (y / s * m + m).floor().clamp_(0, (1 << bits) - 1)
        sh = s.half().float()
        xh = ((q - (m - 0.5)) / m * sh @ hs.T).reshape(rows, _N_KV, _HEAD_DIM)
        return xh, sh.reshape(rows, -1)

    def rel_l2(a, b):
        num = torch.linalg.vector_norm((a.float() - b.float()).flatten()).item()
        den = torch.linalg.vector_norm(b.float().flatten()).item()
        return num / den if den else (0.0 if num == 0 else float("inf"))

    def oracle(q, k, v, length):
        # FP32 grouped decode attention; query row i sees logical rows 0..length+i.
        q_len, n = q.shape[1], k.shape[0]
        qg = q.float().view(1, q_len, _N_KV, _N_Q // _N_KV, _HEAD_DIM)
        scores = torch.einsum("bqghd,tgd->bqght", qg, k.float()) / (_HEAD_DIM ** 0.5)
        over = torch.arange(n, device=device)[None, :] > (length + torch.arange(q_len, device=device))[:, None]
        probs = scores.masked_fill(over.view(1, q_len, 1, 1, n), float("-inf")).softmax(-1)
        return torch.einsum("bqght,tgd->bqghd", probs, v.float()).reshape(1, q_len, _N_Q, _HEAD_DIM)

    def gather(paged, order, rows):
        return torch.cat([paged[i] for i in order], 0)[:rows]

    def live_mask(order, rows):
        m = torch.zeros((n_pages, PAGE_SIZE), dtype=torch.bool, device=device)
        for t in range(rows):
            m[order[t // PAGE_SIZE], t % PAGE_SIZE] = True
        return m

    def seq(v):
        return torch.full((1,), v, dtype=torch.int32, device=device)

    def make_layer(kb, vb):
        layer = CacheLayer_quant(None, SimpleNamespace(num_kv_heads=_N_KV, head_dim=_HEAD_DIM),
                                 1, _CAPACITY, kb, vb)
        layer.alloc(device)
        return layer

    with torch.inference_mode(), torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(_SEED)
        gen = torch.Generator().manual_seed(_SEED)
        hs = h32()
        assert torch.allclose(hs @ hs.T, torch.eye(32, device=device), atol=1e-6), "H32 reference not orthonormal"
        order = torch.randperm(n_pages, generator=gen).tolist()
        table = torch.tensor([order], dtype=torch.int32, device=device)
        base_k = torch.randn((full, _N_KV, _HEAD_DIM), generator=gen, dtype=torch.float32).half().to(device)
        base_v = torch.randn((full, _N_KV, _HEAD_DIM), generator=gen, dtype=torch.float32).half().to(device)
        queries = {q: (torch.randn((1, q, _N_Q, _HEAD_DIM), generator=gen, dtype=torch.float32) * 0.125)
                   .half().to(device) for q in (1, 7)}
        assert torch.isfinite(base_k).all() and torch.isfinite(base_v).all(), "non-finite synthetic cache rows"

        for kb, vb in _PAIRS:
            layer = make_layer(kb, vb)
            qk, sk, qv, sv, ekb, evb = layer.get_qkv()
            assert (ekb, evb) == (kb, vb), ("bitrate echo", ekb, evb)
            assert layer.compand_a == 0.0, "expected inherited linear compand_a=0"
            qk.fill_(_SENT_I32)
            qv.fill_(_SENT_I32)
            sk.fill_(float("nan"))
            sv.fill_(float("nan"))
            # Two appends; the second crosses the logical page boundary at row 256.
            layer.update_kv_direct(seq(0), table, base_k[:_L1].unsqueeze(0), base_v[:_L1].unsqueeze(0), _L1)
            layer.update_kv_direct(seq(_L1), table, base_k[_L1:total].unsqueeze(0), base_v[_L1:total].unsqueeze(0), _L2)
            torch.cuda.synchronize()
            got_k, got_v = layer.get_kv(seq(total), table)
            torch.cuda.synchronize()
            nat_k, nat_v = gather(got_k, order, total), gather(got_v, order, total)
            assert torch.isfinite(nat_k).all() and torch.isfinite(nat_v).all(), "non-finite paged dequant"
            ref_k, ref_sk = ref_roundtrip(base_k[:total], kb, hs)
            ref_v, ref_sv = ref_roundtrip(base_v[:total], vb, hs)
            k_err, v_err = rel_l2(nat_k, ref_k), rel_l2(nat_v, ref_v)
            k_abs = (nat_k.float() - ref_k).abs().max().item()
            v_abs = (nat_v.float() - ref_v).abs().max().item()
            assert k_err < _KERNEL_REL and v_err < _KERNEL_REL, ("native-vs-reference", kb, vb, k_err, v_err)
            assert k_abs < _KERNEL_ABS and v_abs < _KERNEL_ABS, ("native-vs-reference abs", k_abs, v_abs)
            nat_sk, nat_sv = gather(sk, order, total), gather(sv, order, total)
            assert torch.isfinite(nat_sk).all() and torch.isfinite(nat_sv).all(), "non-finite native scales"
            assert torch.allclose(nat_sk.float(), ref_sk, rtol=1e-3, atol=1e-3), "K scale mismatch vs reference"
            assert torch.allclose(nat_sv.float(), ref_sv, rtol=1e-3, atol=1e-3), "V scale mismatch vs reference"
            dead = ~live_mask(order, total)
            assert (qk[dead] == _SENT_I32).all(), "packed K written outside logical rows"
            assert (qv[dead] == _SENT_I32).all(), "packed V written outside logical rows"
            assert torch.isnan(sk[dead]).all() and torch.isnan(sv[dead]).all(), "scales written outside logical rows"
            direct_k = torch.full_like(got_k, float("nan"))
            direct_v = torch.full_like(got_v, float("nan"))
            extension.dequant_cache_paged(qk, sk, direct_k, qv, sv, direct_v, seq(total), table, PAGE_SIZE, -1, 0.0)
            torch.cuda.synchronize()
            assert torch.equal(gather(direct_k, order, total), nat_k), "get_kv K diverges from direct dequant"
            assert torch.equal(gather(direct_v, order, total), nat_v), "get_kv V diverges from direct dequant"
            assert torch.isnan(direct_k[dead]).all() and torch.isnan(direct_v[dead]).all(), "dequant hit sentinel rows"
            # Lossy quality vs unquantized tensors: recorded, never asserted as kernel failure.
            rec_k, rec_v = rel_l2(nat_k, base_k[:total]), rel_l2(nat_v, base_v[:total])
            # Extra live rows so q_len=1 uses length=full-1 and q_len=7 uses length=full-7.
            layer.update_kv_direct(seq(total), table, base_k[total:full].unsqueeze(0),
                                   base_v[total:full].unsqueeze(0), _EXTRA)
            torch.cuda.synchronize()
            all_k, all_v = layer.get_kv(seq(full), table)
            torch.cuda.synchronize()
            dq_k, dq_v = gather(all_k, order, full), gather(all_v, order, full)
            for q_len in (1, 7):
                length = full - q_len
                out = torch.empty_like(queries[q_len])
                paged_attn_triton_decode(q=queries[q_len], k=None, v=None, k_cache=qk, v_cache=qv,
                                         block_table=table, cache_seqlens=seq(length), causal=True,
                                         pre_appended_len=q_len, qc=(sk, sv, kb, vb),
                                         n_kv_heads_override=_N_KV, out=out)
                torch.cuda.synchronize()
                assert torch.isfinite(out).all(), ("non-finite online decode", kb, vb, q_len)
                expected = oracle(queries[q_len], dq_k, dq_v, length)
                r = rel_l2(out, expected)
                m = (out.float() - expected.float()).abs().max().item()
                assert r < _DECODE_REL, ("online decode vs FP32 on dequantized cache", kb, vb, q_len, r)
                results.append(dict(pair=[kb, vb], q_len=q_len, decode_rel_l2=r, decode_max_abs=m))
            results.append(dict(pair=[kb, vb], kernel_rel_l2=[k_err, v_err], kernel_max_abs=[k_abs, v_abs],
                                recon_rel_l2=[rec_k, rec_v],
                                packed_bytes=int(qk.numel() * 4 + qv.numel() * 4),
                                scales_bytes=int(sk.numel() * 2 + sv.numel() * 2),
                                packed_bytes_per_token=int((qk.shape[2] + qv.shape[2]) * 4),
                                scales_bytes_per_token=int((sk.shape[2] + sv.shape[2]) * 2),
                                live_rows=total, capacity=_CAPACITY))
            layer.free()

        # Non-default stream: identical packed cache and gathered outputs.
        ref_layer, alt_layer = make_layer(8, 8), make_layer(8, 8)
        ref_layer.update_kv_direct(seq(0), table, base_k[:_L1].unsqueeze(0), base_v[:_L1].unsqueeze(0), _L1)
        ref_layer.update_kv_direct(seq(_L1), table, base_k[_L1:total].unsqueeze(0), base_v[_L1:total].unsqueeze(0), _L2)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            alt_layer.update_kv_direct(seq(0), table, base_k[:_L1].unsqueeze(0), base_v[:_L1].unsqueeze(0), _L1)
            alt_layer.update_kv_direct(seq(_L1), table, base_k[_L1:total].unsqueeze(0),
                                       base_v[_L1:total].unsqueeze(0), _L2)
            alt_k, alt_v = alt_layer.get_kv(seq(total), table)
        stream.synchronize()
        ref_k, ref_v = ref_layer.get_kv(seq(total), table)
        torch.cuda.synchronize()
        rqk, rsk, rqv, rsv, _, _ = ref_layer.get_qkv()
        aqk, ask, aqv, asv, _, _ = alt_layer.get_qkv()
        assert torch.equal(rqk, aqk) and torch.equal(rqv, aqv), "stream changed packed words"
        assert torch.equal(rsk, ask) and torch.equal(rsv, asv), "stream changed scales"
        assert torch.equal(gather(ref_k, order, total), gather(alt_k, order, total)), "stream changed dequant K"
        assert torch.equal(gather(ref_v, order, total), gather(alt_v, order, total)), "stream changed dequant V"
        results.append(dict(nondefault_stream=True, pair=[8, 8]))
        ref_layer.free()
        alt_layer.free()

        # HIP graph replay with changed inputs, after warmup, where supported.
        if not hasattr(torch.cuda, "CUDAGraph"):
            results.append(dict(graph_replay="skipped", reason="torch.cuda.CUDAGraph missing"))
        else:
            try:
                g_layer, r_layer = make_layer(8, 8), make_layer(8, 8)
                gk_in = base_k[:_L1].unsqueeze(0).clone()
                gv_in = base_v[:_L1].unsqueeze(0).clone()
                g_seq = seq(0)
                for _ in range(3):
                    g_layer.update_kv_direct(g_seq, table, gk_in, gv_in, _L1)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    g_layer.update_kv_direct(g_seq, table, gk_in, gv_in, _L1)
                new_k = torch.randn((1, _L1, _N_KV, _HEAD_DIM), generator=gen, dtype=torch.float32).half().to(device)
                new_v = torch.randn((1, _L1, _N_KV, _HEAD_DIM), generator=gen, dtype=torch.float32).half().to(device)
                gk_in.copy_(new_k)
                gv_in.copy_(new_v)
                graph.replay()
                torch.cuda.synchronize()
                r_layer.update_kv_direct(g_seq, table, new_k, new_v, _L1)
                torch.cuda.synchronize()
                gqk, gsk, gqv, gsv, _, _ = g_layer.get_qkv()
                xqk, xsk, xqv, xsv, _, _ = r_layer.get_qkv()
                assert torch.equal(gqk, xqk) and torch.equal(gqv, xqv), "graph replay ignored changed K/V inputs"
                assert torch.equal(gsk, xsk) and torch.equal(gsv, xsv), "graph replay ignored changed scale inputs"
                results.append(dict(graph_replay=True, path="quant_cache_paged", rows=_L1))
                g_layer.free()
                r_layer.free()
            except RuntimeError:
                # A capture/replay failure on an exposed API is a failed check,
                # not evidence that the platform may silently skip validation.
                raise

        if hasattr(extension, "TritonKernel"):
            # Exercise the actual BC ASTSource builder after adding optional
            # nonuniform parameters. Its uniform kernel must still compile and
            # load through the native bridge, with no extra runtime pointer.
            from exllamav3.modules.attention_fn.bc_attn import BCAttn
            bridge = BCAttn.__new__(BCAttn)
            bridge.device, bridge.head_dim = device, _HEAD_DIM
            bridge.num_q_heads, bridge.num_kv_heads = _N_Q, _N_KV
            bridge.k_bits, bridge.v_bits, bridge.quant = 8, 8, True
            bridge.window_size, bridge.sm_scale, bridge.softcap = None, _HEAD_DIM ** -0.5, 0.0
            bridge.sinks, bridge.gate_mode, bridge.g_weight = None, 0, None
            bridge.hidden_size = bridge.hidden_padded = _N_Q * _HEAD_DIM
            bridge.o_dtype = torch.float16
            slots = []
            bridge.bc = SimpleNamespace(configure_slot=lambda *args: slots.append(args))
            bridge._configure(1, 1, True)
            assert len(slots) == 1 and slots[0][9] is not None and slots[0][10] is not None
            results.append(dict(uniform_bc_compile_and_load=True, note="Compilation/bridge loading, not full BC model execution"))
    return results
