"""AsterKV GPU correctness checks; caller verifies the extension and owns the GPU.

No models or external data are loaded. Numerical kernel correctness is checked
against an independent unpacker and NumPy quantizer, separately from lossy error.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def run_checks(torch, extension):
    import numpy as np
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.cache import Cache
    from exllamav3.cache.aster import CacheLayer_aster
    from exllamav3.modules.attention_fn import triton_paged as attn
    from exllamav3.cache.aster_codebook import ASTER5_LLOYD_CENTROIDS
    from exllamav3.modules.attention_fn import attn_dispatch

    spec = importlib.util.spec_from_file_location("aster_reference", Path(__file__).with_name("asterkv_reference.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = SimpleNamespace(num_kv_heads=4, head_dim=256)
    capacity, nq, nk, hd, bsz = 2048, 24, 4, 256, 2
    length = 277
    order = [[2, 0, 6], [1, 7, 5]]
    table = torch.tensor(order, dtype=torch.int32, device=device)
    hs = torch.tensor(ref.hadamard_matrix32(), dtype=torch.float32, device=device)
    results = []

    def layer(cls=CacheLayer_aster):
        obj = cls(None, geometry, 0, capacity, 5, 5)
        obj.alloc(device)
        return obj

    def seq(value):
        return torch.full((bsz,), value, dtype=torch.int32, device=device)

    def gather(paged, count=length):
        return torch.stack([torch.cat([paged[p] for p in pages])[:count] for pages in order])

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

    def unpack(qwords):
        words = qwords.reshape(bsz, length, -1, 5).long()
        d = torch.arange(32, device=device)
        hi = (words[..., d // 8] >> ((d % 8) * 4)) & 15
        lo = (words[..., 4, None] >> d) & 1
        return ((hi << 1) | lo).reshape(bsz, length, nk, hd)

    def independent_dequant(obj, words, scales):
        codes = unpack(gather(words))
        rotated = obj.codebook_tensor[codes].reshape(-1, 32) * gather(scales).reshape(-1, 1)
        return (rotated @ hs.T).reshape(bsz, length, nk, hd).half()

    def oracle(q, k, v, window=-1):
        qlen = q.shape[1]
        grouped = q.float().reshape(bsz, qlen, nk, nq // nk, hd)
        scores = torch.einsum("bqghd,btgd->bqght", grouped, k.float()) / hd ** 0.5
        pos = length - qlen + torch.arange(qlen, device=device)
        keys = torch.arange(length, device=device)
        valid = keys[None, :] <= pos[:, None]
        if window >= 0:
            valid &= keys[None, :] >= pos[:, None] - window
        probs = scores.masked_fill(~valid[None, :, None, None, :], -float("inf")).softmax(-1)
        return torch.einsum("bqght,btgd->bqghd", probs, v.float()).reshape_as(q)

    def span_oracle(q, k, v, past_len, spans):
        outs = []
        for span in spans:
            a, b, nc = span[:3]
            span_q = q[:, a:b]
            span_len = b - a
            readable = past_len + b
            span_k, span_v = k[:, :readable], v[:, :readable]
            grouped = span_q.float().reshape(bsz, span_len, nk, nq // nk, hd)
            scores = torch.einsum("bqghd,btgd->bqght", grouped, span_k.float()) / hd ** 0.5
            pos = past_len + a + torch.arange(span_len, device=device)
            keys = torch.arange(readable, device=device)
            if nc:
                valid = torch.ones((span_len, readable), dtype=torch.bool, device=device)
            else:
                valid = keys[None, :] <= pos[:, None]
            probs = scores.masked_fill(~valid[None, :, None, None, :], -float("inf")).softmax(-1)
            outs.append(torch.einsum("bqght,btgd->bqghd", probs, span_v.float()).reshape_as(span_q))
        return torch.cat(outs, dim=1)

    def attention(obj, q, past, *, window=-1, splits=2, out=None):
        fn = attn.paged_attn_triton_decode if q.shape[1] <= 16 else attn.paged_attn_triton_prefill
        return fn(q, None, None, obj.qk, obj.qv, table, past,
                  causal=True, window_size=(window, 0) if window >= 0 else None,
                  qc=(obj.sk, obj.sv, 5, 5), codebook=obj.codebook_tensor, polynomial=obj.polynomial,
                  pre_appended_len=q.shape[1], n_kv_heads_override=nk,
                  num_splits=splits, out=out)

    with torch.inference_mode(), torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(190927)
        remapping_attention = SimpleNamespace(layer_idx=0, cache_layer_type=lambda default, kw: (CacheLayer_quant, kw))
        incompatible_model = SimpleNamespace(config=None, get_cache_layers=lambda: [remapping_attention])
        try:
            Cache(incompatible_model, capacity, layer_type=CacheLayer_aster, k_bits=5, v_bits=5)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("architecture silently substituted a uniform codec")
        obj = layer()
        # Canaries prove writes stay within mapped logical rows, including partial pages.
        for words in (obj.qk, obj.qv):
            words.fill_(0x61616161)
        for scales in (obj.sk, obj.sv):
            scales.fill_(float("nan"))
        k = torch.randn((bsz, length, nk, hd), device=device, dtype=torch.float16)
        v = torch.randn_like(k)
        # Zero, tiny, constant and high-dynamic-range groups exercise scale behavior.
        for data in (k, v):
            data[:, 0] = 0
            data[:, 1] *= 1e-5
            data[:, 2] = 3.0
            data[:, 3, :, 0] = 400.0
        obj.update_kv_direct(seq(0), table, k[:, :250], v[:, :250], 250)
        obj.update_kv_direct(seq(250), table, k[:, 250:], v[:, 250:], length - 250)
        torch.cuda.synchronize()
        expected_k = independent_dequant(obj, obj.qk, obj.sk)
        expected_v = independent_dequant(obj, obj.qv, obj.sv)
        deq_k, deq_v = obj.get_kv(seq(length), table)
        for name, actual, expected, original, words in (
            ("k", gather(deq_k), expected_k, k, obj.qk),
            ("v", gather(deq_v), expected_v, v, obj.qv),
        ):
            assert torch.isfinite(actual).all(), f"{name} nonfinite dequant"
            error = rel(actual, expected)
            assert error < 8e-4, ("unpack/inverse rotation", name, error)
            reference, codes, _ = ref.nonuniform_roundtrip(original.cpu().numpy(), obj.centroids)
            assignments = unpack(gather(words)).cpu().numpy()
            assert np.all(assignments[:, 0] == 15), "zero midpoint tie must select lower code"
            constant_codes = assignments[:, 2].reshape(-1, 32)
            assert np.all(constant_codes[:, 0] == 31) and np.all(constant_codes[:, 1:] == 15), "constant-group tie rule"
            mismatch = float(np.mean(assignments != codes))
            # Matrix multiplication and butterfly summation can land on opposite
            # sides of exact midpoint ties. Accept only adjacent codes at a
            # numerically indistinguishable threshold, not a global loose tolerance.
            rotated = ref.hadamard_rotate(original.cpu().numpy()).reshape(-1, 32)
            normalized = rotated / (np.abs(rotated).max(axis=1, keepdims=True) + 1e-10)
            wanted, got = codes.reshape(-1, 32), assignments.reshape(-1, 32)
            changed = wanted != got
            grid = np.asarray(obj.centroids)
            boundary = (grid[wanted] + grid[got]) * 0.5
            assert np.all(np.abs(wanted.astype(int)[changed] - got[changed]) == 1), "nonadjacent encoder mismatch"
            assert np.all(np.abs(normalized[changed] - boundary[changed]) < 2e-6), "encoder mismatch away from midpoint"
            # Random, non-degenerate groups additionally constrain reconstructed
            # error; exact-zero/constant inputs have separate checks above/below.
            cpu_error = rel(actual[:, 4:], torch.tensor(reference[:, 4:], device=device))
            assert cpu_error < 2e-3, ("end-to-end reference", name, cpu_error)
            assert torch.count_nonzero(actual[:, 0]) == 0, "zero groups must remain zero"
            results.append(dict(side=name, unpack_rel_l2=error, reference_rel_l2=cpu_error,
                                code_mismatch_fraction=mismatch, lossy_rel_l2=rel(actual, original)))
        live = torch.zeros((capacity // 256, 256), dtype=torch.bool, device=device)
        for pages in order:
            live[pages[0], :] = True
            live[pages[1], :length - 256] = True
        assert (obj.qk[~live] == 0x61616161).all() and (obj.qv[~live] == 0x61616161).all()
        assert torch.isnan(obj.sk[~live]).all() and torch.isnan(obj.sv[~live]).all()
        measured = sum(t.numel() * t.element_size()
                       for t in obj.get_tensors() + obj.get_metadata_tensors())
        assert measured == obj.storage_size() == capacity * 1408 + 128
        assert all(t.shape[0] == capacity // 256 for t in obj.get_tensors()), "metadata exposed as page storage"
        results.append(dict(packed_and_metadata_bytes=measured, partial_page_canaries=True))

        # Test physical-input updates independently from the contiguous-input writer.
        paged_k, paged_v = torch.zeros_like(deq_k), torch.zeros_like(deq_v)
        for b, pages in enumerate(order):
            paged_k[pages[0]] = k[b, :256]
            paged_v[pages[0]] = v[b, :256]
            paged_k[pages[1], :length - 256] = k[b, 256:]
            paged_v[pages[1], :length - 256] = v[b, 256:]
        second = layer()
        second.update_kv(seq(0), table, paged_k, paged_v, length)
        for a, b in zip(obj.get_tensors()[:4], second.get_tensors()[:4]):
            assert torch.equal(a[live], b[live]), "physical-input writer disagrees"
        second.copy_page(obj, 2, 3, 113)
        assert torch.equal(second.qk[3, :113], obj.qk[2, :113])
        uniform = layer(CacheLayer_quant)
        for dest, source in ((obj, uniform), (uniform, obj)):
            try:
                dest.copy_page(source, 0, 1, 1)
            except ValueError:
                pass
            else:
                raise AssertionError("mixed codecs were reinterpreted during page copy")
        for src_page, dst_page in ((-1, 0), (0, -1), (8, 0), (0, 8)):
            try:
                second.copy_page(obj, src_page, dst_page, 1)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid page copy accepted")
        uniform.free()
        results.append(dict(physical_input=True, page_copy=True, incompatible_copy_rejected=True))

        # Both split/combine and direct-output attention, masked tails, windowed
        # decode/prefill, and MTP-length decodes.
        for qlen, window, splits in ((1, -1, 1), (2, -1, 2), (3, -1, 2), (4, -1, 2),
                                     (5, -1, 2), (6, -1, 2), (7, -1, 2), (7, 71, 2),
                                     (8, -1, 2), (17, -1, 1), (257, -1, 2), (33, 71, 2)):
            q = torch.randn((bsz, qlen, nq, hd), device=device, dtype=torch.float16) * 0.125
            out = attention(obj, q, seq(length - qlen), window=window, splits=splits)
            expected = oracle(q, expected_k, expected_v, window)
            error = rel(out, expected)
            assert torch.isfinite(out).all() and error < 4e-3, ("attention", qlen, window, splits, error)
            results.append(dict(q_len=qlen, window=window, splits=splits, attention_rel_l2=error))
        window_k, window_v = obj.get_kv(seq(length), table, sliding_window=19)
        assert rel(gather(window_k)[:, -19:], expected_k[:, -19:]) < 8e-4
        assert rel(gather(window_v)[:, -19:], expected_v[:, -19:]) < 8e-4

        # Original Lloyd lookup grid (polynomial=None) regression: same packing
        # layout, independent dequant, then decode + prefill oracle checks.
        lloyd = CacheLayer_aster(None, geometry, 0, capacity, 5, 5,
                                 centroids=ASTER5_LLOYD_CENTROIDS)
        lloyd.alloc(device)
        assert lloyd.polynomial is None
        assert len(lloyd.get_qkv()) == 8 and lloyd.get_qkv()[7] is None
        lloyd.update_kv_direct(seq(0), table, k[:, :250], v[:, :250], 250)
        lloyd.update_kv_direct(seq(250), table, k[:, 250:], v[:, 250:], length - 250)
        torch.cuda.synchronize()
        expected_k_lloyd = independent_dequant(lloyd, lloyd.qk, lloyd.sk)
        expected_v_lloyd = independent_dequant(lloyd, lloyd.qv, lloyd.sv)
        deq_k_lloyd, deq_v_lloyd = lloyd.get_kv(seq(length), table)
        assert rel(gather(deq_k_lloyd), expected_k_lloyd) < 8e-4
        assert rel(gather(deq_v_lloyd), expected_v_lloyd) < 8e-4
        for lloyd_qlen in (1, 17):
            q = torch.randn((bsz, lloyd_qlen, nq, hd), device=device, dtype=torch.float16) * 0.125
            out = attention(lloyd, q, seq(length - lloyd_qlen))
            expected = oracle(q, expected_k_lloyd, expected_v_lloyd)
            error = rel(out, expected)
            assert torch.isfinite(out).all() and error < 4e-3, ("lloyd attention", lloyd_qlen, error)
            results.append(dict(centroids="lloyd", q_len=lloyd_qlen, attention_rel_l2=error))
        lloyd.free()

        # Serving-path dispatch over the Aster 8-tuple: decode, prefill, and a
        # mixed causal/non-causal span chunk, each against the FP32 oracle.
        disp = layer()
        disp.update_kv_direct(seq(0), table, k[:, :250], v[:, :250], 250)
        disp.update_kv_direct(seq(250), table, k[:, 250:], v[:, 250:], length - 250)
        torch.cuda.synchronize()
        assert len(disp.get_qkv()) == 8
        expected_k_disp = independent_dequant(disp, disp.qk, disp.sk)
        expected_v_disp = independent_dequant(disp, disp.qv, disp.sv)
        for disp_qlen, want in ((1, "fn_triton_paged_attn_decode_qc"),
                                (17, "fn_triton_paged_attn_prefill_qc")):
            q = torch.randn((bsz, disp_qlen, nq, hd), device=device, dtype=torch.float16) * 0.125
            past_len = length - disp_qlen
            hint = {}
            out = attn_dispatch(q, k[:, past_len:].contiguous(), v[:, past_len:].contiguous(),
                                cache=disp, block_table=table,
                                cache_seqlens=seq(past_len), dispatch_cache=hint)
            assert hint.get("fn_qc") is not None and hint["fn_qc"].__name__ == want, hint
            expected = oracle(q, expected_k_disp, expected_v_disp)
            error = rel(out, expected)
            assert torch.isfinite(out).all() and error < 4e-3, ("dispatch", disp_qlen, error)
            results.append(dict(dispatch=want, q_len=disp_qlen, attention_rel_l2=error))
        spans = [(0, 4, True), (4, 8, False)]
        q = torch.randn((bsz, 8, nq, hd), device=device, dtype=torch.float16) * 0.125
        hint = {}
        out = attn_dispatch(q, k[:, length - 8:].contiguous(), v[:, length - 8:].contiguous(),
                            cache=disp, block_table=table, cache_seqlens=seq(length - 8),
                            non_causal_spans=spans, dispatch_cache=hint)
        assert hint.get("fn_qc") is not None, hint
        assert hint["fn_qc"].__name__ == "fn_triton_paged_attn_prefill_qc", hint
        expected = span_oracle(q, expected_k_disp, expected_v_disp, length - 8, spans)
        error = rel(out, expected)
        assert torch.isfinite(out).all() and error < 4e-3, ("dispatch spans", error)
        results.append(dict(dispatch="fn_triton_paged_attn_prefill_qc", spans=True, attention_rel_l2=error))
        disp.free()

        # Nondefault stream plus graph replay: changed K/V, query, and lengths.
        q = torch.randn((bsz, 1, nq, hd), device=device, dtype=torch.float16) * 0.125
        past = seq(length - 1)
        new_k, new_v = k[:, -1:].clone(), v[:, -1:].clone()
        output = torch.empty_like(q)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                second.update_kv_direct(past, table, new_k, new_v, 1)
                attention(second, q, past, out=output)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            second.update_kv_direct(past, table, new_k, new_v, 1)
            attention(second, q, past, out=output)
        past.fill_(length - 2)
        q.mul_(1.2)
        new_k.neg_()
        new_v.add_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        replay = output.clone()
        obj.update_kv_direct(past, table, new_k, new_v, 1)
        eager = attention(obj, q, past)
        assert torch.equal(gather(obj.qk), gather(second.qk)), "graph encoder used stale inputs"
        assert torch.equal(replay, eager), ("graph attention used stale state", rel(replay, eager))
        results.append(dict(nondefault_stream=True, graph_replay_changed_state=True))
        # Construct a mixed policy and run dispatch through the real Cache lookup.
        modules = [SimpleNamespace(layer_idx=i, num_kv_heads=nk, head_dim=hd, cache_layers=[])
                   for i in (3, 7, 11)]
        fake = SimpleNamespace(config=None, cache_weakrefs={}, recurrent_state_cls=None,
            get_cache_layers=lambda: modules, get_recurrent_layers=lambda: [],
            get_layer_instances=lambda i: [(i, 0)])
        selected = Cache(fake, capacity, layer_type=CacheLayer_aster, k_bits=5, v_bits=5,
                         layer_overrides={7: dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=6, compand_a=0),
                                          11: dict(layer_type=CacheLayer_quant, k_bits=6, v_bits=8, compand_a=0)})
        assert type(selected.layers[3, 0]) is CacheLayer_aster
        for idx in (3, 7, 11):
            selected_layer = selected.layers[idx, 0]
            selected_layer.alloc(device)
            selected_layer.update_kv_direct(seq(0), table, k, v, length)
            dk, dv = selected_layer.get_kv(seq(length), table)
            query = torch.randn((bsz, 7, nq, hd), device=device, dtype=torch.float16) * .125
            hints = {}
            actual = attn_dispatch(query, k[:, -7:].contiguous(), v[:, -7:].contiguous(),
                                   cache=selected, cache_idx=idx, block_table=table,
                                   cache_seqlens=seq(length-7), dispatch_cache=hints)
            error = rel(actual, oracle(query, gather(dk), gather(dv)))
            assert error < .003, (idx, error)
            results.append(dict(check='policy_layer_dispatch', layer_idx=idx,
                                k_bits=selected_layer.k_bits, v_bits=selected_layer.v_bits, relative_l2=error))
            selected_layer.free()
        selected.detach_from_model()
        assert not fake.cache_weakrefs and all(not m.cache_layers for m in modules)
        try:
            Cache(fake, capacity, layer_overrides={999: dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)})
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown policy layer was silently ignored')
        obj.free()
        second.free()
        try:
            obj.get_qkv()
        except RuntimeError:
            pass
        else:
            raise AssertionError("freed cache accessor did not fail explicitly")
    return results
