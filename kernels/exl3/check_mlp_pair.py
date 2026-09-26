"""Paired MLP: exact separate-path equivalence, guards, streams and replay.

The separate path is also compared with independently decoded CPU weights.
"""
import numpy as np

from kernels.exl3.check_smallm import environment
from quantlab.methods.exl3.oracle import reconstruct


def run_checks(torch, extension):
    rng = np.random.default_rng(928)
    records = []

    def fixture(bits, cb, k, n):
        packed = rng.integers(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=np.int16)
        su = rng.choice([-0.25, 0.25], k).astype(np.float16)
        sv = rng.choice([-0.5, 0.5], n).astype(np.float16)
        tensors = tuple(torch.from_numpy(t).cuda() for t in (packed, su, sv))
        return tensors, reconstruct(packed, bits, su, sv, codebook=cb).astype(np.float64)

    def check(bits, cb, gate, up, wg, wu, rows, warps, capture):
        k, n = wg.shape
        cpu = (rng.standard_normal((rows, k)) * 0.1).astype(np.float16)
        x = torch.from_numpy(cpu).cuda()
        buffers = []

        def guarded(shape):
            buffer = torch.full((int(np.prod(shape)) + 32,), 23., device='cuda', dtype=torch.half)
            buffers.append(buffer)
            return buffer[16:-16].view(shape)

        xh, gu, y = guarded((2, rows, k)), guarded((2, rows, n)), guarded((rows, n))
        b1, su1, sv1 = gate
        b2, su2, sv2 = up
        args = [x, b1, b2, su1, su2, sv1, sv2, xh, gu, y, bits, cb == 2]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with environment(EXL3_GEMV_SPLITK='0' if warps == 1 else '1',
                         EXL3_GEMV_SPLITK_WARPS=str(max(4, warps))):
            with torch.cuda.stream(stream):
                extension.exl3_mlp_gate_up(*args)
                if capture:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        extension.exl3_mlp_gate_up(*args)
                    cpu = (-cpu * np.float16(0.5)).astype(np.float16)
                    x.copy_(torch.from_numpy(cpu).cuda())
                    graph.replay()
                g, u = torch.empty_like(y), torch.empty_like(y)
                ref_xh = torch.empty_like(xh)
                for i, (b, su, sv, dest) in enumerate(((b1, su1, sv1, g), (b2, su2, sv2, u))):
                    tag = extension.exl3_gemm(x, b, dest, su, ref_xh[i], sv, -1, False, cb == 2, 0)
                    assert tag == 90, ('separate dispatch', tag)
                expected = torch.empty_like(y)
                extension.silu_mul(g, u, expected, 0.)
            stream.synchronize()
        assert torch.equal(xh, ref_xh), ('input Hadamard changed', bits, cb, rows, warps)
        assert torch.equal(y, expected), ('fused output changed', bits, cb, rows, warps,
                                          float((y.float() - expected.float()).abs().max()))
        errors = []
        for dest, weight in ((g, wg), (u, wu)):
            ref = cpu.astype(np.float64) @ weight
            got = dest.float().cpu().numpy().astype(np.float64)
            error = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12))
            assert np.isfinite(got).all() and error < 0.005, (bits, cb, rows, error)
            errors.append(error)
        for buffer in buffers:
            assert bool((buffer[:16] == 23).all() and (buffer[-16:] == 23).all()), 'Canary overwritten'
        records.append(dict(bits=bits, codebook=cb, k=k, n=n, rows=rows, warps=warps,
                            capture=capture, nondefault_stream=True, exact_separate=True,
                            relative_l2=errors))
        return args

    with environment(EXL3_GEMV='2', EXL3_SMALLM='1', EXL3_SMALLM_WMMA='0',
                     EXL3_GEMV_LDS='0', EXL3_GEMV_GRAPH='0'):
        for cb, widths in ((0, (2, 3, 4)), (2, (2, 3, 4, 5, 6))):
            for bits in widths:
                gate, wg = fixture(bits, cb, 640, 384)
                up, wu = fixture(bits, cb, 640, 384)
                for rows in (1, 2, 3, 5):
                    for warps in (1, 4, 8, 16):
                        args = check(bits, cb, gate, up, wg, wu, rows, warps, capture=warps == 8)
        for k, n in ((4096, 256), (12288, 128)):
            gate, wg = fixture(5, 2, k, n)
            up, wu = fixture(5, 2, k, n)
            for rows in (1, 3, 5):
                args = check(5, 2, gate, up, wg, wu, rows, 8, capture=True)

        # Reject malformed buffers before launching. Rejected calls must leave
        # the previous output intact and must not poison the GPU stream.
        y_before = args[9].clone()
        variants = []
        for idx, value, label in (
                (0, args[0].cpu(), 'CPU input'), (1, args[1].float(), 'wrong packed dtype'),
                (2, args[2][:, :-1].contiguous(), 'wrong packed shape'),
                (3, args[3][:-1], 'wrong scales'), (7, args[7][:, :-1], 'wrong scratch'),
                (9, args[8][0], 'overlapping buffers'), (10, 7, 'unsupported bits'),
                (11, False, 'unsupported codebook')):
            bad = list(args)
            bad[idx] = value
            variants.append((label, bad))
        bad = list(args)
        bad[9] = torch.empty(args[9].numel() + 1, device='cuda', dtype=torch.half)[1:].view_as(args[9])
        variants.append(('misaligned output', bad))
        for label, bad in variants:
            try:
                extension.exl3_mlp_gate_up(*bad)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError('Expected rejection: ' + label)
            torch.cuda.synchronize()
            assert torch.equal(args[9], y_before), label
            records.append(dict(rejected=label))
    return records
