"""Independent numerical, tail, stream and external-capture checks for ABI 1.

The public validator owns binary verification, execution permission, GPU lease,
memory caps and timeout. No cooperative or native MLP graph is needed here.
"""
import numpy as np

from kernels.exl3.check_smallm import environment
from quantlab.methods.exl3.oracle import decode_trellis, reconstruct


def run_checks(torch, extension):
    rng = np.random.default_rng(926)
    records = []

    def fixture(bits, cb, k, n):
        packed = rng.integers(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=np.int16)
        su = rng.choice([-0.25, 0.25], k).astype(np.float16)
        sv = rng.choice([-0.5, 0.5], n).astype(np.float16)
        tensors = tuple(torch.from_numpy(t).cuda() for t in (packed, su, sv))
        decoded = torch.empty((k, n), device='cuda', dtype=torch.float16)
        extension.reconstruct(decoded, tensors[0], bits, False, cb == 2)
        np.testing.assert_array_equal(decoded.cpu().float().numpy(), decode_trellis(packed, bits, codebook=cb))
        return tensors, reconstruct(packed, bits, su, sv, codebook=cb).astype(np.float64)

    def check(bits, cb, tensors, weight, rows, dtype, warps, capture=False):
        b, su, sv = tensors
        k, n = weight.shape
        cpu = (rng.standard_normal((rows, k)) * 0.1).astype(np.float16)
        x = torch.from_numpy(cpu).cuda()
        ah_storage = torch.full((rows * k + 32,), 23., dtype=torch.float16, device='cuda')
        ah = ah_storage[16:-16].view(rows, k)
        storage = torch.full((rows * n + 32,), 23., dtype=dtype, device='cuda')
        y = storage[16:-16].view(rows, n)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def call():
            tag = extension.exl3_gemm(x, b, y, su, ah, sv, -1, False, cb == 2, 0)
            assert tag == 90, ('unexpected dispatch', rows, bits, cb, tag)

        with environment(EXL3_GEMV_SPLITK_WARPS=str(warps), EXL3_SMALLM_MLP_WARPS=str(warps)):
            with torch.cuda.stream(stream):
                call()  # Also prewarms the native GEMV parameter allocation.
                if capture:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        call()
                    cpu = (-cpu * np.float16(0.5)).astype(np.float16)
                    x.copy_(torch.from_numpy(cpu).cuda())
                    graph.replay()
            stream.synchronize()
        got = y.float().cpu().numpy().astype(np.float64)
        expected = cpu.astype(np.float64) @ weight
        error = float(np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1e-12))
        assert np.isfinite(got).all() and error < 0.005, (bits, cb, rows, warps, error)
        for buffer in (storage, ah_storage):
            assert bool((buffer[:16] == 23).all() and (buffer[-16:] == 23).all()), 'Canary overwritten'
        records.append(dict(bits=bits, codebook=cb, k=k, n=n, rows=rows, dtype=str(dtype),
                            warps=warps, capture=capture, nondefault_stream=True, relative_l2=error))

    with environment(EXL3_GEMV='2', EXL3_SMALLM='1', EXL3_PACKED_MID='1',
                     EXL3_SMALLM_WMMA='0', EXL3_GEMV_SPLITK='1'):
        for cb, widths in ((0, (2, 3, 4)), (2, (2, 3, 4, 5, 6))):
            for bits in widths:
                tensors, weight = fixture(bits, cb, 128, 256)
                # Every admitted row count, both output precisions, all split-K counts.
                for rows in range(10, 65):
                    for dtype in (torch.float16, torch.float32):
                        check(bits, cb, tensors, weight, rows, dtype, (4, 8, 16)[rows % 3])
                for rows in (10, 17, 33, 63, 64):
                    check(bits, cb, tensors, weight, rows, torch.float32, 8, capture=True)
        # Uneven split-K chunks and multiple Hadamard groups catch tile/address errors.
        for bits, cb in ((2, 0), (5, 2), (6, 2)):
            tensors, weight = fixture(bits, cb, 640, 384)
            for warps in (4, 8, 16):
                check(bits, cb, tensors, weight, 33, torch.float16, warps, capture=True)
    return records
