"""Independent CPU-oracle checks for tiled packed-prefill ABI 1.

Run through check_exl3_kernels.py for verified loading, GPU ownership and limits.
"""
import numpy as np

from kernels.exl3.check_smallm import environment
from quantlab.methods.exl3.oracle import reconstruct


def run_checks(torch, extension):
    rng = np.random.default_rng(927)
    records = []

    def fixture(bits, cb, k, n):
        packed = rng.integers(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=np.int16)
        su = rng.choice([-0.25, 0.25], k).astype(np.float16)
        sv = rng.choice([-0.5, 0.5], n).astype(np.float16)
        tensors = tuple(torch.from_numpy(t).cuda() for t in (packed, su, sv))
        weight = reconstruct(packed, bits, su, sv, codebook=cb).astype(np.float64)
        return tensors, weight

    def check(bits, cb, tensors, weight, rows, dtype, capture=False):
        b, su, sv = tensors
        k, n = weight.shape
        cpu = (rng.standard_normal((rows, k)) * 0.1).astype(np.float16)
        x = torch.from_numpy(cpu).cuda()
        scratch = torch.full((rows * k + 32,), 23., dtype=torch.float16, device='cuda')
        ah = scratch[16:-16].view(rows, k)
        storage = torch.full((rows * n + 32,), 23., dtype=dtype, device='cuda')
        y = storage[16:-16].view(rows, n)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def call():
            tag = extension.exl3_gemm(x, b, y, su, ah, sv, -1, False, cb == 2, 0)
            assert tag == 90, ('unexpected dispatch', rows, bits, cb, tag)

        with torch.cuda.stream(stream):
            call()
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
        assert np.isfinite(got).all() and error < 0.005, (bits, cb, rows, k, error)
        for buffer in (storage, scratch):
            assert bool((buffer[:16] == 23).all() and (buffer[-16:] == 23).all()), 'Canary overwritten'
        records.append(dict(bits=bits, codebook=cb, k=k, n=n, rows=rows, dtype=str(dtype),
                            capture=capture, nondefault_stream=True, relative_l2=error))

    with environment(EXL3_GEMV='2', EXL3_SMALLM='1', EXL3_PACKED_PREFILL='1'):
        for cb, widths in ((0, (2, 3, 4)), (2, (2, 3, 4, 5, 6))):
            for bits in widths:
                tensors, weight = fixture(bits, cb, 384, 256)
                for rows in (65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1024, 1025, 2048, 4096):
                    for dtype in (torch.float16, torch.float32):
                        check(bits, cb, tensors, weight, rows, dtype, capture=rows in (65, 129, 513))
        # Long reductions catch WMMA error accumulation and the final partial fold.
        for k in (4096, 12288, 17408, 17536):
            tensors, weight = fixture(5, 2, k, 128)
            for dtype in (torch.float16, torch.float32):
                check(5, 2, tensors, weight, 129, dtype, capture=True)
    return records
