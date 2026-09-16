"""Packed projection checks for a verified extension under an acquired GPU lease.

The caller owns binary loading, permissions, timeouts and resource limits.
Synthetic weights avoid model downloads. References decode independently on CPU.
"""
import os
from contextlib import contextmanager

import numpy as np

from quantlab.methods.exl3.oracle import decode_trellis, reconstruct


@contextmanager
def environment(**values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_checks(torch, extension):
    """Check cb0/cb2 decode, dot/WMMA, output bounds, streams and graph replay."""
    rng = np.random.default_rng(2916)
    device = torch.device('cuda', torch.cuda.current_device())
    results = []

    def fixture(bits, cb, k, n):
        packed = rng.integers(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=np.int16)
        su = rng.choice([-0.25, 0.25], k).astype(np.float16)
        sv = rng.choice([-0.5, 0.5], n).astype(np.float16)
        tensors = tuple(torch.from_numpy(t).to(device) for t in (packed, su, sv))
        decoded = torch.empty((k, n), device=device, dtype=torch.float16)
        extension.reconstruct(decoded, tensors[0], bits, False, cb == 2)
        np.testing.assert_array_equal(decoded.cpu().float().numpy(), decode_trellis(packed, bits, codebook=cb))
        reference = reconstruct(packed, bits, su, sv, codebook=cb)
        return tensors, reference

    def compare(actual, reference, label, tolerance=0.005):
        value = actual.detach().float().cpu().numpy().astype(np.float64)
        assert np.isfinite(value).all(), (label, 'nonfinite')
        delta = value - reference
        relative = float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12))
        assert relative < tolerance, (label, relative, float(np.max(np.abs(delta))))
        return relative

    def projection(bits, cb, tensors, reference, rows, dtype, kernel, *, capture=False, nondefault=False):
        b, su, sv = tensors
        k, n = reference.shape
        cpu = (rng.standard_normal((rows, k)) * 0.1).astype(np.float16)
        x = torch.from_numpy(cpu).to(device)
        ah = torch.empty_like(x)
        # A contiguous subview with guards catches linear out-of-bounds stores.
        storage = torch.full((rows * n + 32,), 23.0, dtype=dtype, device=device)
        y = storage[16:-16].view(rows, n)
        stream = torch.cuda.Stream() if nondefault else torch.cuda.current_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with environment(EXL3_SMALLM_WMMA=str(kernel)):
            with torch.cuda.stream(stream):
                tag = extension.exl3_gemm(x, b, y, su, ah, sv, -1, False, cb == 2, 0)
                assert tag == 90, ('unexpected dispatch', tag, rows, cb)
                if capture:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        extension.exl3_gemm(x, b, y, su, ah, sv, -1, False, cb == 2, 0)
                    # Change values after capture: stale-output replay must fail.
                    cpu = (-cpu * np.float16(0.5)).astype(np.float16)
                    x.copy_(torch.from_numpy(cpu).to(device))
                    graph.replay()
            stream.synchronize()
        label = dict(bits=bits, codebook=cb, rows=rows, k=k, n=n,
                     dtype=str(dtype), kernel=kernel, capture=capture, nondefault=nondefault)
        error = compare(y, cpu.astype(np.float64) @ reference.astype(np.float64), label)
        assert bool((storage[:16] == 23).all()) and bool((storage[-16:] == 23).all()), label
        results.append(dict(**label, relative_l2=error))

    with environment(EXL3_SMALLM='1', EXL3_GEMV='2', EXL3_GEMV_GRAPH='1',
                     EXL3_SMALLM_GRAPH='1', EXL3_ROCM_HIP_GRAPHS='1', EXL3_GEMV_SPLITK='1'):
        for cb in (0, 2):
            for bits in (2, 3, 4):
                tensors, reference = fixture(bits, cb, 128, 256)
                for rows in range(1, 10):
                    for dtype in (torch.float16, torch.float32):
                        for kernel in (0, 1, 2):
                            projection(bits, cb, tensors, reference, rows, dtype, kernel)
                for kernel in (0, 1, 2):
                    projection(bits, cb, tensors, reference, 9, torch.float32, kernel, capture=True, nondefault=True)
        for warps in ('4', '8', '16'):
            with environment(EXL3_GEMV_SPLITK_WARPS=warps):
                tensors, reference = fixture(3, 2, 640, 128)
                for kernel in (0, 1, 2):
                    projection(3, 2, tensors, reference, 7, torch.float32, kernel, nondefault=True)
        with environment(EXL3_GEMV_SPLITK='0'):
            tensors, reference = fixture(4, 2, 256, 128)
            for kernel in (0, 1, 2):
                projection(4, 2, tensors, reference, 5, torch.float16, kernel)
        results.extend(_native_graph_checks(torch, extension, fixture, compare, device, rng))
    return results


def _native_graph_checks(torch, extension, fixture, compare, device, rng):
    """Exercise the runtime's own graph parameter patching through a gated MLP."""
    results = []
    k = 128
    for cb in (0, 2):
        for kernel in (0, 1, 2):
            with environment(EXL3_SMALLM_WMMA=str(kernel)):
                weights = [fixture(bits, cb, k, k) for bits in (2, 3, 4)]
                handles = [extension.BC_LinearEXL3(*tensors, bits, None, False, cb == 2,
                           torch.empty((1, k), dtype=torch.float16, device=device))
                           for bits, (tensors, _) in zip((2, 3, 4), weights)]
                def empty(*shape):
                    return torch.empty(shape, device=device, dtype=torch.float16)
                mlp = extension.BC_GatedMLP(empty(2, 8, k), empty(2, 8, k), empty(1, 8, k), empty(1, 8, k),
                                          None, None, None, 2, False, cb == 2, True, False, False,
                                          *handles, 0.0)
                for rows in (2, 5, 8):
                    # First call warms up, second captures, third patches new pointers.
                    for repeat in range(3):
                        x = torch.from_numpy((rng.standard_normal((rows, k)) * 0.1).astype(np.float16)).to(device)
                        y = empty(rows, k)
                        mlp.run_bszN(x, y)
                        torch.cuda.synchronize()
                        cpu = x.cpu().float().numpy().astype(np.float64)
                        gate = cpu @ weights[0][1]
                        up = cpu @ weights[1][1]
                        reference = (gate / (1 + np.exp(-gate)) * up) @ weights[2][1]
                        label = dict(native_graph=True, codebook=cb, kernel=kernel, rows=rows, repeat=repeat)
                        relative = compare(y, reference, label, tolerance=0.01)
                        results.append(dict(**label, relative_l2=relative))
    return results
