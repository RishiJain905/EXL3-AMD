"""Numerical and memory-boundary checks for an explicitly loaded native extension.

Call run_checks(torch, extension) from an environment that has already verified
the binary and acquired the GPU. This module never loads or builds an extension.
"""
from __future__ import annotations


def run_checks(torch, extension):
    results = []
    device = torch.device('cuda', torch.cuda.current_device())

    def check(m, k, n, *, strided=False, dtype=None, stream=None):
        dtype = dtype or torch.float32
        a = torch.randn((m, k), device=device, dtype=torch.float16) * (k ** -0.5)
        b = torch.randn((k, n), device=device, dtype=torch.float16)
        sentinel = 12345.0 if dtype == torch.float32 else 123.0
        backing = torch.full((m, n + 32), sentinel, device=device, dtype=dtype)
        c = backing[:, 16:16+n] if strided else torch.empty((m, n), device=device, dtype=dtype)
        # Small cases use an independent CPU FP64 reference. Large-K cases use
        # full FP32 inputs, avoiding the native FP16-input GEMM dispatch.
        if k <= 256:
            reference = (a.double().cpu() @ b.double().cpu()).to(device=device, dtype=dtype)
        else:
            reference = torch.mm(a.float(), b.float()).to(dtype)
        if stream is None:
            extension.hgemm(a, b, c)
            torch.cuda.synchronize()
        else:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                extension.hgemm(a, b, c)
            stream.synchronize()
        error = (c.float() - reference.float()).abs()
        relative_rms = float(error.square().mean().sqrt() /
                             reference.float().square().mean().sqrt().clamp_min(1e-12))
        limit = 5e-5 if dtype == torch.float32 else 1e-3
        assert torch.isfinite(c).all(), 'non-finite GEMM output'
        assert relative_rms < limit, ('relative RMS', relative_rms, m, k, n)
        assert torch.allclose(c, reference, rtol=limit*4, atol=limit*4), ('max error', error.max().item())
        if dtype == torch.float32 and m:
            assert (c != c.half().float()).any(), 'FP32 output was rounded through FP16'
        if strided:
            assert (backing[:, :16] == sentinel).all(), 'prefix canary overwritten'
            assert (backing[:, 16+n:] == sentinel).all(), 'suffix canary overwritten'
        results.append(dict(shape=[m, k, n], strided=strided, dtype=str(dtype),
                            nondefault_stream=stream is not None,
                            relative_rms=relative_rms, max_absolute=float(error.max())))

    with torch.inference_mode(), torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(4217)
        for m in (1, 63, 64, 65, 127, 128, 255, 256, 257, 511, 512, 513, 777, 1024):
            check(m, 128, 128)
        check(65, 256, 256, strided=True)
        check(513, 256, 256, strided=True)
        check(257, 17408, 128)
        check(513, 17408, 128)
        check(1024, 5120, 10240)
        check(65, 128, 128, dtype=torch.float16, strided=True)
        check(65, 128, 128, stream=torch.cuda.Stream())
        check(513, 128, 128, stream=torch.cuda.Stream())
        # Unsupported WMMA geometry must still compute correctly via BLAS.
        check(65, 80, 96, strided=True)

        a3 = torch.randn((2, 65, 128), device=device, dtype=torch.float16)
        b3 = torch.randn((128, 128), device=device, dtype=torch.float16)
        c3 = torch.empty((2, 65, 128), device=device, dtype=torch.float32)
        extension.hgemm(a3, b3, c3)
        reference = (a3.double().cpu() @ b3.double().cpu()).to(device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        assert torch.allclose(c3, reference, rtol=2e-4, atol=2e-4)
        results.append(dict(batched_flatten=True))

        # WMMA must preserve subnormal inputs; a flushed reference would hide
        # a real loss of precision on small activations or weights.
        for rows in (64, 513):
            for operand in ('a', 'b', 'both'):
                for value in (2**-24, -2**-20, 2**-14):
                    tiny_a = torch.full((rows, 128), value if operand != 'b' else 1,
                                        device=device, dtype=torch.float16)
                    tiny_b = torch.full((128, 128), value if operand != 'a' else 1,
                                        device=device, dtype=torch.float16)
                    tiny_c = torch.empty((rows, 128), device=device, dtype=torch.float32)
                    extension.hgemm(tiny_a, tiny_b, tiny_c)
                    torch.cuda.synchronize()
                    expected = 128 * (value * value if operand == 'both' else value)
                    assert (tiny_c == expected).all(), ('subnormal lost', rows, operand, value)
                    results.append(dict(rows=rows, subnormal_operand=operand, value=value))

        a = torch.ones((64, 128), device=device, dtype=torch.float16)
        b = torch.ones((128, 128), device=device, dtype=torch.float16)
        c = torch.empty((64, 128), device=device, dtype=torch.float32)
        overlap_a = torch.ones((64, 256), device=device, dtype=torch.float16)
        alias_a = overlap_a.flatten()[:64*128].view(64, 128)
        overlap_b = torch.ones((128, 128), device=device, dtype=torch.float16)
        invalid = [
            ('input dtype', (a.float(), b, c)),
            ('input stride', (a[:, ::2], b[:64].contiguous(), c)),
            ('weight stride', (a, b.T, c)),
            ('output capacity', (a, b, c[:63])),
            ('output columns', (a, b, c[:, :127])),
            ('output dtype', (a, b, c.to(torch.int32))),
            ('CPU device', (a.cpu(), b.cpu(), c.cpu())),
            ('zero K', (a[:, :0].contiguous(), b[:0], c)),
            ('output aliases input', (alias_a, b, overlap_a.view(torch.float32))),
            ('output aliases weights', (a, overlap_b, overlap_b.view(torch.float32).view(64, 128))),
            ('native integer range', (
                torch.empty((0, 2**31), device=device, dtype=torch.float16),
                torch.empty((2**31, 0), device=device, dtype=torch.float16),
                torch.empty((0, 0), device=device, dtype=torch.float32))),
        ]
        for name, tensors in invalid:
            try:
                extension.hgemm(*tensors)
            except RuntimeError:
                results.append(dict(rejected=name))
            else:
                raise AssertionError('Invalid GEMM input accepted: '+name)
        extension.hgemm(a[:0], b, c[:0])
        torch.cuda.synchronize()
        results.append(dict(empty_rows=True))

        # Replay must consume current inputs on the captured stream.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            extension.hgemm(a, b, c)
        a.mul_(2)
        graph.replay()
        torch.cuda.synchronize()
        assert (c == 256).all(), 'captured GEMM did not consume updated input'
        results.append(dict(graph_replay=True))

        # Cover the prefetch branch and an incomplete 256-row tile on replay.
        wide_a = torch.ones((513, 128), device=device, dtype=torch.float16)
        wide_b = torch.ones((128, 128), device=device, dtype=torch.float16)
        wide_c = torch.empty((513, 128), device=device, dtype=torch.float32)
        wide_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(wide_graph):
            extension.hgemm(wide_a, wide_b, wide_c)
        wide_a.fill_(3)
        wide_b.fill_(2)
        wide_graph.replay()
        torch.cuda.synchronize()
        assert (wide_c == 768).all(), 'prefetch graph did not consume updated inputs'
        results.append(dict(graph_replay=True, rows=513))
    return results
