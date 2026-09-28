"""Checks for the narrow dense GEMMs of narrow-gemm ABI 1.

FP16 through extension.hgemm and BF16 through extension.narrow_gemm_bf16. Call
run_checks(torch, extension) after the binary is verified, the GPU lease is held
and EXL3_NARROW_GEMM=1 is set (configure_native does this for a binary with the
ABI). Covers every supported row count and column widths, FP32/FP16/BF16 output,
strided output with canaries, run-to-run determinism, a non-default stream,
changed-input graph replay, shapes outside the FP16 envelope (BLAS) and BF16
rejections. The reference is an independent CPU FP64 product.
"""
from __future__ import annotations


def run_checks(torch, extension):
    import os
    if os.environ.get('EXL3_NARROW_GEMM') != '1':
        raise ValueError('EXL3_NARROW_GEMM=1 is required; the binary lacks narrow-gemm ABI 1 or it is disabled')
    results = []
    device = torch.device('cuda', torch.cuda.current_device())

    def check(m, k, n, *, dtype=torch.float32, strided=False, stream=None, graph=False, bf16=False):
        operand = torch.bfloat16 if bf16 else torch.float16
        a = torch.randn((m, k), device=device, dtype=operand)
        b = (torch.randn((k, n), device=device, dtype=torch.float32) * (k ** -0.5)).to(operand)
        sentinel = 12345.0 if dtype == torch.float32 else 123.0
        backing = torch.full((m, n + 64), sentinel, device=device, dtype=dtype)
        c = backing[:, 32:32 + n] if strided else torch.empty((m, n), device=device, dtype=dtype)

        def run():
            if bf16:
                extension.narrow_gemm_bf16(a, b, c)
            else:
                extension.hgemm(a, b, c)

        if graph:
            run()
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                with torch.cuda.graph(captured, stream=side):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            a.copy_(torch.randn_like(a))
            captured.replay()
            torch.cuda.synchronize()
        elif stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            stream.synchronize()
        else:
            run()
            torch.cuda.synchronize()
        reference = (a.double().cpu() @ b.double().cpu())
        output = c.double().cpu()
        error = (output - reference).abs()
        scale = reference.square().mean().sqrt().clamp_min(1e-12)
        relative_rms = float(error.square().mean().sqrt() / scale)
        limit = {torch.float32: 2e-6, torch.float16: 1e-3, torch.bfloat16: 5e-3}[dtype]
        assert torch.isfinite(c).all(), 'non-finite output'
        assert relative_rms < limit, ('relative RMS', relative_rms, m, k, n, str(dtype))
        assert float(error.max() / scale) < limit * 20, ('max error', float(error.max()), m, k, n)
        if strided:
            assert (backing[:, :32] == sentinel).all(), 'prefix canary overwritten'
            assert (backing[:, 32 + n:] == sentinel).all(), 'suffix canary overwritten'
        first = c.clone()
        run()
        torch.cuda.synchronize()
        deterministic = bool(torch.equal(first, c))
        assert deterministic, 'repeated call differs'
        if bf16:
            # Same inputs through BLAS: at most a few BF16 ulps apart after its own rounding.
            blas = torch.matmul(a, b).double().cpu()
            assert float((blas - output).abs().max() / scale) < 0.05, ('BLAS difference', m, k, n)
        results.append(dict(shape=[m, k, n], dtype=str(dtype), operands=str(operand), strided=strided, graph=graph,
                            nondefault_stream=stream is not None, relative_rms=relative_rms,
                            max_absolute=float(error.max()), deterministic=deterministic))

    with torch.inference_mode(), torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(928)
        for m in range(1, 9):
            for n in (8, 16, 32, 48, 64, 96, 120, 128):
                check(m, 4096, n)
        for k in (8, 136, 256, 1000, 5120, 12288):
            check(3, k, 32)
            check(3, k, 48)
        for m in (1, 3, 8):
            check(m, 4096, 32, dtype=torch.float16)
            check(m, 4096, 64, strided=True)
            check(m, 4096, 32, dtype=torch.float16, strided=True)
        check(3, 4096, 32, stream=torch.cuda.Stream())
        check(3, 4096, 32, graph=True)
        # Outside the envelope: BLAS keeps these, results must still be correct.
        for m, n in ((9, 32), (3, 12), (3, 136), (3, 160)):
            check(m, 4096, n)
        # BF16 MTP projections: 4096 <-> 12288, 8192 -> 4096, 4096 -> 1024/8192.
        for m in (1, 2, 3, 4):
            for k, n in ((4096, 12288), (12288, 4096), (8192, 4096), (4096, 1024), (4096, 8192), (136, 72)):
                check(m, k, n, dtype=torch.bfloat16, bf16=True)
        check(1, 4096, 8200, dtype=torch.bfloat16, bf16=True, strided=True)
        check(2, 4096, 1024, dtype=torch.bfloat16, bf16=True, stream=torch.cuda.Stream())
        check(1, 4096, 1024, dtype=torch.bfloat16, bf16=True, graph=True)
        a = torch.randn((5, 4096), device=device, dtype=torch.bfloat16)
        b = torch.randn((4096, 64), device=device, dtype=torch.bfloat16)
        for bad in (lambda: extension.narrow_gemm_bf16(a, b, torch.empty((5, 64), device=device, dtype=torch.bfloat16)),
                    lambda: extension.narrow_gemm_bf16(a[:1], b[:, :60].contiguous(),
                                                       torch.empty((1, 60), device=device, dtype=torch.bfloat16)),
                    lambda: extension.narrow_gemm_bf16(a[:1], b, torch.empty((1, 64), device=device, dtype=torch.float16))):
            try:
                bad()
            except RuntimeError:
                pass
            else:
                raise AssertionError('unsupported BF16 request was accepted')
        results.append(dict(bf16_rejections=3))
    return results
