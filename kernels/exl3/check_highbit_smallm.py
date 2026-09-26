"""Independent packed-matrix checks for small-M high-bit ABI 1.

Called under a verified binary, exclusive GPU lease and external resource guard.
No JIT build or model execution. Real model slices are optional additional codec
checks, not full-model evidence.
"""
import os
import numpy as np
from quantlab.methods.exl3.oracle import decode_trellis, reconstruct
from kernels.exl3.check_smallm import environment


def run_checks(torch, extension, real_slices=()):
    rng = np.random.default_rng(925)
    records = []
    fixtures = []
    for bits in (5, 6):
        packed = rng.integers(-32768, 32768, (8, 16, 16*bits), dtype=np.int16)
        su = rng.choice([-0.25, 0.25], 128).astype(np.float16)
        sv = rng.choice([-0.5, 0.5], 256).astype(np.float16)
        fixtures.append((f'synthetic-k{bits}', bits, packed, su, sv))
    fixtures.extend(real_slices)
    for name, bits, packed, su, sv in fixtures:
        k, n = packed.shape[0]*16, packed.shape[1]*16
        tensors = [torch.from_numpy(a.copy()).cuda() for a in (packed, su, sv)]
        b, suh, svh = tensors
        decoded = torch.empty((k,n), device='cuda', dtype=torch.float16)
        extension.reconstruct(decoded, b, bits, False, True)
        np.testing.assert_array_equal(decoded.float().cpu().numpy(), decode_trellis(packed, bits, codebook=2))
        weight = reconstruct(packed, bits, su, sv, codebook=2).astype(np.float64)
        for warps in (1,4,8,16):
            with environment(EXL3_SMALLM='1', EXL3_SMALLM_WMMA='0', EXL3_GEMV='2',
                             EXL3_GEMV_SPLITK='0' if warps==1 else '1',
                             EXL3_GEMV_SPLITK_WARPS=str(max(4,warps)), EXL3_GEMV_GRAPH='0'):
                for rows in (2,3,5):
                    inputs = (rng.standard_normal((rows,k))*0.1).astype(np.float16)
                    x = torch.from_numpy(inputs).cuda()
                    ah = torch.empty_like(x)
                    for dtype in (torch.float16, torch.float32):
                        storage = torch.full((rows*n+32,),23.0,device='cuda',dtype=dtype)
                        y = storage[16:-16].view(rows,n)
                        tag = extension.exl3_gemm(x,b,y,suh,ah,svh,-1,False,True,0)
                        torch.cuda.synchronize()
                        if tag != 90:
                            raise AssertionError(('highbit did not take GEMV/small-M dispatch',tag))
                        if not bool((storage[:16]==23).all() and (storage[-16:]==23).all()):
                            raise AssertionError('Output guard overwritten')
                        got = y.float().cpu().numpy().astype(np.float64)
                        expected = inputs.astype(np.float64) @ weight
                        relative = float(np.linalg.norm(got-expected)/max(np.linalg.norm(expected),1e-12))
                        if not np.isfinite(got).all() or relative >= 0.005:
                            raise AssertionError((name,rows,str(dtype),warps,relative))
                        # Same frozen operands, ordinary one-row native path.
                        reference = torch.empty_like(y)
                        with environment(EXL3_SMALLM='0'):
                            for row in range(rows):
                                extension.exl3_gemm(x[row:row+1],b,reference[row:row+1],suh,ah[row:row+1],svh,-1,False,True,0)
                        torch.cuda.synchronize()
                        records.append(dict(fixture=name,bits=bits,rows=rows,dtype=str(dtype),warps=warps,
                                            shape=[k,n],dispatch=tag,relative_l2=relative,
                                            max_absolute_error=float(np.abs(got-expected).max()),
                                            exact_sequential=bool(torch.equal(y,reference))))
    return records
