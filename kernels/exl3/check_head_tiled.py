"""Wide packed projections: independent oracle, exact controls and capture."""
import ctypes
from contextlib import contextmanager
import os
import numpy as np

from quantlab.methods.exl3.oracle import reconstruct


@contextmanager
def environment(**values):
    old = {key: os.environ.get(key) for key in values}
    for key,value in values.items():
        if value is None: os.environ.pop(key,None)
        else: os.environ[key] = value
    try:
        yield
    finally:
        for key,value in old.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key] = value


def run_checks(torch, extension):
    library = ctypes.CDLL(extension.__file__)
    abi = library.quantlab_exl3_head_tiled_abi
    assert abi() in (1, 2)
    count = library.quantlab_exl3_head_tiled_calls
    count.restype = ctypes.c_uint64
    assert library.quantlab_exl3_head_repacked_abi() == 3
    repacked_count = library.quantlab_exl3_head_repacked_calls
    repacked_count.restype = ctypes.c_uint64
    rng = np.random.default_rng(930)
    records = []

    def check(bits, cb, k, n, rows, capture=False, repacked=False, half_output=False):
        packed = rng.integers(-32768,32768,(k//16,n//16,16*bits),dtype=np.int16)
        su = rng.choice([-0.25,0.25],k).astype(np.float16)
        sv = rng.choice([-0.5,0.5],n).astype(np.float16)
        cpu = (rng.standard_normal((rows,k))*0.1).astype(np.float16)
        b, suh, svh, x = (torch.from_numpy(t).cuda() for t in (packed,su,sv,cpu))
        tiles = b.view(k//16,n//128,8,16*bits).permute(1,0,2,3).contiguous() if repacked else None
        ah = torch.empty_like(x)
        buf = torch.full((rows*n+32,),23.,device='cuda',dtype=torch.half if half_output else torch.float32)
        y = buf[16:-16].view(rows,n)
        reference = torch.empty_like(y)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        def launch(out):
            tag = extension.exl3_gemm(x,b,out,suh,ah,svh,-1,False,cb==2,0)
            assert tag == 90, tag
        def candidate():
            if repacked:
                extension.exl3_head_repacked(x,tiles,suh,ah,svh,y,bits,cb)
            else:
                launch(y)
        calls = repacked_count if repacked else count
        with torch.cuda.stream(stream):
            with environment(EXL3_HEAD_TILED='0'): launch(reference)
            before = calls()
            with environment(EXL3_HEAD_TILED='1'):
                candidate()
                assert calls() == before+1, 'Candidate did not dispatch'
                if capture:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream): candidate()
                    cpu = (-cpu*np.float16(0.5)).astype(np.float16)
                    x.copy_(torch.from_numpy(cpu).to(x.device))
                    graph.replay()
            if capture:
                with environment(EXL3_HEAD_TILED='0'): launch(reference)
        stream.synchronize()
        assert torch.equal(y,reference), ('Changed arithmetic',bits,cb,k,n,rows,
                                          float((y-reference).abs().max()))
        assert bool((buf[:16]==23).all() and (buf[-16:]==23).all())
        weight = reconstruct(packed[:,:16].copy(),bits,su,sv[:256],codebook=cb).astype(np.float64)
        expected = cpu.astype(np.float64) @ weight
        actual = y[:,:256].double().cpu().numpy()
        relative = float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12))
        assert np.isfinite(actual).all() and relative < 0.005, relative
        records.append(dict(bits=bits,codebook=cb,k=k,n=n,rows=rows,capture=capture,repacked=repacked,half_output=half_output,
                            exact_control=True,nondefault_stream=True,relative_l2=relative))
        return x,b,y,suh,ah,svh

    with environment(EXL3_GEMV='2',EXL3_SMALLM='1',EXL3_SMALLM_WMMA='0',
                     EXL3_GEMV_LDS='0',EXL3_GEMV_GRAPH='0',EXL3_GEMV_SPLITK='1',
                     EXL3_SMALLM_HEAD_WARPS=None,EXL3_GEMV_SPLITK_WARPS=None):
        for cb,widths in ((0,(2,3,4)),(2,(2,3,4,5,6))):
            for bits in widths:
                for rows in (1,2,3,5):
                    args = check(bits,cb,640,32896,rows,capture=rows==3)
                    check(bits,cb,640,32896,rows,capture=rows==3,repacked=True)
                    check(bits,cb,640,32896,rows,capture=rows==3,repacked=True,half_output=True)
        for k,n,bits,cb in ((4096,248320,6,2),(5120,248320,4,0),
                            (5120,248320,3,2),(2048,33024,4,2),(8192,65536,4,0)):
            for rows in (1,2,3,5):
                args = check(bits,cb,k,n,rows,capture=True)
                check(bits,cb,k,n,rows,capture=True,repacked=True)
                check(bits,cb,k,n,rows,capture=True,repacked=True,half_output=True)
        # Overrides are exact legacy dispatch controls, not alternative arithmetic.
        x,b,y,suh,ah,svh = args
        for flag,value in (('EXL3_SMALLM_HEAD_WARPS','4'),('EXL3_SMALLM_WMMA','1'),
                           ('EXL3_GEMV_LDS','1'),('EXL3_GEMV_SPLITK','0'),
                           ('EXL3_GEMV_SPLITK_WARPS','8')):
            before = count()
            with environment(EXL3_HEAD_TILED='1',**{flag:value}):
                extension.exl3_gemm(x,b,y,suh,ah,svh,-1,False,False,0)
            torch.cuda.synchronize()
            assert count() == before, ('Override ignored',flag)
            records.append(dict(override=flag,old_dispatch=True))
        # The split-K arithmetic boundary must stay on the old dispatch.
        boundary_b = b[:,:2048].contiguous()
        boundary_sv = svh[:32768].contiguous()
        boundary_y = torch.empty((x.shape[0],32768),device=x.device,dtype=torch.float32)
        before = count()
        with environment(EXL3_HEAD_TILED='1'):
            extension.exl3_gemm(x,boundary_b,boundary_y,suh,ah,boundary_sv,-1,False,False,0)
        torch.cuda.synchronize()
        assert count() == before
        records.append(dict(n=32768,old_dispatch=True))

        # The explicit layout entry rejects malformed buffers before launching.
        rows,k,n,bits = 3,128,32896,4
        x = torch.zeros((rows,k),device='cuda',dtype=torch.half)
        packed = torch.zeros((n//128,k//16,8,16*bits),device='cuda',dtype=torch.int16)
        suh = torch.ones(k,device='cuda',dtype=torch.half)
        svh = torch.ones(n,device='cuda',dtype=torch.half)
        scratch = torch.empty_like(x)
        output = torch.empty((rows,n),device='cuda',dtype=torch.float32)
        good = dict(x=x,packed=packed,suh=suh,input_scratch=scratch,svh=svh,output=output,bits=bits,codebook=0)
        def invalid(name, **changes):
            before = repacked_count()
            try:
                extension.exl3_head_repacked(**(good|changes))
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError(('Malformed input was accepted', name))
            assert repacked_count() == before
            records.append(dict(rejected=name))
        invalid('CPU input', x=x.cpu())
        invalid('input dtype', x=x.float())
        invalid('packed dtype', packed=packed.to(torch.int32))
        invalid('output dtype', output=output.double())
        invalid('input rank', x=x.unsqueeze(0))
        invalid('packed layout', packed=packed.flatten())
        invalid('packed shape', packed=packed[:-1])
        invalid('noncontiguous input', x=torch.empty((k,rows),device='cuda',dtype=torch.half).t())
        invalid('scratch shape', input_scratch=scratch[:1])
        invalid('scratch overlap', input_scratch=x)
        invalid('scale size', suh=suh[:-4])
        invalid('output alignment', output=torch.empty(rows*n+1,device='cuda')[1:].view(rows,n))
        invalid('input alignment', x=torch.empty(rows*k+1,device='cuda',dtype=torch.half)[1:].view(rows,k))
        invalid('codebook', codebook=1)
        invalid('bits', bits=5)
        invalid('row count', x=x[:0])
        invalid('split boundary', output=output[:,:32768].contiguous())

        # Exercise the actual adapter lifecycle, beyond the raw entry point.
        from types import SimpleNamespace
        from exllamav3.modules.quant.exl3 import LinearEXL3
        from quantlab.methods.exl3.packed_head import install_packed_head
        original_packed = torch.randint(-32768,32767,(k//16,n//16,16*bits),device='cuda',dtype=torch.int16)
        layer = LinearEXL3(None,k,n,trellis=original_packed,suh=suh,svh=svh,
                           out_dtype=torch.float32,key='head-validation')
        model = SimpleNamespace(modules=[SimpleNamespace(inner=layer)],logit_layer_idx=0)
        x = torch.randn((1,rows,k),device='cuda',dtype=torch.half)*0.1
        reference = type(layer).forward(layer,x,{})
        stats = install_packed_head(model,model,enabled=True,memory_fraction=0.5)
        assert len(stats) == 1, 'Shared target/draft head was duplicated'
        actual = layer.forward(x,{})
        assert torch.equal(actual,reference) and stats[0]['packed_calls'] == 1
        first_view = layer._quantlab_head_state['packed']
        assert stats[0]['packed_bytes'] == original_packed.numel()*original_packed.element_size()
        layer.forward(x,{})
        assert layer._quantlab_head_state['packed'] is first_view
        layer.trellis = layer.trellis.clone()
        assert torch.equal(layer.forward(x,{}),reference)
        assert layer._quantlab_head_state['packed'] is not first_view
        del first_view
        records.append(dict(adapter='shared-copy-reuse-and-source-replacement',pass_check=True))

        # Captured work must consume changed input, using the prepared view.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): graph_output = layer.forward(x,{})
        x.mul_(-0.5)
        graph.replay()
        assert torch.equal(graph_output,type(layer).forward(layer,x,{}))
        del graph, graph_output
        records.append(dict(adapter='prepared-graph-changed-input',pass_check=True))

        before = stats[0]['packed_calls']
        assert torch.equal(layer.forward(x,{'ovr':{}}),type(layer).forward(layer,x,{'ovr':{}}))
        assert torch.equal(layer.forward(x,{},torch.half),type(layer).forward(layer,x,{},torch.half))
        assert stats[0]['packed_calls'] == before + 1
        install_packed_head(model,enabled=False,memory_fraction=0.5)
        assert layer._quantlab_head_state['packed'] is None and stats[0]['packed_bytes'] == 0
        assert torch.equal(layer.forward(x,{}),type(layer).forward(layer,x,{}))
        records.append(dict(adapter='semantic-fallback-and-disable-release',pass_check=True))

        install_packed_head(model,enabled=True,memory_fraction=0.00001)
        before = stats[0]['packed_calls']
        assert torch.equal(layer.forward(x,{}),type(layer).forward(layer,x,{}))
        assert stats[0]['packed_calls'] == before and stats[0]['packed_bytes'] == 0
        assert stats[0]['memory_limited']
        records.append(dict(adapter='allocator-budget-fallback',pass_check=True))
        install_packed_head(model,enabled=True,memory_fraction=0.5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): cold_output = layer.forward(x,{})
        graph.replay()
        assert torch.equal(cold_output,type(layer).forward(layer,x,{}))
        assert stats[0]['packed_bytes'] == 0, 'Allocated packed view during capture'
        del graph, cold_output
        layer.forward(x,{})
        assert stats[0]['packed_bytes'] > 0
        layer.unload()
        assert layer._quantlab_head_state['packed'] is None and stats[0]['packed_bytes'] == 0
        records.append(dict(adapter='cold-capture-fallback-and-unload-release',pass_check=True))

        # Real model heads use FP16 by default; explicit FP32 must also work.
        layer = LinearEXL3(None,k,n,trellis=original_packed,suh=suh,svh=svh,
                           key='head-explicit-fp32')
        model = SimpleNamespace(modules=[SimpleNamespace(inner=layer)],logit_layer_idx=0)
        assert layer.default_out_dtype == torch.half
        reference = type(layer).forward(layer,x,{},torch.float32)
        stats = install_packed_head(model,enabled=True,memory_fraction=0.5)
        assert len(stats) == 1, 'Skipped a head with per-call FP32 logits'
        assert torch.equal(layer.forward(x,{}),type(layer).forward(layer,x,{}))
        assert stats[0]['packed_calls'] == 1
        actual = layer.forward(x,{},torch.float32)
        assert torch.equal(actual,reference) and stats[0]['packed_calls'] == 2
        layer.unload()
        records.append(dict(adapter='fp16-default-and-explicit-fp32-logits',pass_check=True))
    return records
