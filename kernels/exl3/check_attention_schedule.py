"""Attention policy checks on real paged caches, streams and changed-input graphs."""
import math
from types import SimpleNamespace


def run_checks(torch, extension):
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.modules.attention_fn import triton_paged as attention
    from exllamav3.modules.attention_fn.schedule import decode_options
    records=[]
    torch.manual_seed(932)
    props=torch.cuda.get_device_properties(0)
    arch=props.gcnArchName.split(':',1)[0]
    with torch.inference_mode():
        fixtures = [(h,4,256,b,b,length,rows) for h,b,length,rows in (
            (16,0,16377,3),(24,8,32761,3),(16,6,16377,1),(24,4,16377,1),
            (24,5,32761,3),(16,8,65529,1),(16,8,32761,5),(24,8,32761,7),
            (16,0,32761,7),(24,0,32761,5))]
        fixtures += [(h,kv,d,b,b,32761,m) for h,kv,d in
            ((32,8,128),(16,2,128),(32,4,128),(32,8,256)) for b in (0,8) for m in (1,3)]
        fixtures += [(h,4,256,k,v,32761,m) for h in (16,24)
            for k,v in ((8,4),(6,4),(8,6),(5,8)) for m in (1,3)]
        fixtures += [(h,4,256,b,b,32761,m) for h in (16,24)
            for b in (5,6) for m in (3,7)]
        fixtures += [(h,4,256,8,8,32761,m) for h in (16,24) for m in (2,4,6,8)]
        for heads,kv,dim,bits,vbits,length,rows in fixtures:
            page=256
            capacity=math.ceil((length+rows)/page)*page
            pages=capacity//page
            table=torch.randperm(pages,device='cuda').to(torch.int32).view(1,-1)
            lengths=torch.tensor([length],dtype=torch.int32,device='cuda')
            total=length+rows
            k=torch.randn((1,total,kv,dim),device='cuda',dtype=torch.half)*0.1
            v=torch.randn_like(k)*0.1
            q=torch.randn((1,rows,heads,dim),device='cuda',dtype=torch.half)*0.1
            layer=None
            kw=dict(q=q,k=None,v=None,block_table=table,cache_seqlens=lengths,
                    causal=True,pre_appended_len=rows)
            if bits:
                layer=CacheLayer_quant(None,SimpleNamespace(num_kv_heads=kv,head_dim=dim),1,capacity,bits,vbits)
                layer.alloc(torch.device('cuda:0'))
                layer.update_kv_direct(torch.zeros_like(lengths),table,k,v,total)
                kc,sk,vc,sv,kb,vb=layer.get_qkv()
                kw.update(k_cache=kc,v_cache=vc,qc=(sk,sv,kb,vb),n_kv_heads_override=kv)
            else:
                kp=torch.zeros((capacity,kv,dim),device='cuda',dtype=torch.half)
                vp=torch.zeros_like(kp)
                kp[:total]=k[0];vp[:total]=v[0]
                kc=torch.empty((pages,page,kv,dim),device='cuda',dtype=torch.half)
                vc=torch.empty_like(kc)
                kc[table[0].long()]=kp.view_as(kc);vc[table[0].long()]=vp.view_as(vc)
                kw.update(k_cache=kc,v_cache=vc)
            options=decode_options(arch=arch,batch=1,query_rows=rows,query_heads=heads,
                kv_heads=kv,head_dim=dim,occupied_bound=capacity,k_bits=bits,v_bits=vbits)
            assert options, ('Policy did not admit fixture',heads,bits,length,rows)
            storage=torch.full((q.numel()+32,),23.,device='cuda',dtype=torch.half)
            out=storage[16:-16].view_as(q)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                attention.paged_attn_triton_decode(out=out,**kw,**options)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    attention.paged_attn_triton_decode(out=out,**kw,**options)
                for step in (0,1):
                    if step:
                        q.mul_(-0.5)
                        lengths.sub_(17)  # New length must control masks during replay.
                    reference=attention.paged_attn_triton_decode(**kw)
                    graph.replay()
                    actual=out.clone()
                    stream.synchronize()
                    relative=float(torch.linalg.vector_norm(actual.float()-reference.float())/
                                   torch.linalg.vector_norm(reference.float()).clamp_min(1e-12))
                    assert torch.isfinite(actual).all() and relative<0.01,(heads,bits,step,relative)
                    assert bool((storage[:16]==23).all() and (storage[-16:]==23).all())
                    records.append(dict(heads=heads,kv_heads=kv,head_dim=dim,bits=bits,vbits=vbits,length=length-step*17,rows=rows,
                        options=options,nondefault_stream=True,changed_input_graph=True,relative_l2=relative))
            torch.cuda.current_stream().wait_stream(stream)
            if layer is not None: layer.free()
    return records
