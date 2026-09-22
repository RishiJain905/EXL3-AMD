"""Bounded MiMo compatibility pilots, supervised through the registered runtime.

Private artifacts only. Gaussian fixtures measure operator compatibility, not
language-model quality. No installations, JIT builds or full-model conversion.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
HARNESS_SOURCE = Path(__file__).read_bytes()
sys.path.insert(0, str(ROOT))
from scripts import launch_runtime as launcher
from scripts.check_exl3_kernels import verified_binary


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_native(request):
    binary = verified_binary(request['runtime']['extension_dir'], request['runtime']['extension_sha256'])
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'vendor/rocm-exl3'), str(ROOT/'scripts')]
    os.environ['EXL3_BC_ATTN'] = '0'
    import torch
    import torch.utils.cpp_extension as cpp
    def no_build(*args, **kwargs):
        raise RuntimeError('Native JIT forbidden')
    cpp.load = cpp.load_inline = no_build
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('exllamav3_ext', binary)
    extension = importlib.util.module_from_spec(spec)
    sys.modules['exllamav3_ext'] = extension
    spec.loader.exec_module(extension)
    return torch, extension


def inspect_protocol(request):
    from inspect_exl3_source import inspect, mapped_text_config
    from exllamav3 import Model, Tokenizer
    from transformers import AutoTokenizer
    source, run = Path(request['source']), Path(request['run'])
    result = inspect(source, body_bits=4, head_bits_options=(6,))
    launcher.write_json(run/'mapping.json', result)
    for key in ('missing_required_tensors', 'shape_mismatches', 'unhandled_leaf_modules', 'unconsumed_text_mtp_tensors'):
        if result[key]:
            raise ValueError((key, result[key]))
    config, _ = mapped_text_config(source)
    try:
        Model.from_config(config, component='mtp')
    except (ValueError, KeyError, AssertionError) as error:
        mtp_error = str(error)
    else:
        raise ValueError('Unavailable MTP did not fail')
    tokenizer = Tokenizer(config)
    hf = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if hf.get_vocab() != tokenizer.get_vocab_dict():
        raise ValueError('Full source/runtime vocabulary mapping differs')
    tools = [dict(type='function', function=dict(name='add', description='Add two integers',
             parameters=dict(type='object', properties=dict(a=dict(type='integer'), b=dict(type='integer')), required=['a','b'])))]
    simple = [dict(role='user', content='Return the sum of 2 and 3.')]
    history = simple + [dict(role='assistant', content='', tool_calls=[dict(type='function', function=dict(name='add', arguments=dict(a=2,b=3)))]),
                        dict(role='tool', name='add', content='5'), dict(role='user', content='Explain the result.')]
    fixtures = []
    for name, messages, kwargs in [('default', simple, {}), ('thinking_on', simple, dict(enable_thinking=True)),
                                  ('thinking_off', simple, dict(enable_thinking=False)), ('tool_history', history, dict(tools=tools, enable_thinking=False))]:
        rendered = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        expected = hf.encode(rendered, add_special_tokens=False)
        actual = tokenizer.encode(rendered, add_bos=False, add_eos=False, encode_special_tokens=True).flatten().tolist()
        runtime_ids = tokenizer.hf_chat_template(messages, add_generation_prompt=True, **kwargs).flatten().tolist()
        if expected != actual or expected != runtime_ids:
            raise ValueError('Tokenizer/template mismatch: '+name)
        fixtures.append(dict(name=name, rendered=rendered, ids=expected, text_sha256=hashlib.sha256(rendered.encode()).hexdigest()))
    stops = json.loads((source/'generation_config.json').read_text())['eos_token_id']
    if set(stops) != set(config.eos_token_id_list):
        raise ValueError(('Stop IDs differ', stops, config.eos_token_id_list))
    launcher.write_json(run/'token-protocol.json', dict(fixtures=fixtures, stops=stops, base_vocab_size=hf.vocab_size,
                                                      tokenizer_entries=len(hf), model_vocab_size=config.vocab_size,
                                                      full_vocabulary_exact_match=True,
                                                      mtp_request_error=mtp_error, exact_match=True))
    print(json.dumps(dict(stage='inspection_passed', text_tensors=427, vision_tensors=333, token_fixtures=len(fixtures), stops=stops)), flush=True)


def encode(request, torch):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    source, run = Path(request['source']), Path(request['run'])
    index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    families = dict(mlp='model.language_model.layers.0.mlp.up_proj.weight',
                    attention='model.language_model.layers.3.self_attn.q_proj.weight',
                    gdn='model.language_model.layers.0.linear_attn.out_proj.weight')
    cases = [(family, key, bits, 256, 256) for family,key in families.items() for bits in (4,5,6)]
    cases += [('head', 'lm_head.weight', 6, 128, 4096), ('mlp_repeat', families['mlp'],4,256,256)]
    results = []
    for family, key, bits, rows, cols in cases:
        name = f'{family}-k{bits}'
        with safe_open(source/index[key], framework='pt', device='cpu') as handle:
            original = handle.get_slice(key)[:rows,:cols].contiguous()
        if original.dtype != torch.bfloat16 or tuple(original.shape) != (rows,cols):
            raise ValueError('Source slice geometry/dtype differs')
        cal = torch.randn(512, cols, generator=torch.Generator().manual_seed(100))
        h = dict(H=(cal.T@cal).cuda(), count=512, finalized=False, device='cuda:0', first_key=key)
        qa = dict(K=bits, seed=20260922, apply_out_scales=True, sigma_reg=0.025, devices=['cuda:0'],
                  compact_cpu_buffers=True, mul1=True)
        before = dict(qa)
        start = time.monotonic()
        _, proxy, packed = quantize_exl3(original.half().T.contiguous(), h, qa, False, verbose=False)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        packed = {k:v.cpu().contiguous() for k,v in packed.items()}
        if set(packed) != {'trellis','suh','svh','mul1'} or packed['mul1'].view(torch.uint32).item() != 0x83DCD12D:
            raise ValueError('Incorrect codebook payload')
        target = run/(name+'.safetensors')
        save_file(packed, str(target))
        save_file(dict(weight=original, calibration=cal), str(run/(name+'-fixture.safetensors')))
        entry = dict(name=name, bits=bits, source_key=key, shape=[rows,cols], seconds=elapsed,
                     proxy_error=proxy, quant_args_requested=before, quant_args_actual=qa,
                     artifact_sha256=digest(target), fixture_sha256=digest(run/(name+'-fixture.safetensors')),
                     file_bytes=target.stat().st_size,
                     tensor_bytes=sum(v.numel()*v.element_size() for v in packed.values()))
        results.append(entry)
        launcher.write_json(run/'encodings.json', results)
        print(json.dumps(dict(stage='encoded', **entry)), flush=True)
        del h, cal, original, packed
        torch.cuda.empty_cache()
    # Content equality is checked during independent reload, too: serialization
    # order can vary independently of tensor identity.
    return results


def compare(actual, expected):
    import numpy as np
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    error = float(np.linalg.norm(a-e)/max(np.linalg.norm(e),1e-12))
    if not np.isfinite(a).all() or error > .005:
        raise ValueError(f'Native relative L2 {error} exceeds 0.005')
    return error


def reload(request, torch, ext):
    import numpy as np
    from safetensors.torch import load_file
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.model.config import NullConfig
    from quantlab.methods.exl3.compat import install, smallm_supported
    from quantlab.methods.exl3.oracle import reconstruct, decode_trellis
    run, source = Path(request['run']), Path(request['input'])
    config = NullConfig()
    install(config, native_smallm=True, native_smallm_max_rows=9, native_smallm_codebooks=(0,2))
    results = []
    entries = json.loads((source/'encodings.json').read_text())
    for entry in entries:
        name, bits = entry['name'], entry['bits']
        if digest(source/(name+'.safetensors')) != entry['artifact_sha256'] or digest(source/(name+'-fixture.safetensors')) != entry['fixture_sha256']:
            raise ValueError('Checkpoint/fixture hash changed')
        p = load_file(str(source/(name+'.safetensors')))
        fixture = load_file(str(source/(name+'-fixture.safetensors')))
        rows, cols = entry['shape']
        decoded = decode_trellis(p['trellis'].numpy(),bits,codebook=2)
        reference = reconstruct(p['trellis'].numpy(),bits,p['suh'].numpy(),p['svh'].numpy(),codebook=2)
        gpu = {k:v.cuda() for k,v in p.items()}
        native = torch.empty((cols, rows), dtype=torch.half, device='cuda')
        ext.reconstruct(native,gpu['trellis'],bits,False,True)
        mismatches = int(np.count_nonzero(native.float().cpu().numpy()!=decoded))
        if mismatches:
            raise ValueError(('Raw decoder mismatch', name, mismatches))
        linear = LinearEXL3(config,cols,rows,**gpu,key=name)
        x = torch.randn(128,cols,generator=torch.Generator().manual_seed(200)).half()*.1
        for count in (1,2,9,16,128):
            for dtype in (torch.half, torch.float):
                actual = linear.forward(x[:count].cuda(),{},dtype).float().cpu().numpy()
                cpu = x[:count].float().numpy()
                expected = cpu.astype(np.float64)@reference.astype(np.float64)
                rel = compare(actual,expected)
                source_ref = cpu.astype(np.float64)@fixture['weight'].float().T.numpy().astype(np.float64)
                source_error = float(np.linalg.norm(actual-source_ref)/max(np.linalg.norm(source_ref),1e-12))
                result = dict(name=name,bits=bits,rows=count,dtype=str(dtype),relative_l2=rel,
                              source_bf16_relative_l2=source_error,decode_mismatches=mismatches,
                              route='reconstruct_hgemm' if count>9 else 'native_smallm' if smallm_supported(linear) else 'rowwise_native')
                results.append(result)
        print(json.dumps(dict(stage='reloaded', name=name, max_relative_l2=max(v['relative_l2'] for v in results if v['name']==name))),flush=True)
        launcher.write_json(run/'reload-results.json',results)
        del linear, gpu, native
        torch.cuda.empty_cache()
    first=load_file(str(source/'mlp-k4.safetensors'))
    repeat=load_file(str(source/'mlp_repeat-k4.safetensors'))
    equality = all(torch.equal(first[k],repeat[k]) for k in first)
    launcher.write_json(run/'repeatability.json',dict(exact_packed_tensor_equality=equality,
        identical_file_hash=digest(source/'mlp-k4.safetensors')==digest(source/'mlp_repeat-k4.safetensors')))
    if not equality:
        raise ValueError('Repeated same-seed encoding changed packed tensors')


def head_chunks(request, torch, ext):
    import numpy as np
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.model.config import NullConfig
    from quantlab.methods.exl3.oracle import reconstruct
    from safetensors.torch import save_file
    # Cross the production 32768-column chunk boundary at true MiMo input width.
    k,n,bits=4096,32896,6
    rng=np.random.default_rng(20260922)
    p=rng.integers(-32768,32768,(k//16,n//16,16*bits),dtype=np.int16)
    su=rng.choice([-.125,.125],k).astype(np.float16)
    sv=rng.choice([-.125,.125],n).astype(np.float16)
    packed=dict(trellis=torch.from_numpy(p),suh=torch.from_numpy(su),svh=torch.from_numpy(sv),
                mul1=torch.tensor(0x83DCD12D,dtype=torch.int64).int())
    fixture=Path(request['run'])/'head-fixture.safetensors'
    save_file(packed,str(fixture))
    linear=LinearEXL3(NullConfig(),k,n,**{key:value.cuda() for key,value in packed.items()},key='head_chunk_fixture')
    x=torch.randn(1024,k,generator=torch.Generator().manual_seed(200)).half()*.1
    results=[]
    calls=[]
    originals={name:getattr(ext,name) for name in ('reconstruct_slice','reconstruct_had_slice')}
    for name,original in originals.items():
        def observed(*args,_name=name,_original=original):
            calls.append(dict(function=_name,offset=args[-1],shape=list(args[0].shape)))
            return _original(*args)
        setattr(ext,name,observed)
    try:
        for count in (16,1024):
            for dtype in (torch.half,torch.float):
                actual=linear.forward(x[:count].cuda(),dict(reconstruct=True),dtype).float().cpu().numpy()
                if not np.isfinite(actual).all(): raise ValueError('Nonfinite head output')
                for start in (0,32640,32768):
                    end=start+128
                    ref=reconstruct(p[:,start//16:end//16],bits,su,sv[start:end],codebook=2)
                    error=compare(actual[:,start:end], x[:count].float().numpy().astype(np.float64)@ref.astype(np.float64))
                    results.append(dict(input_features=k,output_features=n,rows=count,dtype=str(dtype),start=start,end=end,relative_l2=error))
    finally:
        for name,original in originals.items(): setattr(ext,name,original)
    launcher.write_json(Path(request['run'])/'head-chunks.json',results)
    launcher.write_json(Path(request['run'])/'head-dispatch.json',dict(calls=calls,fixture_sha256=digest(fixture)))
    print(json.dumps(dict(stage='head_chunks_passed',checks=len(results))),flush=True)


def worker(request):
    (Path(request['run'])/'harness.py').write_bytes(HARNESS_SOURCE)
    torch,ext=load_native(request)
    if request['mode']=='inspect':
        inspect_protocol(request)
        return 0
    properties=torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(8*2**30/properties.total_memory)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False
    record=dict(device=properties.name,arch=properties.gcnArchName,torch=torch.__version__,hip=torch.version.hip,
                extension_sha256=request['runtime']['extension_sha256'],harness_sha256=hashlib.sha256(HARNESS_SOURCE).hexdigest(),
                matmul_allow_tf32=False,allocator_cap_bytes=8*2**30)
    launcher.write_json(Path(request['run'])/'runtime.json',record)
    if request['mode']=='encode': encode(request,torch)
    elif request['mode']=='reload': reload(request,torch,ext)
    else: head_chunks(request,torch,ext)
    record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  free_total_bytes=list(torch.cuda.mem_get_info()),status='passed')
    launcher.write_json(Path(request['run'])/'runtime.json',record)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mode',required=True,choices=['inspect','encode','reload','head'])
    parser.add_argument('--input',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    config=tomllib.loads(args.config.read_text())
    if not args.execute or not all(config['execution'].get(k) is True for k in ('allow_backend_probes','allow_local_inference')):
        parser.error('Requires --execute and both configured execution permissions')
    if args.mode=='reload' and not args.input: parser.error('reload requires --input')
    source,run=args.source.resolve(),args.output.resolve()
    if source==run or source in run.parents or run in source.parents: parser.error('Source/output must be separate and nonnested')
    runtime=dict(config['runtime'])
    for key in ('python','sdk','torch_lib','hsa_preload','server_deps','extension_dir'):
        if runtime.get(key): runtime[key]=launcher.linux_path(runtime[key])
    verified_binary(runtime['extension_dir'],runtime['extension_sha256'])
    run.mkdir(parents=True,exist_ok=False)
    request_file=run/'request.json'
    request=dict(root=str(ROOT),run=str(run),source=str(source),mode=args.mode,input=str(args.input) if args.input else None,
                 runtime=runtime,timeout=900,limits=dict(max_host_memory_fraction=.7,min_free_ram_gib=4,min_free_disk_gib=120),
                 argv=[runtime['python'],str(Path(__file__).resolve()),'--worker',str(request_file)])
    launcher.write_json(request_file,request)
    with launcher.lease(Path(launcher.linux_path(runtime['lease_file'])),run):
        return launcher.worker(request_file)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        raise SystemExit(worker(json.loads(Path(sys.argv[2]).read_text())))
    raise SystemExit(main())
