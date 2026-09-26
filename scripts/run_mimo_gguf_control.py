"""Bounded Windows llama-server control runs against frozen MiMo token inputs.

Uses an explicitly supplied existing executable. No build/install or source
mutation. Model files, protocol and suite are read-only; output must be new.
"""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.launch_runtime import lease
from quantlab.telemetry import TelemetryCollector

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def summarize_teacher_forced(top,target,valid_size=248077,stored_size=248320):
    """Deterministic summary of one full-vocabulary presampling logprob row.

    Pure: validates the returned row and derives target NLL normalized over
    valid IDs only, deterministic top-1 (smallest ID on exact ties, matching
    argmax over ascending vocabulary) and excluded mass from the explicitly
    returned invalid IDs normalized by the returned valid+excluded mass.
    Raises RuntimeError on any malformed, incomplete or nonfinite input.
    """
    if valid_size<=0 or stored_size<=0 or valid_size>stored_size:
        raise ValueError('Invalid vocabulary sizes')
    if not isinstance(top,list) or len(top)!=stored_size:
        raise RuntimeError('Full-vocabulary probabilities unavailable; NLL not inferred from top-k')
    if not isinstance(target,int) or isinstance(target,bool) or not 0<=target<valid_size:
        raise RuntimeError('Invalid target probability')
    values={}
    for entry in top:
        if not isinstance(entry,dict) or 'id' not in entry or 'logprob' not in entry:
            raise RuntimeError('Full-vocabulary probabilities unavailable; NLL not inferred from top-k')
        tid=entry['id']
        logprob=entry['logprob']
        if not isinstance(tid,int) or isinstance(tid,bool):
            raise RuntimeError('Full-vocabulary probabilities unavailable; NLL not inferred from top-k')
        if not isinstance(logprob,(int,float)) or isinstance(logprob,bool) or not math.isfinite(logprob):
            raise RuntimeError('Invalid target probability')
        if tid in values:
            raise RuntimeError('Full-vocabulary probabilities unavailable; NLL not inferred from top-k')
        values[tid]=float(logprob)
    if len(values)!=stored_size or set(values)!=set(range(stored_size)):
        raise RuntimeError('Full-vocabulary probabilities unavailable; NLL not inferred from top-k')
    target_logprob=values[target]
    if target_logprob<-1e30:
        raise RuntimeError('Target probability underflowed; exact NLL is unavailable')
    try:
        valid_mass=math.fsum(math.exp(values[k]) for k in range(valid_size))
        excluded_mass=math.fsum(math.exp(values[k]) for k in range(valid_size,stored_size))
    except OverflowError:
        raise RuntimeError('Invalid target probability')
    if not math.isfinite(valid_mass) or not valid_mass>0:
        raise RuntimeError('Invalid target probability')
    if not math.isfinite(excluded_mass) or excluded_mass<0:
        raise RuntimeError('Invalid target probability')
    total=valid_mass+excluded_mass
    if not math.isfinite(total) or not total>0:
        raise RuntimeError('Invalid target probability')
    excluded_fraction=excluded_mass/total
    if not math.isfinite(excluded_fraction):
        raise RuntimeError('Invalid target probability')
    target_nll=-target_logprob+math.log(valid_mass)
    if not math.isfinite(target_nll):
        raise RuntimeError('Invalid target probability')
    best=max(values[k] for k in range(valid_size))
    top1=min(k for k in range(valid_size) if values[k]==best)
    return dict(target_id=target,target_nll=target_nll,top1_id=top1,
        valid_vocabulary_size=valid_size,excluded_probability_mass=excluded_fraction,
        returned_vocabulary_size=len(values))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','executable','model','protocol','suite','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('quality','speed','logprobs'),required=True)
    p.add_argument('--device',required=True,help='Exact --list-devices identifier for the target GPU')
    p.add_argument('--hip-visible-devices',help='Optional process-local HIP device filter; device IDs must match its inventory')
    p.add_argument('--port',type=int,default=8094)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    environment=dict(os.environ)
    if args.hip_visible_devices is not None:
        environment['HIP_VISIBLE_DEVICES']=args.hip_visible_devices
    config=tomllib.loads(args.config.read_text(encoding='utf-8'))
    if not args.execute or not all(config['execution'].get(k) is True for k in ('allow_backend_probes','allow_local_inference')):
        p.error('Requires --execute and both local permissions')
    if sys.platform!='win32':p.error('This monitor targets the registered Windows backend')
    if args.output.exists():p.error('Output must be new')
    for path in (args.executable,args.model,args.protocol,args.suite):
        if not path.is_file():p.error('Missing input: '+str(path))
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    suite=json.loads(args.suite.read_text(encoding='utf-8'))
    if digest(args.suite)!=protocol['suite_file_sha256']:p.error('Suite/protocol hash mismatch')
    quality_limit=protocol['max_new_tokens']
    if (type(quality_limit) is not int or not 1<=quality_limit<=2048
            or suite['quality_max_tokens']!=quality_limit):
        p.error('Suite output budget differs from supported frozen protocol')
    if protocol['thinking'] and args.mode!='quality':
        p.error('Thinking-enabled supplement is quality-only')
    if protocol['context']!=4096 or protocol['cache']!='f16' or protocol['mtp']:
        p.error('Unexpected frozen protocol')
    if protocol['valid_vocabulary_size']!=248077 or protocol['stored_vocabulary_size']!=248320:
        p.error('Unexpected frozen vocabulary')
    args.output.mkdir(parents=True)
    (args.output/'harness.py').write_bytes(Path(__file__).read_bytes())
    cases={row['id']:row for row in protocol['cases']}
    command=[str(args.executable),'-m',str(args.model),'--device',args.device,'--split-mode','none',
             '-c','4096','-ngl','99','-fa','on',
             '-ctk','f16','-ctv','f16','-b','256','-ub','256','-np','1',
             '--spec-type','none','--no-mmproj','-lv','5','--host','127.0.0.1','--port',str(args.port)]
    # llama.cpp auto-classifies three FIM markers as EOG, unlike this source's
    # two-token generation config. These completion-only controls do not use
    # the FIM API: alias those metadata slots to the existing EOS so the actual
    # logged EOG set and ignore_eos mask match the frozen source protocol.
    if set(protocol['stop_ids'])!={248044,248046}:p.error('Unexpected source stop IDs')
    for field in ('fim_pad_token_id','fim_rep_token_id','fim_sep_token_id'):
        command+=['--override-kv','tokenizer.ggml.'+field+'=int:248046']
    write(args.output/'request.json',dict(argv=command,mode=args.mode,protocol_sha256=digest(args.protocol),
        suite_sha256=digest(args.suite),executable_sha256=digest(args.executable),
        model_path=str(args.model),model_bytes=args.model.stat().st_size,timeout=900,
        server_cwd=str(args.executable.parent),hip_visible_devices=environment.get('HIP_VISIBLE_DEVICES'),
        thinking=protocol['thinking'],quality_max_tokens=quality_limit,
        limits=dict(min_host_available_bytes=4*2**30,max_dedicated_fraction=.95,min_artifact_free_gib=120)))
    # Full model hash occurs before inference and is retained separately from timing.
    write(args.output/'model-identity.json',dict(file=args.model.name,bytes=args.model.stat().st_size,sha256=digest(args.model)))
    url='http://127.0.0.1:'+str(args.port)
    def api(path,payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        request=urllib.request.Request(url+path,data=data,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=60) as response:
            raw=response.read()
        return json.loads(raw),hashlib.sha256(raw).hexdigest()
    collector=TelemetryCollector('mimo-'+args.mode,target_gpu_substring='7800')
    if not collector.gpu_total_dedicated_bytes:raise RuntimeError('GPU telemetry unavailable')
    import shutil
    if shutil.disk_usage(args.output).free<120*2**30:raise RuntimeError('Artifact disk preflight failed')
    stopped=threading.Event();state={'reason':None};results=[];child=None;start=time.monotonic()
    def monitor():
        with (args.output/'windows-telemetry.jsonl').open('x',encoding='utf-8') as stream:
            while not stopped.is_set():
                sample=collector.sample_now();stream.write(json.dumps(sample)+'\n');stream.flush()
                if sample.get('host_available_bytes',0)<4*2**30:state['reason']='host_memory_guard'
                gpu=sample.get('dedicated_gpu_bytes')
                if isinstance(gpu,int) and gpu>.95*collector.gpu_total_dedicated_bytes:state['reason']='dedicated_gpu_guard'
                if time.monotonic()-start>900:state['reason']='timeout'
                if state['reason']:
                    if child is not None and child.poll() is None:child.terminate()
                    return
                stopped.wait(1)
    def complete(name,ids,limit,fixed=False,target=None,warmup=False):
        payload=dict(prompt=ids,n_predict=limit,temperature=0,top_k=1,top_p=1,min_p=0,
                     repeat_penalty=1,seed=20260922,cache_prompt=False,return_tokens=True,
                     ignore_eos=fixed,stream=False,post_sampling_probs=False)
        if not fixed:payload['stop']=['<|im_end|>','<|endoftext|>']
        if target is not None:payload['n_probs']=248320
        began=time.monotonic();response,response_hash=api('/completion',payload);elapsed=time.monotonic()-began
        tokens=response.get('tokens',[])
        if not isinstance(tokens,list) or any(not isinstance(x,int) or not 0<=x<248077 for x in tokens):
            raise RuntimeError('Missing or invalid returned token IDs')
        if fixed and len(tokens)!=limit:raise RuntimeError('Fixed output count mismatch')
        result=dict(name=name,warmup=warmup,input_tokens=len(ids),input_ids=ids,
                    output_tokens=len(tokens),output_token_ids=tokens,output_text=response.get('content',''),
                    fixed_eos_policy='suppress_stop_logits' if fixed else None,
                    end_to_end_seconds=elapsed,timings=response.get('timings'),response_sha256=response_hash,
                    stop_type=response.get('stop_type'),truncated=response.get('truncated'))
        if target is not None:
            if not (args.output/'first-probability-response.json').exists():
                write(args.output/'first-probability-response.json',response)
            probabilities=response.get('completion_probabilities',response.get('probs',[]))
            if len(probabilities)!=1:raise RuntimeError('Expected exactly one probability row')
            row=probabilities[0]
            top=row.get('top_logprobs')
            if top is None:raise RuntimeError('Backend did not return presampling top_logprobs')
            result['teacher_forced']=summarize_teacher_forced(top,target)
        write(args.output/(name+'-result.json'),result)
        results.append(result)
        print(json.dumps(dict(case=name,tokens=len(tokens),seconds=elapsed)),flush=True)
    with lease(Path(config['runtime']['lease_file']),args.output):
        monitor_thread=None
        try:
            # Refuse to repurpose an already active listener.
            import socket
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1',args.port))==0:raise RuntimeError('Requested port already occupied')
            with (args.output/'server.log').open('wb') as log:
                child=subprocess.Popen(command,cwd=args.executable.parent,env=environment,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                collector.set_pid(child.pid)
                collector.set_stage('load', 'Loading fixed MiMo GGUF control')
                write(args.output/'process.json',dict(pid=child.pid,argv=command))
                monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
                while True:
                    if child.poll() is not None:raise RuntimeError('Server exited before readiness')
                    try:
                        health,_=api('/health')
                        if health.get('status')=='ok':break
                    except (urllib.error.URLError,TimeoutError,ConnectionError):pass
                    if time.monotonic()-start>240:raise RuntimeError('Readiness timeout')
                    time.sleep(1)
                checks=[]
                stop_checks=[]
                for marker in ('<|im_end|>','<|endoftext|>'):
                    tokenized,_=api('/tokenize',dict(content=marker,add_special=False,parse_special=True))
                    if len(tokenized['tokens'])!=1:raise RuntimeError('Stop marker is not a single token')
                    stop_checks.extend(tokenized['tokens'])
                server_log=(args.output/'server.log').read_text(encoding='utf-8',errors='replace')
                eog_ids=set(map(int,re.findall(r'EOG token\s*=\s*(\d+)',server_log)))
                if set(stop_checks)!=set(protocol['stop_ids']) or eog_ids!=set(protocol['stop_ids']):
                    raise RuntimeError('Backend stop IDs differ from frozen protocol: '+repr((stop_checks,sorted(eog_ids))))
                write(args.output/'stop-checks.json',dict(marker_ids=stop_checks,backend_eog_ids=sorted(eog_ids)))
                for case in protocol['cases']:
                    tokenized,_=api('/tokenize',dict(content=case['rendered'],add_special=False,parse_special=True))
                    matched=tokenized['tokens']==case['input_ids']
                    checks.append(dict(id=case['id'],match=matched))
                    if not matched:raise RuntimeError('Tokenizer prefix mismatch: '+case['id'])
                write(args.output/'tokenizer-checks.json',checks)
                collector.set_stage(args.mode, 'Frozen MiMo comparison protocol')
                first=cases[suite['tasks'][0]['id']]
                complete('warmup',first['input_ids'],32,fixed=True,warmup=True)
                if args.mode=='quality':
                    for task in suite['tasks']:
                        complete(task['id'],cases[task['id']]['input_ids'],quality_limit)
                elif args.mode=='speed':
                    for task in (suite['tasks'][0],suite['tasks'][4]):
                        ids=cases[task['id']]['input_ids']
                        complete(task['id']+'-warmup',ids,128,fixed=True,warmup=True)
                        for rep in range(3):complete(task['id']+'-'+str(rep),ids,128,fixed=True)
                else:
                    for probe in protocol['teacher_forced_probes']:
                        complete(probe['id'],probe['input_ids'],1,fixed=True,target=probe['target_id'])
                if state['reason'] is not None:
                    raise RuntimeError('Resource monitor stopped run: '+state['reason'])
                state['status']='completed'
        except BaseException as error:
            state.update(status='failed',error=type(error).__name__+': '+str(error))
            raise
        finally:
            stopped.set()
            if monitor_thread is not None:monitor_thread.join(timeout=5)
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True)
                    child.wait(timeout=10)
            collector.stop()
            write(args.output/'windows-summary.json',collector.summary())
            state.update(process_seconds=time.monotonic()-start,results=results,
                         server_exit_code=child.returncode if child else None,
                         timing_scope='Client end-to-end plus llama.cpp native timings; native decode timing is not the same boundary as EXL3 after-first-iteration timing')
            write(args.output/'result.json',state)

if __name__=='__main__':main()
