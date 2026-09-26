"""Freeze MiMo calibration from archived public dataset-viewer responses.

CPU only; does not download data or execute dataset content. Sources and their
licenses/revisions are fixed below. Keep response archives and outputs private.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

SOURCES = {
    'ot': dict(dataset='open-thoughts/OpenThoughts-114k', revision='bd093c3994fd54d2390985b66988ddf282a55eb6', license='apache-2.0', generator='DeepSeek-R1; dataset-authored traces'),
    'nous': dict(dataset='NousResearch/hermes-function-calling-v1', revision='dae3e1d28cfbcf4b915c04ea1e072030529b4bda', license='apache-2.0', generator='dataset-authored synthetic traces; exact generating run unknown'),
    'dolly': dict(dataset='databricks/databricks-dolly-15k', revision='bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a', license='cc-by-sa-3.0', generator='dataset contributors'),
}
SEED = 20260922
COUNTS = {'code':48,'math':32,'tools':16,'json':16,'general':16}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def normalize_tools(row):
    tools=json.loads(row['tools'])
    messages=[]
    for turn in row['conversations']:
        role={'human':'user','gpt':'assistant'}.get(turn['from'],turn['from'])
        value=turn['value']
        if role=='system':
            # Replace the source's JSON-call convention with MiMo's own tool
            # template; actual signatures are supplied through tools=.
            messages.append(dict(role='system',content='Use the supplied tools when needed. Do not invent missing arguments.'))
        elif role=='assistant' and '<tool_call>' in value:
            calls=[]
            for body in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',value,re.S):
                call=json.loads(body)
                calls.append(dict(type='function',function=dict(name=call['name'],arguments=call['arguments'])))
            if not calls: raise ValueError('Unparsed tool call')
            content=re.sub(r'<tool_call>.*?</tool_call>','',value,flags=re.S).strip()
            messages.append(dict(role=role,content=content,tool_calls=calls))
        elif role=='tool':
            bodies=re.findall(r'<tool_response>\s*(.*?)\s*</tool_response>',value,re.S)
            if not bodies: raise ValueError('Unparsed tool response')
            for body in bodies:
                response=json.loads(body)
                messages.append(dict(role='tool',name=response.get('name',''),content=json.dumps(response.get('content',response),ensure_ascii=False)))
        elif role in ('system','user','assistant'):
            messages.append(dict(role=role,content=value))
        else: raise ValueError('Unsupported role '+role)
    return messages,tools

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--responses',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output must be new')
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(args.source,local_files_only=True,trust_remote_code=False)
    groups={k:[] for k in COUNTS}; seen=set(); skipped=[]; response_hashes={}
    paths=sorted(args.responses.glob('ot-rows-*.json'))+sorted(args.responses.glob('dolly-rows*.json'))
    paths += [args.responses/'nous-tools.json',args.responses/'nous-json.json']
    paths=[x for x in paths if not x.name.endswith('-request.json')]
    for path in paths:
        raw=path.read_bytes();response_hashes[path.name]=sha(raw)
        obj=json.loads(raw)
        for entry in obj['rows']:
            row=entry['row']; tools=None
            identity=(path.name.split('-rows')[0],entry['row_idx']) if path.name.startswith(('ot-','dolly-')) else (path.stem,entry['row_idx'])
            if identity in seen:continue
            seen.add(identity)
            if entry.get('truncated_cells'):
                skipped.append(dict(identity=identity,reason='viewer_truncated_cells'));continue
            try:
                if path.name.startswith('ot-'):
                    family=row['domain']
                    if family not in ('code','math'):continue
                    messages=[dict(role='user',content=row['problem']),dict(role='assistant',content=row['deepseek_solution'],reasoning_content=row['deepseek_reasoning'])]
                    source='ot'; upstream=row['source']
                elif path.name.startswith('dolly-'):
                    family='general';source='dolly';upstream=row['category']
                    messages=[dict(role='user',content=row['instruction']+ ('\n\n'+row['context'] if row['context'] else '')),dict(role='assistant',content=row['response'])]
                elif path.stem=='nous-tools':
                    family='tools';source='nous';upstream='func_calling';messages,tools=normalize_tools(row)
                else:
                    family='json';source='nous';upstream='json_mode_agentic'
                    messages=[dict(role={'human':'user','gpt':'assistant'}.get(t['from'],t['from']),content=t['value']) for t in row['conversations']]
                rendered=tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=False,enable_thinking=True)
                tokens=tokenizer.encode(rendered,add_special_tokens=False)
                if not tokens:raise ValueError('empty tokens')
                groups[family].append(dict(identity=identity,source=source,upstream=upstream,response_file=path.name,
                    source_record_sha256=sha(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()),
                    rendered_sha256=sha(rendered.encode()),tokens=tokens))
            except (KeyError,ValueError,TypeError) as error:
                skipped.append(dict(identity=identity,reason=str(error)))
    rows=[];docs=[];rng=random.Random(SEED);fingerprints=set()
    for family,count in COUNTS.items():
        candidates=groups[family];rng.shuffle(candidates)
        stream=[];segments=[]
        for doc in candidates:
            tokens=doc.pop('tokens');fingerprint=sha(json.dumps(tokens).encode())
            if fingerprint in fingerprints:continue
            fingerprints.add(fingerprint)
            # Limit each conversation to one 2048-token window so long traces
            # cannot dominate. Cycle prefix/middle/suffix windows across docs.
            mode=len(segments)%3
            start=0 if mode==0 else max(0,(len(tokens)-2048)//2) if mode==1 else max(0,len(tokens)-2048)
            window=tokens[start:start+2048]
            doc.update(family=family,original_tokens=len(tokens),window_start=start,window_tokens=len(window),stream_start=len(stream))
            docs.append(doc);segments.append(doc);stream.extend(window)
            if len(stream)>=count*2048:break
        if len(stream)<count*2048:raise ValueError(f'{family}: only {len(stream)} of {count*2048} required tokens')
        for idx in range(count):
            start,end=idx*2048,(idx+1)*2048
            spans=[dict(identity=d['identity'],row_start=max(start,d['stream_start'])-start,
                        row_end=min(end,d['stream_start']+d['window_tokens'])-start,
                        source_token_start=d['window_start']+max(0,start-d['stream_start']))
                   for d in segments if d['stream_start']<end and d['stream_start']+d['window_tokens']>start]
            rows.append(dict(family=family,family_row=idx,tokens=stream[start:end],spans=spans))
    rng.shuffle(rows)
    tensor=torch.tensor([r.pop('tokens') for r in rows],dtype=torch.int64)
    assert tuple(tensor.shape)==(128,2048) and int(tensor.min())>=0 and int(tensor.max())<248320
    args.output.mkdir(parents=True)
    save_file({'input_ids':tensor},str(args.output/'calibration.safetensors'))
    manifest=dict(schema=1,seed=SEED,shape=list(tensor.shape),family_rows=COUNTS,sources=SOURCES,
                  response_sha256=response_hashes,documents=docs,rows=rows,skipped=skipped,
                  tokenizer_sha256=sha((args.source/'tokenizer.json').read_bytes()),
                  template_sha256=sha((args.source/'chat_template.jinja').read_bytes()),
                  script_sha256=sha(Path(__file__).read_bytes()),
                  tensor_sha256=sha(tensor.numpy().tobytes()),
                  artifact_sha256=sha((args.output/'calibration.safetensors').read_bytes()),
                  purpose='Calibration only; not BF16 references or quality evaluation',
                  transform='MiMo chat template; native tool-call normalization; deduplicate rendered tokens; seeded order; one prefix/middle/suffix window per document; concatenate without padding, retain exact span map')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({k:manifest[k] for k in ('shape','family_rows','artifact_sha256','tensor_sha256')}),flush=True)

if __name__=='__main__':main()
