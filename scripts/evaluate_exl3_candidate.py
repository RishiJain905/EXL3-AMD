"""Bounded saved-candidate speed, task and occupied-context evaluation.

Run under the external serial GPU monitor. All model and runtime inputs are read-only.
"""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import tomllib
import traceback

from exl3_timing import TIMING_VERSION, FirstIterationTiming
from run_exl3_model_smoke import (
    ROOT, candidate_preflight, check_required_tensors, digest, inspect_logits,
    mapped_text_config, observe_sampler_input, observe_mtp_head,
)

def _cache_precision():
    try:
        from quantlab.methods.exl3 import cache_precision as module
        return module
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'quantlab_cache_precision', ROOT / 'src' / 'quantlab' / 'methods' / 'exl3' / 'cache_precision.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_RESOURCES = None


def _resources():
    global _RESOURCES
    if _RESOURCES is not None:
        return _RESOURCES
    try:
        import exl3_resources as module
        _RESOURCES = module
        return module
    except ImportError:
        pass
    try:
        import scripts.exl3_resources as module
        _RESOURCES = module
        return module
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location(
        'exl3_resources', ROOT / 'scripts' / 'exl3_resources.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RESOURCES = module
    return module


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'candidate', 'source-dir', 'extension-dir', 'suite', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--candidate-manifest', type=Path)
    p.add_argument('--expected-extension-sha256', required=True)
    p.add_argument('--mode', choices=('speed', 'quality', 'context', 'context-speed', 'profile', 'generate'), required=True)
    p.add_argument('--context-speed-task', choices=('docstring', 'canonical'), default='docstring',
                   help='Occupied-coding question: 192-token docstring protocol or exact suite first_index task with 128 tokens; canonical requires context-speed mode')
    p.add_argument('--decode-fusions', choices=('off', 'gdn', 'gdn-mlp', 'gdn-mlp-mgemv'), default='off')
    p.add_argument('--gemv-splitk-warps', type=int, choices=(4, 8, 16))
    p.add_argument('--smallm-kernel', choices=('dot','wmma','wmma-register'), default='dot')
    p.add_argument('--head-warps', type=int, choices=(1,4,8,16))
    p.add_argument('--cache-mtp', choices=('off','fc','attention','mlp','all'), default='off')
    p.add_argument('--draft-step-graph', action='store_true', help='Experimental whole MTP step graph; requires GPU drafting and metadata')
    p.add_argument('--native-attention', action='store_true', help='Experimental native attention graph on ROCm')
    p.add_argument('--shortlist-groups', type=int, choices=(8,16,32,64,128), default=0, help='Experimental draft vocabulary groups of 128 tokens')
    p.add_argument('--shortlist-mode', choices=('packed','dense'), default='packed')
    p.add_argument('--gpu-embedding', action='store_true')
    p.add_argument('--batch-greedy', action='store_true')
    p.add_argument('--gpu-draft', action='store_true')
    p.add_argument('--gpu-draft-metadata', action='store_true')
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--mtp', action='store_true')
    p.add_argument('--draft-tokens', type=int, choices=range(1, 9), default=2)
    p.add_argument('--draft-confidence', type=float, default=None,
                   help='Adaptive MTP truncation target acceptance in (0,1); requires MTP')
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--native-smallm', action='store_true',
                   help='Use the separately built and hash-pinned K2/3/4 two/three-row kernel')
    p.add_argument('--native-smallm-max-rows', type=int, choices=(3,5,9), default=3)
    p.add_argument('--native-smallm-graph', action='store_true',
                   help='Enable two/three-row GDN/MLP graphs in the v2 experimental binary')
    p.add_argument('--expected-results', type=Path,
                   help='Retained result.json; fail immediately on an input/output token mismatch')
    p.add_argument('--execute', action='store_true')
    _resources().add_allocator_arg(p)
    _cache_precision().add_cache_precision_args(p)
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        _resources().validate_fraction(args.gpu_memory_fraction, '--gpu-memory-fraction')
    except ValueError as exc:
        p.error(str(exc))
    if args.context_speed_task != 'docstring' and args.mode != 'context-speed':
        p.error('--context-speed-task canonical requires context-speed mode')
    if args.draft_step_graph and (not args.gpu_draft_metadata or args.shortlist_groups or args.draft_confidence is not None):
        p.error('Draft step graph requires GPU draft metadata, fixed MTP, and full draft head')
    if args.shortlist_groups and (not args.mtp or args.draft_confidence is not None):
        p.error('Draft shortlist requires fixed-depth MTP')
    if args.gpu_draft and (not args.gpu_embedding or not args.mtp):
        p.error('--gpu-draft requires --gpu-embedding and MTP')
    if args.gpu_draft_metadata and not args.gpu_draft:
        p.error('--gpu-draft-metadata requires --gpu-draft')
    if args.cache_mtp != 'off' and (not args.mtp or args.decode_fusions not in ('off','gdn')):
        p.error('--cache-mtp requires MTP and off/gdn decode fusions')
    if args.draft_confidence is not None:
        if not 0 < args.draft_confidence < 1:
            p.error('--draft-confidence must be in (0,1)')
        if not args.mtp:
            p.error('--draft-confidence requires MTP')
        if args.gpu_draft:
            p.error('--draft-confidence conflicts with --gpu-draft')
    try:
        cache_k, cache_v = _cache_precision().resolve_cache_types(args)
    except ValueError as exc:
        p.error(str(exc))
    if (cache_k, cache_v) != ('f16', 'f16') and (args.native_attention or args.draft_step_graph):
        p.error('Quantized KV cache cannot combine with --native-attention or --draft-step-graph in this integration')
    if args.mode == 'generate' and (args.prompt_file is None or not 1 <= args.max_tokens <= 8192):
        p.error('generate requires --prompt-file and --max-tokens in [1,8192]')
    if args.native_smallm_graph and (not args.native_smallm or args.decode_fusions not in ('gdn', 'gdn-mlp')):
        p.error('Small-M graph requires --native-smallm and gdn or gdn-mlp fusions')
    config = tomllib.loads(args.config.read_text())
    if not args.execute or not all(config['execution'].get(k) is True for k in
                                  ('allow_local_inference', 'allow_backend_probes')):
        p.error('Requires --execute and both local permissions')
    if not 1024 <= args.context <= 262144 or args.context % 256:
        p.error('Context must be a multiple of 256 in [1024, 262144]')
    candidate = args.candidate.resolve()
    output = args.output.resolve()
    if candidate == output or candidate in output.parents or output in candidate.parents:
        p.error('Output and candidate must be separate nonnested paths')
    candidate_preflight(candidate)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'harness.py').write_bytes(Path(__file__).read_bytes())
    suite = json.loads(args.suite.read_text())
    expected = None
    if args.expected_results:
        reference = json.loads(args.expected_results.read_text())
        expected = {row['name']: row for row in reference['results']}
    (output / 'suite.json').write_text(json.dumps(suite, indent=2))
    started = time.monotonic()
    status = dict(status='running', mode=args.mode, mtp=args.mtp, cache_tokens=args.context,
                  cache_k=cache_k, cache_v=cache_v,
                  draft_tokens=args.draft_tokens if args.mtp else 0,
                  dynamic_draft_tokens=args.draft_confidence is not None,
                  draft_confidence=args.draft_confidence,
                  native_smallm=args.native_smallm,
                  native_smallm_max_rows=args.native_smallm_max_rows,
                  native_smallm_graph=args.native_smallm_graph,
                  decode_fusions=args.decode_fusions, fusion_counters=[],
                  gemv_splitk_warps=args.gemv_splitk_warps,
                  results=[], full_model_bf16_fidelity=None)
    events = (output / 'events.jsonl').open('x', encoding='utf-8')
    torch = None

    def memory():
        if torch is None:
            return None
        return dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())

    def record(stage, **fields):
        row = dict(stage=stage, elapsed_seconds=time.monotonic() - started, **fields)
        events.write(json.dumps(row, allow_nan=False) + '\n')
        events.flush()
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        current_files = {x.name: x for x in candidate.iterdir() if x.is_file()}
        previous_files = json.loads(args.candidate_manifest.read_text()) if args.candidate_manifest else None
        if previous_files is not None:
            if set(current_files) != {x['file'] for x in previous_files}:
                raise ValueError('Candidate file inventory changed')
            for item in previous_files:
                if current_files[item['file']].stat().st_size != item['bytes']:
                    raise ValueError('Candidate file size changed: ' + item['file'])
        (output / 'candidate-identity.json').write_text(json.dumps(dict(
            prior_manifest_sha256=digest(args.candidate_manifest) if args.candidate_manifest else None,
            prior_files=previous_files,
            current_sizes={k: v.stat().st_size for k, v in current_files.items()},
            current_mtime_ns={k: v.stat().st_mtime_ns for k, v in current_files.items()},
            scope='Prior full hashes reused when supplied; otherwise identity is inventory/size/mtime only. No content-hash claim without manifest.'), indent=2))
        binary, = args.extension_dir.glob('exllamav3_ext*.so')
        if digest(binary) != args.expected_extension_sha256:
            raise ValueError('Extension hash mismatch')
        for key, value in dict(EXL3_BC_ATTN='1' if args.native_attention else '0', EXL3_GEMV='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1').items():
            os.environ[key] = value
        os.environ['EXL3_SMALLM_GRAPH'] = '1' if args.native_smallm_graph else '0'
        os.environ['EXL3_QC_DECODE_PROFILE'] = getattr(args, 'attention_profile', 'default')
        if (cache_k, cache_v) != ('f16', 'f16'):
            # Packed attention without full-cache FP16 staging.
            os.environ['EXL3_QC_STAGING'] = '0'
        if args.native_smallm_graph:
            os.environ['EXL3_GEMV_GRAPH'] = '1'
        if args.gemv_splitk_warps is not None:
            os.environ['EXL3_GEMV_SPLITK_WARPS'] = str(args.gemv_splitk_warps)
        if args.decode_fusions == 'gdn-mlp-mgemv':
            os.environ['EXL3_MGEMV'] = '1'
        import torch as torch_module
        torch = torch_module
        import torch.utils.cpp_extension as cpp
        def no_build(*unused, **kwargs):
            raise RuntimeError('Native JIT build forbidden')
        cpp.load = cpp.load_inline = no_build
        torch.set_num_threads(2)
        # The external worker enforces timeout and resource limits. Repeated
        # asynchronous stack dumps crashed this Python/Triton combination.
        faulthandler.enable()
        spec = importlib.util.spec_from_file_location('exllamav3_ext', binary)
        extension = importlib.util.module_from_spec(spec)
        sys.modules['exllamav3_ext'] = extension
        spec.loader.exec_module(extension)
        from quantlab.methods.exl3.optimizations import configure_native
        status['native_optimizations'] = configure_native(binary,
            smallm_kernel=args.smallm_kernel, head_warps=args.head_warps)
        sys.path.insert(0, str(args.source_dir))
        from exllamav3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job, ArgmaxSampler
        from quantlab.methods.exl3.compat import install, prepare_loaded_module
        from quantlab.methods.exl3.decode_diagnostics import prepare_decode_fusions, profile_modules
        from transformers import AutoTokenizer
        props = torch.cuda.get_device_properties(0)
        gpu_fraction = getattr(args, 'gpu_memory_fraction', _resources().GPU_MEMORY_FRACTION_DEFAULT)
        allocator_limit = _resources().allocator_bytes(gpu_fraction, props.total_memory)
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
        record('runtime', device=props.name, torch=torch.__version__, hip=torch.version.hip,
               extension_sha256=args.expected_extension_sha256, harness_sha256=digest(__file__),
               suite_sha256=digest(args.suite), allocator_limit_bytes=allocator_limit,
               gpu_memory_fraction=gpu_fraction)
        record('adaptive', dynamic_draft_tokens=args.draft_confidence is not None,
               draft_confidence=args.draft_confidence)
        raw_config = json.loads((candidate / 'config.json').read_text())
        if raw_config.get('model_type') in ('qwen3_5', 'qwen3_5_moe') and 'text_config' in raw_config:
            cfg, _ = mapped_text_config(candidate)
        else:
            cfg = Config.from_directory(str(candidate))
        max_context = raw_config.get('text_config', raw_config).get('max_position_embeddings')
        if max_context is None or args.context > max_context:
            raise ValueError('Requested context exceeds or lacks model metadata limit')
        install(cfg, native_smallm=args.native_smallm, native_smallm_max_rows=args.native_smallm_max_rows,
                native_attention=args.native_attention)
        record('gemv_environment', values={k: os.environ.get(k) for k in (
            'EXL3_GEMV', 'EXL3_GEMV_SPLITK', 'EXL3_GEMV_SPLITK_WARPS',
            'EXL3_GEMV_LDS', 'EXL3_GEMV_GRAPH', 'EXL3_MGEMV',
            'EXL3_SMALLM', 'EXL3_SMALLM_GRAPH')})
        if args.decode_fusions == 'gdn-mlp-mgemv':
            # Create native handles at load; the guard exposes them only at
            # one row, where this pin has a noncooperative multi-matrix GEMV.
            cfg.infer_params.use_mgemm = lambda *unused, **kwargs: True
        model = Model.from_config(cfg)
        draft = Model.from_config(cfg, component='mtp') if args.mtp else None
        check_required_tensors(cfg, (model, draft) if draft is not None else (model,))
        cache_options = _cache_precision().cache_kwargs(cache_k, cache_v, quant_layer_type=CacheLayer_quant)
        cache = Cache(model, max_num_tokens=args.context, max_batch_size=1, max_history=args.draft_tokens if args.mtp else 0,
                      **cache_options)
        draft_cache = Cache(draft, max_num_tokens=args.context, max_batch_size=1, **cache_options) if args.mtp else None
        torch.cuda.synchronize()
        load_start = time.monotonic()
        def prepare(module):
            if args.decode_fusions == 'off':
                prepare_loaded_module(module)
            else:
                status['fusion_counters'].extend(prepare_decode_fusions(
                    module, args.decode_fusions, native_smallm_graph=args.native_smallm_graph,
                    native_smallm_max_rows=args.native_smallm_max_rows))
        if args.mtp:
            draft.load(device='cuda:0')
            for module in draft.modules:
                prepare(module)
        model.load(device='cuda:0')
        for module in model.modules:
            prepare(module)
        from quantlab.methods.exl3.optimizations import install_optimizations
        status['optimizations'] = install_optimizations(model, draft,
            gpu_embedding=args.gpu_embedding, batch_greedy=args.batch_greedy,
            gpu_draft=args.gpu_draft, gpu_draft_metadata=args.gpu_draft_metadata)
        if args.draft_step_graph:
            from quantlab.methods.exl3.draft_graph import DraftStepGraph
            model.quantlab_draft_step = DraftStepGraph(draft)
            status['draft_step_graph'] = model.quantlab_draft_step.stats
        if args.native_attention:
            from quantlab.methods.exl3.optimizations import observe_native_attention
            status['native_attention'] = observe_native_attention(model, draft)
        if args.shortlist_groups:
            from quantlab.methods.exl3.shortlist import install_draft_shortlist
            status['draft_shortlist'] = install_draft_shortlist(model, draft, args.shortlist_groups, args.shortlist_mode)
        if args.cache_mtp != 'off':
            from quantlab.methods.exl3.cached_projection import cache_mtp_projections
            status['mtp_projection_cache'] = cache_mtp_projections(draft, args.cache_mtp)
        status['cache_storage_target'] = _cache_precision().cache_storage(cache, cache_k=cache_k, cache_v=cache_v)
        status['cache_storage_draft'] = _cache_precision().cache_storage(draft_cache, cache_k=cache_k, cache_v=cache_v)
        torch.cuda.synchronize()
        record('loaded', seconds=time.monotonic() - load_start, allocator=memory(),
               embedding_device=str(model.modules[0].embedding.weight.device) if hasattr(model.modules[0], 'embedding') else None,
               model_max_context=max_context)
        tokenizer = Tokenizer.from_config(cfg)
        tokenizer.hf_tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True, trust_remote_code=False)
        record('tokenizer', tokenizer_sha256=digest(candidate / 'tokenizer.json'),
               template_sha256=hashlib.sha256(tokenizer.hf_tokenizer.chat_template.encode()).hexdigest(),
               eos_ids=cfg.eos_token_id_list, actual_vocab_size=tokenizer.actual_vocab_size)

        def encode_chat(prompt):
            messages = [dict(role='system', content=suite['system']), dict(role='user', content=prompt)]
            rendered = tokenizer.hf_render_chat_template(messages, enable_thinking=False)
            # The same runtime tokenizer is used for all modes; preserve rendered text and actual IDs.
            ids = tokenizer.encode(rendered, add_bos=False, add_eos=False, encode_special_tokens=True)
            return rendered, ids

        def run_case(name, prompt, limit, *, fixed=False, warmup=False, ids_override=None, metadata=None, profiling=None):
            rendered, ids = encode_chat(prompt)
            if ids_override is not None:
                ids = ids_override
                rendered = tokenizer.hf_tokenizer.decode(ids.flatten().tolist(), skip_special_tokens=False)
            if ids.numel() + limit + 4 > args.context:
                raise ValueError('Prompt/output exceeds context')
            (output / (name + '-input.json')).write_text(json.dumps(dict(
                prompt=prompt, rendered=rendered, input_token_ids=ids.flatten().tolist(), metadata=metadata), indent=2))
            gen = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=256, max_q_size=3,
                            draft_model=draft if args.mtp else None, draft_cache=draft_cache,
                            num_draft_tokens=args.draft_tokens if args.mtp else None, enable_defrag=False,
                            cpu_cache_size=0, recurrent_cache_size=1024**3, record_draft_stats=True,
                            dynamic_draft_tokens=args.draft_confidence is not None,
                            draft_confidence=args.draft_confidence if args.draft_confidence is not None else 0.4)
            # This pin requires a non-None, nonzero checkpoint cache to create
            # and retain live recurrent state. Fresh generators prevent reuse
            # across requests; retain the verified 1-GiB in-request budget.
            sampler = ArgmaxSampler()
            original_draft_sample = draft.sample_from_state if args.mtp else None
            if warmup:
                checked = []
                def check_once(logits, component='target'):
                    if not checked or args.native_smallm:
                        result = inspect_logits(logits, logits.shape[1], gen.padded_vocab_size)
                        record('warmup_raw_logits', component=component, **result)
                        if not result['logits_valid']:
                            raise RuntimeError('Nonfinite warmup logits')
                        checked.append(True)
                observe_sampler_input(sampler, check_once)
                if args.native_smallm and args.mtp:
                    observe_mtp_head(draft, model, lambda logits: check_once(logits, 'draft'))
            job = Job(input_ids=ids, max_new_tokens=limit + 1 + gen.num_draft_tokens, sampler=sampler,
                      stop_conditions=[] if fixed else cfg.eos_token_id_list, token_healing=False,
                      return_logits=False)
            original_prefill = job.prefill
            prefill_seconds = 0.0
            prefill_calls = 0
            def timed_prefill(results):
                nonlocal prefill_seconds, prefill_calls
                if job.is_prefill_done():
                    return original_prefill(results)
                if model.caps.get('recurrent_states') and job.recurrent_state is None:
                    raise RuntimeError('Recurrent model job lacks live state; invalid harness configuration')
                if prefill_calls == 0:
                    record('live_state_checked', name=name, present=job.recurrent_state is not None)
                torch.cuda.synchronize()
                t0 = time.monotonic()
                value = original_prefill(results)
                torch.cuda.synchronize()
                prefill_seconds += time.monotonic() - t0
                prefill_calls += 1
                return value
            job.prefill = timed_prefill
            tokens = []
            emission_batches = []
            timing = FirstIterationTiming()
            eos_reason = None
            last_progress = 0.0
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.monotonic()
            record('case_started', name=name, input_tokens=ids.numel(), output_limit=limit, warmup=warmup)
            gen.enqueue(job)
            while gen.num_remaining_jobs():
                if profiling is not None:
                    profiling['enabled'] = job.is_prefill_done()
                try:
                    batch = gen.iterate()
                finally:
                    if profiling is not None:
                        profiling['enabled'] = False
                torch.cuda.synchronize()
                now = time.monotonic()
                event_counts = []
                for event in batch:
                    emitted = event.get('token_ids')
                    emitted = emitted.flatten().tolist() if emitted is not None else []
                    if any(x < 0 or x >= tokenizer.actual_vocab_size for x in emitted):
                        raise RuntimeError('Generated token outside actual vocabulary')
                    tokens.extend(emitted)
                    event_counts.append(len(emitted))
                    if event.get('eos'):
                        eos_reason = event.get('eos_reason')
                    if now - last_progress >= 10:
                        record('progress', name=name, emitted_tokens=len(tokens),
                               prompt_progress=event.get('curr_progress'), text=job.full_completion,
                               allocator=memory())
                        last_progress = now
                batch_tokens = timing.observe(now, event_counts)
                if batch_tokens:
                    emission_batches.append(dict(elapsed_seconds=now - begin,
                                                 cumulative_tokens=len(tokens),
                                                 batch_tokens=batch_tokens))
                status['active_case'] = dict(name=name, emitted_tokens=len(tokens), text=job.full_completion)
            end = time.monotonic()
            if original_draft_sample is not None:
                draft.sample_from_state = original_draft_sample
            decode_seconds = timing.decode_seconds()
            decode_tokens = timing.decode_tokens(len(tokens))
            result = dict(name=name, warmup=warmup, input_tokens=ids.numel(), output_tokens=len(tokens),
                          output_limit=limit, fixed_length=fixed, eos_reason=eos_reason,
                          input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                          output_token_ids=tokens, output_text=job.full_completion,
                          prefill_call_seconds=prefill_seconds, prefill_calls=prefill_calls,
                          prefill_tokens_per_second=(ids.numel()-1)/prefill_seconds if prefill_seconds else None,
                          time_to_first_tokens_seconds=timing.first_time-begin if timing.first_time else None,
                          first_emitted_batch_tokens=timing.first_count, decode_after_first_seconds=decode_seconds,
                          emission_batches=emission_batches, timing_version=TIMING_VERSION,
                          cached_prompt_tokens=job.cached_tokens,
                          decode_after_first_tokens=decode_tokens,
                          decode_tokens_per_second=timing.tokens_per_second(len(tokens)),
                          end_to_end_seconds=end-begin, allocator=memory(), metadata=metadata,
                          cache_attention_backends_target=_cache_precision().attention_backends(cache),
                          cache_attention_backends_draft=_cache_precision().attention_backends(draft_cache),
                          attention_profile=_cache_precision().attention_profile_status(),
                          accepted_draft_tokens=job.accepted_draft_tokens if args.mtp else None,
                          rejected_draft_tokens=job.rejected_draft_tokens if args.mtp else None,
                          draft_windows=list(job.draft_stats),
                          calibrated_labels=gen.draft_calibrator.total if gen.draft_calibrator is not None else None)
            status['results'].append(result)
            (output / (name + '-result.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
            record('case_completed', **result)
            if expected is not None:
                ref = expected[name]
                matched = (ref['input_sha256'] == result['input_sha256']
                           and ref['output_token_ids'] == result['output_token_ids'])
                record('retained_output_equivalence', name=name, exact_match=matched,
                       reference_sha256=digest(args.expected_results))
                if not matched:
                    raise RuntimeError('Input/output tokens differ from retained reference: ' + name)
            if fixed and len(tokens) != limit:
                raise RuntimeError('Fixed-length token count mismatch')
            return result

        if args.mode in ('speed', 'quality', 'profile', 'generate'):
            run_case('warmup', suite['tasks'][0]['prompt'], 32, fixed=True, warmup=True)
        if args.mode == 'generate':
            run_case('answer', args.prompt_file.read_text(encoding='utf-8'), args.max_tokens)
        elif args.mode == 'profile':
            task = suite['tasks'][4]
            run_case('profile-warmup', task['prompt'], 128, fixed=True, warmup=True)
            profiling, summarize_profile = profile_modules(model, projections=True)
            if args.mtp:
                profile_modules(draft, state=profiling, scope='draft')
            run_case('profile-decode', task['prompt'], 32, fixed=True, profiling=profiling)
            status['module_profile'] = summarize_profile()
        elif args.mode == 'speed':
            for task in (suite['tasks'][0], suite['tasks'][4]):
                # Each exact measured prompt/shape gets its own discarded warmup.
                run_case(task['id'] + '-warmup', task['prompt'], 128, fixed=True, warmup=True)
                for repeat in range(3):
                    run_case(task['id'] + '-' + str(repeat), task['prompt'], 128, fixed=True)
        elif args.mode == 'quality':
            for task in suite['tasks']:
                run_case(task['id'], task['prompt'], suite['quality_max_tokens'])
        elif args.mode in ('context', 'context-speed'):
            # Construct tokens directly so actual occupancy is exact; decode each repeated filler block for inspection.
            # Occupied-coding throughput protocol: same exact-occupancy retrieval
            # recipe/filler/head/tail as context, but the tail question requests a
            # fixed 192-token code answer. Throughput only, no quality claim.
            filler = tokenizer.encode('Routine audit record: service healthy; change reviewed; backup verified.\n',
                                      add_bos=False, add_eos=False, encode_special_tokens=False).flatten().tolist()
            prefix, _ = encode_chat('PLACEHOLDER_CONTEXT')
            before, after = prefix.split('PLACEHOLDER_CONTEXT')
            intro = 'Read the following records. Remember the values of audit_alpha, audit_beta and audit_gamma.\n'
            if args.mode == 'context':
                question = '\nReturn only the three exact values of audit_alpha, audit_beta and audit_gamma, in that order.\n'
            elif args.context_speed_task == 'canonical':
                matches = [task['prompt'] for task in suite['tasks'] if task['id'] == 'first_index']
                if len(matches) != 1:
                    raise ValueError('Suite must contain exactly one first_index task')
                question = '\n' + matches[0] + '\n'
            else:
                question = '\nReturn only Python code defining first_index(values, target). values is an ascending sorted list of integers that may contain duplicates. Return the first index equal to target, or -1 if absent. Use O(log n) time and O(1) extra space. Include a docstring explaining the invariant and five assert examples after the function.\n'
            head = tokenizer.encode(before + intro, add_bos=False, add_eos=False, encode_special_tokens=True).flatten().tolist()
            tail = tokenizer.encode(question + after, add_bos=False, add_eos=False, encode_special_tokens=True).flatten().tolist()
            pairs = [('audit_alpha', 'violet-5831'), ('audit_beta', 'copper-9264'), ('audit_gamma', 'meadow-1478')]
            needles = [tokenizer.encode('\n' + k + ' = ' + v + '\n', add_bos=False, add_eos=False,
                                        encode_special_tokens=False).flatten().tolist() for k, v in pairs]
            target = args.context - 256
            room = target - len(head) - len(tail) - sum(map(len, needles))
            body = (filler * ((room // len(filler)) + 1))[:room]
            positions = []
            for proportion, needle in reversed(list(zip((0.1, 0.5, 0.9), needles))):
                at = int(room * proportion)
                body[at:at] = needle
            ids_list = head + body + tail
            for needle in needles:
                positions.append(next(i for i in range(len(ids_list)) if ids_list[i:i+len(needle)] == needle))
            if len(ids_list) != target:
                raise ValueError('Context construction did not reach exact occupancy')
            ids = torch.tensor([ids_list], dtype=torch.long)
            if args.mode == 'context':
                result = run_case('occupied-context', 'Synthetic token-level record retrieval; see input IDs.', 96,
                                  ids_override=ids, metadata=dict(needle_positions=positions, expected=[v for k, v in pairs]))
                status['retrieval_matches'] = [v in result['output_text'] for k, v in pairs]
            else:
                if args.context_speed_task == 'canonical':
                    run_case('occupied-coding-canonical', 'Synthetic token-level occupied coding; see input IDs.', 128,
                             fixed=True, ids_override=ids, metadata=dict(question=question, protocol='context-speed-canonical-128',
                                                                         task_id='first_index', needle_positions=positions,
                                                                         expected=[v for k, v in pairs]))
                else:
                    run_case('occupied-coding', 'Synthetic token-level occupied coding; see input IDs.', 192,
                             fixed=True, ids_override=ids, metadata=dict(question=question, protocol='context-speed',
                                                                         needle_positions=positions,
                                                                         expected=[v for k, v in pairs]))
        status['smallm_python_dispatch'] = [
            dict(key=child.key, calls=child.inner._quantlab_smallm_calls)
            for component in (model, draft) if component is not None for child in component
            if hasattr(getattr(child, 'inner', None), '_quantlab_smallm_calls')]
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error))
        (output / 'failure.txt').write_text(traceback.format_exc())
        record('failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        status['attention_profile'] = _cache_precision().attention_profile_status()
        status.update(process_seconds=time.monotonic()-started, allocator=memory(), timing_version=TIMING_VERSION,
                      timing_scope='WSL monotonic, GPU synchronized. Prefill sums Job.prefill calls; decode excludes whole first emitting GPU iteration (all token events sharing its synchronized timestamp). Fresh generator per request; 1-GiB in-request recurrent checkpoint cache.')
        (output / 'result.json').write_text(json.dumps(status, indent=2, allow_nan=False))
        events.close()


if __name__ == '__main__':
    main()
