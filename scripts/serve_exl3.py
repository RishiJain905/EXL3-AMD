"""Serve the verified EXL3 loader under launch_runtime's resource monitor."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import tomllib

from exl3_timing import TIMING_VERSION, FirstIterationTiming
from run_exl3_model_smoke import (
    ROOT, candidate_preflight, check_required_tensors, digest, mapped_text_config,
)


def _sanitize(value):
    """Strip terminal escapes and controls from console/log metadata."""
    text = str(value)
    return ''.join(c for c in text if c.isprintable() and c not in '\x1b\x07').strip()[:200]


def _say(message):
    """One bounded console line; the launcher relays stdout to the terminal."""
    print('exl3: ' + _sanitize(message)[:500], flush=True)

class Engine:
    """One loaded model; each request has a fresh Job, optionally reusing one generator."""

    def __init__(self, args):
        self.args = args
        self.model_name = args.alias or 'exl3'
        self.context = args.context
        self.draft_confidence = args.draft_confidence
        self.reasoning = getattr(args, 'reasoning', 'auto')
        self.reasoning_format = getattr(args, 'reasoning_format', 'auto')
        self.chunk_size = getattr(args, 'prefill_chunk', 1024)
        self.sampling_defaults = dict(
            temperature=getattr(args, 'temperature', 0.0),
            top_p=getattr(args, 'top_p', 1.0), top_k=getattr(args, 'top_k', 0),
            min_p=getattr(args, 'min_p', 0.0),
            repetition_penalty=getattr(args, 'repetition_penalty', 1.0),
            presence_penalty=getattr(args, 'presence_penalty', 0.0),
            frequency_penalty=getattr(args, 'frequency_penalty', 0.0),
            seed=getattr(args, 'seed', None))
        try:
            self.cache_k, self.cache_v = _cache_precision().resolve_cache_types(args)
        except ValueError as exc:
            raise ValueError('Invalid cache precision: ' + str(exc))
        self.ready = False
        self.active = False
        self.started = self.completed = self.cancelled = self.failed = 0
        self.invalid_tool_outputs = 0
        self.prefix_cache = getattr(args, 'prefix_cache', 'off') == 'on'
        self._prefix_gen = None
        self.prefix_hits = self.prefix_misses = 0
        self.prefix_cached_tokens = self.prefix_computed_tokens = 0
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = (self.output / 'events.jsonl').open('x', encoding='utf-8')
        self.t0 = time.monotonic()
        _say(f'stage=verify model={self.model_name} context={self.context}')
        candidate_preflight(args.candidate)
        files = {p.name: p for p in args.candidate.iterdir() if p.is_file()}
        prior = json.loads(args.candidate_manifest.read_text()) if args.candidate_manifest else None
        if prior is not None:
            if set(files) != {r['file'] for r in prior}:
                raise ValueError('Candidate file inventory changed')
            if any(files[r['file']].stat().st_size != r['bytes'] for r in prior):
                raise ValueError('Candidate file size changed')
        identity = dict(prior_manifest_sha256=digest(args.candidate_manifest) if prior else None,
                        prior_files=prior, current_sizes={k: p.stat().st_size for k, p in files.items()},
                        current_mtime_ns={k: p.stat().st_mtime_ns for k, p in files.items()},
                        scope='Reuse prior full hashes; current checks are inventory/size/mtime only.')
        (self.output / 'candidate-identity.json').write_text(json.dumps(identity, indent=2))
        binary, = args.extension_dir.glob('exllamav3_ext*.so')
        if digest(binary) != args.expected_extension_sha256:
            raise ValueError('Extension hash mismatch')
        os.environ.update(EXL3_BC_ATTN='1' if args.native_attention else '0', EXL3_GEMV='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          EXL3_SMALLM_GRAPH='1' if args.native_smallm_graph else '0')
        os.environ['EXL3_QC_DECODE_PROFILE'] = getattr(args, 'attention_profile', 'default')
        if (self.cache_k, self.cache_v) != ('f16', 'f16'):
            # Packed attention without full-cache FP16 staging.
            os.environ['EXL3_QC_STAGING'] = '0'
        if args.native_smallm_graph:
            os.environ['EXL3_GEMV_GRAPH'] = '1'
        if args.gemv_splitk_warps is not None:
            os.environ['EXL3_GEMV_SPLITK_WARPS'] = str(args.gemv_splitk_warps)
        import torch
        import torch.utils.cpp_extension as cpp
        self.torch = torch
        def no_build(*unused, **kwargs):
            raise RuntimeError('Native JIT build forbidden')
        cpp.load = cpp.load_inline = no_build
        torch.set_num_threads(2)
        spec = importlib.util.spec_from_file_location('exllamav3_ext', binary)
        extension = importlib.util.module_from_spec(spec)
        sys.modules['exllamav3_ext'] = extension
        spec.loader.exec_module(extension)
        self.extension = extension
        _say('stage=extension ok=1')
        from quantlab.methods.exl3.optimizations import configure_native
        self.record('native_optimizations', **configure_native(binary,
            smallm_kernel=args.smallm_kernel, head_warps=args.head_warps,
            prefill_gemm=getattr(args, 'prefill_gemm', 'blas')))
        sys.path.insert(0, str(args.source_dir))
        from exllamav3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job, ArgmaxSampler
        from exllamav3.generator.sampler.presets import ComboSampler
        from quantlab.methods.exl3.compat import install, prepare_loaded_module
        from quantlab.methods.exl3.decode_diagnostics import prepare_decode_fusions
        from transformers import AutoTokenizer
        self.Generator, self.Job, self.Sampler, self.ComboSampler = Generator, Job, ArgmaxSampler, ComboSampler
        props = torch.cuda.get_device_properties(0)
        gpu_fraction = getattr(args, 'gpu_memory_fraction', _resources().GPU_MEMORY_FRACTION_DEFAULT)
        allocator_limit = _resources().allocator_bytes(gpu_fraction, props.total_memory)
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
        self.record('runtime', device=props.name, torch=torch.__version__, hip=torch.version.hip,
                    extension_sha256=args.expected_extension_sha256, engine_sha256=digest(__file__),
                    allocator_limit_bytes=allocator_limit, gpu_memory_fraction=gpu_fraction,
                    context=args.context, draft_tokens=args.draft_tokens,
                    cache_k=self.cache_k, cache_v=self.cache_v,
                    attention_profile=_cache_precision().attention_profile_status(), timing_version=TIMING_VERSION,
                    reasoning=self.reasoning, reasoning_format=self.reasoning_format,
                    prefill_chunk=self.chunk_size, sampling_defaults=self.sampling_defaults)
        vram_gib = props.total_memory / (1024 ** 3)
        _say(f'stage=device name={props.name} vram_gib={vram_gib:.1f} '
             f'context={args.context} kv={self.cache_k}/{self.cache_v} '
             f'mtp={args.draft_tokens if args.mtp else 0} chunk={self.chunk_size} '
             f'gemm={getattr(args, "prefill_gemm", "blas")}')
        self.record('adaptive', dynamic_draft_tokens=args.draft_confidence is not None,
                    draft_confidence=args.draft_confidence)
        raw = json.loads((args.candidate / 'config.json').read_text())
        if raw.get('model_type') in ('qwen3_5', 'qwen3_5_moe') and 'text_config' in raw:
            cfg, _ = mapped_text_config(args.candidate)
        else:
            cfg = Config.from_directory(str(args.candidate))
        self.cfg = cfg
        self._apply_cpu_moe(args, raw)
        max_context = raw.get('text_config', raw).get('max_position_embeddings')
        if max_context is None or args.context > max_context:
            raise ValueError('Requested context exceeds or lacks model metadata limit')
        install(cfg, native_smallm=args.native_smallm, native_smallm_max_rows=args.native_smallm_max_rows,
                native_attention=args.native_attention)
        _say('stage=load model=1')
        self.model = Model.from_config(cfg)
        self.draft = Model.from_config(cfg, component='mtp') if args.mtp else None
        check_required_tensors(cfg, (self.model, self.draft) if self.draft else (self.model,))
        self.depth = args.draft_tokens if args.mtp else 0
        cache_options = _cache_precision().cache_kwargs(self.cache_k, self.cache_v, quant_layer_type=CacheLayer_quant)
        self.cache = Cache(self.model, max_num_tokens=args.context, max_batch_size=1, max_history=self.depth,
                           **cache_options)
        self.draft_cache = Cache(self.draft, max_num_tokens=args.context, max_batch_size=1,
                                 **cache_options) if self.draft else None
        self.fusion_counters = []
        with torch.inference_mode():
            for model in (self.draft, self.model):
                if model is None:
                    continue
                component = 'draft' if model is self.draft else 'target'
                _say(f'stage=weights component={component} loading=1')
                load_start = time.monotonic()
                metrics = cfg.stc.metrics
                bytes_before, read_before = metrics.bytes_loaded, metrics.time_elapsed
                model.load(device='cuda:0')
                load_seconds = time.monotonic() - load_start
                read_bytes = metrics.bytes_loaded - bytes_before
                read_seconds = metrics.time_elapsed - read_before
                self.record('weights_loaded', component=component, seconds=load_seconds,
                            loader_bytes=read_bytes, loader_seconds=read_seconds)
                _say(f'stage=weights component={component} loaded=1 seconds={load_seconds:.2f} '
                     f'read_mib={read_bytes / 1024**2:.0f}')
                for module in model.modules:
                    if args.decode_fusions == 'off':
                        prepare_loaded_module(module)
                    else:
                        self.fusion_counters.extend(prepare_decode_fusions(
                            module, args.decode_fusions, native_smallm_graph=args.native_smallm_graph,
                            native_smallm_max_rows=args.native_smallm_max_rows))
        from quantlab.methods.exl3.optimizations import install_optimizations
        with torch.inference_mode():
            self.optimizations = install_optimizations(self.model, self.draft,
                gpu_embedding=args.gpu_embedding, batch_greedy=args.batch_greedy,
                gpu_draft=args.gpu_draft, gpu_draft_metadata=args.gpu_draft_metadata)
            if args.draft_step_graph:
                from quantlab.methods.exl3.draft_graph import DraftStepGraph
                self.model.quantlab_draft_step = DraftStepGraph(self.draft)
                self.optimizations['draft_step_graph'] = self.model.quantlab_draft_step.stats
            if args.native_attention:
                from quantlab.methods.exl3.optimizations import observe_native_attention
                self.optimizations['native_attention'] = observe_native_attention(self.model, self.draft)
            if args.shortlist_groups:
                from quantlab.methods.exl3.shortlist import install_draft_shortlist
                self.optimizations['draft_shortlist'] = install_draft_shortlist(self.model, self.draft, args.shortlist_groups, args.shortlist_mode)
            if args.cache_mtp != 'off':
                from quantlab.methods.exl3.cached_projection import cache_mtp_projections
                self.optimizations['mtp_projection_cache'] = cache_mtp_projections(self.draft, args.cache_mtp)
        self.record('optimizations', **self.optimizations)
        self.tokenizer = Tokenizer.from_config(cfg)
        self.tokenizer.hf_tokenizer = AutoTokenizer.from_pretrained(
            args.candidate, local_files_only=True, trust_remote_code=False)
        template_source = 'model'
        template_file = getattr(args, 'chat_template_file', None)
        if template_file is not None:
            override = Path(template_file).read_text(encoding='utf-8')
            if not override.strip():
                raise ValueError('Chat template file is empty')
            self.tokenizer.hf_tokenizer.chat_template = override
            template_source = 'file'
        from quantlab.tool_calls import protocol_for_template
        self.tool_protocol = protocol_for_template(self.tokenizer.hf_tokenizer.chat_template)
        torch.cuda.synchronize()
        self.ready = True
        self.record('loaded', model_load_count=1, allocator=self.memory(),
                    tokenizer_sha256=digest(args.candidate / 'tokenizer.json'),
                    template_sha256=hashlib.sha256(self.tokenizer.hf_tokenizer.chat_template.encode()).hexdigest(),
                    template_source=template_source,
                    cache_k=self.cache_k, cache_v=self.cache_v, tool_protocol=self.tool_protocol,
                    attention_profile=_cache_precision().attention_profile_status(),
                    cache_storage_target=_cache_precision().cache_storage(self.cache, cache_k=self.cache_k, cache_v=self.cache_v),
                    cache_storage_draft=_cache_precision().cache_storage(self.draft_cache, cache_k=self.cache_k, cache_v=self.cache_v))
        mem = self.memory()
        _say(f"ready=1 model={self.model_name} template={template_source} "
             f"tools={self.tool_protocol or 'none'} reasoning={self.reasoning} "
             f"load_seconds={time.monotonic()-self.t0:.2f} reserved_mib={mem['reserved_bytes'] / (1024 ** 2):.0f}")

    def _apply_cpu_moe(self, args, raw):
        """Map --n-cpu-moe/--cpu-moe onto the vendored CPU-expert worker."""
        count = getattr(args, 'n_cpu_moe', 0) or 0
        if getattr(args, 'cpu_moe', 'off') == 'all':
            count = 1 << 30  # covers every MoE layer without knowing the depth
        self.cpu_moe_layers = count
        if not count:
            return
        text = raw.get('text_config', raw)
        experts = text.get('num_experts', text.get('num_local_experts', 0)) or 0
        if not experts or not text.get('num_hidden_layers'):
            raise ValueError('CPU MoE offload requires a block-sparse MoE model')
        for symbol in ('exl3_moe_cpu_make_layer', 'exl3_moe_cpu_forward'):
            if not hasattr(self.extension, symbol):
                raise ValueError(f'Extension lacks CPU MoE symbol: {symbol}')
        params = self.cfg.infer_params
        params.moe_cpu_offload = min(count, 1 << 30)
        threads = getattr(args, 'moe_cpu_threads', None)
        if threads is not None:
            params.moe_cpu_threads = threads
        self.record('cpu_moe', layers=params.moe_cpu_offload, threads=threads)
        _say(f'stage=cpu-moe layers={params.moe_cpu_offload} threads={threads or "auto"} '
             'note=first-N-layers-only')

    def record(self, stage, **fields):
        self.events.write(json.dumps(dict(stage=stage, elapsed_seconds=time.monotonic()-self.t0, **fields),
                                     allow_nan=False) + '\n')
        self.events.flush()

    def memory(self):
        return dict(allocated_bytes=self.torch.cuda.memory_allocated(),
                    reserved_bytes=self.torch.cuda.memory_reserved(),
                    peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=self.torch.cuda.max_memory_reserved())

    def _build_generator(self):
        # One construction site for both paths: fresh per request by default,
        # or one persistent generator when --prefix-cache on. Memory stays
        # bounded by the existing 1 GiB recurrent stash; no CPU KV tier and
        # no checkpoint-interval overrides, so reuse granularity is whatever
        # the vendored defaults (model interval, 32K prompt-prefill interval)
        # plus last-page/decode checkpoints provide.
        return self.Generator(self.model, self.cache, self.tokenizer, max_batch_size=1,
                              max_chunk_size=self.chunk_size, max_q_size=3, draft_model=self.draft,
                              draft_cache=self.draft_cache, num_draft_tokens=self.depth or None,
                              enable_defrag=False, cpu_cache_size=0, recurrent_cache_size=1024**3,
                              record_draft_stats=True, dynamic_draft_tokens=self.draft_confidence is not None,
                              draft_confidence=self.draft_confidence if self.draft_confidence is not None else 0.4)

    def _prefix_status(self):
        gen = getattr(self, '_prefix_gen', None)
        live = None
        if gen is not None:
            try:
                recurrent = getattr(gen, 'recurrent_cache', None)
                pagetable = getattr(gen, 'pagetable', None)
                live = dict(recurrent_checkpoints=len(recurrent) if recurrent is not None else 0,
                            recurrent_bytes=getattr(recurrent, 'current_size', 0),
                            recurrent_limit_bytes=getattr(recurrent, 'max_size', 0),
                            unreferenced_pages=(len(getattr(pagetable, 'unreferenced_pages', {}))
                                                if pagetable is not None else 0))
            except Exception:
                live = None
        return dict(enabled=getattr(self, 'prefix_cache', False),
                    hits=getattr(self, 'prefix_hits', 0), misses=getattr(self, 'prefix_misses', 0),
                    cached_tokens=getattr(self, 'prefix_cached_tokens', 0),
                    computed_tokens=getattr(self, 'prefix_computed_tokens', 0), live=live)

    def _release_generator(self, gen):
        if gen is getattr(self, '_prefix_gen', None):
            self._prefix_gen = None
        pool = getattr(gen, 'filter_pool', None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def status(self):
        return dict(ready=self.ready, active=self.active, model=self.model_name, context=self.context,
                    model_load_count=1, started=self.started, completed=self.completed,
                    cancelled=self.cancelled, failed=self.failed, allocator=self.memory(),
                    tool_protocol=self.tool_protocol, invalid_tool_outputs=self.invalid_tool_outputs,
                    reasoning=self.reasoning, reasoning_format=self.reasoning_format,
                    prefix_cache=self._prefix_status(),
                    prefill_chunk=self.chunk_size,
                    prefill_gemm=getattr(self.args, 'prefill_gemm', 'blas'),
                    attention_profile=_cache_precision().attention_profile_status())

    def shutdown(self):
        if self.events.closed:
            return
        if getattr(self, '_prefix_gen', None) is not None:
            gen = self._prefix_gen
            try:
                gen.clear_queue()
            except Exception:
                pass
            finally:
                self._release_generator(gen)
        self.ready = False
        self.torch.cuda.synchronize()
        snapshot = self.status()
        self.record('shutdown', **snapshot)
        (self.output/'shutdown.json').write_text(json.dumps(snapshot, indent=2))
        self.events.close()
        _say(f"shutdown=1 completed={snapshot['completed']} cancelled={snapshot['cancelled']} "
             f"failed={snapshot['failed']}")

    def prepare(self, *, messages=None, prompt=None, max_tokens=256,
                tools=None, tool_choice=None, parallel_tool_calls=True,
                sampling=None, template_kwargs=None):
        if not self.ready:
            raise ValueError('Engine requires restart after a generation failure')
        policy = None
        params = dict(self.sampling_defaults)
        if sampling:
            params.update({k: v for k, v in sampling.items() if v is not None})
        from quantlab.sampling import is_plain_greedy
        if getattr(self.args, 'batch_greedy', False) and not is_plain_greedy(params):
            raise ValueError('--batch-greedy requires greedy sampling without penalties')
        thinking = None  # auto: respect the saved template default
        if self.reasoning == 'on':
            thinking = True
        elif self.reasoning == 'off':
            thinking = False
        if template_kwargs and template_kwargs.get('enable_thinking') is not None:
            thinking = bool(template_kwargs['enable_thinking'])
        render_kwargs = {} if thinking is None else dict(enable_thinking=thinking)
        if messages is not None:
            if tools is not None or tool_choice is not None:
                from quantlab.tool_calls import prepare_tools
                messages, exposed, policy = prepare_tools(messages, tools, tool_choice,
                    parallel_tool_calls, self.tokenizer.hf_tokenizer.chat_template)
                rendered = self.tokenizer.hf_render_chat_template(messages, tools=exposed, **render_kwargs)
            else:
                rendered = self.tokenizer.hf_render_chat_template(messages, **render_kwargs)
            template = self.tokenizer.hf_tokenizer.chat_template
            recognize_reasoning = ('<think>' in template and '</think>' in template
                                   or self.reasoning_format == 'deepseek')
            split_reasoning = (self.reasoning_format != 'none' and thinking is not False
                               and recognize_reasoning)
            prefix_open = False
            if recognize_reasoning:
                from quantlab.reasoning import has_unclosed_think
                prefix_open = has_unclosed_think(rendered)
        else:
            rendered = prompt
            recognize_reasoning = False
            split_reasoning, prefix_open = False, False  # raw completions stay raw
        ids = self.tokenizer.encode(rendered, add_bos=False, add_eos=False, encode_special_tokens=True)
        if ids.numel() < 1:
            raise ValueError('Prompt must encode at least one token')
        if ids.numel() + max_tokens + self.depth + 8 > self.context:
            raise ValueError('Prompt plus output and draft reserve exceeds configured context')
        return dict(ids=ids, max_tokens=max_tokens, tool_policy=policy, sampling=params,
                    split_reasoning=split_reasoning, prefix_open=prefix_open,
                    recognize_reasoning=recognize_reasoning)

    async def generate(self, prepared):
        from quantlab.tool_calls import ToolCallError, parse_response
        from quantlab.sampling import DEFAULTS, build_sampler
        from quantlab.methods.exl3.prefix_cache import (
            cached_credit, prefill_telemetry, split_prefill_advance)
        if self.active or not self.ready:
            raise RuntimeError('Engine unavailable or concurrent generation attempted')
        self.active = True
        self.started += 1
        serial = self.started
        gen = job = batch = event = original_prefill = None
        tokens, texts = [], []
        reasoning_texts: list[str] = []
        timing = FirstIterationTiming()
        begin = time.monotonic()
        done = False
        response_finished = False
        reason = None
        prefill_done_at = None
        prefill_seconds = 0.0
        prefill_tokens = 0
        prefill_computed = prefill_cached = 0
        prefill_wall_seconds = None
        prefix_hit = None
        prompt_tokens = int(prepared['ids'].numel())
        split = prepared.get('split_reasoning', False)
        splitter = None
        if split and prepared.get('tool_policy') is None:
            from quantlab.reasoning import ReasoningSplitter
            splitter = ReasoningSplitter(prefix_open=prepared.get('prefix_open', False))
        try:
            params = dict(DEFAULTS)
            params.update(prepared.get('sampling') or {})
            sampler = build_sampler(params, ArgmaxSampler=self.Sampler, ComboSampler=self.ComboSampler)
            if getattr(self, 'prefix_cache', False):
                gen = getattr(self, '_prefix_gen', None)
                if gen is None:
                    gen = self._build_generator()
                    self._prefix_gen = gen
                else:
                    # Reused page metadata and recurrent checkpoints only. Job,
                    # sampler, and adaptive draft state stay per request, exactly
                    # as the fresh path constructs them.
                    shortlist = getattr(self.model, 'quantlab_shortlist', None)
                    if shortlist is not None:
                        shortlist.reset()
                    calibrator = getattr(gen, 'draft_calibrator', None)
                    if calibrator is not None:
                        gen.draft_calibrator = type(calibrator)(
                            calibrator.confidence, bin_width=calibrator.bin_width,
                            decay=calibrator.decay, min_count=calibrator.min_count,
                            burn_in=calibrator.burn_in)
                    gen._draft_conf_round = None
            else:
                gen = self._build_generator()
            # This upstream pin subtracts one in Job.__init__ and draft depth at EOS.
            job = self.Job(input_ids=prepared['ids'], max_new_tokens=prepared['max_tokens']+1+self.depth,
                           sampler=sampler, seed=params.get('seed'), stop_conditions=self.cfg.eos_token_id_list,
                           token_healing=False, return_logits=False)
            original_prefill = job.prefill
            def checked_prefill(results):
                nonlocal prefill_seconds, prefill_tokens, prefill_done_at, prefill_computed
                nonlocal prefill_cached, prefill_wall_seconds, prefix_hit
                if not job.is_prefill_done() and self.model.caps.get('recurrent_states') and job.recurrent_state is None:
                    raise RuntimeError('Recurrent request lacks live state')
                before = sum(seq.kv_position for seq in job.sequences)
                credit_before = cached_credit(job)
                prefill_start = time.monotonic()
                value = original_prefill(results)
                advanced = sum(seq.kv_position for seq in job.sequences) - before
                computed, _ = split_prefill_advance(advanced, credit_before, cached_credit(job))
                if advanced > 0:
                    self.torch.cuda.synchronize()
                    if computed > 0:
                        prefill_seconds += time.monotonic() - prefill_start
                    prefill_tokens += advanced
                    prefill_computed += computed
                if prefill_done_at is None and job.is_prefill_done():
                    prefill_done_at = time.monotonic()
                    # Include generator setup, page lookup and recurrent-state
                    # restoration, which occur before the first prefill call.
                    prefill_wall_seconds = prefill_done_at - begin
                    prefill_cached = cached_credit(job)
                    prefill_tokens = prefill_computed + prefill_cached
                    prefix_hit = prefill_cached > 0
                    if getattr(self, 'prefix_cache', False):
                        self.prefix_hits += 1 if prefix_hit else 0
                        self.prefix_misses += 0 if prefix_hit else 1
                        self.prefix_cached_tokens += prefill_cached
                        self.prefix_computed_tokens += prefill_computed
                    fields = prefill_telemetry(computed_tokens=prefill_computed, cached_tokens=prefill_cached,
                                               compute_seconds=prefill_seconds, wall_seconds=prefill_wall_seconds)
                    rate = fields['prefill_tokens_per_second']
                    line = (f'request={serial} prefill tokens={prefill_tokens} seconds={prefill_seconds:.3f} '
                            f'rate={rate:.1f}t/s' if rate is not None else
                            f'request={serial} prefill tokens={prefill_tokens} seconds={prefill_seconds:.3f} rate=n/a')
                    if getattr(self, 'prefix_cache', False):
                        line += (f' computed={prefill_computed} cached={prefill_cached} '
                                 f'wall={prefill_wall_seconds:.3f}s')
                    _say(line)
                return value
            job.prefill = checked_prefill
            gen.enqueue(job)
            self.record('request_started', serial=serial, input_token_ids=prepared['ids'].flatten().tolist(),
                        max_tokens=prepared['max_tokens'])
            _say(f'request={serial} start prompt_tokens={prompt_tokens} max_tokens={prepared["max_tokens"]}')
            first_announced = False
            while gen.num_remaining_jobs():
                # Pull one bounded GPU iteration on the event-loop thread. There is
                # no background decoder that can outrun a slow/disconnected client.
                batch = gen.iterate()
                self.torch.cuda.synchronize()
                now = time.monotonic()
                event_counts = []
                for event in batch:
                    ids = event.get('token_ids')
                    emitted = ids.flatten().tolist() if ids is not None else []
                    if any(t < 0 or t >= self.tokenizer.actual_vocab_size for t in emitted):
                        raise RuntimeError('Generated token outside actual vocabulary')
                    tokens.extend(emitted)
                    if len(tokens) > prepared['max_tokens']:
                        raise RuntimeError('Generation exceeded requested token limit')
                    event_counts.append(len(emitted))
                    text = event.get('text', '')
                    texts.append(text)
                    if event.get('eos'):
                        done, reason = True, event.get('eos_reason')
                    if text and prepared.get('tool_policy') is None:
                        if splitter is not None:
                            thinking, visible = splitter.feed(text)
                            if thinking:
                                reasoning_texts.append(thinking)
                            item = dict(text=visible, done=False)
                            if thinking:
                                item['reasoning'] = thinking
                            yield item
                        else:
                            yield dict(text=text, done=False)
                timing.observe(now, event_counts)
                if not first_announced and timing.first_time is not None:
                    first_announced = True
                    _say(f'request={serial} first_token seconds={timing.first_time - begin:.2f} '
                         f'batch_tokens={timing.first_count}')
                await asyncio.sleep(0)
            if not done:
                raise RuntimeError('Generator ended without an EOS event')
            if splitter is not None:
                thinking, visible = splitter.flush()
                if thinking or visible:
                    if thinking:
                        reasoning_texts.append(thinking)
                    yield dict(text=visible, reasoning=thinking, done=False)
            self.completed += 1
            total_seconds = time.monotonic() - begin
            decode_seconds = timing.decode_seconds()
            decode_tokens = timing.decode_tokens(len(tokens))
            timings = dict(prefill_seconds=prefill_seconds, prefill_tokens=prefill_tokens,
                           first_token_seconds=timing.first_time - begin if timing.first_time else None,
                           decode_seconds=decode_seconds, decode_tokens=decode_tokens, total_seconds=total_seconds,
                           decode_tokens_per_second=timing.tokens_per_second(len(tokens)))
            # Compute-only rate plus the cached/computed split; without reuse the
            # rate and totals reduce exactly to the historical fields above.
            timings.update(prefill_telemetry(computed_tokens=prefill_computed, cached_tokens=prefill_cached,
                                             compute_seconds=prefill_seconds, wall_seconds=prefill_wall_seconds))
            usage = dict(prompt_tokens=prompt_tokens, completion_tokens=len(tokens),
                         total_tokens=prompt_tokens + len(tokens), timings=timings)
            if self.depth:
                accepted = job.accepted_draft_tokens
                rejected = job.rejected_draft_tokens
                proposed = accepted + rejected
                usage['draft_tokens'] = dict(accepted=accepted, rejected=rejected)
                acceptance = f'{accepted/proposed:.1%}' if proposed else 'n/a'
                _say(f'request={serial} mtp accepted={accepted} rejected={rejected} acceptance={acceptance}')
            finish = 'length' if reason == 'max_new_tokens' else 'stop'
            if prepared.get('tool_policy') is not None:
                raw_text = ''.join(texts)
                thinking_text = ''
                if prepared.get('recognize_reasoning', split):
                    from quantlab.reasoning import split_complete
                    thinking_text, raw_text = split_complete(raw_text, prefix_open=prepared.get('prefix_open', False))
                    if thinking_text:
                        reasoning_texts.append(thinking_text)
                # Tool-looking text inside reasoning is never parsed as calls.
                visible, calls = parse_response(raw_text, prepared['tool_policy'], finish)
                if thinking_text and not split:
                    # Display raw reasoning when requested, but never parse it
                    # as executable tool-call data, regardless of display mode.
                    visible = '<think>' + thinking_text + '</think>' + visible
                    thinking_text = ''
                if visible or calls or thinking_text:
                    item = dict(text=visible, tool_calls=calls, done=False)
                    if thinking_text:
                        item['reasoning'] = thinking_text
                    yield item
                if calls:
                    finish = 'tool_calls'
            rate = timing.tokens_per_second(len(tokens))
            decode_text = f'{decode_seconds:.3f}s' if decode_seconds is not None else 'n/a'
            rate_text = f'{rate:.1f}t/s' if rate is not None else 'n/a'
            _say(f'request={serial} done tokens={len(tokens)} decode_tokens={decode_tokens} decode={decode_text} '
                 f'rate={rate_text} total={total_seconds:.2f}s finish={finish}')
            response_finished = True
            yield dict(text='', done=True, finish_reason=finish, usage=usage)
        except (GeneratorExit, asyncio.CancelledError):
            if not done:
                _say(f'request={serial} cancelled tokens={len(tokens)}')
            raise
        except ToolCallError as exc:
            self.invalid_tool_outputs += 1
            self.record('request_protocol_error', serial=serial, error=str(exc))
            _say(f'request={serial} protocol_error=1')
            raise  # The GPU completed normally; keep the loaded engine healthy.
        except Exception as exc:
            self.failed += 1
            self.ready = False  # A GPU error can poison native state; fail closed.
            self.record('request_error', serial=serial, error=type(exc).__name__+': '+str(exc))
            _say(f'request={serial} error={type(exc).__name__}')
            raise
        finally:
            # Preserve the already collected verification shapes for private
            # diagnostics before releasing the job and its GPU state.
            draft_rounds = list(getattr(job, 'draft_stats', ()))
            cleanup_ok = False
            try:
                if gen is not None and self.ready:
                    gen.clear_queue()
                    self.torch.cuda.synchronize()
                    cleanup_ok = True
            except Exception as exc:
                self.ready = False
                self.failed += 1
                self.record('request_cleanup_error', serial=serial, error=type(exc).__name__+': '+str(exc))
                raise
            finally:
                if not done and self.ready:
                    self.cancelled += 1
                if gen is not None and (gen is not getattr(self, '_prefix_gen', None)
                                        or not response_finished or not cleanup_ok or not self.ready):
                    # Cancellation, failure, or any uncertain cleanup discards
                    # reusable state; the next request rebuilds from scratch.
                    self._release_generator(gen)
                gen = job = batch = event = original_prefill = None
                gc.collect()  # Generator/job cycles must not retain per-request state.
                self.active = False
                decode_seconds = timing.decode_seconds()
                self.record('request_finished', serial=serial, completed=done, eos_reason=reason,
                            output_token_ids=tokens, output_text=''.join(texts),
                            draft_rounds=draft_rounds,
                            reasoning_text=''.join(reasoning_texts),
                            timing_version=TIMING_VERSION, first_emitted_batch_tokens=timing.first_count,
                            first_token_seconds=timing.first_time-begin if timing.first_time else None,
                            prefill_seconds=prefill_seconds, prefill_tokens=prefill_tokens,
                            prefill_computed_tokens=prefill_computed, prefill_cached_tokens=prefill_cached,
                            prefill_wall_seconds=prefill_wall_seconds, prefix_hit=prefix_hit,
                            decode_tokens=timing.decode_tokens(len(tokens)), decode_seconds=decode_seconds,
                            decode_tokens_per_second=timing.tokens_per_second(len(tokens)),
                            elapsed_request_seconds=time.monotonic()-begin, allocator=self.memory(),
                            cache_attention_backends_target=_cache_precision().attention_backends(self.cache),
                            cache_attention_backends_draft=_cache_precision().attention_backends(self.draft_cache),
                            attention_profile=_cache_precision().attention_profile_status())


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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'candidate', 'source-dir', 'extension-dir', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--candidate-manifest', type=Path)
    p.add_argument('--expected-extension-sha256', required=True)
    p.add_argument('--context', type=int, default=4096)
    p.add_argument('--mtp', action='store_true')
    p.add_argument('--draft-tokens', type=int, choices=range(1, 9), default=4)
    p.add_argument('--draft-confidence', type=float, default=None,
                   help='Adaptive MTP truncation target acceptance in (0,1); requires MTP')
    p.add_argument('--decode-fusions', choices=('off', 'gdn', 'gdn-mlp'), default='gdn')
    p.add_argument('--native-smallm', action='store_true')
    p.add_argument('--native-smallm-max-rows', type=int, choices=(3, 5, 9), default=5)
    p.add_argument('--native-smallm-graph', action='store_true')
    p.add_argument('--gemv-splitk-warps', type=int, choices=(4, 8, 16))
    p.add_argument('--smallm-kernel', choices=('dot','wmma','wmma-register'), default='dot')
    p.add_argument('--prefill-gemm', choices=('blas','wmma'), default='blas')
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
    p.add_argument('--host', choices=('127.0.0.1',), default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--alias')
    p.add_argument('--request-timeout', type=int, default=120)
    p.add_argument('--reasoning', choices=('on', 'off', 'auto'), default='auto',
                   help='Thinking templates: force on/off or respect the template default')
    p.add_argument('--reasoning-format', choices=('auto', 'deepseek', 'none'), default='auto',
                   help='Split <think>...</think> into reasoning_content; none keeps raw text')
    p.add_argument('--chat-template-file', type=Path, default=None,
                   help='Local trusted Jinja template file overriding the model template')
    p.add_argument('--jinja', action='store_true',
                   help='Accepted for harness compatibility; rendering is always Jinja via Transformers')
    p.add_argument('--temp', '--temperature', dest='temperature', type=float, default=0.0, help='Serve sampling default in [0,2]; 0 is greedy')
    p.add_argument('--top-p', type=float, default=1.0, help='Serve sampling default in (0,1]')
    p.add_argument('--top-k', type=int, default=0, help='Serve sampling default in [0,1000]; 0 disables')
    p.add_argument('--min-p', type=float, default=0.0, help='Serve sampling default in [0,1)')
    p.add_argument('--repeat-penalty', '--repetition-penalty', dest='repetition_penalty', type=float, default=1.0, help='Serve sampling default in (0,2]')
    p.add_argument('--presence-penalty', type=float, default=0.0, help='Serve sampling default in [-2,2]')
    p.add_argument('--frequency-penalty', type=float, default=0.0, help='Serve sampling default in [-2,2]')
    p.add_argument('--seed', type=int, default=None, help='Serve sampling default seed in [0,2**63)')
    p.add_argument('-b', '--batch-size', '--prefill-chunk', dest='prefill_chunk', type=int, default=1024,
                   help='Prompt tokens processed per prefill step, 256-8192, multiple of 256; default 1024')
    p.add_argument('--prefix-cache', choices=('off', 'on'), default='off',
                   help='Reuse one generator across serialized requests so repeated prompt prefixes skip recompute; off rebuilds per request')
    p.add_argument('-ncmoe', '--n-cpu-moe', dest='n_cpu_moe', type=int, default=0,
                   help='Keep routed experts of the first N MoE layers on CPU (llama.cpp -ncmoe analogue)')
    p.add_argument('-cmoe', '--cpu-moe', nargs='?', const='all', choices=('off', 'all'), default='off',
                   help='Offload every MoE layer experts to CPU; not dynamic caching')
    p.add_argument('--moe-cpu-threads', type=int, default=None, help='CPU MoE worker threads, 1-256')
    _resources().add_allocator_arg(p)
    _cache_precision().add_cache_precision_args(p)
    return p


def reserve_socket(host, port):
    """Reserve a listener before model loading; allow Linux TIME_WAIT reuse."""
    sock = socket.socket()
    try:
        if os.name == 'nt':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        # Listening now also keeps a second reuse-enabled process from binding
        # during the potentially long model load. Uvicorn reuses this socket.
        sock.listen(128)
        return sock
    except BaseException:
        sock.close()
        raise


def main():
    p = parser()
    args = p.parse_args()
    try:
        _resources().validate_fraction(args.gpu_memory_fraction, '--gpu-memory-fraction')
    except ValueError as exc:
        p.error(str(exc))
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
    config = tomllib.loads(args.config.read_text())
    if not all(config['execution'].get(k) is True for k in
               ('allow_local_inference', 'allow_backend_probes')):
        p.error('Requires both configured local execution permissions')
    if not 0 <= args.temperature <= 2:
        p.error('--temperature must be in [0,2]')
    if not 0 < args.top_p <= 1:
        p.error('--top-p must be in (0,1]')
    if not 0 <= args.min_p < 1:
        p.error('--min-p must be in [0,1)')
    if not 0 < args.repetition_penalty <= 2:
        p.error('--repetition-penalty must be in (0,2]')
    if not -2 <= args.presence_penalty <= 2:
        p.error('--presence-penalty must be in [-2,2]')
    if not -2 <= args.frequency_penalty <= 2:
        p.error('--frequency-penalty must be in [-2,2]')
    if not 0 <= args.top_k <= 1000:
        p.error('--top-k must be in [0,1000]')
    if args.seed is not None and not 0 <= args.seed < 2 ** 63:
        p.error('--seed must be in [0,2**63)')
    greedy = args.temperature == 0 or args.top_k == 1
    if args.batch_greedy and (not greedy or args.repetition_penalty != 1 or args.presence_penalty != 0 or args.frequency_penalty != 0):
        p.error('--batch-greedy requires greedy sampling without penalties')
    if not 256 <= args.prefill_chunk <= 8192 or args.prefill_chunk % 256:
        p.error('--prefill-chunk must be a multiple of 256 in [256,8192]')
    if args.n_cpu_moe < 0 or args.n_cpu_moe > 1024:
        p.error('--n-cpu-moe must be in [0,1024]')
    if args.cpu_moe == 'all' and args.n_cpu_moe:
        p.error('--cpu-moe all conflicts with --n-cpu-moe')
    if args.moe_cpu_threads is not None and not 1 <= args.moe_cpu_threads <= 256:
        p.error('--moe-cpu-threads must be in [1,256]')
    if args.chat_template_file is not None:
        name = str(args.chat_template_file)
        if '://' in name or not args.chat_template_file.is_file():
            p.error('--chat-template-file must name a local file')
        if args.chat_template_file.stat().st_size > 1024 * 1024:
            p.error('--chat-template-file exceeds 1 MiB')
    if args.context < 1024 or args.context % 256:
        p.error('Context must be a multiple of 256 and at least 1024')
    if not 1 <= args.port <= 65535 or not 1 <= args.request_timeout <= 900:
        p.error('Invalid port or request timeout')
    candidate, output = args.candidate.resolve(), args.output.resolve()
    if candidate == output or candidate in output.parents or output in candidate.parents:
        p.error('Output and model must be separate nonnested paths')
    # Reserve the loopback socket before expensive loading; retain it for uvicorn.
    sock = reserve_socket(args.host, args.port)
    import uvicorn
    from quantlab.server import create_app
    engine = Engine(args)
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            # Uvicorn re-raises SIGTERM after lifespan shutdown. Save completion
            # here, before that signal can bypass ordinary Python finally blocks.
            engine.shutdown()
    app = create_app(engine, request_timeout=args.request_timeout)
    app.router.lifespan_context = lifespan
    server = uvicorn.Server(uvicorn.Config(app,
                                          host=args.host, port=args.port, access_log=False,
                                          timeout_graceful_shutdown=3))
    try:
        with engine.torch.inference_mode():
            server.run(sockets=[sock])
    finally:
        engine.shutdown()
        sock.close()


if __name__ == '__main__':
    main()
