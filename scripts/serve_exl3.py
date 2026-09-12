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


class Engine:
    """One loaded model; a fresh generator owns each request's recurrent state."""

    def __init__(self, args):
        self.args = args
        self.model_name = args.alias or 'exl3'
        self.context = args.context
        self.draft_confidence = args.draft_confidence
        try:
            self.cache_k, self.cache_v = _cache_precision().resolve_cache_types(args)
        except ValueError as exc:
            raise ValueError('Invalid cache precision: ' + str(exc))
        self.ready = False
        self.active = False
        self.started = self.completed = self.cancelled = self.failed = 0
        self.invalid_tool_outputs = 0
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = (self.output / 'events.jsonl').open('x', encoding='utf-8')
        self.t0 = time.monotonic()
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
        from quantlab.methods.exl3.optimizations import configure_native
        self.record('native_optimizations', **configure_native(binary,
            smallm_kernel=args.smallm_kernel, head_warps=args.head_warps))
        sys.path.insert(0, str(args.source_dir))
        from exllamav3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job, ArgmaxSampler
        from quantlab.methods.exl3.compat import install, prepare_loaded_module
        from quantlab.methods.exl3.decode_diagnostics import prepare_decode_fusions
        from transformers import AutoTokenizer
        self.Generator, self.Job, self.Sampler = Generator, Job, ArgmaxSampler
        props = torch.cuda.get_device_properties(0)
        gpu_fraction = getattr(args, 'gpu_memory_fraction', _resources().GPU_MEMORY_FRACTION_DEFAULT)
        allocator_limit = _resources().allocator_bytes(gpu_fraction, props.total_memory)
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
        self.record('runtime', device=props.name, torch=torch.__version__, hip=torch.version.hip,
                    extension_sha256=args.expected_extension_sha256, engine_sha256=digest(__file__),
                    allocator_limit_bytes=allocator_limit, gpu_memory_fraction=gpu_fraction,
                    context=args.context, draft_tokens=args.draft_tokens,
                    cache_k=self.cache_k, cache_v=self.cache_v,
                    attention_profile=_cache_precision().attention_profile_status(), timing_version=TIMING_VERSION)
        self.record('adaptive', dynamic_draft_tokens=args.draft_confidence is not None,
                    draft_confidence=args.draft_confidence)
        raw = json.loads((args.candidate / 'config.json').read_text())
        if raw.get('model_type') in ('qwen3_5', 'qwen3_5_moe') and 'text_config' in raw:
            cfg, _ = mapped_text_config(args.candidate)
        else:
            cfg = Config.from_directory(str(args.candidate))
        self.cfg = cfg
        max_context = raw.get('text_config', raw).get('max_position_embeddings')
        if max_context is None or args.context > max_context:
            raise ValueError('Requested context exceeds or lacks model metadata limit')
        install(cfg, native_smallm=args.native_smallm, native_smallm_max_rows=args.native_smallm_max_rows,
                native_attention=args.native_attention)
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
                model.load(device='cuda:0')
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
        from quantlab.tool_calls import protocol_for_template
        self.tool_protocol = protocol_for_template(self.tokenizer.hf_tokenizer.chat_template)
        torch.cuda.synchronize()
        self.ready = True
        self.record('loaded', model_load_count=1, allocator=self.memory(),
                    tokenizer_sha256=digest(args.candidate / 'tokenizer.json'),
                    template_sha256=hashlib.sha256(self.tokenizer.hf_tokenizer.chat_template.encode()).hexdigest(),
                    cache_k=self.cache_k, cache_v=self.cache_v, tool_protocol=self.tool_protocol,
                    attention_profile=_cache_precision().attention_profile_status(),
                    cache_storage_target=_cache_precision().cache_storage(self.cache, cache_k=self.cache_k, cache_v=self.cache_v),
                    cache_storage_draft=_cache_precision().cache_storage(self.draft_cache, cache_k=self.cache_k, cache_v=self.cache_v))

    def record(self, stage, **fields):
        self.events.write(json.dumps(dict(stage=stage, elapsed_seconds=time.monotonic()-self.t0, **fields),
                                     allow_nan=False) + '\n')
        self.events.flush()

    def memory(self):
        return dict(allocated_bytes=self.torch.cuda.memory_allocated(),
                    reserved_bytes=self.torch.cuda.memory_reserved(),
                    peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=self.torch.cuda.max_memory_reserved())

    def status(self):
        return dict(ready=self.ready, active=self.active, model=self.model_name, context=self.context,
                    model_load_count=1, started=self.started, completed=self.completed,
                    cancelled=self.cancelled, failed=self.failed, allocator=self.memory(),
                    tool_protocol=self.tool_protocol, invalid_tool_outputs=self.invalid_tool_outputs,
                    attention_profile=_cache_precision().attention_profile_status())

    def shutdown(self):
        if self.events.closed:
            return
        self.ready = False
        self.torch.cuda.synchronize()
        snapshot = self.status()
        self.record('shutdown', **snapshot)
        (self.output/'shutdown.json').write_text(json.dumps(snapshot, indent=2))
        self.events.close()

    def prepare(self, *, messages=None, prompt=None, max_tokens=256,
                tools=None, tool_choice=None, parallel_tool_calls=True):
        if not self.ready:
            raise ValueError('Engine requires restart after a generation failure')
        policy = None
        if messages is not None:
            if tools is not None or tool_choice is not None:
                from quantlab.tool_calls import prepare_tools
                messages, exposed, policy = prepare_tools(messages, tools, tool_choice,
                    parallel_tool_calls, self.tokenizer.hf_tokenizer.chat_template)
                rendered = self.tokenizer.hf_render_chat_template(messages, tools=exposed, enable_thinking=False)
            else:
                rendered = self.tokenizer.hf_render_chat_template(messages, enable_thinking=False)
        else:
            rendered = prompt
        ids = self.tokenizer.encode(rendered, add_bos=False, add_eos=False, encode_special_tokens=True)
        if ids.numel() < 1:
            raise ValueError('Prompt must encode at least one token')
        if ids.numel() + max_tokens + self.depth + 8 > self.context:
            raise ValueError('Prompt plus output and draft reserve exceeds configured context')
        return dict(ids=ids, max_tokens=max_tokens, tool_policy=policy)

    async def generate(self, prepared):
        from quantlab.tool_calls import ToolCallError, parse_response
        if self.active or not self.ready:
            raise RuntimeError('Engine unavailable or concurrent generation attempted')
        self.active = True
        self.started += 1
        serial = self.started
        gen = job = batch = event = original_prefill = None
        tokens, texts = [], []
        timing = FirstIterationTiming()
        begin = time.monotonic()
        done = False
        reason = None
        try:
            gen = self.Generator(self.model, self.cache, self.tokenizer, max_batch_size=1,
                                 max_chunk_size=256, max_q_size=3, draft_model=self.draft,
                                 draft_cache=self.draft_cache, num_draft_tokens=self.depth or None,
                                 enable_defrag=False, cpu_cache_size=0, recurrent_cache_size=1024**3,
                                 record_draft_stats=True, dynamic_draft_tokens=self.draft_confidence is not None,
                                 draft_confidence=self.draft_confidence if self.draft_confidence is not None else 0.4)
            # This upstream pin subtracts one in Job.__init__ and draft depth at EOS.
            job = self.Job(input_ids=prepared['ids'], max_new_tokens=prepared['max_tokens']+1+self.depth,
                           sampler=self.Sampler(), stop_conditions=self.cfg.eos_token_id_list,
                           token_healing=False, return_logits=False)
            original_prefill = job.prefill
            def checked_prefill(results):
                if not job.is_prefill_done() and self.model.caps.get('recurrent_states') and job.recurrent_state is None:
                    raise RuntimeError('Recurrent request lacks live state')
                return original_prefill(results)
            job.prefill = checked_prefill
            gen.enqueue(job)
            self.record('request_started', serial=serial, input_token_ids=prepared['ids'].flatten().tolist(),
                        max_tokens=prepared['max_tokens'])
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
                        yield dict(text=text, done=False)
                timing.observe(now, event_counts)
                await asyncio.sleep(0)
            if not done:
                raise RuntimeError('Generator ended without an EOS event')
            self.completed += 1
            usage = dict(prompt_tokens=prepared['ids'].numel(), completion_tokens=len(tokens),
                         total_tokens=prepared['ids'].numel()+len(tokens))
            finish = 'length' if reason == 'max_new_tokens' else 'stop'
            if prepared.get('tool_policy') is not None:
                visible, calls = parse_response(''.join(texts), prepared['tool_policy'], finish)
                if visible or calls:
                    yield dict(text=visible, tool_calls=calls, done=False)
                if calls:
                    finish = 'tool_calls'
            yield dict(text='', done=True, finish_reason=finish, usage=usage)
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except ToolCallError as exc:
            self.invalid_tool_outputs += 1
            self.record('request_protocol_error', serial=serial, error=str(exc))
            raise  # The GPU completed normally; keep the loaded engine healthy.
        except Exception as exc:
            self.failed += 1
            self.ready = False  # A GPU error can poison native state; fail closed.
            self.record('request_error', serial=serial, error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            try:
                if gen is not None and self.ready:
                    gen.clear_queue()
                    self.torch.cuda.synchronize()
            finally:
                if not done and self.ready:
                    self.cancelled += 1
                gen = job = batch = event = original_prefill = None
                gc.collect()  # Generator/job cycles must not retain per-request state.
                self.active = False
                decode_seconds = timing.decode_seconds()
                self.record('request_finished', serial=serial, completed=done, eos_reason=reason,
                            output_token_ids=tokens, output_text=''.join(texts),
                            timing_version=TIMING_VERSION, first_emitted_batch_tokens=timing.first_count,
                            first_token_seconds=timing.first_time-begin if timing.first_time else None,
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
    p.add_argument('--execute', action='store_true')
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
    if not args.execute or not all(config['execution'].get(k) is True for k in
                                   ('allow_local_inference', 'allow_backend_probes')):
        p.error('Requires --execute and both local execution permissions')
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
