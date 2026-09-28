"""Per-kernel GPU self-time of warm decode rounds without a device profiler.

rocprofv3 cannot trace kernels on the WSL ROCm stack (see docs/DECODE-PROFILING.md),
so this tool times every GPU operation of a real decode round with HIP events:

1. normal: uninstrumented rounds; wall time per round and emitted tokens.
2. segments: after every host synchronization a spin kernel holds the GPU while
   the host enqueues the next segment, so the segment then runs back to back.
   One event pair per segment gives GPU busy time without host gaps.
3. kernels: the same pre-queued rounds with one event after every operation
   (native extension call, packed projection, Triton launch or Torch op),
   attributed to phase, module and shape.

normal minus segments is time the GPU waits for the host (launch overhead,
synchronization round trips and bookkeeping); kernels minus segments is event
overhead. A segment whose host enqueue time exceeds its spin time was starved
and is reported. The engine is serve_exl3.Engine with the given serve flags,
including startup warm-up; decoding is greedy with fresh generators.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import launch_runtime as launcher

# Baseline serving flags of the MiMo 131K run (docs/OPTIMIZATION-PLAN.md) minus HTTP options.
DEFAULT_SERVE_FLAGS = [
    '--context', '131072', '--decode-fusions', 'off', '--native-smallm', '--native-smallm-max-rows', '9',
    '--prefill-chunk', '1024', '--mtp', '--draft-tokens', '2', '--mtp-dtype', 'bf16', '--smallm-kernel', 'dot',
    '--prefill-gemm', 'auto', '--gpu-memory-fraction', '0.99', '--cache-type-k', 'q6', '--cache-type-v', 'q6',
    '--attention-profile', 'auto', '--prefix-cache', 'off', '--temperature', '0']
PROMPT = ('Return only Python code defining first_index(values, target). values is an ascending sorted list '
          'of integers that may contain duplicates. Return the first index equal to target, or -1 if absent. '
          'Use O(log n) time and O(1) extra space. Include a docstring explaining the invariant and five '
          'assert examples after the function.')
FILLER = 'Routine audit record: service healthy; change reviewed; backup verified.\n'
# Torch ops that allocate or alias without launching a kernel.
NO_KERNEL = frozenset((
    'empty', 'empty_like', 'empty_strided', 'new_empty', 'new_empty_strided', 'view', '_unsafe_view',
    'reshape', 'as_strided', 'slice', 'select', 'unsqueeze', 'squeeze', 'permute', 't', 'transpose',
    'expand', 'detach', 'alias', 'split', 'split_with_sizes', 'narrow', 'unbind', 'chunk', 'lift_fresh',
    'view_as', 'flatten', 'unflatten', 'set_', 'resize_', 'record_stream', 'is_pinned', '_pin_memory',
    'pin_memory', 'contiguous', 'clone_meta', 'sym_size', 'sym_stride', 'sym_numel', 'dim', 'size'))


def projection_role(key):
    """'model.layers.7.mlp.down_proj' -> 'mlp.down_proj'; keeps the layer type, drops the index."""
    parts = [p for p in (key or '').split('.') if p and not p.isdigit() and p not in ('model', 'layers')]
    return '.'.join(parts[-2:]) if parts else '?'


def summarize_rounds(rounds, skip=2):
    """Aggregate per-round {label: [calls, ms]} dicts after skipping the first decode rounds."""
    measured = [r for r in rounds if not r['prefill']][skip:]
    if not measured:
        return dict(rounds=0, labels=[])
    totals = defaultdict(lambda: [0, 0.0])
    for row in measured:
        for label, (calls, ms) in row['ops'].items():
            totals[label][0] += calls
            totals[label][1] += ms
    n = len(measured)
    labels = sorted(({'label': k, 'calls_per_round': v[0] / n, 'ms_per_round': v[1] / n}
                     for k, v in totals.items()), key=lambda x: -x['ms_per_round'])
    return dict(rounds=n, labels=labels)


class Profiler:
    """Event marks around GPU operations; one instance, installed only while profiling."""

    def __init__(self, torch, per_op, spin_cycles):
        self.torch = torch
        self.per_op = per_op
        self.spin_cycles = spin_cycles
        self.active = False
        self.depth = 0
        self.context = []
        self.phase = 'other'
        self.pool = [torch.cuda.Event(enable_timing=True) for _ in range(4096)]
        self.used = 0
        self.marks = []
        self.segments = []
        self.blockers = []

    def event(self):
        if self.used == len(self.pool):
            self.pool.append(self.torch.cuda.Event(enable_timing=True))
        event = self.pool[self.used]
        self.used += 1
        event.record()
        return event

    def mark(self, label):
        self.marks.append((self.event(), label, time.perf_counter()))

    def start_segment(self):
        # Hold the GPU so the host can enqueue the whole next segment first.
        spin_start = self.event()
        self.torch.cuda._sleep(self.spin_cycles)
        self.segments.append(dict(spin_start=spin_start, host_start=time.perf_counter(), base=len(self.marks)))
        self.mark(None)

    def end_segment(self, reason):
        segment = self.segments[-1]
        segment['host_seconds'] = time.perf_counter() - segment['host_start']
        segment['reason'] = reason
        self.mark('segment_end' if not self.per_op else None)
        segment['end'] = len(self.marks) - 1

    def op_done(self, label):
        if self.per_op:
            self.mark(label)

    def label(self, kind, detail=''):
        module = self.context[-1] if self.context else ''
        return f'{self.phase} | {kind}{(" " + detail) if detail else ""}{(" @" + module) if module else ""}'

    def begin_round(self):
        self.used = 0
        self.marks, self.segments, self.blockers = [], [], []
        self.active = True
        self.start_segment()

    def end_round(self):
        self.end_segment('round_end')
        self.active = False
        self.torch.cuda.synchronize()
        ops = defaultdict(lambda: [0, 0.0])
        busy = starved = 0.0
        segments = []
        for segment in self.segments:
            first, last = segment['base'], segment['end']
            spin = segment['spin_start'].elapsed_time(self.marks[first][0])
            seg_ms = self.marks[first][0].elapsed_time(self.marks[last][0])
            busy += seg_ms
            if segment['host_seconds'] * 1000 >= spin:
                starved += seg_ms
            # The longest host interval between marks shows where the host blocked on the GPU.
            waits = sorted(((self.marks[i][2] - self.marks[i - 1][2]) * 1000, str(self.marks[i][1]))
                           for i in range(first + 1, last + 1))[-2:]
            segments.append(dict(reason=segment['reason'], gpu_ms=seg_ms, spin_ms=spin,
                                 host_ms=segment['host_seconds'] * 1000, marks=last - first,
                                 longest_host_waits=waits))
            for index in range(first + 1, last + 1):
                label = self.marks[index][1]
                if label is None or label == 'segment_end':
                    continue
                ops[label][0] += 1
                ops[label][1] += self.marks[index - 1][0].elapsed_time(self.marks[index][0])
        return dict(busy_ms=busy, starved_ms=starved, segments=segments, ops=dict(ops),
                    blockers=[b for b in self.blockers if 'aten:copy_' not in b[0] or b[1] < 40])


def install_hooks(prof, engine):
    """Wrap extension functions, EXL3 projections, module context, Triton and Torch ops. Returns restore()."""
    torch = prof.torch
    from torch.utils._python_dispatch import TorchDispatchMode
    import triton.runtime.jit as triton_jit
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.module import Module
    from exllamav3.modules.quant.exl3 import LinearEXL3
    restore = []

    def watched(describe, call):
        # Any call in which the host blocks for a spin-length time is an undetected synchronization.
        start = time.perf_counter()
        result = call()
        if prof.active and time.perf_counter() - start > 0.005:
            prof.blockers.append((describe(), (time.perf_counter() - start) * 1000))
        return result

    def leaf(label_fn, call):
        if not prof.active or prof.depth:
            return call()
        prof.depth += 1
        try:
            result = watched(label_fn, call)
        finally:
            prof.depth -= 1
        prof.op_done(label_fn())
        return result

    for name in dir(ext):
        original = getattr(ext, name)
        if name.startswith('_') or type(original).__name__ != 'builtin_function_or_method':
            continue
        def make(name, original):
            def wrapped(*args, **kwargs):
                return leaf(lambda: prof.label('ext:' + name), lambda: original(*args, **kwargs))
            return wrapped
        setattr(ext, name, make(name, original))
        restore.append((ext, name, original))

    linear_forward = LinearEXL3.forward
    def linear(self, x, params, out_dtype=None):
        def label():
            rows = x.numel() // x.shape[-1]
            return prof.label(f'linear K{self.K} {self.in_features}x{self.out_features} r{rows}',
                              projection_role(self.key))
        return leaf(label, lambda: linear_forward(self, x, params, out_dtype))
    LinearEXL3.forward = linear
    restore.append((LinearEXL3, 'forward', linear_forward))

    classes = {type(m) for component in (engine.model, engine.draft) if component is not None for m in component}
    wrapped = set()
    for cls in classes:
        for owner in cls.__mro__:
            if owner in wrapped or owner is LinearEXL3 or 'forward' not in owner.__dict__ or not issubclass(owner, Module):
                continue
            wrapped.add(owner)
            original = owner.__dict__['forward']
            def make(original, owner):
                def forward(self, *args, **kwargs):
                    prof.context.append(f'{owner.__name__}({projection_role(self.key)})')
                    try:
                        return original(self, *args, **kwargs)
                    finally:
                        prof.context.pop()
                return forward
            setattr(owner, 'forward', make(original, owner))
            restore.append((owner, 'forward', original))

    def phase(target, method, name):
        original = getattr(target, method)
        def wrapped(*args, **kwargs):
            previous, prof.phase = prof.phase, name
            try:
                return original(*args, **kwargs)
            finally:
                prof.phase = previous
        restore.append((target, method, vars(target).get(method)))
        setattr(target, method, wrapped)
    phase(engine.model, 'forward', 'target')
    if engine.draft is not None:
        for method, name in (('forward', 'draft'), ('sample_from_state', 'draft_sample'),
                             ('update_kv_from_target', 'draft_update'), ('prefill', 'draft_prefill')):
            if hasattr(engine.draft, method):
                phase(engine.draft, method, name)

    triton_run = triton_jit.JITFunction.run
    def run(self, *args, **kwargs):
        if kwargs.get('warmup'):
            return triton_run(self, *args, **kwargs)
        return leaf(lambda: prof.label('triton:' + getattr(self, '__name__', '?')),
                    lambda: triton_run(self, *args, **kwargs))
    triton_jit.JITFunction.run = run
    restore.append((triton_jit.JITFunction, 'run', triton_run))

    def synchronizing(name, fn):
        def wrapped(*args, **kwargs):
            if not prof.active or prof.depth:
                return fn(*args, **kwargs)
            prof.end_segment(name)
            prof.depth += 1
            try:
                return fn(*args, **kwargs)
            finally:
                prof.depth -= 1
                prof.start_segment()
        return wrapped
    for owner, name in ((torch.cuda, 'synchronize'), (torch.cuda.Event, 'synchronize'),
                        (torch.cuda.Stream, 'synchronize')):
        original = getattr(owner, name)
        setattr(owner, name, synchronizing('cuda.' + name, original))
        restore.append((owner, name, original))

    def tensors(args, kwargs):
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (list, tuple)):
                yield from (v for v in value if isinstance(v, torch.Tensor))

    def host_wait(name, args, kwargs):
        # Ops after which the host has waited for the GPU queue: scalar reads, blocking
        # CPU<->GPU copies, and any copy through pageable host memory (HIP stages it in-stream).
        if name in ('_local_scalar_dense', 'nonzero', 'equal', 'is_nonzero'):
            return any(t.is_cuda for t in tensors(args, kwargs))
        if name == 'copy_':
            destination, source = args[0], args[1]
            blocking = not kwargs.get('non_blocking', args[2] if len(args) > 2 else False)
        elif name in ('_to_copy', 'to'):
            # aten.to is composite and aliases when nothing changes, so it is seen before _to_copy.
            source = args[0]
            device = kwargs.get('device')
            if device is None and name == 'to' and len(args) > 1:
                other = args[1]
                device = other.device if isinstance(other, torch.Tensor) else (
                    other if isinstance(other, (torch.device, str)) else None)
            if device is None or torch.device(device).type == source.device.type:
                return False
            destination = None
            flags = [a for a in args[2:] if isinstance(a, bool)]
            blocking = not kwargs.get('non_blocking', flags[0] if flags else False)
        else:
            return False
        if destination is None:
            host_pinned = source.is_pinned() if source.device.type == 'cpu' else bool(kwargs.get('pin_memory'))
            crosses = True
        else:
            crosses = destination.device.type != source.device.type
            host = destination if destination.device.type == 'cpu' else source
            host_pinned = host.is_pinned()
        return crosses and (blocking or not host_pinned)

    class Ops(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func.overloadpacket.__name__
            if prof.active and not prof.depth and name == 'to' and host_wait(name, args, kwargs):
                return synchronizing('aten:to', lambda: func(*args, **kwargs))()
            if not prof.active or prof.depth or name in NO_KERNEL or getattr(func, 'is_view', False):
                if not prof.active:
                    return func(*args, **kwargs)
                return watched(lambda: f'nested aten:{name} depth={prof.depth}', lambda: func(*args, **kwargs))
            found = list(tensors(args, kwargs))
            if host_wait(name, args, kwargs):
                return synchronizing('aten:' + name, lambda: func(*args, **kwargs))()
            if not any(t.is_cuda for t in found):
                return watched(lambda: 'host aten:' + name + ' ' + str([(t.device.type, t.is_pinned()) for t in found]),
                               lambda: func(*args, **kwargs))
            shape = 'x'.join(str(s) for s in found[0].shape) if found else ''
            return leaf(lambda: prof.label('aten:' + name, shape), lambda: func(*args, **kwargs))

    mode = Ops()
    mode.__enter__()

    def undo():
        mode.__exit__(None, None, None)
        for owner, name, original in reversed(restore):
            if original is None:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
    return undo


def spin_cycles_for(torch, milliseconds):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    torch.cuda.synchronize()
    per_ms = 10_000_000 / start.elapsed_time(end)
    return int(per_ms * milliseconds), per_ms


def dispatch_floor(torch, count=2000):
    """GPU time per back-to-back trivial kernel when the queue is already full."""
    x = torch.zeros(1, device='cuda')
    cycles, _ = spin_cycles_for(torch, 50)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(cycles)
    start.record()
    for _ in range(count):
        x.add_(1)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / count


def build_ids(engine, occupied):
    torch = engine.torch
    tokenizer = engine.tokenizer
    def encode(text):
        rendered = tokenizer.hf_render_chat_template(
            [dict(role='user', content=text)], enable_thinking=False)
        return tokenizer.encode(rendered, add_bos=False, add_eos=False, encode_special_tokens=True)
    ids = encode(PROMPT)
    if occupied:
        filler = tokenizer.encode(FILLER, add_bos=False, add_eos=False).flatten().tolist()
        head = encode('Read the following records.\nPLACEHOLDER\n' + PROMPT).flatten().tolist()
        room = occupied - len(head)
        body = (filler * (room // len(filler) + 1))[:room]
        marker = tokenizer.encode('PLACEHOLDER', add_bos=False, add_eos=False).flatten().tolist()
        at = next(i for i in range(len(head)) if head[i:i + len(marker)] == marker)
        ids = torch.tensor([head[:at] + body + head[at + len(marker):]], dtype=torch.long)
    return ids


def generate(engine, ids, tokens, prof=None):
    torch = engine.torch
    gen = engine._build_generator()
    job = engine.Job(input_ids=ids, max_new_tokens=tokens + 1 + engine.depth, sampler=engine.Sampler(),
                     seed=0, stop_conditions=[], token_healing=False, return_logits=False)
    output, rounds = [], []
    try:
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            decode = job.is_prefill_done()
            torch.cuda.synchronize()
            start = time.perf_counter()
            if prof is not None and decode:
                prof.begin_round()
            batch = gen.iterate()
            detail = prof.end_round() if prof is not None and decode else None
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            emitted = []
            for event in batch:
                ids_out = event.get('token_ids')
                if ids_out is not None:
                    emitted += ids_out.flatten().tolist()
            output += emitted
            row = dict(prefill=not decode, ms=seconds * 1000, tokens=len(emitted))
            if detail is not None:
                row.update(detail)
            rounds.append(row)
        gen.clear_queue()
    finally:
        engine._release_generator(gen)
    return output, rounds


def median_rounds(rounds, key, skip=2):
    values = [r[key] for r in rounds if not r['prefill']][skip:]
    return statistics.median(values) if values else None


def run_profile(request):
    """Internal child entry point; the parent owns the GPU lease and monitor."""
    sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
    import hashlib
    import os
    os.environ.update(request.get('env', {}))
    import serve_exl3
    args = serve_exl3.parser().parse_args(request['serve_argv'])
    engine = serve_exl3.Engine(args)
    torch = engine.torch
    run = Path(request['run'])
    result = dict(request=request, cases=[], device=torch.cuda.get_device_name(0),
                  extension_sha256=args.expected_extension_sha256)
    spin, per_ms = spin_cycles_for(torch, request['spin_ms'])
    result['spin'] = dict(ms=request['spin_ms'], cycles=spin, cycles_per_ms=per_ms)
    result['dispatch_floor_us'] = dispatch_floor(torch)
    with torch.inference_mode():
        for occupied in request['occupied']:
            ids = build_ids(engine, occupied)
            case = dict(occupied=occupied, input_tokens=ids.numel(), passes=[])
            reference = None
            for mode in request['passes']:
                prof = None if mode == 'normal' else Profiler(torch, mode == 'kernels', spin)
                undo = install_hooks(prof, engine) if prof is not None else None
                try:
                    tokens, rounds = generate(engine, ids, request['tokens'], prof)
                finally:
                    if undo is not None:
                        undo()
                if reference is None:
                    reference = tokens
                decode = [r for r in rounds if not r['prefill']][2:]
                entry = dict(mode=mode, tokens_match=tokens == reference, output_tokens=len(tokens),
                             output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                             output_token_ids=tokens,
                             decode_tokens_per_second=(sum(r['tokens'] for r in decode)
                                                       / (sum(r['ms'] for r in decode) / 1000)) if decode else None,
                             decode_rounds=len(decode),
                             tokens_per_round=sum(r['tokens'] for r in decode) / max(1, len(decode)),
                             median_round_ms=median_rounds(rounds, 'ms'),
                             mean_round_ms=statistics.fmean(r['ms'] for r in decode) if decode else None)
                if prof is not None:
                    entry.update(median_busy_ms=median_rounds(rounds, 'busy_ms'),
                                 mean_busy_ms=statistics.fmean(r['busy_ms'] for r in decode) if decode else None,
                                 starved_ms_total=sum(r['starved_ms'] for r in decode),
                                 segments_per_round=statistics.fmean(len(r['segments']) for r in decode) if decode else None,
                                 example_segments=decode[len(decode) // 2]['segments'] if decode else [],
                                 profile=summarize_rounds(rounds))
                case['passes'].append(entry)
                (run / f'rounds-{occupied}-{mode}.json').write_text(json.dumps(
                    [{k: v for k, v in r.items() if k != 'ops'} for r in rounds], indent=1))
                launcher.write_json(run / 'result.json', result | dict(partial=True))
                print(json.dumps({k: v for k, v in entry.items()
                                  if k not in ('profile', 'example_segments', 'output_token_ids')}),
                      flush=True)
            result['cases'].append(case)
    launcher.write_json(run / 'result.json', result)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / '.runtime/installation.toml')
    parser.add_argument('--output', type=Path, required=True, help='New private result directory')
    parser.add_argument('-m', '--model', type=Path, required=True)
    parser.add_argument('--occupied', type=int, nargs='+', default=[0],
                        help='Prompt lengths to profile; 0 is the short prompt, N pads with filler to N tokens')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--passes', nargs='+', choices=('normal', 'segments', 'kernels'),
                        default=['normal', 'segments', 'kernels', 'normal'])
    parser.add_argument('--spin-ms', type=float, default=60.0, help='GPU hold after each host synchronization')
    parser.add_argument('--wait', type=float, default=0.0, help='Seconds to wait for a held GPU lease')
    parser.add_argument('--serve-flag', action='append', default=[], metavar='FLAG',
                        help='Extra serve_exl3 flag appended to the serve flags, e.g. --serve-flag=--no-narrow-gemm')
    parser.add_argument('--env', action='append', default=[], metavar='EXL3_NAME=VALUE',
                        help='Diagnostic EXL3_* dispatch variable for the backend process')
    parser.add_argument('serve_flags', nargs=argparse.REMAINDER,
                        help='-- then serve_exl3 flags; default: the MiMo 131K serving configuration')
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        parser.error('Run this profiler inside Linux or WSL')
    config = tomllib.loads(args.config.read_text(encoding='utf-8'))
    if not all(config.get('execution', {}).get(key) is True for key in
               ('allow_backend_probes', 'allow_local_inference')):
        parser.error('Requires both configured local execution permissions')
    runtime = dict(config['runtime'])
    for key in ('python', 'sdk', 'torch_lib', 'hsa_preload', 'server_deps', 'extension_dir'):
        if runtime.get(key):
            runtime[key] = launcher.linux_path(runtime[key])
    flags = args.serve_flags[1:] if args.serve_flags[:1] == ['--'] else args.serve_flags
    env = dict(item.split('=', 1) for item in args.env)
    if any(not key.startswith('EXL3_') for key in env):
        parser.error('--env accepts EXL3_* variables only')
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    serve_argv = ['--config', str(args.config.resolve()), '--candidate', str(args.model.resolve()),
                  '--source-dir', launcher.linux_path(ROOT / runtime['source_dir']),
                  '--extension-dir', runtime['extension_dir'],
                  '--expected-extension-sha256', runtime['extension_sha256'],
                  '--output', str(run / 'engine')] + (flags or DEFAULT_SERVE_FLAGS) + args.serve_flag
    request_file = run / 'request.json'
    timeout = config.get('limits', {}).get('max_capture_seconds', 900)
    request = dict(run=str(run), root=str(ROOT), runtime=runtime, timeout=min(int(timeout), 1800),
                   serve_argv=serve_argv, occupied=args.occupied, tokens=args.tokens, passes=args.passes,
                   spin_ms=args.spin_ms, env=env,
                   argv=[runtime['python'], str(Path(__file__).resolve()), '--worker', str(request_file)])
    launcher.write_json(request_file, request)
    lock = Path(launcher.linux_path(runtime.get('lease_file', 'artifacts/locks/local-gpu.lock')))
    deadline = time.monotonic() + args.wait
    while True:
        try:
            with launcher.lease(lock, run):
                return launcher.worker(request_file)
        except FileExistsError:
            if time.monotonic() >= deadline:
                parser.error('GPU lease is held: ' + str(lock))
            time.sleep(2)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        sys.exit(run_profile(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))))
    sys.exit(main())
