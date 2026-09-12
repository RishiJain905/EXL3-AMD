"""Run the configured EXL3 model with serial, bounded Windows/WSL monitoring."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import posixpath
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


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
        import scripts.exl3_resources as module
        _RESOURCES = module
        return module
    except ImportError:
        pass
    try:
        import exl3_resources as module
        _RESOURCES = module
        return module
    except ImportError:
        pass
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'exl3_resources', ROOT / 'scripts' / 'exl3_resources.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RESOURCES = module
    return module

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def linux_path(value, root=ROOT):
    value = str(value).replace('\\', '/')
    if len(value) > 2 and value[1:3] == ':/':
        return posixpath.normpath('/mnt/' + value[0].lower() + value[2:])
    if value.startswith('/'):
        return posixpath.normpath(value)
    return linux_path(root / value)


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@contextmanager
def lease(path, run):
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid4().hex
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), run=str(run), nonce=nonce), stream)
    try:
        yield
    finally:
        try:
            if json.loads(path.read_text(encoding='utf-8')).get('nonce') == nonce:
                path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def runtime_environment(runtime):
    env = os.environ.copy()
    # Isolate benchmark dispatch from inherited experiments.
    for key in tuple(env):
        if key.startswith('EXL3_'):
            env.pop(key)
    sdk = runtime['sdk']
    env.update(LD_PRELOAD=runtime['hsa_preload'], ROCM_PATH=sdk, ROCM_HOME=sdk,
               HIP_PATH=sdk, EXL3_BACKEND='rocm', PYTORCH_ROCM_ARCH=runtime['gpu_arch'],
               GPU_ARCHS=runtime['gpu_arch'], MAX_JOBS='2', PYTHONUTF8='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env['PATH'] = str(Path(runtime['python']).parent) + ':' + sdk + '/bin:' + sdk + '/llvm/bin:' + env.get('PATH','')
    env.pop('PYTHONPATH', None)
    if runtime.get('server_deps'):
        env['PYTHONPATH'] = runtime['server_deps']
    env['LD_LIBRARY_PATH'] = runtime['torch_lib'] + ':' + sdk + '/lib:' + str(Path(runtime['hsa_preload']).parent)
    return env


def group_rss(pid):
    total = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pid:
                total += int((entry / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            pass
    return total


def terminate_group(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
    except ProcessLookupError:
        child.wait()


def clean_server_stop(request, run, code, reason, error):
    """Classify an intentional, confirmed HTTP shutdown without masking limits."""
    if (request.get('mode') != 'serve' or reason != 'external_stop' or error
            or code not in (0, -signal.SIGTERM) or (run/'external-stop.json').exists()):
        return False
    try:
        state = json.loads((run/'result/shutdown.json').read_text())
        return (state.get('ready') is False and state.get('active') is False
                and state.get('failed') == 0 and state.get('model_load_count') == 1)
    except (OSError, ValueError):
        return False


def worker(request_file):
    request = json.loads(Path(request_file).read_text(encoding='utf-8'))
    run = Path(request['run'])
    start = time.monotonic()
    peak = 0
    minimum_ram = minimum_disk = None
    reason = None
    error = None
    # Older saved requests lack request['limits']; wsl_limits() supplies
    # sanitized portable defaults.
    limits = _resources().wsl_limits(request)
    ram_guard = _resources().free_bytes(limits['min_free_ram_gib'])
    disk_guard = _resources().free_bytes(limits['min_free_disk_gib'])
    with (run / 'process.log').open('xb') as log:
        child = subprocess.Popen(request['argv'], cwd=request['root'], env=runtime_environment(request['runtime']),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(run / 'child.json', dict(pid=child.pid))
        try:
            while child.poll() is None:
                rss = group_rss(child.pid)
                mem_total, available = _resources().parse_meminfo(Path('/proc/meminfo').read_text())
                free = shutil.disk_usage(run).free
                rss_limit = _resources().rss_limit_bytes(limits['max_host_memory_fraction'], mem_total) if mem_total else None
                peak = max(peak, rss)
                if available is not None:
                    minimum_ram = available if minimum_ram is None else min(minimum_ram, available)
                minimum_disk = free if minimum_disk is None else min(minimum_disk, free)
                if (run / 'stop').exists(): reason = 'external_stop'
                elif request['timeout'] is not None and time.monotonic() - start >= request['timeout']: reason = 'timeout'
                elif rss_limit is not None and rss > rss_limit: reason = 'group_rss'
                elif available is not None and available < ram_guard: reason = 'wsl_available_ram'
                elif free < disk_guard: reason = 'artifact_disk'
                if reason:
                    terminate_group(child)
                    break
                time.sleep(1)
            code = child.wait()
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            reason = reason or 'monitor_error'
        finally:
            if child.poll() is None:
                terminate_group(child)
            code = child.returncode if child.returncode is not None else 1
    stopped_cleanly = clean_server_stop(request, run, code, reason, error)
    write_json(run / 'monitor.json', dict(exit_code=code, stop_reason='user_stop' if stopped_cleanly else reason, error=error,
               clean_server_shutdown=stopped_cleanly,
               elapsed_seconds=time.monotonic()-start, peak_group_rss_bytes=peak,
               min_wsl_available_bytes=minimum_ram, min_artifact_free_bytes=minimum_disk))
    return 0 if stopped_cleanly else (code if code else (1 if reason else 0))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', nargs='?', default='generate',
                   choices=('generate', 'speed', 'quality', 'context', 'context-speed', 'profile', 'serve'))
    p.add_argument('--context-speed-task', choices=('docstring', 'canonical'), default='docstring',
                   help='Occupied-coding question: 192-token docstring protocol or exact suite first_index task with 128 tokens; canonical requires context-speed mode')
    p.add_argument('--config', type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument('-m', '--model', type=Path, help='Local EXL3 model directory; required with an installed runtime')
    p.add_argument('--model-manifest', type=Path, help='Optional trusted file/hash manifest for --model')
    p.add_argument('-c', '--ctx-size', '--context', dest='context', type=int, default=4096)
    p.add_argument('--mtp', '--spec-draft-n-max', dest='mtp', type=int, choices=range(0, 9), default=None,
                   help='Draft tokens per verification; 0 disables MTP')
    p.add_argument('--spec-type', choices=('none','draft-mtp'), default=None,
                   help='Integrated MTP drafting; no separate draft model required')
    p.add_argument('-p', '--prompt', help='Prompt for generate mode')
    p.add_argument('-f', '--file', '--prompt-file', dest='prompt_file', type=Path, help='UTF-8 prompt file for generate mode')
    p.add_argument('-n', '--n-predict', '--max-tokens', dest='max_tokens', type=int, default=256,
                   help='Generation token limit; speed mode uses its fixed 128-token protocol')
    p.add_argument('--decode-fusions', choices=('off','gdn','gdn-mlp'), default='gdn')
    p.add_argument('--warps', type=int, choices=(4,8,16))
    p.add_argument('--smallm-kernel', choices=('dot','wmma','wmma-register'), default='dot', help='Experimental packed projection kernel')
    p.add_argument('--head-warps', type=int, choices=(1,4,8,16), help='Experimental wide-projection split-K override')
    p.add_argument('--cache-mtp', choices=('off','fc','attention','mlp','all'), default='off',
                   help='Cache selected reconstructed draft projections on GPU (extra VRAM)')
    p.add_argument('--draft-step-graph', action='store_true', help='Experimental whole MTP step graph; requires GPU drafting and metadata')
    p.add_argument('--native-attention', action='store_true', help='Experimental native attention graph on ROCm')
    p.add_argument('--shortlist-groups', type=int, choices=(8,16,32,64,128), default=0, help='Experimental draft vocabulary groups of 128 tokens')
    p.add_argument('--shortlist-mode', choices=('packed','dense'), default='packed')
    p.add_argument('--gpu-embedding', action='store_true', help='Experimental GPU-resident embedding table')
    p.add_argument('--batch-greedy', action='store_true', help='Batch independent greedy verification samples')
    p.add_argument('--gpu-draft', action='store_true', help='Keep integrated MTP token IDs on GPU; requires --gpu-embedding')
    p.add_argument('--gpu-draft-metadata', action='store_true', help='Keep draft attention metadata on GPU; requires --gpu-draft')
    p.add_argument('--draft-confidence', type=float, default=None,
                   help='Adaptive MTP truncation target acceptance in (0,1); requires positive MTP')
    p.add_argument('--expected-results', type=Path)
    p.add_argument('--host', choices=('127.0.0.1',), default='127.0.0.1', help='serve: loopback binding')
    p.add_argument('--port', type=int, default=8000, help='serve: HTTP port')
    p.add_argument('--alias', help='serve: public API model name (default: exl3)')
    p.add_argument('--request-timeout', type=int, default=120, help='serve: per-request seconds, including queue wait')
    p.add_argument('--output', type=Path, help='New private run directory; never overwritten')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--telemetry-gpu', type=str, default=None,
                   help='Windows monitoring adapter description substring; selects the monitored DXGI adapter only, not the Torch device. '
                   'Without it, monitoring auto-selects only when exactly one physical AMD adapter with dedicated VRAM above 1 GiB exists; '
                   'otherwise it reports unknown capacity.')
    _resources().add_resource_args(p)
    _cache_precision().add_cache_precision_args(p)
    return p


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    if len(sys.argv) == 3 and sys.argv[1] == '_worker':
        raise SystemExit(worker(sys.argv[2]))
    p = parser()
    args = p.parse_args()
    for name in ('gpu_memory_fraction', 'max_gpu_memory_fraction', 'max_host_memory_fraction'):
        try:
            _resources().validate_fraction(getattr(args, name), '--' + name.replace('_', '-'))
        except ValueError as exc:
            p.error(str(exc))
    for name in ('min_free_ram_gib', 'min_free_disk_gib'):
        try:
            _resources().validate_positive_gib(getattr(args, name), '--' + name.replace('_', '-'))
        except ValueError as exc:
            p.error(str(exc))
    if args.context_speed_task != 'docstring' and args.mode != 'context-speed':
        p.error('--context-speed-task canonical requires context-speed mode')
    if args.spec_type == 'none' and args.mtp:
        p.error('--spec-type none conflicts with a positive draft count')
    if args.spec_type == 'draft-mtp' and args.mtp == 0:
        p.error('--spec-type draft-mtp conflicts with zero draft tokens')
    if args.mtp is None:
        args.mtp = 2 if args.spec_type == 'draft-mtp' else 0
    if args.draft_step_graph and (not args.gpu_draft_metadata or args.shortlist_groups or args.draft_confidence is not None):
        p.error('Draft step graph requires GPU draft metadata, fixed MTP, and full draft head')
    if args.shortlist_groups and (not args.mtp or args.draft_confidence is not None):
        p.error('Draft shortlist requires fixed-depth MTP')
    if args.gpu_draft and (not args.gpu_embedding or not args.mtp):
        p.error('--gpu-draft requires --gpu-embedding and positive MTP depth')
    if args.gpu_draft_metadata and not args.gpu_draft:
        p.error('--gpu-draft-metadata requires --gpu-draft')
    if args.cache_mtp != 'off' and (not args.mtp or args.decode_fusions == 'gdn-mlp'):
        p.error('--cache-mtp requires MTP and off/gdn decode fusions')
    if args.draft_confidence is not None:
        if not 0 < args.draft_confidence < 1:
            p.error('--draft-confidence must be in (0,1)')
        if not args.mtp:
            p.error('--draft-confidence requires positive MTP')
        if args.gpu_draft:
            p.error('--draft-confidence conflicts with --gpu-draft')
    try:
        cache_k, cache_v = _cache_precision().resolve_cache_types(args)
    except ValueError as exc:
        p.error(str(exc))
    if (cache_k, cache_v) != ('f16', 'f16') and (args.native_attention or args.draft_step_graph):
        p.error('Quantized KV cache cannot combine with --native-attention or --draft-step-graph in this integration')
    if args.config is not None:
        config_path = args.config
        if not config_path.is_file():
            p.error(f'Config not found: {config_path}')
    else:
        installation = ROOT / '.runtime' / 'installation.toml'
        legacy = ROOT / 'configs' / 'local.toml'
        if installation.exists():
            config_path = installation
        elif legacy.is_file():
            config_path = legacy
        else:
            p.error(f'No runtime metadata: {installation} or {legacy}')
    try:
        config = tomllib.loads(config_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError):
        p.error(f'Cannot read runtime metadata: {config_path}')
    except tomllib.TOMLDecodeError as exc:
        p.error(f'Invalid runtime metadata {config_path}: {exc}')
    if not all(isinstance(config.get(section), dict) for section in ('execution', 'limits', 'runtime')):
        p.error(f'Invalid runtime metadata {config_path}: requires [execution], [limits] and [runtime]')
    if not args.execute or not all(config['execution'].get(k) is True for k in
                                   ('allow_local_inference','allow_backend_probes')):
        p.error('Requires --execute and both configured local execution permissions')
    max_capture = config['limits'].get('max_capture_seconds')
    if isinstance(max_capture, bool) or not isinstance(max_capture, int) or max_capture < 1:
        p.error(f'Invalid runtime metadata {config_path}: [limits] max_capture_seconds must be a positive integer')
    if args.context < 1024 or args.context % 256:
        p.error('--context must be a multiple of 256 and at least 1024')
    if not 1 <= args.max_tokens <= 8192:
        p.error('--max-tokens must be in [1,8192]')
    if args.prompt and args.prompt_file:
        p.error('Use one of --prompt and --prompt-file')
    if args.mode == 'generate' and not (args.prompt or args.prompt_file):
        p.error('generate requires --prompt or --prompt-file')
    if args.mode != 'generate' and (args.prompt or args.prompt_file or args.max_tokens != 256):
        p.error('Custom prompt and token-limit options apply only to generate mode')
    prompt = args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else args.prompt
    if args.expected_results and not args.expected_results.is_file():
        p.error('--expected-results must name an existing file')
    if args.mode == 'serve' and args.expected_results:
        p.error('--expected-results applies to benchmark modes, not serve')
    if not 1 <= args.port <= 65535 or not 1 <= args.request_timeout <= 900:
        p.error('Port must be in [1,65535] and request timeout in [1,900]')
    runtime = dict(config['runtime'])
    for key in ('python','sdk','torch_lib','hsa_preload','source_dir','extension_dir','extension_sha256'):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            p.error(f'Invalid runtime metadata {config_path}: [runtime] {key} must be a non-empty string')
    is_installation = config_path.resolve() == (ROOT / '.runtime' / 'installation.toml').resolve()
    if is_installation:
        for key in ('distribution', 'gpu_arch'):
            if not isinstance(runtime.get(key), str) or not runtime[key]:
                p.error(f'Invalid installation: [runtime] {key} must be a non-empty string')
        digest = runtime['extension_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
            p.error('Invalid installation: extension_sha256 must be 64 hex characters')
        rows = runtime.get('native_smallm_max_rows', 3)
        if type(rows) is not int or rows not in (3, 5, 9):
            p.error('Invalid installation: native_smallm_max_rows must match a supported 3, 5 or 9 row build')
    if args.model is not None:
        model = str(args.model).replace('\\', '/')
        runtime['candidate'] = model if model.startswith('/') or (len(model) > 1 and model[1] == ':') else args.model.resolve()
    elif is_installation or 'candidate' not in runtime:
        p.error('Requires -m MODEL_DIRECTORY with model-free runtime metadata')
    for key in ('python','sdk','torch_lib','hsa_preload','source_dir','extension_dir','candidate'):
        runtime[key] = linux_path(runtime[key])
    if args.mode == 'serve':
        runtime['server_deps'] = linux_path(runtime.get('server_deps', '.runtime/server-deps'))
    else:
        runtime.pop('server_deps', None)
    model_manifest = args.model_manifest or (None if args.model or is_installation else runtime.get('candidate_manifest'))
    run = args.output or ROOT / 'artifacts' / ('RUN-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8])
    run = run.resolve()
    candidate_path = linux_path(runtime['candidate']).rstrip('/')
    output_path = linux_path(run).rstrip('/')
    if os.name == 'nt':
        candidate_path, output_path = candidate_path.casefold(), output_path.casefold()
    if output_path == candidate_path or output_path.startswith(candidate_path+'/') or candidate_path.startswith(output_path+'/'):
        p.error('Output and model must be separate nonnested paths')
    run.mkdir(parents=True, exist_ok=False)
    entry = 'serve_exl3.py' if args.mode == 'serve' else 'evaluate_exl3_candidate.py'
    argv = [runtime['python'], linux_path(ROOT/'scripts'/entry),
            '--config', linux_path(config_path.resolve()), '--candidate', runtime['candidate'],
            '--source-dir', runtime['source_dir'],
            '--extension-dir', runtime['extension_dir'], '--expected-extension-sha256', runtime['extension_sha256'],
            '--output', linux_path(run/'result'), '--context', str(args.context),
            '--decode-fusions', args.decode_fusions, '--native-smallm', '--native-smallm-max-rows',
            str(runtime.get('native_smallm_max_rows',3)), '--execute']
    if args.mode == 'serve':
        argv += ['--host', args.host, '--port', str(args.port), '--request-timeout', str(args.request_timeout)]
        if args.alias:
            argv += ['--alias', args.alias]
    else:
        argv += ['--suite', linux_path(ROOT/'configs/evaluation.json'), '--mode', args.mode]
    if args.decode_fusions != 'off': argv += ['--native-smallm-graph']
    if model_manifest: argv += ['--candidate-manifest',linux_path(model_manifest)]
    if args.mtp: argv += ['--mtp','--draft-tokens',str(args.mtp)]
    if args.draft_confidence is not None: argv += ['--draft-confidence', str(args.draft_confidence)]
    if args.warps: argv += ['--gemv-splitk-warps',str(args.warps)]
    argv += ['--smallm-kernel',args.smallm_kernel]
    argv += ['--gpu-memory-fraction', str(args.gpu_memory_fraction)]
    if args.head_warps is not None: argv += ['--head-warps',str(args.head_warps)]
    argv += ['--cache-type-k',cache_k,'--cache-type-v',cache_v]
    argv += ['--attention-profile',args.attention_profile]
    if args.cache_mtp != 'off': argv += ['--cache-mtp',args.cache_mtp]
    if args.draft_step_graph: argv += ['--draft-step-graph']
    if args.native_attention: argv += ['--native-attention']
    if args.shortlist_groups: argv += ['--shortlist-groups',str(args.shortlist_groups),'--shortlist-mode',args.shortlist_mode]
    for flag in ('gpu_embedding','batch_greedy','gpu_draft','gpu_draft_metadata'):
        if getattr(args, flag): argv += ['--' + flag.replace('_','-')]
    if args.expected_results: argv += ['--expected-results',linux_path(args.expected_results.resolve())]
    if args.mode == 'context-speed': argv += ['--context-speed-task', args.context_speed_task]
    if args.mode == 'generate':
        (run/'prompt.txt').write_text(prompt,encoding='utf-8')
        argv += ['--prompt-file',linux_path(run/'prompt.txt'),'--max-tokens',str(args.max_tokens)]
    request = dict(root=linux_path(ROOT), run=linux_path(run), runtime=runtime, argv=argv, mode=args.mode,
                   limits=_resources().limits_from_args(args),
                   timeout=(None if args.mode == 'serve' else min(900,max_capture)))
    write_json(run/'request.json',request)
    print('Run artifacts: ' + str(run),flush=True)
    if args.mode == 'serve':
        print(f'Server: http://{args.host}:{args.port} (available after model load; runs until stopped)', flush=True)
    telemetry = None
    start = time.monotonic()
    lock_value = config['runtime'].get('lease_file','artifacts/locks/local-gpu.lock')
    lock = local_path(lock_value) if os.name == 'nt' else Path(linux_path(lock_value))
    with lease(lock,run):
        if os.name == 'nt':
            sys.path.insert(0,str(ROOT/'src'))
            from quantlab.telemetry import TelemetryCollector
            telemetry = TelemetryCollector(run.name, target_gpu_substring=args.telemetry_gpu, interval_sec=0.5)
            telemetry.start()
            child = subprocess.Popen(['wsl','-d',runtime['distribution'],'--','python3',
                       linux_path(ROOT/'run.py'),'_worker',linux_path(run/'request.json')])
            try:
                ram_guard = _resources().free_bytes(args.min_free_ram_gib)
                disk_guard = _resources().free_bytes(args.min_free_disk_gib)
                while child.poll() is None:
                    sample = telemetry.sample_now()
                    reason = None
                    host_avail = sample.get('host_available_bytes')
                    dedicated = sample.get('dedicated_gpu_bytes')
                    gpu_total = getattr(telemetry, 'gpu_total_dedicated_bytes', 0)
                    gpu_known = isinstance(gpu_total, (int, float)) and not isinstance(gpu_total, bool) and gpu_total > 0
                    if request['timeout'] is not None and time.monotonic()-start >= request['timeout']: reason='windows_timeout'
                    elif host_avail is not None and isinstance(host_avail, (int, float)) and host_avail < ram_guard: reason='windows_available_ram'
                    elif (dedicated is not None and isinstance(dedicated, (int, float)) and gpu_known
                          and dedicated > args.max_gpu_memory_fraction * gpu_total): reason='adapter_dedicated'
                    elif shutil.disk_usage(os.environ['SystemDrive']+'\\').free < disk_guard: reason='system_disk'
                    if reason:
                        if not (run/'external-stop.json').exists(): write_json(run/'external-stop.json',dict(reason=reason,sample=sample))
                        (run/'stop').touch()
                    time.sleep(1)
                code=child.wait()
            except KeyboardInterrupt:
                (run/'stop').touch()
                code=child.wait(timeout=20)
            finally:
                try:
                    if child.poll() is None:
                        (run/'stop').touch()
                        child.wait(timeout=20)
                finally:
                    telemetry.stop()
                    telemetry.write_csv(run/'telemetry.csv')
                    write_json(run/'telemetry.json',telemetry.summary())
        else:
            code = worker(run/'request.json')
    write_json(run/'launch.json',dict(exit_code=code,elapsed_seconds=time.monotonic()-start,
                                    windows_adapter_monitored=telemetry is not None))
    result_path=run/'result/result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf-8'))
        for row in result['results']:
            if row.get('warmup'): continue
            if args.mode in ('generate','quality'): print('\n'+row['name']+'\n'+row['output_text'])
            rate = row['decode_tokens_per_second']
            rate_text = f'{rate:.2f}' if rate is not None else 'unavailable'
            print(f"{row['name']}: {rate_text} tok/s decode; {row['input_tokens']} input, {row['output_tokens']} output")
    if code: print('Run failed; inspect '+str(run/'process.log'),file=sys.stderr)
    raise SystemExit(code)
