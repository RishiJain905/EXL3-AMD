"""Run model-free native correctness checks in a registered Linux/WSL environment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import launch_runtime as launcher


def verified_file(binary, expected):
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
        raise ValueError('Expected SHA-256 must contain 64 hex digits')
    with binary.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected.lower():
        raise ValueError(f'Native binary hash mismatch: {binary.name}')
    return binary


def verified_binary(directory, expected):
    binaries = list(Path(directory).glob('exllamav3_ext*.so'))
    if len(binaries) != 1:
        raise ValueError('Expected exactly one native extension')
    return verified_file(binaries[0], expected)


def run_checks(request):
    """Internal child entry point; the parent owns the GPU lease and monitor."""
    binary = verified_binary(request['extension_dir'], request['expected_sha256'])
    if request.get('wmma_probe'):
        probe = verified_file(Path(request['wmma_probe']), request['wmma_probe_sha256'])
        subprocess.run([str(probe)], check=True, timeout=120)
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    import torch.utils.cpp_extension as cpp

    def no_build(*args, **kwargs):
        raise RuntimeError('Native JIT builds are forbidden during validation')

    cpp.load = cpp.load_inline = no_build
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.5)
    props = torch.cuda.get_device_properties(0)
    spec = importlib.util.spec_from_file_location('exllamav3_ext', binary)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    from quantlab.methods.exl3.optimizations import configure_native
    options = configure_native(binary, native_smallm=True, prefill_gemm='wmma')
    if options['smallm_codebooks'] != [0, 2]:
        raise ValueError('These checks require a build with mul1 small-M support')
    results = dict(device=props.name, arch=props.gcnArchName, torch=torch.__version__,
                   hip=torch.version.hip, extension_sha256=request['expected_sha256'],
                   native_options=options, checks={})
    if request.get('wmma_probe'):
        results['wmma_probe_sha256'] = request['wmma_probe_sha256']
    output = Path(request['run']) / 'results.json'
    for name in ('smallm', 'hgemm'):
        spec = importlib.util.spec_from_file_location('check_' + name,
                    ROOT / 'kernels/exl3' / ('check_' + name + '.py'))
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        results['checks'][name] = checker.run_checks(torch, extension)
        launcher.write_json(output, results)
        print(f"{name}: {len(results['checks'][name])} checks passed", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / '.runtime/installation.toml')
    parser.add_argument('--output', type=Path, required=True, help='New private result directory')
    parser.add_argument('--extension-dir', type=Path, help='Candidate build; requires its explicit hash')
    parser.add_argument('--expected-extension-sha256')
    parser.add_argument('--wmma-probe', type=Path, help='Optional compiled check_wmma.hip executable')
    parser.add_argument('--expected-wmma-sha256', help='Required with --wmma-probe')
    args = parser.parse_args(argv)
    if sys.platform != 'linux':
        parser.error('Run this validator inside Linux or WSL')
    if bool(args.extension_dir) != bool(args.expected_extension_sha256):
        parser.error('Candidate extension directory and expected SHA-256 must be supplied together')
    if bool(args.wmma_probe) != bool(args.expected_wmma_sha256):
        parser.error('WMMA probe and expected SHA-256 must be supplied together')
    config = tomllib.loads(args.config.read_text(encoding='utf-8'))
    if not all(config.get('execution', {}).get(key) is True for key in
               ('allow_backend_probes', 'allow_local_inference')):
        parser.error('Requires both configured local execution permissions')
    runtime = dict(config['runtime'])
    for key in ('python', 'sdk', 'torch_lib', 'hsa_preload', 'server_deps', 'extension_dir'):
        if runtime.get(key):
            runtime[key] = launcher.linux_path(runtime[key])
    directory = args.extension_dir or Path(runtime['extension_dir'])
    expected = args.expected_extension_sha256 or runtime['extension_sha256']
    try:
        verified_binary(directory, expected)
        if args.wmma_probe:
            verified_file(args.wmma_probe, args.expected_wmma_sha256)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    timeout = config.get('limits', {}).get('max_capture_seconds', 900)
    if type(timeout) is not int or timeout <= 0:
        parser.error('max_capture_seconds must be a positive integer')
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    request_file = run / 'request.json'
    request = dict(run=str(run), root=str(ROOT), runtime=runtime, timeout=min(timeout, 900),
                   extension_dir=str(directory.resolve()), expected_sha256=expected.lower(),
                   argv=[runtime['python'], str(Path(__file__).resolve()), '--worker', str(request_file)])
    if args.wmma_probe:
        request.update(wmma_probe=str(args.wmma_probe.resolve()),
                       wmma_probe_sha256=args.expected_wmma_sha256.lower())
    launcher.write_json(request_file, request)
    lock = Path(launcher.linux_path(runtime.get('lease_file', 'artifacts/locks/local-gpu.lock')))
    with launcher.lease(lock, run):
        return launcher.worker(request_file)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        sys.exit(run_checks(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))))
    sys.exit(main())
