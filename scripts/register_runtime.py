"""Register an already-built EXL3 runtime as model-free installation metadata.

Reads an existing local TOML config and writes only machine-specific
execution, limits and installation keys. Never builds, installs, overwrites,
modifies the source config, or grants execution permission.
"""
import argparse
import string
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / '.runtime' / 'installation.toml'

RUNTIME_KEYS = ('distribution', 'python', 'sdk', 'torch_lib', 'hsa_preload',
                'source_dir', 'extension_dir', 'extension_sha256', 'gpu_arch',
                'native_smallm_max_rows', 'lease_file', 'server_deps')
REQUIRED_RUNTIME = ('distribution', 'python', 'sdk', 'torch_lib', 'hsa_preload',
                    'source_dir', 'extension_dir', 'extension_sha256', 'gpu_arch')
VALID_ROWS = (3, 5, 9)


def _toml_value(value):
    """Encode a bool/int/float/str scalar as TOML; roundtrips through tomllib."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError('non-finite float has no TOML encoding')
        return repr(value)
    if isinstance(value, str):
        if "'" not in value and not any(ord(c) < 0x20 or ord(c) == 0x7f for c in value):
            return "'" + value + "'"
        out = []
        for char in value:
            code = ord(char)
            if char == '"':
                out.append('\\"')
            elif char == '\\':
                out.append('\\\\')
            elif char == '\n':
                out.append('\\n')
            elif char == '\r':
                out.append('\\r')
            elif char == '\t':
                out.append('\\t')
            elif code < 0x20 or code == 0x7f:
                out.append(f'\\u{code:04X}')
            else:
                out.append(char)
        return '"' + ''.join(out) + '"'
    raise TypeError(f'unsupported TOML scalar: {type(value).__name__}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path, help='Existing local runtime TOML to register')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                   help='Installation metadata path (never overwritten)')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not args.source.is_file():
        p.error(f'source config not found: {args.source}')
    try:
        text = args.source.read_text(encoding='utf-8')
    except OSError:
        p.error(f'Cannot read source config: {args.source}')
    try:
        config = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        p.error(f'Invalid source config {args.source}: {exc}')
    execution = config.get('execution')
    if not isinstance(execution, dict):
        execution = {}
    allow_inference = execution.get('allow_local_inference') is True
    allow_probes = execution.get('allow_backend_probes') is True
    limits = config.get('limits')
    if not isinstance(limits, dict):
        p.error(f'Invalid source config {args.source}: missing [limits]')
    max_capture = limits.get('max_capture_seconds')
    if isinstance(max_capture, bool) or not isinstance(max_capture, int) or max_capture < 1:
        p.error(f'Invalid source config {args.source}: [limits] max_capture_seconds must be a positive integer')
    raw_runtime = config.get('runtime')
    if not isinstance(raw_runtime, dict):
        p.error(f'Invalid source config {args.source}: missing [runtime]')
    for key in REQUIRED_RUNTIME:
        if not isinstance(raw_runtime.get(key), str) or not raw_runtime[key]:
            p.error(f'Invalid source config {args.source}: [runtime] {key} must be a non-empty string')
    digest = raw_runtime['extension_sha256']
    if len(digest) != 64 or any(char not in string.hexdigits for char in digest):
        p.error(f'Invalid source config {args.source}: [runtime] extension_sha256 must be 64 hex characters')
    for key in ('gpu_arch', 'lease_file', 'server_deps'):
        if key in raw_runtime and (not isinstance(raw_runtime[key], str) or not raw_runtime[key]):
            p.error(f'Invalid source config {args.source}: [runtime] {key} must be a non-empty string')
    if 'native_smallm_max_rows' in raw_runtime:
        rows = raw_runtime['native_smallm_max_rows']
        if isinstance(rows, bool) or not isinstance(rows, int) or rows not in VALID_ROWS:
            p.error(f'Invalid source config {args.source}: [runtime] native_smallm_max_rows must be 3, 5 or 9')
    try:
        lines = ['[execution]',
                 f'allow_backend_probes = {_toml_value(allow_probes)}',
                 f'allow_local_inference = {_toml_value(allow_inference)}',
                 '', '[limits]']
        for key, value in limits.items():
            if isinstance(value, (bool, int, float, str)):
                lines.append(f'{_toml_value(key)} = {_toml_value(value)}')
            else:
                p.error(f'Invalid source config {args.source}: [limits] {key} must be a scalar')
        lines.extend(('', '[runtime]'))
        for key in RUNTIME_KEYS:
            if key in raw_runtime:
                lines.append(f'{key} = {_toml_value(raw_runtime[key])}')
        lines.append('')
        body = '\n'.join(lines)
        tomllib.loads(body)
    except (TypeError, ValueError) as exc:
        p.error(f'Invalid source config {args.source}: {exc}')
    if args.output.exists():
        p.error(f'refusing to overwrite existing: {args.output}')
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'x', encoding='utf-8') as handle:
            handle.write(body)
    except FileExistsError:
        p.error(f'refusing to overwrite existing: {args.output}')
    except OSError as exc:
        p.error(f'Cannot write {args.output}: {exc.strerror or exc}')
    print(f'Registered runtime: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
