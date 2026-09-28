"""Prefill rate at growing occupied context against a running server (prefix cache on).

Sends growing prefixes of one deterministic document (this checkout's vendored Python
sources) to /v1/completions with max_tokens=1. Each request after the first is a
prefix hit, so it computes only the new span, at the occupied context reached so far.
Reports computed tokens, prefill seconds and rate per span from the usage record.
Stdlib only; no model or server management.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def post(port, payload):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3600) as response:
        return json.load(response)


def document(chars):
    parts, total = [], 0
    for path in sorted((ROOT / 'vendor/rocm-exl3/exllamav3').rglob('*.py')):
        text = path.read_text(encoding='utf-8', errors='replace')
        parts.append(f'# file: {path.relative_to(ROOT).as_posix()}\n{text}\n')
        total += len(parts[-1])
        if total >= chars:
            break
    return ''.join(parts)[:chars]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--targets', type=int, nargs='+', default=[8192, 16384, 32768, 49152, 65536, 81920,
                                                              98304, 114688, 120000],
                   help='approximate prompt token counts, ascending')
    args = p.parse_args()
    probe = document(20000)
    ratio = len(probe) / post(args.port, dict(prompt=probe, max_tokens=1, temperature=0))['usage']['prompt_tokens']
    text = document(int(args.targets[-1] * ratio * 1.02))
    rows = []
    for target in args.targets:
        prompt = text[:int(target * ratio)]
        start = time.monotonic()
        usage = post(args.port, dict(prompt=prompt, max_tokens=1, temperature=0))['usage']
        timings = usage['timings']
        computed = timings.get('prefill_computed_tokens')
        seconds = timings.get('prefill_seconds')
        row = dict(prompt_tokens=usage['prompt_tokens'], computed_tokens=computed,
                   cached_tokens=timings.get('prefill_cached_tokens'), prefill_seconds=seconds,
                   rate=computed / seconds if computed and seconds else None,
                   wall_seconds=time.monotonic() - start)
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(chars_per_token=ratio, rows=rows), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
