"""Prefill attention tile/split sweep over a paged FP16 cache (the staged Q6/Q8 path).

With quantized KV, chunked prefill dequantizes the referenced window into an FP16
scratch and runs ``paged_attn_triton_prefill`` over it; this harness times that
kernel for synthetic tensors at a given occupied context. Each candidate is checked
against the default configuration (relative L2) before its time is reported. It
loads no weights and says nothing about model fidelity.

Run inside the WSL runtime interpreter while holding the GPU lease, e.g.
  python scripts/benchmark_exl3_prefill_attention.py --source-dir vendor/rocm-exl3 \
      --extension-dir <lib> --expected-extension-sha256 <sha> --output artifacts/<new> \
      --contexts 8192 32768 131072 --q-lens 1024 2048
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import sys

PAGE_SIZE = 256
REL_L2_GATE = 1e-3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--extension-dir', type=Path, required=True)
    p.add_argument('--expected-extension-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contexts', type=int, nargs='+', default=[8192, 32768, 131072])
    p.add_argument('--q-lens', type=int, nargs='+', default=[1024, 2048])
    p.add_argument('--query-heads', type=int, default=16)
    p.add_argument('--kv-heads', type=int, default=4)
    p.add_argument('--head-dim', type=int, default=256)
    p.add_argument('--block-m', type=int, nargs='+', default=[32, 64, 128])
    p.add_argument('--block-n', type=int, nargs='+', default=[16, 32, 64])
    p.add_argument('--warps', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--stages', type=int, nargs='+', default=[1, 2])
    p.add_argument('--splits', type=int, nargs='+', default=[0], help='0 keeps the automatic split rule')
    p.add_argument('--rows-per-warp', type=int, nargs='+', default=[8, 16, 32])
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--fp32-reference', action='store_true',
                   help='also report relative L2 of each candidate against an FP32 attention reference')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    binary, = args.extension_dir.glob('exllamav3_ext*.so')
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == args.expected_extension_sha256
    spec = importlib.util.spec_from_file_location('exllamav3_ext', binary)
    extension = importlib.util.module_from_spec(spec)
    sys.modules['exllamav3_ext'] = extension
    spec.loader.exec_module(extension)
    sys.path.insert(0, str(args.source_dir))
    from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill

    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(device)
    result = dict(device=props.name, gpu_arch=getattr(props, 'gcnArchName', None), torch=torch.__version__,
                  extension_sha256=args.expected_extension_sha256,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  geometry=dict(query_heads=args.query_heads, kv_heads=args.kv_heads, head_dim=args.head_dim),
                  cases=[])

    def save():
        (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    def timed(fn):
        fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        return statistics.median(times)

    configs = [c for c in itertools.product(args.block_m, args.block_n, args.warps, args.stages, args.splits)
               if c[0] // c[2] in args.rows_per_warp]
    gen = torch.Generator(device=device).manual_seed(7)
    for context in args.contexts:
        for q_len in args.q_lens:
            total = context + q_len
            pages = -(-total // PAGE_SIZE)
            shape = (pages, PAGE_SIZE, args.kv_heads, args.head_dim)
            k_cache = torch.randn(shape, generator=gen, device=device, dtype=torch.half)
            v_cache = torch.randn(shape, generator=gen, device=device, dtype=torch.half)
            q = torch.randn((1, q_len, args.query_heads, args.head_dim), generator=gen, device=device,
                            dtype=torch.half)
            table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
            seqlens = torch.tensor([context], dtype=torch.int32, device=device)

            def run(block_m=None, block_n=None, warps=None, stages=None, splits=None):
                return paged_attn_triton_prefill(q, None, None, k_cache, v_cache, table, seqlens, causal=True,
                                                 pre_appended_len=q_len, block_m=block_m, block_n=block_n,
                                                 num_warps=warps, num_stages=stages, num_splits=splits)

            reference = run().float()
            exact = None
            if args.fp32_reference:
                # Causal GQA attention in FP32: query row i sees keys 0..context+i
                group = args.query_heads // args.kv_heads
                keys = k_cache.view(-1, args.kv_heads, args.head_dim)[:total].float()
                values = v_cache.view(-1, args.kv_heads, args.head_dim)[:total].float()
                exact = torch.empty((q_len, args.query_heads, args.head_dim), device=device)
                for h in range(args.query_heads):
                    kh, vh = keys[:, h // group], values[:, h // group]
                    scores = (q[0, :, h].float() @ kh.T) / args.head_dim ** 0.5
                    causal = torch.arange(total, device=device)[None, :] > (context + torch.arange(q_len, device=device))[:, None]
                    exact[:, h] = torch.softmax(scores.masked_fill(causal, float('-inf')), dim=-1) @ vh
                exact = exact.view(1, q_len, args.query_heads, args.head_dim)
            default_ms = timed(run)
            case = dict(context=context, q_len=q_len, default_ms=default_ms, candidates=[])
            if exact is not None:
                case['default_fp32_relative_l2'] = float((reference - exact).norm() / exact.norm())
            flops = 4 * q_len * (context + q_len / 2) * args.query_heads * args.head_dim
            case['default_tflops'] = flops / default_ms / 1e9
            print(f'context={context} q_len={q_len} default={default_ms:.3f}ms '
                  f'{case["default_tflops"]:.2f}TFLOP/s', flush=True)
            for block_m, block_n, warps, stages, splits in configs:
                kwargs = dict(block_m=block_m, block_n=block_n, warps=warps, stages=stages, splits=splits or None)
                row = dict(block_m=block_m, block_n=block_n, warps=warps, stages=stages, splits=splits)
                try:
                    out = run(**kwargs).float()
                    rel = float((out - reference).norm() / reference.norm())
                    row.update(relative_l2=rel, finite=bool(torch.isfinite(out).all()))
                    if exact is not None:
                        row['fp32_relative_l2'] = float((out - exact).norm() / exact.norm())
                    if rel <= REL_L2_GATE and row['finite']:
                        row['ms'] = timed(lambda: run(**kwargs))
                        row['speedup'] = default_ms / row['ms']
                except Exception as exc:  # out of LDS/registers for this tile
                    row['error'] = f'{type(exc).__name__}: {str(exc)[:160]}'
                case['candidates'].append(row)
            ranked = sorted((r for r in case['candidates'] if 'ms' in r), key=lambda r: r['ms'])
            for r in ranked[:5]:
                print(f"  bm={r['block_m']} bn={r['block_n']} w={r['warps']} s={r['stages']} "
                      f"splits={r['splits']} {r['ms']:.3f}ms x{r['speedup']:.3f}", flush=True)
            result['cases'].append(case)
            save()
            del k_cache, v_cache, q, reference, exact
            torch.cuda.empty_cache()
    save()


if __name__ == '__main__':
    main()
