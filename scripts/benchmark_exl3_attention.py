"""Bounded decode-attention config benchmark over paged EXL3 KV caches.

Run through the existing external monitor with the configured pinned extension.
Uses synthetic tensors only, fixed 24 Q/4 KV heads and D256. Does not load weights or claim model fidelity.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import tomllib
import traceback
from types import SimpleNamespace

MAX_CASES = 8
MAX_CONFIGS_PER_CASE = 12
N_Q_HEADS = 24
N_KV_HEADS = 4
HEAD_DIM = 256
PAGE_SIZE = 256
MAX_CAPACITY = 131072
DEFAULT_SEED = 4197
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 2
REL_L2_GATE = 0.01
WARMUPS = 2
TIMING_BATCHES = 5
TIMING_REPEATS = 10
MEMORY_BUDGET_BYTES = 6 * 2**30


def _is_power_of_2(x):
    return isinstance(x, int) and x > 0 and (x & (x - 1)) == 0


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def _check_bool_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an int')


def validate_plan(plan):
    """Validate a decoded plan object into normalized case dicts."""
    if not isinstance(plan, list) or not plan:
        raise ValueError('plan must be a non-empty JSON list of cases')
    if len(plan) > MAX_CASES:
        raise ValueError(f'plan must list at most {MAX_CASES} cases')
    cases = []
    seen_cases = set()
    for i, case in enumerate(plan):
        where = f'case[{i}]'
        if not isinstance(case, dict):
            raise ValueError(f'{where} must be an object')
        name = case.get('name')
        if not isinstance(name, str) or not name:
            raise ValueError(f'{where}.name must be a non-empty string')
        if name in seen_cases:
            raise ValueError(f'duplicate case name {name!r}')
        seen_cases.add(name)
        cache_type = case.get('cache_type')
        if cache_type not in ('f16', 'q8', 'q4'):
            raise ValueError(f'{where}.cache_type must be one of f16/q8/q4')
        length = case.get('length')
        _check_bool_int(length, f'{where}.length')
        q_len = case.get('q_len')
        _check_bool_int(q_len, f'{where}.q_len')
        if length < 1 or q_len < 1 or q_len > 16:
            raise ValueError(f'{where} needs length >= 1 and 1 <= q_len <= 16')
        capacity = ((length + q_len + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
        if capacity > MAX_CAPACITY:
            raise ValueError(f'{where} capacity {capacity} exceeds {MAX_CAPACITY}')
        seed = case.get('seed', DEFAULT_SEED)
        _check_bool_int(seed, f'{where}.seed')
        if seed < 0:
            raise ValueError(f'{where}.seed must be non-negative')
        table_pad_pages = case.get('table_pad_pages', 1)
        _check_bool_int(table_pad_pages, f'{where}.table_pad_pages')
        if table_pad_pages not in (1, 16):
            raise ValueError(f'{where}.table_pad_pages must be one of 1/16')
        raw_configs = case.get('configs')
        if not isinstance(raw_configs, list) or not raw_configs:
            raise ValueError(f'{where}.configs must be a non-empty list')
        if len(raw_configs) > MAX_CONFIGS_PER_CASE:
            raise ValueError(f'{where}.configs must list at most {MAX_CONFIGS_PER_CASE} configs')
        configs = []
        seen_configs = set()
        for j, raw in enumerate(raw_configs):
            cwhere = f'{where}.configs[{j}]'
            if not isinstance(raw, dict):
                raise ValueError(f'{cwhere} must be an object')
            cname = raw.get('name')
            if not isinstance(cname, str) or not cname:
                raise ValueError(f'{cwhere}.name must be a non-empty string')
            if cname in seen_configs:
                raise ValueError(f'duplicate config name {cname!r} in {where}')
            if cname == 'control':
                raise ValueError(f'{cwhere}.name must not use the reserved control name')
            seen_configs.add(cname)
            block_n = raw.get('block_n')
            if block_n is not None:
                _check_bool_int(block_n, f'{cwhere}.block_n')
                if not _is_power_of_2(block_n) or block_n > 4096:
                    raise ValueError(f'{cwhere}.block_n must be a power of two <= 4096')
            num_splits = raw.get('num_splits')
            if num_splits is not None:
                _check_bool_int(num_splits, f'{cwhere}.num_splits')
                if num_splits < 1 or num_splits > 128:
                    raise ValueError(f'{cwhere}.num_splits must be in 1..128')
            num_warps = raw.get('num_warps', DEFAULT_NUM_WARPS)
            _check_bool_int(num_warps, f'{cwhere}.num_warps')
            if num_warps not in (1, 2, 4, 8):
                raise ValueError(f'{cwhere}.num_warps must be one of 1/2/4/8')
            num_stages = raw.get('num_stages', DEFAULT_NUM_STAGES)
            _check_bool_int(num_stages, f'{cwhere}.num_stages')
            if num_stages < 1 or num_stages > 8:
                raise ValueError(f'{cwhere}.num_stages must be in 1..8')
            head_block = raw.get('head_block')
            if head_block is not None:
                _check_bool_int(head_block, f'{cwhere}.head_block')
                if not _is_power_of_2(head_block) or head_block > 16:
                    raise ValueError(f'{cwhere}.head_block must be a power of two in 1..16')
                product = _next_pow2(q_len) * head_block
                if product < 16 or product > 64:
                    raise ValueError(f'{cwhere}.head_block must satisfy nextpow2(q_len)*head_block in 16..64')
            parallel_combine = raw.get('parallel_combine', False)
            if not isinstance(parallel_combine, bool):
                raise ValueError(f'{cwhere}.parallel_combine must be a bool')
            configs.append(dict(name=cname, block_n=block_n, num_splits=num_splits,
                                num_warps=num_warps, num_stages=num_stages,
                                head_block=head_block, parallel_combine=parallel_combine))
        cases.append(dict(name=name, cache_type=cache_type, length=length, q_len=q_len,
                          seed=seed, capacity=capacity, table_pad_pages=table_pad_pages, configs=configs))
    return cases


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--extension-dir', type=Path, required=True)
    p.add_argument('--expected-extension-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        p.error('Explicit execution and existing local permissions required')
    cfg = tomllib.loads(args.config.read_text(encoding='utf-8'))
    if not all(cfg['execution'].get(k) is True for k in ('allow_backend_probes', 'allow_local_inference')):
        p.error('Explicit execution and existing local permissions required')
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', cases=[])

    def save():
        (args.output / 'result.json').write_text(json.dumps(status, indent=2, allow_nan=False), encoding='utf-8')

    try:
        source = Path(__file__).read_bytes()
        (args.output / 'harness.py').write_bytes(source)
        cases = validate_plan(json.loads(args.plan.read_text(encoding='utf-8')))
        status['plan'] = cases
        save()
        os.environ.update(EXL3_BC_ATTN='0', EXL3_QC_STAGING='0', EXL3_GEMV='2',
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        import torch
        import torch.utils.cpp_extension as cpp

        def no_build(*a, **kw):
            raise RuntimeError('JIT native builds are forbidden')

        cpp.load = cpp.load_inline = no_build
        binary, = args.extension_dir.glob('exllamav3_ext*.so')
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == args.expected_extension_sha256
        spec = importlib.util.spec_from_file_location('exllamav3_ext', binary)
        extension = importlib.util.module_from_spec(spec)
        sys.modules['exllamav3_ext'] = extension
        spec.loader.exec_module(extension)
        sys.path.insert(0, str(args.source_dir))
        from exllamav3.cache import CacheLayer_quant
        from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode

        torch.set_num_threads(2)
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(min(1.0, MEMORY_BUDGET_BYTES / props.total_memory))
        torch.manual_seed(42961)
        status.update(device=props.name, query_heads=N_Q_HEADS, kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, gpu_arch=getattr(props, "gcnArchName", None), multiprocessor_count=props.multi_processor_count,
                      total_memory_bytes=props.total_memory, torch=torch.__version__, hip=torch.version.hip,
                      extension_sha256=args.expected_extension_sha256,
                      harness_sha256=hashlib.sha256(source).hexdigest())
        save()

        def compare(a, b):
            a, b = a.detach().float().cpu(), b.detach().float().cpu()
            assert a.shape == b.shape
            error, norm = torch.linalg.vector_norm(a - b).item(), torch.linalg.vector_norm(b).item()
            return dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                        relative_l2=error / norm if norm else (0.0 if error == 0 else None),
                        max_abs=(a - b).abs().max().item())

        def oracle_attention(q, k, v, length):
            # Float32 grouped decode attention without repeating K/V 4x.
            # Query row i attends logical KV rows 0..length+i (causal, no window/sinks).
            q_len, total = q.shape[1], k.shape[0]
            qg = q.float().view(1, q_len, N_KV_HEADS, N_Q_HEADS // N_KV_HEADS, HEAD_DIM)
            k, v = k.float(), v.float()
            scores = torch.einsum('bqghd,tgd->bqght', qg, k) / (HEAD_DIM ** 0.5)
            kv_pos = torch.arange(total, device=device)
            q_pos = length + torch.arange(q_len, device=device)
            mask = kv_pos[None, :] > q_pos[:, None]
            probs = scores.masked_fill(mask.view(1, q_len, 1, 1, total), -torch.inf).softmax(dim=-1)
            return torch.einsum('bqght,tgd->bqghd', probs, v).reshape(1, q_len, N_Q_HEADS, HEAD_DIM)

        def gpu_usable():
            try:
                torch.cuda.synchronize(device)
                torch.zeros(4, device=device).sum().item()
                torch.cuda.synchronize(device)
                return True
            except BaseException:
                return False

        def is_oom(exc):
            if type(exc).__name__ == 'OutOfMemoryError':
                return True
            try:
                return isinstance(exc, torch.cuda.OutOfMemoryError)
            except BaseException:
                return False

        aborted = None
        for case in cases:
            name, cache_type = case['name'], case['cache_type']
            length, q_len, seed = case['length'], case['q_len'], case['seed']
            total, capacity = length + q_len, case['capacity']
            table_pad_pages = case['table_pad_pages']
            num_pages = capacity // PAGE_SIZE
            table_columns = ((num_pages + table_pad_pages - 1) // table_pad_pages) * table_pad_pages
            row = dict(name=name, cache_type=cache_type, length=length, q_len=q_len,
                       seed=seed, capacity=capacity, table_pad_pages=table_pad_pages,
                       table_columns=table_columns, status='running', configs=[])
            status['cases'].append(row)
            k_all = v_all = queries = table = seqlens = None
            k_cache = v_cache = k_ref = v_ref = expected = out = actual = None
            k_pad = v_pad = blocks_k = blocks_v = order = None
            qk = sk = qv = sv = k_bits = v_bits = None
            call_kw = None
            save()
            torch.cuda.reset_peak_memory_stats(device)
            layer = None
            try:
                gen = torch.Generator().manual_seed(seed)
                k_all = torch.randn((total, N_KV_HEADS, HEAD_DIM), generator=gen,
                                    dtype=torch.float32, device='cpu').half().to(device)
                v_all = torch.randn((total, N_KV_HEADS, HEAD_DIM), generator=gen,
                                    dtype=torch.float32, device='cpu').half().to(device)
                queries = (torch.randn((1, q_len, N_Q_HEADS, HEAD_DIM), generator=gen,
                                       dtype=torch.float32, device='cpu') * 0.125).half().to(device)
                physical_pages = torch.randperm(num_pages, generator=gen).tolist()
                table = torch.tensor([physical_pages + [0] * (table_columns - num_pages)],
                                     dtype=torch.int32, device=device)
                seqlens = torch.tensor([length], dtype=torch.int32, device=device)
                row['data_sha256'] = hashlib.sha256(json.dumps(
                    dict(seed=seed, cache_type=cache_type, length=length, q_len=q_len,
                         n_q_heads=N_Q_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
                         num_pages=num_pages, physical_pages=physical_pages,
                         table_pad_pages=table_pad_pages, table_columns=table_columns),
                    sort_keys=True).encode()).hexdigest()
                row['inputs_finite'] = bool(torch.isfinite(k_all).all() and torch.isfinite(v_all).all()
                                            and torch.isfinite(queries).all())
                # Every live row is written before the sweep; no unwritten rows are read.
                call_kw = dict(q=queries, k=None, v=None, block_table=table, cache_seqlens=seqlens,
                               causal=True, pre_appended_len=q_len)
                if cache_type == 'f16':
                    k_cache = torch.empty((num_pages, PAGE_SIZE, N_KV_HEADS, HEAD_DIM),
                                          dtype=torch.float16, device=device)
                    v_cache = torch.empty_like(k_cache)
                    # Padded logical capacity; only live total rows are copied,
                    # so a partial final page scatters full pages correctly.
                    k_pad = torch.zeros((capacity, N_KV_HEADS, HEAD_DIM),
                                        dtype=torch.float16, device=device)
                    v_pad = torch.zeros_like(k_pad)
                    k_pad[:total] = k_all
                    v_pad[:total] = v_all
                    blocks_k = k_pad.view(num_pages, PAGE_SIZE, N_KV_HEADS, HEAD_DIM)
                    blocks_v = v_pad.view(num_pages, PAGE_SIZE, N_KV_HEADS, HEAD_DIM)
                    order = torch.tensor(physical_pages, dtype=torch.int64, device=device)
                    k_cache[order] = blocks_k
                    v_cache[order] = blocks_v
                    call_kw.update(k_cache=k_cache, v_cache=v_cache)
                    k_ref, v_ref = k_all, v_all
                else:
                    bits = 8 if cache_type == 'q8' else 4
                    layer = CacheLayer_quant(None, SimpleNamespace(num_kv_heads=N_KV_HEADS,
                                                                   head_dim=HEAD_DIM),
                                             1, capacity, bits, bits)
                    layer.alloc(device)
                    layer.update_kv_direct(torch.zeros((1,), dtype=torch.int32, device=device),
                                           table, k_all.unsqueeze(0), v_all.unsqueeze(0), total)
                    qk, sk, qv, sv, k_bits, v_bits = layer.get_qkv()
                    call_kw.update(k_cache=qk, v_cache=qv, qc=(sk, sv, k_bits, v_bits),
                                   n_kv_heads_override=N_KV_HEADS)
                    native_k, native_v = layer.get_kv(torch.tensor([total], dtype=torch.int32, device=device),
                                                      table)
                    k_ref = torch.cat([native_k[i] for i in physical_pages], dim=0)[:total]
                    v_ref = torch.cat([native_v[i] for i in physical_pages], dim=0)[:total]
                    del native_k, native_v
                torch.cuda.synchronize(device)
                expected = oracle_attention(queries, k_ref, v_ref, length)
                torch.cuda.synchronize(device)
                row['oracle_finite'] = bool(torch.isfinite(expected).all().item())

                out = torch.empty_like(queries)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                sweep = [dict(name='control', block_n=None, num_splits=None,
                              num_warps=DEFAULT_NUM_WARPS, num_stages=DEFAULT_NUM_STAGES,
                              head_block=None, parallel_combine=False)]
                sweep.extend(case['configs'])
                for cfg_row in sweep:
                    crow = dict(name=cfg_row['name'], block_n=cfg_row['block_n'],
                                num_splits=cfg_row['num_splits'], num_warps=cfg_row['num_warps'],
                                num_stages=cfg_row['num_stages'], head_block=cfg_row['head_block'],
                                parallel_combine=cfg_row['parallel_combine'], status='running')
                    row['configs'].append(crow)
                    save()
                    try:
                        torch.cuda.synchronize(device)
                        start.record()
                        actual = paged_attn_triton_decode(
                            out=out, block_n=cfg_row['block_n'], num_splits=cfg_row['num_splits'],
                            num_warps=cfg_row['num_warps'], num_stages=cfg_row['num_stages'],
                            head_block=cfg_row['head_block'], parallel_combine=cfg_row['parallel_combine'],
                            **call_kw)
                        end.record()
                        torch.cuda.synchronize(device)
                        crow['cold_ms'] = start.elapsed_time(end)
                        for _ in range(WARMUPS):
                            paged_attn_triton_decode(
                                out=out, block_n=cfg_row['block_n'], num_splits=cfg_row['num_splits'],
                                num_warps=cfg_row['num_warps'], num_stages=cfg_row['num_stages'],
                                head_block=cfg_row['head_block'], parallel_combine=cfg_row['parallel_combine'],
                                **call_kw)
                        torch.cuda.synchronize(device)
                        samples = []
                        wall_start = time.perf_counter()
                        for _ in range(TIMING_BATCHES):
                            start.record()
                            for _ in range(TIMING_REPEATS):
                                paged_attn_triton_decode(
                                    out=out, block_n=cfg_row['block_n'], num_splits=cfg_row['num_splits'],
                                    num_warps=cfg_row['num_warps'], num_stages=cfg_row['num_stages'],
                                    head_block=cfg_row['head_block'], parallel_combine=cfg_row['parallel_combine'],
                                    **call_kw)
                            end.record()
                            torch.cuda.synchronize(device)
                            samples.append(start.elapsed_time(end) / TIMING_REPEATS)
                        crow['wall_s'] = time.perf_counter() - wall_start
                        crow['samples_ms'] = samples
                        crow['median_ms'] = median(samples)
                        result = compare(actual, expected)
                        crow.update(result)
                        ok = result['finite'] and result['relative_l2'] is not None \
                            and result['relative_l2'] <= REL_L2_GATE
                        crow['status'] = 'pass' if ok else 'mismatch'
                    except BaseException as exc:
                        crow.update(status='error', error=f'{type(exc).__name__}: {exc}')
                        if is_oom(exc):
                            # Never probe or continue GPU work after OOM; abort the batch.
                            crow['gpu_usable'] = False
                            aborted = crow['error']
                            save()
                            break
                        if not gpu_usable():
                            crow['gpu_usable'] = False
                            aborted = 'GPU unusable after: ' + crow['error']
                            save()
                            break
                        crow['gpu_usable'] = True
                    save()
                    print(f'{name} {crow["name"]}: {crow["status"]}', flush=True)
                if aborted is not None:
                    row.update(status='failed', error=aborted)
                else:
                    row['status'] = 'completed'
            except BaseException as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                if is_oom(exc):
                    aborted = row['error']
                else:
                    try:
                        usable = gpu_usable()
                    except BaseException:
                        usable = False
                    if not usable:
                        aborted = 'GPU unusable after: ' + row['error']
            finally:
                try:
                    row['peak_bytes'] = torch.cuda.max_memory_allocated(device)
                except BaseException:
                    row['peak_bytes'] = None
                if layer is not None:
                    try:
                        layer.free()
                    except BaseException:
                        pass
                del k_all, v_all, queries, table, seqlens
                del k_cache, v_cache, k_ref, v_ref, expected, out, actual
                del k_pad, v_pad, blocks_k, blocks_v, order
                del qk, sk, qv, sv, k_bits, v_bits
                del call_kw, layer
                try:
                    torch.cuda.empty_cache()
                except BaseException:
                    pass
                save()
            if aborted is not None:
                break
        if aborted is not None:
            status.update(status='failed', error=aborted)
            save()
            raise RuntimeError('Aborted attention batch: ' + aborted)
        status['peak_bytes'] = max((row.get('peak_bytes') or 0 for row in status['cases']), default=0)
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
