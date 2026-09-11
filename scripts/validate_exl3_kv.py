"""Bounded GPU checks for the inherited packed KV cache and attention kernels.

Run through the existing external monitor with the configured pinned extension.
Uses synthetic tensors only; does not load weights or claim model fidelity.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tomllib
import traceback
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--extension-dir', type=Path, required=True)
    p.add_argument('--expected-extension-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    cfg = tomllib.loads(args.config.read_text(encoding='utf-8'))
    if not args.execute or not all(cfg['execution'].get(k) is True for k in ('allow_backend_probes', 'allow_local_inference')):
        p.error('Explicit execution and existing local permissions required')
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', cases=[])

    def save():
        (args.output/'result.json').write_text(json.dumps(status, indent=2, allow_nan=False), encoding='utf-8')

    try:
        source = Path(__file__).read_bytes()
        (args.output/'harness.py').write_bytes(source)
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
        from exllamav3.modules.attention_fn.dispatch import attn_dispatch

        torch.set_num_threads(2)
        device = torch.device('cuda:0')
        props = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(4 * 2**30 / props.total_memory)
        torch.manual_seed(42961)
        status.update(device=props.name, torch=torch.__version__, hip=torch.version.hip,
                      extension_sha256=args.expected_extension_sha256,
                      harness_sha256=hashlib.sha256(source).hexdigest())
        had = torch.ones((1, 1), dtype=torch.float32)
        for _ in range(5):
            had = torch.cat((torch.cat((had, had), dim=1), torch.cat((had, -had), dim=1)), dim=0)
        had /= 32**0.5

        def unpack(words, scales, bits, table, length):
            # Independent little-endian unpack and inverse H32 on CPU.
            words = words.cpu().to(torch.int64).reshape(4, 256, 4, 8, bits)
            codes = torch.stack([(words[..., j*bits//32] >> ((j*bits) % 32)) & ((1 << bits)-1)
                                 for j in range(32)], dim=-1)
            rotated = ((2*codes.float()+1)/(1 << bits)-1) * scales.cpu().reshape(4, 256, 4, 8, 1)
            decoded = (rotated @ had).reshape(4, 256, 4, 256)
            return torch.cat([decoded[i] for i in table], dim=0)[:length]

        def compare(a, b):
            a, b = a.detach().float().cpu(), b.detach().float().cpu()
            assert a.shape == b.shape
            error, norm = torch.linalg.vector_norm(a-b).item(), torch.linalg.vector_norm(b).item()
            return dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                        relative_l2=error/norm if norm else (0.0 if error == 0 else None),
                        max_abs=(a-b).abs().max().item())

        def attention(q, k, v, offset):
            q = q.float().transpose(1, 2)
            k = k.to(device).float().repeat_interleave(4, dim=1).transpose(0, 1).unsqueeze(0)
            v = v.to(device).float().repeat_interleave(4, dim=1).transpose(0, 1).unsqueeze(0)
            scores = (q @ k.transpose(-1, -2)) / 16.0
            mask = torch.arange(k.shape[-2], device=device)[None, :] > (offset + torch.arange(q.shape[-2], device=device))[:, None]
            return (scores.masked_fill(mask, -torch.inf).softmax(dim=-1) @ v).transpose(1, 2)

        physical_pages = [2, 0, 3, 1]
        table = torch.tensor([physical_pages], dtype=torch.int32, device=device)
        for kb, vb in ((8, 8), (4, 4), (8, 4), (4, 8)):
            for family in ('zeros', 'gaussian', 'outliers'):
                layer = CacheLayer_quant(None, SimpleNamespace(num_kv_heads=4, head_dim=256), 1, 1024, kb, vb)
                layer.alloc(device)
                original_k = torch.zeros((1024, 4, 256), dtype=torch.float16, device=device)
                original_v = torch.zeros_like(original_k)
                for offset, length in ((0, 255), (255, 1), (256, 5), (261, 256), (250, 7)):
                    shape = (1, length, 4, 256)
                    k, v = torch.randn(shape, device=device).half(), torch.randn(shape, device=device).half()
                    q = torch.randn((1, length, 16, 256), device=device).half()
                    if family == 'zeros':
                        k.zero_(); v.zero_(); q.zero_()
                    elif family == 'outliers':
                        k[..., 0] *= 12; v[..., 0] *= 12
                        q *= 0.125  # Keep attention scores moderate while testing KV outliers.
                    original_k[offset:offset+length] = k[0]
                    original_v[offset:offset+length] = v[0]
                    seqlens = torch.tensor([offset], dtype=torch.int32, device=device)
                    hint = {}
                    actual = attn_dispatch(q, k, v, cache=layer, block_table=table, cache_seqlens=seqlens, dispatch_cache=hint)
                    torch.cuda.synchronize()
                    active = offset + length
                    k_ref = unpack(layer.qk, layer.sk, kb, physical_pages, active)
                    v_ref = unpack(layer.qv, layer.sv, vb, physical_pages, active)
                    decoded = layer.get_kv(torch.tensor([active], dtype=torch.int32, device=device), table)
                    k_native, v_native = [torch.cat([t[i] for i in physical_pages], dim=0)[:active] for t in decoded]
                    row = dict(k_bits=kb, v_bits=vb, family=family, offset=offset, length=length,
                               attention_path=hint['fn_qc'].__name__,
                               storage_bytes=sum(t.numel()*t.element_size() for t in layer.get_tensors()),
                               declared_storage_bytes=int(layer.storage_size()),
                               key_oracle=compare(k_native, k_ref), value_oracle=compare(v_native, v_ref),
                               attention_oracle=compare(actual, attention(q, k_ref, v_ref, offset)),
                               attention_f16=compare(actual, attention(q, original_k[:active], original_v[:active], offset)))
                    status['cases'].append(row)
                    save()
                    assert row['storage_bytes'] == row['declared_storage_bytes']
                    for key in ('key_oracle', 'value_oracle', 'attention_oracle', 'attention_f16'):
                        result = row[key]
                        gate = (0.03 if min(kb, vb) == 8 else 0.25) if key == 'attention_f16' else 0.01
                        assert result['finite'] and result['relative_l2'] is not None and result['relative_l2'] <= gate, (key, row)
                    print(f'q{kb}/q{vb} {family} offset={offset} length={length}: pass', flush=True)
                layer.free()
        status['status'] = 'completed'
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
