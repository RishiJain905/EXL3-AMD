"""Bounded compressed vocabulary-head view for the verified native entry."""
from functools import wraps
import math
import os
import weakref


def eligible_head(layer, half_dtype):
    # Output dtype is selected per call; keep the model's FP16/FP32 contract.
    k, n = layer.in_features, layer.out_features
    return (not layer.mcg and layer.bias is None
            and 0 < k <= 65536 and k % 128 == 0
            and 32768 < n <= 1048576 and n % 128 == 0
            and layer.K in ((2, 3, 4, 5, 6) if layer.mul1 else (2, 3, 4))
            and layer.trellis.is_cuda and layer.trellis.is_contiguous()
            and tuple(layer.trellis.shape) == (k // 16, n // 16, 16 * layer.K)
            and all(t.dtype == half_dtype and t.is_contiguous()
                    and t.device == layer.trellis.device and t.data_ptr() % 8 == 0
                    for t in (layer.suh, layer.svh))
            and layer.suh.numel() == k and layer.svh.numel() == n)


def install_packed_head(model, draft=None, *, enabled, memory_fraction):
    """Install only after binary verification; a shared MTP head gets one copy.

    Packing is lazy so the allocator check includes the already allocated KV
    cache. Captures can reuse a prepared view but cannot allocate the view.
    Disabling or unloading releases it. No persistent full-precision weights.
    """
    import torch
    from exllamav3.ext import exllamav3_ext as extension
    from exllamav3.modules.quant.exl3 import LinearEXL3

    if type(enabled) is not bool or not math.isfinite(memory_fraction) or not 0 < memory_fraction < 1:
        raise ValueError('Invalid packed head setting or allocator fraction')
    if enabled and not hasattr(extension, 'exl3_head_repacked'):
        raise ValueError('Packed head requires its verified native entry')
    enabled = (enabled and os.environ.get('EXL3_SMALLM_WMMA', '0') == '0'
               and os.environ.get('EXL3_GEMV_LDS', '0') == '0'
               and os.environ.get('EXL3_HEAD_TILED', '0') != '1'
               and os.environ.get('EXL3_GEMV_SPLITK', '1') != '0'
               and 'EXL3_SMALLM_HEAD_WARPS' not in os.environ
               and 'EXL3_GEMV_SPLITK_WARPS' not in os.environ)
    records, seen = [], set()
    for component in (model, draft):
        if component is None or component.logit_layer_idx is None:
            continue
        layer = getattr(component.modules[component.logit_layer_idx], 'inner', None)
        if type(layer) is not LinearEXL3 or id(layer) in seen:
            continue
        seen.add(id(layer))
        if hasattr(layer, '_quantlab_head_state'):
            state = layer._quantlab_head_state
            state['enabled'] = enabled
            state['memory_fraction'] = memory_fraction
            state['memory_blocked'] = False
            state['stats']['memory_limited'] = False
            if not enabled:
                state['packed'] = state['source'] = None
                state['stats']['packed_bytes'] = 0
            records.append(state['stats'])
            continue
        if (not enabled or 'forward' in vars(layer) or 'unload' in vars(layer)
                or not eligible_head(layer, torch.half)):
            continue
        props = torch.cuda.get_device_properties(layer.trellis.device)
        if getattr(props, 'gcnArchName', '').split(':', 1)[0] != 'gfx1101':
            continue
        stats = dict(key=layer.key, packed_calls=0, fallback_calls=0, packed_bytes=0,
                     peak_packed_bytes=0, memory_limited=False)
        state = dict(enabled=True, packed=None, source=None, stats=stats,
                     memory_fraction=memory_fraction, memory_blocked=False)
        layer._quantlab_head_state = state
        records.append(stats)

        def wrap(layer, state):
            reference = weakref.ref(layer)
            original, original_unload = type(layer).forward, type(layer).unload
            stats = state['stats']

            @wraps(original)
            def forward(x, params, out_dtype=None):
                layer = reference()
                k, n = layer.in_features, layer.out_features
                rows = x.numel() // x.shape[-1] if x.ndim >= 2 and x.shape[-1] else 0
                if (not state['enabled'] or state['memory_blocked'] or rows not in (1, 2, 3, 5)
                        or layer.mcg or layer.bias is not None
                        or (out_dtype or layer.default_out_dtype) not in (torch.half, torch.float32)
                        or x.dtype != torch.half or not x.is_contiguous()
                        or x.device != layer.trellis.device or x.shape[-1] != k
                        or x.data_ptr() % 8 or params.get('reconstruct')
                        or any(key in params for key in ('ovr', 'capture'))):
                    stats['fallback_calls'] += 1
                    return original(layer, x, params, out_dtype)
                if state['source'] is not None and state['source']() is not layer.trellis:
                    state['packed'] = state['source'] = None
                    stats['packed_bytes'] = 0
                if state['packed'] is None:
                    need = layer.trellis.numel() * layer.trellis.element_size()
                    # Leave room for later prefill/reconstruction workspaces.
                    reserve = 1024**3
                    if torch.cuda.is_current_stream_capturing():
                        stats['fallback_calls'] += 1
                        return original(layer, x, params, out_dtype)
                    free, total = torch.cuda.mem_get_info(x.device)
                    if (not eligible_head(layer, torch.half) or need > 1024**3
                            or need + reserve > free
                            or torch.cuda.memory_allocated(x.device) + need + reserve > total * state['memory_fraction']):
                        # Retry only on explicit reinstall, not on every token.
                        state['memory_blocked'] = stats['memory_limited'] = True
                        stats['fallback_calls'] += 1
                        return original(layer, x, params, out_dtype)
                    try:
                        state['packed'] = layer.trellis.view(k//16, n//128, 8, 16*layer.K).permute(1, 0, 2, 3).contiguous()
                    except torch.OutOfMemoryError:
                        state['memory_blocked'] = stats['memory_limited'] = True
                        stats['fallback_calls'] += 1
                        return original(layer, x, params, out_dtype)
                    state['source'] = weakref.ref(layer.trellis)
                    stats['packed_bytes'] = need
                    stats['peak_packed_bytes'] = max(stats['peak_packed_bytes'], need)
                output = torch.empty((*x.shape[:-1], n), device=x.device,
                                     dtype=out_dtype or layer.default_out_dtype)
                scratch = torch.empty((rows, k), device=x.device, dtype=torch.half)
                extension.exl3_head_repacked(x.view(rows, k), state['packed'], layer.suh,
                    scratch, layer.svh, output.view(rows, n), layer.K, 2 if layer.mul1 else 0)
                stats['packed_calls'] += 1
                return output

            @wraps(original_unload)
            def unload():
                state['packed'] = state['source'] = None
                stats['packed_bytes'] = 0
                return original_unload(reference())

            return forward, unload

        layer.forward, layer.unload = wrap(layer, state)
    return records
