"""Bounded paired gate/up dispatch; down projections retain their normal path."""
from functools import wraps
import os


def eligible_pair(module, linear_type, packed_type, half_dtype):
    """Reject wrapper behavior the paired native entry cannot reproduce."""
    if (module.num_slices != 1 or module.tp_reduce or module.activation_fn != 'silu'
            or module.act_limit != 0 or module.interm_dtype != half_dtype):
        return False
    for projection in (module.gates[0], module.ups[0]):
        if (type(projection) is not linear_type or type(projection.inner) is not packed_type
                or 'forward' in vars(projection) or 'forward' in vars(projection.inner)
                or projection.lora_a_tensors or projection.pre_scale != 1
                or projection.post_scale != 1 or projection.softcap != 0
                or (projection.trim_padded_out and projection.out_features != projection.out_features_unpadded)):
            return False
        inner = projection.inner
        if (inner.bias is not None or inner.mcg or inner.default_out_dtype != half_dtype
                or inner.in_features <= 0 or inner.out_features <= 0
                or inner.in_features > 65536 or inner.out_features > 65536
                or inner.in_features % 128 or inner.out_features % 128
                or inner.K not in ((2, 3, 4, 5, 6) if inner.mul1 else (2, 3, 4))):
            return False
    gate, up = module.gates[0].inner, module.ups[0].inner
    return ((gate.K, gate.mul1, gate.in_features, gate.out_features, gate.trellis.device)
            == (up.K, up.mul1, up.in_features, up.out_features, up.trellis.device))


def install_mlp_pair(model, draft=None, *, enabled):
    """Called after binary capability verification and model preparation."""
    import torch
    from exllamav3.ext import exllamav3_ext as extension
    from exllamav3.modules import Linear
    from exllamav3.modules.mlp import GatedMLP
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.util.tensor import to2

    if type(enabled) is not bool:
        raise ValueError('Paired MLP setting must be a boolean')
    if enabled and not hasattr(extension, 'exl3_mlp_gate_up'):
        raise ValueError('Paired MLP requires its verified native entry')
    # Preserve explicit diagnostic kernels with different arithmetic/reduction.
    enabled = (enabled and os.environ.get('EXL3_SMALLM_WMMA', '0') == '0'
               and os.environ.get('EXL3_GEMV_LDS', '0') == '0')
    records = []
    for component in (model, draft):
        if component is None:
            continue
        for module in component:
            if type(module) is not GatedMLP:
                continue
            if not hasattr(module, '_quantlab_pair_stats') and (
                    'forward' in vars(module) or getattr(module, 'bc', None) is not None):
                continue
            module._quantlab_pair_enabled = enabled and eligible_pair(module, Linear, LinearEXL3, torch.half)
            if module._quantlab_pair_enabled:
                module._quantlab_pair_identity = (id(module.gates[0].inner), id(module.ups[0].inner))
            if hasattr(module, '_quantlab_pair_stats'):
                records.append(module._quantlab_pair_stats)
                continue
            if not module._quantlab_pair_enabled:
                continue
            stats = dict(key=module.key, paired_calls=0, fallback_calls=0)
            module._quantlab_pair_stats = stats
            records.append(stats)
            original = module.forward
            def wrap(module, original, stats):
                @wraps(original)
                def forward(x, params, out_dtype=None):
                    rows = x.numel() // x.shape[-1]
                    gate, up = module.gates[0].inner, module.ups[0].inner
                    if (not module._quantlab_pair_enabled or rows not in (1, 2, 3, 5)
                            or (id(gate), id(up)) != module._quantlab_pair_identity
                            or x.ndim != 3 or x.dtype != torch.half or not x.is_cuda
                            or not x.is_contiguous() or x.shape[-1] != gate.in_features
                            or x.data_ptr() % 8 != 0 or x.device != gate.trellis.device
                            or 'forward' in vars(gate) or 'forward' in vars(up)
                            or params.get('reconstruct') or any(k in params for k in ('ovr', 'capture', 'q_mlp_slice'))
                            or module.gates[0].lora_a_tensors or module.ups[0].lora_a_tensors):
                        stats['fallback_calls'] += 1
                        return original(x, params, out_dtype)
                    k, n = gate.in_features, gate.out_features
                    xh = torch.empty((2, rows, k), device=x.device, dtype=torch.half)
                    gu = torch.empty((2, rows, n), device=x.device, dtype=torch.half)
                    activated = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.half)
                    extension.exl3_mlp_gate_up(x.view(rows, k), gate.trellis, up.trellis,
                        gate.suh, up.suh, gate.svh, up.svh, xh, gu, activated.view(rows, n), gate.K, gate.mul1)
                    stats['paired_calls'] += 1
                    return to2(module.downs[0].forward(activated, params), out_dtype, module.out_dtype)
                return forward
            module.forward = wrap(module, original, stats)
    return records
