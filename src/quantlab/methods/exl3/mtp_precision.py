"""Opt-in BF16 dense Qwen3.5 MTP projections, with explicit FP16 interfaces.

The ordinary dense loader converts BF16 matrices to FP16. Reloading the original
checkpoint tensors avoids that conversion. This changes only the draft model;
the shared EXL3 target head, norms, attention and cache keep their existing paths.
"""

from functools import partial


def bf16_projection_forward(inner, x, params, out_dtype=None):
    """Use BF16 operands and result, then restore the caller's interface dtype."""
    import torch

    shape = x.shape[:-1] + (inner.out_features,)
    operands = x.reshape(-1, inner.in_features).to(torch.bfloat16)
    result = torch.matmul(operands, inner.weight)
    if inner.bias is not None:
        result += inner.bias
    inner.bf16_calls += 1
    return result.reshape(shape).to(out_dtype or inner.out_dtype or torch.float16)


def preserve_mtp_bf16(draft):
    """Reload eight original BF16 matrices in one-layer dense Qwen3.5 MTP.

    Call after loading, before inference. Unsupported/packed drafts fail closed.
    Native fused projection handles are disabled because they assume FP16
    weights. Each original tensor is reloaded directly, never reconstructed from
    the already-converted FP16 matrix. No model-directory files are written.
    """
    import torch
    from exllamav3.architecture.qwen3_5_mtp import Qwen3_5MTPModel
    from exllamav3.modules import Linear, RMSNorm
    from exllamav3.modules.quant.fp16 import LinearFP16
    from .compat import prepare_loaded_module

    if type(draft) is not Qwen3_5MTPModel or draft.config.mtp_num_hidden_layers != 1:
        raise ValueError('BF16 MTP requires a one-layer dense Qwen3.5 MTP component')
    if getattr(draft, 'quantlab_bf16_mtp', None) is not None:
        raise ValueError('BF16 MTP is already installed')
    modules = list(draft)
    linears = [m for m in modules if isinstance(m, Linear)]
    norms = [m for m in modules if isinstance(m, RMSNorm)]
    if len(linears) != 8 or len(norms) != 7:
        raise ValueError('Unexpected dense MTP tensor inventory')
    metadata = draft.config.stc.list_tensors('mtp', only_serializable=True)
    total_bytes = 0
    for module in linears:
        inner = module.inner
        key = module.key + '.weight'
        meta = metadata.get(key, {})
        if (type(inner) is not LinearFP16 or inner.bias is not None
                or module.is_sliced or not module.transposed_load
                or module.weight_scale != 1.0 or module.used_alt_key
                or meta.get('dtype') != 'torch.bfloat16'
                or meta.get('shape') != [module.out_features, module.in_features]):
            raise ValueError('BF16 MTP requires an ordinary unsliced BF16 matrix: ' + key)
        total_bytes += meta['n_bytes']
    if total_bytes > 512 * 1024**2:
        raise ValueError('BF16 MTP projection payload exceeds 512 MiB')
    if any(m.weight.dtype != torch.bfloat16 for m in norms):
        raise ValueError('MTP norm weights must already preserve BF16')
    for module in draft.modules:
        prepare_loaded_module(module)

    records = []
    for module in linears:
        inner = module.inner
        weight = draft.config.stc.get_tensor(
            module.key + '.weight', module.device, allow_bf16=True,
            transpose=True, no_defer=True,
        )
        if weight.dtype != torch.bfloat16 or weight.shape != inner.weight.shape:
            raise ValueError('BF16 MTP loader returned unexpected tensor: ' + module.key)
        if not bool(torch.isfinite(weight).all()):
            raise ValueError('Nonfinite BF16 MTP weights: ' + module.key)
        inner.bc = None
        inner.weight = weight
        inner.bf16_calls = 0
        inner.forward = partial(bf16_projection_forward, inner)
        inner.quant_type = module.quant_type = 'bf16'
        records.append(dict(key=module.key, dtype=str(weight.dtype),
                            bytes=weight.numel()*weight.element_size()))
    report = dict(weight_dtype='bfloat16', projection_operands='bfloat16',
                  projection_result='bfloat16', target_head='shared EXL3, unchanged',
                  other_arithmetic='existing mixed FP16/BF16/FP32 paths',
                  projection_bytes=total_bytes, norm_count=len(norms), projections=records)
    draft.quantlab_bf16_mtp = report
    return report
