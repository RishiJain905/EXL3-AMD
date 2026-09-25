"""Opt-in sequential attention queries for numerically stable verification."""


def rowwise_attention(dispatch, counters, **kwargs):
    """Append and attend one causal row at a time, retaining packed projections.

    Target projections can still run as a batch. Only short cached attention
    queries use the same shape and causal prefix as ordinary one-token decode.
    """
    import torch

    q = kwargs['q']
    rows = q.shape[1]
    if rows <= 1 or rows > 9 or kwargs.get('cache') is None:
        return dispatch(**kwargs)
    if (not kwargs.get('causal', True) or kwargs.get('non_causal_spans')
            or kwargs['k'].shape[1] != rows or kwargs['v'].shape[1] != rows):
        raise ValueError('Rowwise verification requires aligned causal Q/K/V')
    output = []
    for row in range(rows):
        step = dict(kwargs)
        for key in ('q', 'k', 'v'):
            step[key] = kwargs[key][:, row:row+1].contiguous()
        step['cache_seqlens'] = kwargs['cache_seqlens'] + row
        output.append(dispatch(**step))
    counters['windows'] += 1
    counters['rows'] += rows
    return torch.cat(output, dim=1)


def install_rowwise_verifier_attention(model):
    """Apply only to this target's attention modules, before generator creation."""
    from functools import partial
    from exllamav3.modules.attn import Attention, attn_dispatch

    records = []
    for module in model:
        if not isinstance(module, Attention):
            continue
        if hasattr(module, '_quantlab_attn_dispatch'):
            raise ValueError('Verification attention override already installed')
        record = dict(key=module.key, windows=0, rows=0)
        module._quantlab_attn_dispatch = partial(rowwise_attention, attn_dispatch, record)
        records.append(record)
    if not records:
        raise ValueError('Target has no attention modules')
    return dict(mode='rowwise', max_rows=9, modules=records)
