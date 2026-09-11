"""Single-sequence diagnostics and opt-in decode fusion for the pinned runtime."""
from functools import wraps
import time

from .compat import prepare_loaded_module, smallm_supported


def guard_single_row(module, saved, counters, *, allowed_rows=(1,)):
    """Restore handles for explicitly supported rows; always restore fallback."""
    original = module.forward
    disabled = {key: getattr(module, key) for key in saved}

    @wraps(original)
    def forward(x, params, *args, **kwargs):
        if x.numel() // x.shape[-1] not in allowed_rows or params.get('reconstruct') or 'ovr' in params:
            counters['fallback_calls'] += 1
            return original(x, params, *args, **kwargs)
        counters['fused_calls'] += 1
        for key, value in saved.items():
            setattr(module, key, value)
        try:
            return original(x, params, *args, **kwargs)
        finally:
            for key, value in disabled.items():
                setattr(module, key, value)
    module.forward = forward


def prepare_decode_fusions(module, mode, *, native_smallm_graph=False, native_smallm_max_rows=3):
    """Preserve existing load-created handles before applying the safe fallback.

    Single sequence and synchronous Python dispatch only. This does not enable
    cooperative multirow GEMM or change any model/extension bytes.
    """
    from exllamav3.modules.gated_delta_net import GatedDeltaNet
    from exllamav3.modules.mlp import GatedMLP, MLP
    from exllamav3.modules.quant.exl3 import LinearEXL3
    captures = []
    for child in module:
        selected = isinstance(child, GatedDeltaNet) or (
            mode in ('gdn-mlp', 'gdn-mlp-mgemv') and isinstance(child, (GatedMLP, MLP)))
        if not selected or getattr(child, 'bc', None) is None:
            continue
        saved = {'bc': child.bc}
        if isinstance(child, GatedDeltaNet):
            saved['bc_split'] = child.bc_split
        if isinstance(child, GatedMLP):
            saved['multi_gu'] = child.multi_gu
        rows = (1,)
        if native_smallm_graph:
            if mode == 'gdn-mlp-mgemv':
                raise ValueError('Small-M MultiLinear graph dispatch is unsupported')
            packed = [desc.inner for desc in child
                      if isinstance(getattr(desc, 'inner', None), LinearEXL3)]
            if not packed or not all(smallm_supported(inner) for inner in packed):
                raise ValueError('Native graph contains unsupported packed projections')
            rows = tuple(range(1, native_smallm_max_rows + 1))
        captures.append((child, saved, rows))
    prepare_loaded_module(module)
    records = []
    for child, saved, rows in captures:
        counts = dict(key=child.key, kind=type(child).__name__, fused_calls=0, fallback_calls=0)
        counts['allowed_rows'] = list(rows)
        if 'multi_gu' in saved:
            counts['multi_gu_handles'] = sum(x is not None for x in saved['multi_gu'])
        guard_single_row(child, saved, counts, allowed_rows=rows)
        records.append(counts)
    return records


def profile_modules(model, *, state=None, scope='target', projections=False):
    """Record inclusive outer-module timings for selected decode iterations.

    CUDA event elapsed time includes submission gaps; it is not kernel self time.
    This instrumented run is diagnostic, separate from ordinary speed runs.
    """
    import torch
    if state is None:
        state = dict(enabled=False, samples=[])
    modules = list(model.modules)
    if projections:
        # One recurrent and one full-attention block; native GDN internals are
        # opaque, so retain its outer measurement and ordinary packed linears.
        from exllamav3.modules import TransformerBlock
        selected = {}
        for module in model.modules:
            if isinstance(module, TransformerBlock):
                selected.setdefault(type(module.attn).__name__, module)
        modules = []
        for block in selected.values():
            modules.extend(child for child in block if child is not block)
    for index, module in enumerate(modules):
        original = module.forward
        def make_forward(original, index, module):
            @wraps(original)
            def forward(*args, **kwargs):
                if not state['enabled']:
                    return original(*args, **kwargs)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                cpu_start = time.perf_counter()
                result = original(*args, **kwargs)
                cpu_ms = (time.perf_counter() - cpu_start) * 1000
                end.record()
                state['samples'].append((scope, index, type(module).__name__, module.key, start, end, cpu_ms))
                return result
            return forward
        module.forward = make_forward(original, index, module)

    def summarize():
        torch.cuda.synchronize()
        rows = {}
        for scope, index, kind, key, start, end, cpu_ms in state['samples']:
            row = rows.setdefault((scope, index), dict(scope=scope, index=index, kind=kind, key=key, calls=0,
                                              cuda_elapsed_ms=0.0, cpu_submission_ms=0.0))
            row['calls'] += 1
            row['cuda_elapsed_ms'] += start.elapsed_time(end)
            row['cpu_submission_ms'] += cpu_ms
        return list(rows.values())
    return state, summarize
