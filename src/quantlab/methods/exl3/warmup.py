"""Startup warm-up: first-use Triton compilation before serving reports ready.

Triton compiles one binary per specialization: the constexpr arguments,
num_warps/num_stages, and, for plain integer and pointer arguments, whether
the value is divisible by 16 (plus a <2 GiB flag for tensors on AMD).
Serving reaches new paged-attention specializations as the block-table width,
query rows and schedule change, so a cold server compiled kernels for 3-8 s
each in the middle of requests.

Two parts cover this. Engine runs a few short real generations to warm the
model path (GEMMs, FLA/conv1d Triton kernels and autotuning, packed head,
MTP draft, samplers). warm_attention() then drives the real attention
dispatch over every query-row count and a block-table width set that crosses
each 16-page boundary. Its Triton launches are compile-only, so it is cheap
and reads or writes no attention output. Selection logic stays in the
vendored dispatch; nothing here copies its thresholds.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

# Mirror of exllamav3.constants.PAGE_SIZE (see prefix_cache.py).
PAGE_SIZE = 256

# End-to-end prefill segment lengths besides the full chunk, each at both
# 16-residue classes. conv1d runs natively up to 32 rows, then as one Triton
# kernel per power-of-two row bucket (64/128/256) and as split output/state
# kernels above 256; the FLA chunked delta rule starts at the number of value
# heads (32 for MiMo/Qwen3.5). The full chunk covers the divisible >256 class.
SEGMENT_TAILS = (47, 48, 95, 96, 191, 192, 257)

# Decode rows: the paged decode kernels take up to 16 query rows (MTP verify,
# prefill tails and prefix-hit remainders); longer queries use prefill kernels.
MAX_DECODE_ROWS = 16

WARMUP_TEXT = (
    'The river bends past the old mill, where the miller keeps careful records of '
    'every harvest. In spring the water rises, and the wheel turns faster; in late '
    'summer it slows, and the village gathers to repair the stone channel. '
)


def prefill_segments(chunk):
    """Prefill lengths the end-to-end warm-up jobs compute, full chunk first."""
    return [chunk] + [rows for rows in SEGMENT_TAILS if rows < chunk]


def table_widths(max_pages):
    """Block-table widths (pages) covering both sides of every 16-page boundary.

    Target tables are padded to 16 pages but draft and prefill tables are
    not. Schedule thresholds and Triton's divisible-by-16 specialization
    therefore see every width class; all widths up to 16 plus 16k-1, 16k and
    16k+1 give each threshold region both residue classes.
    """
    widths = set(range(1, min(16, max_pages) + 1))
    for edge in range(16, max_pages + 1, 16):
        widths.update((edge - 1, edge, edge + 1))
    widths.add(max_pages)
    return sorted(w for w in widths if 1 <= w <= max_pages)


def query_rows(chunk):
    """Query-row counts: every decode count, then prefill rows up to the chunk.

    Prefill kernels tile 32-128 rows per block and specialize on rows % 16,
    so 17 plus 32k and 32k+1 reach every block count in both residue
    classes, including both sides of the 256/1024-row schedule limits.
    """
    rows = set(range(1, MAX_DECODE_ROWS + 1))
    rows.add(MAX_DECODE_ROWS + 1)
    for edge in range(32, chunk + 1, 32):
        rows.update(n for n in (edge, edge + 1) if n <= chunk)
    return sorted(rows)


@contextmanager
def compile_only(module, jit_type):
    """Route `kernel[grid](...)` launches of the module's `*_kernel` functions
    to Triton's compile-only path: specialize, compile or load from the disk
    cache, and load the binary, without running it. Helper @triton.jit
    functions called from inside kernels keep their module globals.

    Yields a dict of the distinct compiled kernels that were requested.
    """
    kernels = {}

    class CompileOnly:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                kernel = self.fn.run(*args, grid=grid, warmup=True, **kwargs)
                if kernel is not None:
                    init = getattr(kernel, '_init_handles', None)
                    if init is not None:
                        init()  # load the module now, not on the first request
                    kernels[id(kernel)] = kernel
                return kernel
            return launch

    originals = {name: value for name, value in vars(module).items()
                 if isinstance(value, jit_type) and name.endswith('_kernel')}
    try:
        for name, fn in originals.items():
            setattr(module, name, CompileOnly(fn))
        yield kernels
    finally:
        for name, fn in originals.items():
            setattr(module, name, fn)


@contextmanager
def attention_telemetry_preserved(triton_paged):
    """Keep the attention-profile call counters request-only across warm-up."""
    saved = triton_paged._auto_decode_calls, triton_paged._qc_long_decode_calls
    try:
        yield
    finally:
        triton_paged._auto_decode_calls, triton_paged._qc_long_decode_calls = saved


def attention_layers(model, cache):
    """(attention module, cache layer) pairs of the standard paged attention."""
    from exllamav3.modules import Attention

    return [(attn, cache.layers[instance])
            for attn in model.get_cache_layers() if isinstance(attn, Attention)
            for instance in model.get_layer_instances(attn.layer_idx)]


def _signature(attn, layer):
    tensors = tuple((tuple(t.shape), str(t.dtype)) for t in layer.get_tensors() if t is not None)
    sinks = None if attn.sinks is None else tuple(attn.sinks.shape)
    return (type(layer).__name__, tensors, getattr(layer, 'k_bits', None), getattr(layer, 'v_bits', None),
            getattr(layer, 'compand_a', None), attn.num_q_heads, attn.num_kv_heads, attn.head_dim,
            attn.sm_scale, attn.sliding_window, attn.logit_softcapping, sinks)


def warm_attention(torch, pairs, *, chunk):
    """Compile every paged-attention launch serving can reach for these layers.

    Calls the real dispatch with synthetic metadata: an empty cache
    (cache_seqlens 0) and block tables of every width class. Kernel choice
    uses only metadata, never cache contents, and Triton launches are
    compile-only. Quantized layers still quantize the synthetic K/V into the
    first table pages; no page table exists yet, so nothing can reuse them.
    Layers with identical geometry share one pass.
    """
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.modules.attention_fn import dispatch, triton_paged
    from triton.runtime.jit import JITFunction

    seen, skipped, calls = set(), 0, 0
    picks = set(triton_paged._qc_prefill_ns_cache)
    rows = query_rows(chunk)
    with compile_only(triton_paged, JITFunction) as kernels:
        for attn, layer in pairs:
            signature = _signature(attn, layer)
            if signature in seen:
                continue
            seen.add(signature)
            direct = (dispatch._qc_attn and dispatch.has_triton and layer.compand_a == 0.0
                      and attn.head_dim <= 512 and dispatch._is_power_of_2(attn.head_dim))
            if isinstance(layer, CacheLayer_quant) and not direct:
                # Outside quant-direct dispatch every call dequantizes the whole cache.
                skipped += 1
                continue
            device = attn.device
            q = torch.zeros((1, chunk, attn.num_q_heads, attn.head_dim), dtype=torch.half, device=device)
            k = torch.zeros((1, chunk, attn.num_kv_heads, attn.head_dim), dtype=torch.half, device=device)
            v = torch.zeros_like(k)
            seqlens = torch.zeros((1,), dtype=torch.int32, device=device)
            for width in table_widths(layer.max_num_tokens // PAGE_SIZE):
                table = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
                for n in rows:
                    if n > width * PAGE_SIZE:
                        break
                    dispatch.attn_dispatch(q[:, :n], k[:, :n], v[:, :n], cache=layer,
                                           block_table=table, cache_seqlens=seqlens, causal=True,
                                           sm_scale=attn.sm_scale, window_size=attn.sliding_window,
                                           softcap=attn.logit_softcapping, sinks=attn.sinks)
                    calls += 1
    # The prefill stage-count pick times real launches; the real generation
    # already made it for these layers. A pick first made here timed no-op
    # launches, so drop it and let serving measure again.
    reset = set(triton_paged._qc_prefill_ns_cache) - picks
    for key in reset:
        del triton_paged._qc_prefill_ns_cache[key]
    names = Counter(getattr(kernel, 'name', '?') for kernel in kernels.values())
    return dict(layers=len(seen), skipped_layers=skipped, calls=calls, variants=len(kernels),
                kernels=dict(sorted(names.items())), stage_picks_reset=len(reset))


class JitMonitor:
    """Count Triton specializations first used by this process.

    Uses Triton's post-compile hook, which fires for fresh compiles and
    on-disk cache loads alike, and chains any previously installed hook.
    """

    def __init__(self, knobs):
        self.count = 0
        self.kernels = Counter()
        self.listener = None
        self._previous = knobs.runtime.jit_post_compile_hook
        knobs.runtime.jit_post_compile_hook = self._observe

    def _observe(self, *args, **kwargs):
        name = getattr(kwargs.get('fn'), 'name', '?')
        self.count += 1
        self.kernels[name] += 1
        if self.listener is not None:
            self.listener(name)
        if self._previous is not None:
            return self._previous(*args, **kwargs)
        return None


def monitor_jit():
    """A JitMonitor on the installed Triton, or None without Triton knobs."""
    try:
        from triton import knobs
    except ImportError:
        return None
    return JitMonitor(knobs)
