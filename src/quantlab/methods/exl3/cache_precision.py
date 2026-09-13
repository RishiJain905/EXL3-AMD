"""Shared EXL3 attention-cache precision plumbing; Torch-free by design.

Exposes the pinned upstream packed integer KV caches (CacheLayer_quant at
8/4 bits, no compander) through one argparse/resolve/kwarg/storage surface
shared by the launcher, evaluator and server. Defaults preserve the existing
f16 cache; only global-attention layers are affected, recurrent state never
is. These are integer packed formats, not floating point: no fp8 alias.
"""
import sys

CACHE_TYPES = ("f16", "q8", "q4")
ATTENTION_PROFILES = ("default", "long")
_BITS = {"q8": 8, "q4": 4}


def add_cache_precision_args(parser):
    """Add --cache-type shorthand plus per-side -ctk/-ctv overrides."""
    parser.add_argument("--cache-type", choices=CACHE_TYPES, default=None,
                        help="Attention KV cache precision for K and V (shorthand)")
    parser.add_argument("-ctk", "--cache-type-k", choices=CACHE_TYPES, default=None,
                        help="Attention K cache precision; defaults to --cache-type, then f16")
    parser.add_argument("-ctv", "--cache-type-v", choices=CACHE_TYPES, default=None,
                        help="Attention V cache precision; defaults to --cache-type, then f16")
    parser.add_argument("--attention-profile", choices=ATTENTION_PROFILES, default="default",
                        help="Bounded long-context decode attention profile; default preserves inherited attention")
    return parser


def attention_profile_status():
    """Live vendor decode-profile counters without importing Torch or GPU state.

    Reads exllamav3.modules.attention_fn.triton_paged.qc_decode_profile_status()
    only when that module is already imported; otherwise reports explicit
    absence with None values, never synthesized call counts.
    """
    module = sys.modules.get("exllamav3.modules.attention_fn.triton_paged")
    if module is None:
        return {"profile": None, "applied_calls": None}
    status_fn = getattr(module, "qc_decode_profile_status", None)
    if not callable(status_fn):
        return {"profile": None, "applied_calls": None}
    status = status_fn()
    if not isinstance(status, dict):
        return {"profile": None, "applied_calls": None}
    return {"profile": status.get("profile"), "applied_calls": status.get("applied_calls")}


def resolve_cache_types(args):
    """Resolve the effective (K, V) precision pair; defaults f16.

    Raises ValueError on conflicting shorthand, unknown values, or f16 mixed
    with quantized sides, which the upstream class does not support.
    """
    shorthand = getattr(args, "cache_type", None)
    k = getattr(args, "cache_type_k", None)
    v = getattr(args, "cache_type_v", None)
    for value in (shorthand, k, v):
        if value is not None and value not in CACHE_TYPES:
            raise ValueError(f"Unknown cache precision {value!r}; expected one of {', '.join(CACHE_TYPES)}")
    if shorthand is not None:
        if k is not None and k != shorthand:
            raise ValueError(f"--cache-type {shorthand} conflicts with --cache-type-k {k}")
        if v is not None and v != shorthand:
            raise ValueError(f"--cache-type {shorthand} conflicts with --cache-type-v {v}")
        k = shorthand if k is None else k
        v = shorthand if v is None else v
    k = k if k is not None else "f16"
    v = v if v is not None else "f16"
    if (k == "f16") != (v == "f16"):
        raise ValueError(f"Unsupported KV cache mix K={k} V={v}; "
                         "CacheLayer_quant needs both sides quantized (q8/q4) or both f16")
    return k, v


def is_quantized(cache_k, cache_v):
    """True when the resolved pair uses the packed integer cache."""
    return cache_k != "f16" or cache_v != "f16"


def cache_kwargs(cache_k, cache_v, *, quant_layer_type=None):
    """Cache constructor kwargs; {} preserves the f16 default path.

    The caller passes the imported upstream CacheLayer_quant; the import
    stays lazy here so host-side parsing never needs Torch.
    """
    if cache_k == "f16" and cache_v == "f16":
        return {}
    if (cache_k == "f16") != (cache_v == "f16"):
        raise ValueError(f"Unsupported KV cache mix K={cache_k} V={cache_v}; "
                         "CacheLayer_quant needs both sides quantized (q8/q4) or both f16")
    try:
        k_bits, v_bits = _BITS[cache_k], _BITS[cache_v]
    except KeyError:
        raise ValueError(f"Unknown cache precision K={cache_k} V={cache_v}; "
                         f"expected one of {', '.join(CACHE_TYPES)}") from None
    if quant_layer_type is None:
        from exllamav3.cache import CacheLayer_quant
        quant_layer_type = CacheLayer_quant
    return {"layer_type": quant_layer_type, "k_bits": k_bits, "v_bits": v_bits, "compand_a": 0}


def _tensor_bytes(tensor):
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def attention_backends(cache):
    """Last selected attention functions observed on the cache's actual layers."""
    if cache is None:
        return []
    return sorted({fn.__name__ for layer in cache.layers.values()
                   for fn in getattr(layer.attention, 'dispatch_cache', {}).values()
                   if callable(fn)})


def cache_storage(cache, *, cache_k=None, cache_v=None):
    """Measured attention-cache tensors only; recurrent state excluded.

    A None cache (absent draft) reports explicit absence: present False with
    zero counts/bytes and None precision, not unknown values.
    """
    if cache is None:
        return {"present": False, "cache_k": None, "cache_v": None,
                "attention_layers": 0, "tensor_bytes": 0, "layers": []}
    layers = []
    total = 0
    for layer in (getattr(cache, "layers", None) or {}).values():
        tensors = layer.get_tensors() or []
        shapes = [list(tensor.shape) for tensor in tensors if tensor is not None]
        size = sum(_tensor_bytes(tensor) for tensor in tensors)
        total += size
        layers.append({"class": type(layer).__name__, "tensor_shapes": shapes, "tensor_bytes": size})
    return {"present": True, "cache_k": cache_k, "cache_v": cache_v,
            "attention_layers": len(layers), "tensor_bytes": total, "layers": layers}
