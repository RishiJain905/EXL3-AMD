"""Shared EXL3 attention-cache precision plumbing; Torch-free by design.

Exposes the pinned upstream packed integer KV caches (CacheLayer_quant at
8/6/5/4 bits, no compander) through one argparse/resolve/kwarg/storage surface
shared by the launcher, evaluator and server. Defaults preserve the existing
f16 cache; only global-attention layers are affected, recurrent state never
is. These are integer packed formats, not floating point: no fp8 alias.
Q6/Q5 are inherited uniform rotated integer formats, not AsterKV or TurboQuant.
The experimental aster5 format has a separate nonuniform encoder/decoder.
"""
import sys
from pathlib import Path

CACHE_TYPES = ("f16", "q8", "q6", "q5", "q4", "aster5")
ATTENTION_PROFILES = ("auto", "default", "long")
PREFILL_STAGING = ("on", "off")
_BITS = {"q8": 8, "q6": 6, "q5": 5, "q4": 4}


def _cache_policy():
    # The host launcher also loads this module by filename before src is on sys.path.
    if __package__:
        from . import cache_policy
        return cache_policy
    import importlib.util
    spec = importlib.util.spec_from_file_location("quantlab_cache_policy", Path(__file__).with_name("cache_policy.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_cache_precision_args(parser):
    """Add --cache-type shorthand plus per-side -ctk/-ctv overrides."""
    parser.add_argument("--cache-type", choices=CACHE_TYPES, default=None,
                        help="Attention KV cache precision for K and V (shorthand)")
    parser.add_argument("-ctk", "--cache-type-k", choices=CACHE_TYPES, default=None,
                        help="Attention K cache precision; defaults to --cache-type, then f16")
    parser.add_argument("-ctv", "--cache-type-v", choices=CACHE_TYPES, default=None,
                        help="Attention V cache precision; defaults to --cache-type, then f16")
    parser.add_argument("--attention-profile", choices=ATTENTION_PROFILES, default="auto",
                        help="Automatic measured attention scheduling (default); default and long retain diagnostic profiles")
    parser.add_argument("--prefill-staging", choices=PREFILL_STAGING, default="on",
                        help="Quantized KV: dequantize the referenced window to an FP16 scratch for chunked "
                             "prefill (default on); off keeps in-kernel dequantization (diagnostic)")
    parser.add_argument("--cache-policy", type=Path, default=None,
                        help="Experimental offline JSON precision profile for every attention layer; requires a quantized cache type")
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
                         "CacheLayer_quant needs both sides quantized (q8/q6/q5/q4) or both f16")
    if (k == "aster5") != (v == "aster5"):
        raise ValueError("Unsupported KV cache mix: experimental aster5 requires both K and V")
    policy_path = getattr(args, "cache_policy", None)
    if policy_path is not None:
        if k == "f16":
            raise ValueError("--cache-policy requires a quantized --cache-type")
        _cache_policy().load_cache_policy(policy_path)
    return k, v


def qc_staging_env(cache_k, cache_v, prefill_staging):
    """EXL3_QC_STAGING value for the backend, or None to keep the vendored default (1).

    Staging dequantizes each prefill chunk's referenced window once into a shared FP16
    scratch (one layer's K/V for the whole pool) and runs the FP16 prefill kernel over it.
    See docs/PREFILL-ATTENTION.md.
    """
    if prefill_staging not in PREFILL_STAGING:
        raise ValueError(f"Unknown prefill staging {prefill_staging!r}")
    return "0" if is_quantized(cache_k, cache_v) and prefill_staging == "off" else None


def is_quantized(cache_k, cache_v):
    """True when the resolved pair uses the packed integer cache."""
    return cache_k != "f16" or cache_v != "f16"


def cache_kwargs(cache_k, cache_v, *, quant_layer_type=None, aster_layer_type=None):
    """Cache constructor kwargs; {} preserves the f16 default path.

    The caller passes the imported upstream CacheLayer_quant; the import
    stays lazy here so host-side parsing never needs Torch.
    """
    if cache_k == "f16" and cache_v == "f16":
        return {}
    if cache_k == "aster5" or cache_v == "aster5":
        if cache_k != cache_v:
            raise ValueError("Unsupported KV cache mix: experimental aster5 requires both K and V")
        if aster_layer_type is None:
            from exllamav3.cache.aster import CacheLayer_aster
            aster_layer_type = CacheLayer_aster
        return {"layer_type": aster_layer_type, "k_bits": 5, "v_bits": 5, "compand_a": 0}
    if (cache_k == "f16") != (cache_v == "f16"):
        raise ValueError(f"Unsupported KV cache mix K={cache_k} V={cache_v}; "
                         "CacheLayer_quant needs both sides quantized (q8/q6/q5/q4) or both f16")
    try:
        k_bits, v_bits = _BITS[cache_k], _BITS[cache_v]
    except KeyError:
        raise ValueError(f"Unknown cache precision K={cache_k} V={cache_v}; "
                         f"expected one of {', '.join(CACHE_TYPES)}") from None
    if quant_layer_type is None:
        from exllamav3.cache import CacheLayer_quant
        quant_layer_type = CacheLayer_quant
    return {"layer_type": quant_layer_type, "k_bits": k_bits, "v_bits": v_bits, "compand_a": 0}


def model_cache_options(args, config_bytes, models, *, quant_layer_type=None, aster_layer_type=None):
    """Validate all active components before constructing any cache or loading weights."""
    k, v = resolve_cache_types(args)
    base = cache_kwargs(k, v, quant_layer_type=quant_layer_type, aster_layer_type=aster_layer_type)
    options = {name: dict(base) for name in models}
    policy_path = getattr(args, "cache_policy", None)
    if policy_path is None:
        return options, None
    policy_module = _cache_policy()
    policy = policy_module.load_cache_policy(policy_path)
    descriptors = {name: [dict(layer_idx=layer.layer_idx, kv_heads=layer.num_kv_heads,
                              head_dim=layer.head_dim) for layer in model.get_cache_layers()]
                   for name, model in models.items()}
    assignments = policy_module.validate_cache_policy(policy, config_bytes, descriptors)
    for name, layers in assignments.items():
        options[name]["layer_overrides"] = {
            idx: cache_kwargs(ck, cv, quant_layer_type=quant_layer_type, aster_layer_type=aster_layer_type)
            for idx, (ck, cv) in layers.items()}
    return options, policy_module.policy_summary(policy)


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
    for instance, layer in (getattr(cache, "layers", None) or {}).items():
        tensors = layer.get_tensors() or []
        metadata = getattr(layer, "get_metadata_tensors", lambda: [])() or []
        tensors = [*tensors, *metadata]
        shapes = [list(tensor.shape) for tensor in tensors if tensor is not None]
        size = sum(_tensor_bytes(tensor) for tensor in tensors)
        total += size
        fmt = getattr(layer, "cache_format", None)
        k = "aster5" if fmt == "aster5" else "q" + str(layer.k_bits) if hasattr(layer, "k_bits") else cache_k
        v = "aster5" if fmt == "aster5" else "q" + str(layer.v_bits) if hasattr(layer, "v_bits") else cache_v
        layers.append({"class": type(layer).__name__, "tensor_shapes": shapes, "tensor_bytes": size,
                       "layer_idx": instance[0] if isinstance(instance, tuple) else instance,
                       "cache_k": k, "cache_v": v})
    if len({row['cache_k'] for row in layers}) > 1:
        cache_k = "mixed"
    elif layers:
        cache_k = layers[0]['cache_k']
    if len({row['cache_v'] for row in layers}) > 1:
        cache_v = "mixed"
    elif layers:
        cache_v = layers[0]['cache_v']
    return {"present": True, "cache_k": cache_k, "cache_v": cache_v,
            "attention_layers": len(layers), "tensor_bytes": total, "layers": layers}
