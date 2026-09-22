"""Inspect mapped EXL3 text/MTP modules without loading tensor payloads.

Run with the pinned EXL3 Python environment and source on PYTHONPATH. Output is
private inventory: module names are retained; no conversion or GPU work runs.
"""

import argparse
import json
from pathlib import Path
import sys
import struct

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantlab.methods.exl3.source_map import tensor_name_fixes


MTP_STATUS_PRESENT = "present"
MTP_STATUS_ABSENT_SUPPRESSED = "absent_suppressed"
MTP_STATUS_ABSENT_UNDECLARED = "absent_undeclared"


def canonical_tensor_name(source, fixes):
    """Apply EXL3 suffix fixes exactly like SafetensorsCollection header loading."""
    destination = source
    for old, new in fixes.items():
        if destination.endswith(old):
            destination = destination[: len(destination) - len(old)] + new
    return destination


def is_mtp_tensor_name(canonical):
    """True for top-level/compiled ``mtp.*`` keys and nested ``*.mtp.*`` keys."""
    return canonical == "mtp" or canonical.startswith("mtp.") or ".mtp." in canonical


def count_mtp_tensors(source_keys, fixes):
    """Count source keys whose canonical destination indicates MTP tensors."""
    return sum(1 for key in source_keys if is_mtp_tensor_name(canonical_tensor_name(key, fixes)))


def estimate_budgets(linears, selected, *, body_bits=2, head_bits_options=(2, 3, 4), mtp_bits=4, mtp_present=True):
    """Quantized-size budgets from inspected linear geometry; no payload loaded."""
    quantized = [m for m in linears if m["qmap"] is not None]
    replaced = {m["key"] + ".weight" for m in quantized}
    unchanged = sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in selected.items() if k not in replaced)
    budgets = []
    for head_bits in head_bits_options:
        payload = scales = padding = 0
        for m in quantized:
            bits = mtp_bits if m["component"] == "mtp" else head_bits if m["qbits_key"] == "head_bits" else body_bits
            rows, cols = m["out_features"], m["in_features"]
            payload += rows * cols * bits // 8
            scales += 2 * (rows + cols) + 4  # fp16 suh/svh and scalar int32 mul1 marker
            source_rows, source_cols = m["source_shape"]
            padding += (rows * cols - source_rows * source_cols) * bits // 8
        budgets.append({"text_bits": body_bits, "head_bits": head_bits, "mtp_bits": mtp_bits if mtp_present else None,
                        "unchanged_source_tensor_bytes": unchanged, "packed_trellis_bytes": payload,
                        "padding_bytes_included_in_trellis": padding,
                        "scale_and_codebook_marker_bytes": scales,
                        "derived_tensor_bytes": unchanged + payload + scales,
                        "container_and_config_bytes": None,
                        "quantized_linear_count": len(quantized)})
    return budgets


def mapped_text_config(model_dir):
    """Construct the real VL text/MTP adapter without requiring vision metadata.

    Reusable by a run-local converter wrapper in place of Config.from_directory
    for this source. No upstream classes or source checkpoint files are patched.
    When the checkpoint carries no MTP tensors, the optional MTP component is
    suppressed in memory and the discrepancy recorded on the returned config.
    """
    from exllamav3.architecture.qwen3_5 import (
        Qwen3_5VLConfig, Qwen3_5VLBaseConfig, Qwen3_5VLModel,
    )
    from exllamav3.architecture.qwen3_5_mtp import Qwen3_5MTPModel

    index_path = Path(model_dir) / "model.safetensors.index.json"
    if index_path.is_file():
        keys = json.loads(index_path.read_text())["weight_map"]
    else:
        # A single-shard compiled candidate has no index. Read headers only;
        # preserve duplicate names so tensor_name_fixes can reject collisions.
        keys = []
        for shard in sorted(Path(model_dir).glob("*.safetensors")):
            with shard.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                if not 0 < header_size <= 100 * 1024**2:
                    raise ValueError("Invalid safetensors header length")
                header = json.loads(stream.read(header_size))
                keys.extend(k for k in header if k != "__metadata__")
        if not keys:
            raise ValueError("No safetensors tensors found")
    fixes = tensor_name_fixes(keys)
    mtp_source_tensors = count_mtp_tensors(keys, fixes)
    class TextOnlyConfig(Qwen3_5VLConfig):
        def get_tensor_name_fixes(self):
            return fixes

        def __init__(self):
            # The pinned base accepts optional component classes. Supplying no
            # vision class skips its preprocessor_config.json read entirely.
            Qwen3_5VLBaseConfig.__init__(
                self, str(model_dir), text_cfg="text_config",
                text_model=Qwen3_5VLModel, vision_model=None,
                mtp_model=Qwen3_5MTPModel,
            )
            # In memory only; source files are never mutated. Complete absence
            # of MTP tensor names suppresses the optional draft component so a
            # declared-but-shipped-without-MTP checkpoint does not masquerade as
            # an incomplete MTP tensor set. Any MTP name at all preserves MTP
            # construction so missing-required checks reveal partial sets.
            self.mtp_source_tensors = mtp_source_tensors
            self.mtp_suppressed_missing_tensors = (
                mtp_source_tensors == 0 and "mtp" in self.model_classes
            )
            if self.mtp_suppressed_missing_tensors:
                del self.model_classes["mtp"]

    return TextOnlyConfig(), len(fixes)


def construct_models(model_dir):
    from exllamav3 import Model

    config, fix_count = mapped_text_config(model_dir)

    def forbid_payload(*args, **kwargs):
        raise RuntimeError("Metadata inspection attempted to load a tensor payload")

    config.stc.get_tensor = forbid_payload
    model = Model.from_config(config)
    mtp = Model.from_config(config, component="mtp") if "mtp" in config.model_classes else None
    return config, model, mtp, fix_count


def inspect(model_dir, *, body_bits=2, head_bits_options=(2, 3, 4), mtp_bits=4):
    from exllamav3.modules import Linear, Embedding, RMSNorm, GatedDeltaNet

    config, model, mtp, fix_count = construct_models(model_dir)
    headers = {}
    for header in config.stc.file_headers.values():
        headers.update({k: v for k, v in header.items() if k not in ("__metadata__", "_header_offset")})
    selected = {k: v for k, v in headers.items() if not k.startswith("model.visual.")}
    missing, mismatches, unhandled = [], [], []
    consumed = set()
    linears = []

    def require(key, shape=None):
        if key not in headers:
            missing.append(key)
            return
        consumed.add(key)
        if shape is not None and headers[key]["shape"] != shape:
            mismatches.append({"key": key, "actual": headers[key]["shape"], "expected": shape})

    components = [("text", model)]
    if mtp is not None:
        components.append(("mtp", mtp))
    for component, instance in components:
        for module in instance:
            if isinstance(module, Linear):
                require(module.key + ".weight", [module.out_features_unpadded, module.in_features_unpadded])
                if module.key + ".bias" in headers:
                    consumed.add(module.key + ".bias")
                linears.append({"key": module.key, "component": component,
                                "qmap": module.qmap, "qbits_key": module.qbits_key,
                                "in_features": module.in_features, "out_features": module.out_features,
                                "source_shape": headers.get(module.key + ".weight", {}).get("shape")})
            elif isinstance(module, Embedding):
                require(module.key + ".weight", [config.vocab_size, config.hidden_size])
            elif isinstance(module, RMSNorm):
                if not module.unweighted:
                    width = config.head_dim if module.key.endswith((".self_attn.q_norm", ".self_attn.k_norm")) else config.hidden_size
                    require(module.tensor_key, [width])
            elif isinstance(module, GatedDeltaNet):
                require(module.key_a_log, [module.num_v_heads])
                require(module.key_dt_bias, [module.num_v_heads])
                require(module.key_conv1d_weight, [module.fdim_qkv, 1, module.conv_kernel_size])
                require(module.norm.key + ".weight", [module.v_head_dim])
            elif not module.modules:
                # GatedRMSNorm is explicitly accounted through GatedDeltaNet.
                if type(module).__name__ != "GatedRMSNorm":
                    unhandled.append({"key": module.key, "class": type(module).__name__})

    def nbytes(entry):
        return entry["data_offsets"][1] - entry["data_offsets"][0]

    budgets = estimate_budgets(linears, selected, body_bits=body_bits,
                               head_bits_options=head_bits_options, mtp_bits=mtp_bits,
                               mtp_present=mtp is not None)
    if mtp is not None:
        mtp_status = MTP_STATUS_PRESENT
    elif config.mtp_num_hidden_layers:
        mtp_status = MTP_STATUS_ABSENT_SUPPRESSED
    else:
        mtp_status = MTP_STATUS_ABSENT_UNDECLARED
    scope = "Derived successful mul1 encoding: K-bit trellis plus fp16 row/column scales, int32 marker, unchanged nonquantized source tensors. No serialized container/config overhead or runtime allocation included. No tensor payload loaded."
    if mtp is None:
        scope += " No MTP tensors present; no draft recipe included."
    return {"architecture": config.architecture, "name_fix_count": fix_count,
            "text_modules": len(model.modules), "mtp_modules": len(mtp.modules) if mtp is not None else 0,
            "mtp_declared_layers": config.mtp_num_hidden_layers,
            "mtp_source_tensors": config.mtp_source_tensors,
            "mtp_status": mtp_status,
            "missing_required_tensors": sorted(set(missing)), "shape_mismatches": mismatches,
            "unhandled_leaf_modules": unhandled,
            "unconsumed_text_mtp_tensors": sorted(set(selected) - consumed),
            "excluded_vision_tensors": len(headers) - len(selected),
            "excluded_vision_source_bytes": sum(nbytes(v) for k, v in headers.items() if k.startswith("model.visual.")),
            "linears": linears, "size_estimates": budgets,
            "estimate_scope": scope,
            "embedding_default_placement": "CPU preference; GPU placement may be explicitly overridden and measured"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.model_dir)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    summary = {k: v for k, v in result.items() if k != "linears"}
    print(json.dumps(summary, indent=2))
    return int(bool(result["missing_required_tensors"] or result["shape_mismatches"] or result["unhandled_leaf_modules"] or result["unconsumed_text_mtp_tensors"]))


if __name__ == "__main__":
    raise SystemExit(main())
