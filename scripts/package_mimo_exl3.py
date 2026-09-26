"""Package a completed MiMo text EXL3 conversion with preserved vision weights.

Stdlib-only audit/packaging CLI. Reads the original BF16 source directory, a
completed compiled text candidate, a step1 vision preservation pilot and the
step1 source audit, then writes a new final candidate directory plus an
external JSON report. Never modifies inputs. Fails closed before writing
output whenever expected inputs are missing or incomplete.
"""

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
from pathlib import Path

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
               "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
MUL1_MARKER = 0x83DCD12D
BODY_BITS = 4
HEAD_BITS = 6
EXPECTED_QUANTIZED = 201
EXPECTED_VISION = 333
EXPECTED_ARCH = "Qwen3_5ForConditionalGeneration"
EXPECTED_HIDDEN = 4096
EXPECTED_LAYERS = 32
HEAD_BASE = "lm_head"
EMBED_KEY = "model.language_model.embed_tokens.weight"
VISION_FILENAME = "vision.safetensors"
INCOMPLETE_FILENAME = "PACKAGING-INCOMPLETE.json"
HEADER_SIZE_CAP = 100 * 1024**2
HASH_CHUNK = 8 * 1024**2

# Fallback-only suffix map used when --mapping is absent. Keys are module-base
# suffixes (source tensor key minus trailing ".weight").
QUANT_SUFFIXES = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
SOURCE_METADATA_SUFFIXES = (".json", ".jinja", ".txt")
TEXT_METADATA_FILES = ("quantization_config.json", "conversion_identity.json")
IGNORED_METADATA_SUFFIXES = (".bin", ".ckpt", ".pth", ".pt", ".log")


class PackagingError(Exception):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PackagingError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise PackagingError(f"Cannot read {path}: {error}")
    try:
        return json.loads(text, object_pairs_hook=unique_object)
    except ValueError as error:
        raise PackagingError(f"Invalid JSON in {path}: {error}")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_payload_hash(path, header_size, begin, end):
    digest = hashlib.sha256()
    remaining = end - begin
    with open(path, "rb") as stream:
        stream.seek(8 + header_size + begin)
        while remaining:
            chunk = stream.read(min(HASH_CHUNK, remaining))
            if not chunk:
                raise PackagingError(f"Truncated payload in {Path(path).name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def read_scalar_u32(path, header_size, begin):
    with open(path, "rb") as stream:
        stream.seek(8 + header_size + begin)
        raw = stream.read(4)
    if len(raw) != 4:
        raise PackagingError(f"Truncated scalar payload in {Path(path).name}")
    return struct.unpack("<I", raw)[0]


def within_or_equal(path, other):
    return path == other or other in path.parents


def check_paths(source, text_dir, vision_dir, audit_path, mapping_path, output, report,
                allocation_path=None):
    inputs = {"--source": source, "--text-dir": text_dir, "--vision-dir": vision_dir}
    for name, path in inputs.items():
        if not path.is_dir():
            raise PackagingError(f"{name} is not a directory: {path}")
    for name, path in (("--source-audit", audit_path),):
        if not path.is_file():
            raise PackagingError(f"{name} is not a file: {path}")
    if mapping_path is not None and not mapping_path.is_file():
        raise PackagingError(f"--mapping is not a file: {mapping_path}")
    if allocation_path is not None and not allocation_path.is_file():
        raise PackagingError(f"--allocation is not a file: {allocation_path}")
    for first, second in (("--source", "--text-dir"), ("--source", "--vision-dir"),
                          ("--text-dir", "--vision-dir")):
        first_path, second_path = inputs[first], inputs[second]
        if within_or_equal(first_path, second_path) or within_or_equal(second_path, first_path):
            raise PackagingError(f"{first} and {second} must be separate, non-nested directories")
    for name, path in inputs.items():
        if within_or_equal(output, path) or within_or_equal(path, output):
            raise PackagingError(f"--output must be separate from and not nested with {name}")
        if within_or_equal(report, path):
            raise PackagingError(f"--report must be outside {name}")
    if within_or_equal(report, output) or within_or_equal(output, report):
        raise PackagingError("--report must be outside the --output candidate")
    if within_or_equal(audit_path, output):
        raise PackagingError("--source-audit must be outside --output")
    if mapping_path is not None and within_or_equal(mapping_path, output):
        raise PackagingError("--mapping must be outside --output")
    if allocation_path is not None and within_or_equal(allocation_path, output):
        raise PackagingError("--allocation must be outside --output")
    if report.exists() or output.exists():
        which = "--report" if report.exists() else "--output"
        raise PackagingError(f"{which} already exists; refusing to overwrite")
    if not report.parent.is_dir():
        raise PackagingError(f"--report parent directory does not exist: {report.parent}")
    for path in (source, text_dir, vision_dir):
        if path.is_symlink():
            raise PackagingError(f"Input directory is a symlink: {path}")


def inspect_shard(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise PackagingError(f"Truncated safetensors prefix: {path.name}")
        header_size, = struct.unpack("<Q", prefix)
        if not 0 < header_size <= min(HEADER_SIZE_CAP, size - 8):
            raise PackagingError(f"Invalid header length: {path.name}")
        try:
            header = json.loads(stream.read(header_size), object_pairs_hook=unique_object)
        except ValueError as error:
            raise PackagingError(f"Invalid header in {path.name}: {error}")
        if not isinstance(header, dict):
            raise PackagingError(f"Safetensors header must be an object: {path.name}")
        payload_size = size - 8 - header_size
        tensors = []
        for key, entry in header.items():
            if key == "__metadata__":
                if not isinstance(entry, dict) or any(not isinstance(v, str) for v in entry.values()):
                    raise PackagingError(f"Safetensors metadata must contain string values: {path.name}")
                continue
            if not isinstance(entry, dict):
                raise PackagingError(f"Invalid tensor declaration: {key}")
            shape, dtype, offsets = entry.get("shape"), entry.get("dtype"), entry.get("data_offsets")
            if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(dtype, str) or dtype not in DTYPE_BYTES):
                raise PackagingError(f"Invalid shape or unsupported dtype: {key}")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)):
                raise PackagingError(f"Invalid offsets: {key}")
            begin, end = offsets
            expected = math.prod(shape) * DTYPE_BYTES[dtype]
            if not 0 <= begin <= end <= payload_size or end - begin != expected:
                raise PackagingError(f"Out-of-bounds or incorrect payload size: {key}")
            tensors.append({"key": key, "shape": shape, "dtype": dtype,
                            "data_offsets": [begin, end], "bytes": expected})
        cursor = 0
        for tensor in sorted(tensors, key=lambda t: tuple(t["data_offsets"])):
            begin, end = tensor["data_offsets"]
            if begin != cursor:
                raise PackagingError(f"Overlapping or uncovered payload: {tensor['key']}")
            cursor = end
        if cursor != payload_size:
            raise PackagingError(f"Uncovered trailing payload: {path.name}")
    return {"file": path.name, "file_bytes": size, "header_and_prefix_bytes": 8 + header_size,
            "header_size": header_size, "payload_bytes": payload_size,
            "sha256": file_sha256(path), "tensors": tensors}


def collect_shards(directory, label):
    shards = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            raise PackagingError(f"{label} contains a symlink: {path.name}")
        if path.is_file() and path.suffix == ".safetensors":
            shards.append(path)
    if not shards:
        raise PackagingError(f"{label} has no top-level safetensors shards: {directory}")
    return shards


def merge_shard_tensors(shards, label):
    merged = {}
    for shard in shards:
        for tensor in shard["tensors"]:
            key = tensor["key"]
            if key in merged:
                raise PackagingError(f"Duplicate tensor across {label} shards: {key}")
            merged[key] = {"shard": shard["file"], **tensor}
    return merged


def check_source_config(config):
    if EXPECTED_ARCH not in (config.get("architectures") or []):
        raise PackagingError(f"Source architectures must include {EXPECTED_ARCH}")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise PackagingError("Source config has no text_config object")
    if text.get("hidden_size") != EXPECTED_HIDDEN:
        raise PackagingError("Source text hidden_size is not 4096")
    if text.get("num_hidden_layers") != EXPECTED_LAYERS:
        raise PackagingError("Source text num_hidden_layers is not 32")
    if not isinstance(config.get("vision_config"), dict):
        raise PackagingError("Source config has no vision_config object")


def apply_allocation(quantized, allocation):
    if not isinstance(allocation, dict):
        raise PackagingError("Allocation is not a JSON object")
    expected = set(quantized)
    got = set(allocation)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing:
        raise PackagingError(f"Allocation is missing {len(missing)} projections: {missing[:8]}")
    if extra:
        raise PackagingError(f"Allocation has {len(extra)} unexpected projections: {extra[:8]}")
    for base in sorted(allocation):
        bits = allocation[base]
        if type(bits) is not int:
            raise PackagingError(f"Allocation bitrate for {base} is not an integer: {bits!r}")
        if not 1 <= bits <= 8:
            raise PackagingError(f"Allocation bitrate for {base} is outside 1..8: {bits}")
    if allocation.get(HEAD_BASE) != HEAD_BITS:
        raise PackagingError(
            f"Allocation head rate must be K{HEAD_BITS}: {HEAD_BASE}={allocation.get(HEAD_BASE)!r}")
    for base, bits in allocation.items():
        quantized[base]["bits"] = bits


def build_expected_text(source_audit, mapping, allocation=None):
    tensors = source_audit.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise PackagingError("Source audit has no tensors inventory")
    for key in tensors:
        if key.startswith("mtp."):
            raise PackagingError(f"Unexpected MTP tensor in source audit: {key}")
    quantized = {}
    if mapping is not None:
        linears = mapping.get("linears")
        if not isinstance(linears, list) or not linears:
            raise PackagingError("Mapping has no linears inventory")
        for entry in mapping["linears"]:
            if not isinstance(entry, dict) or entry.get("qmap") is None:
                continue
            base = entry.get("key")
            in_features, out_features = entry.get("in_features"), entry.get("out_features")
            if not base or not isinstance(in_features, int) or not isinstance(out_features, int):
                raise PackagingError(f"Mapping entry lacks key/dimensions: {entry}")
            if base in quantized:
                raise PackagingError(f"Duplicate mapping linear: {base}")
            source_key = base + ".weight" if base != HEAD_BASE else "lm_head.weight"
            source_entry = tensors.get(source_key)
            if source_entry is None:
                raise PackagingError(f"Mapping linear has no source tensor: {source_key}")
            if source_entry.get("shape") != [out_features, in_features]:
                raise PackagingError(f"Mapping/source shape mismatch: {source_key}")
            role = "head" if entry.get("qbits_key") == "head_bits" else "body"
            quantized[base] = {"in_features": in_features, "out_features": out_features,
                               "bits": HEAD_BITS if role == "head" else BODY_BITS, "role": role}
        heads = sorted(base for base, spec in quantized.items() if spec["role"] == "head")
        if heads != [HEAD_BASE]:
            raise PackagingError(f"Mapping must declare exactly one head_bits linear ({HEAD_BASE}): {heads}")
    else:
        for key, entry in tensors.items():
            if key.startswith("model.visual.") or not key.endswith(".weight"):
                continue
            base = key[: -len(".weight")]
            if base == HEAD_BASE:
                role, bits = "head", HEAD_BITS
            elif base.split("model.language_model.")[-1] in QUANT_SUFFIXES or any(
                    base.endswith("." + suffix) for suffix in QUANT_SUFFIXES):
                role, bits = "body", BODY_BITS
            else:
                continue
            shape = entry.get("shape")
            if not isinstance(shape, list) or len(shape) != 2:
                raise PackagingError(f"Source projection shape is not 2-D: {key}")
            out_features, in_features = shape
            if base in quantized:
                raise PackagingError(f"Duplicate projection base: {base}")
            quantized[base] = {"in_features": in_features, "out_features": out_features,
                               "bits": bits, "role": role}
        if HEAD_BASE not in quantized or quantized[HEAD_BASE]["role"] != "head":
            raise PackagingError("Fallback projection scan found no lm_head projection")
    if len(quantized) != EXPECTED_QUANTIZED:
        raise PackagingError(f"Expected {EXPECTED_QUANTIZED} quantized projections, found {len(quantized)}")
    if allocation is not None:
        apply_allocation(quantized, allocation)
    quantized_weight_keys = {"lm_head.weight" if base == HEAD_BASE else base + ".weight"
                             for base in quantized}
    passthrough = {}
    for key, entry in tensors.items():
        if key.startswith("model.visual.") or key in quantized_weight_keys:
            continue
        passthrough[key] = {"shape": entry.get("shape"), "dtype": entry.get("dtype"),
                            "bytes": entry.get("bytes")}
    if EMBED_KEY not in passthrough:
        raise PackagingError(f"Source audit has no embedding tensor: {EMBED_KEY}")
    expected_names = set(passthrough)
    for base in quantized:
        expected_names.update([f"{base}.trellis", f"{base}.suh", f"{base}.svh", f"{base}.mul1"])
    return {"quantized": quantized, "passthrough": passthrough, "expected_names": expected_names}


def audit_text_candidate(text_dir, expected, source, source_audit):
    shard_paths = collect_shards(text_dir, "text candidate")
    shards = [inspect_shard(path) for path in shard_paths]
    merged = merge_shard_tensors(shards, "text")
    mtp_keys = sorted(key for key in merged if key.startswith("mtp."))
    if mtp_keys:
        raise PackagingError(f"Text candidate contains MTP tensors: {mtp_keys[:5]}")
    expected_names = expected["expected_names"]
    missing = sorted(expected_names - set(merged))
    unexpected = sorted(set(merged) - expected_names)
    if missing:
        raise PackagingError(f"Text candidate is missing {len(missing)} tensors: {missing[:8]}")
    if unexpected:
        raise PackagingError(f"Text candidate has {len(unexpected)} unexpected tensors: {unexpected[:8]}")
    by_file = {shard["file"]: shard for shard in shards}
    bit_map, conversions = {}, []
    body_trellis_bytes, body_params = 0, 0
    for base in sorted(expected["quantized"]):
        spec = expected["quantized"][base]
        in_features, out_features, bits = spec["in_features"], spec["out_features"], spec["bits"]
        trellis = merged[f"{base}.trellis"]
        if trellis["dtype"] != "I16" or trellis["shape"] != [in_features // 16, out_features // 16, bits * 16]:
            raise PackagingError(
                f"Bad trellis type/geometry for {base}: dtype={trellis['dtype']} shape={trellis['shape']}")
        measured = trellis["shape"][-1] // 16
        if measured != bits:
            raise PackagingError(f"Bad stored rate for {base}: K{measured}, expected K{bits}")
        bit_map[base] = measured
        if spec["role"] == "body":
            body_trellis_bytes += trellis["bytes"]
            body_params += in_features * out_features
        for suffix, dtype, shape in (("suh", "F16", [in_features]), ("svh", "F16", [out_features])):
            entry = merged[f"{base}.{suffix}"]
            if entry["dtype"] != dtype or entry["shape"] != shape:
                raise PackagingError(f"Bad {suffix} type/geometry for {base}: "
                                     f"dtype={entry['dtype']} shape={entry['shape']}")
        marker = merged[f"{base}.mul1"]
        if marker["dtype"] != "I32" or marker["shape"] != []:
            raise PackagingError(f"Bad mul1 marker type/shape for {base}: "
                                 f"dtype={marker['dtype']} shape={marker['shape']}")
        shard = by_file[marker["shard"]]
        value = read_scalar_u32(text_dir / shard["file"], shard["header_size"], marker["data_offsets"][0])
        if value != MUL1_MARKER:
            raise PackagingError(f"Bad mul1 marker value for {base}: {value:#010x}")
    for key in sorted(expected["passthrough"]):
        spec = expected["passthrough"][key]
        entry = merged[key]
        if entry["shape"] != spec["shape"]:
            raise PackagingError(f"Passthrough shape changed for {key}: {entry['shape']}")
        if entry["dtype"] not in ("F16", "BF16"):
            raise PackagingError(f"Passthrough dtype not F16/BF16 for {key}: {entry['dtype']}")
        if entry["dtype"] != spec["dtype"]:
            conversions.append({"key": key, "from": spec["dtype"], "to": entry["dtype"]})
    embed = merged[EMBED_KEY]
    if embed["dtype"] != "BF16":
        raise PackagingError(f"Embedding must remain BF16, found {embed['dtype']}")
    source_entry = source_audit["tensors"][EMBED_KEY]
    source_shard = source / source_entry["shard"]
    if not source_shard.is_file():
        raise PackagingError(f"Source shard for embedding is missing: {source_entry['shard']}")
    source_header = inspect_shard(source_shard)
    source_tensors = {tensor["key"]: tensor for tensor in source_header["tensors"]}
    if EMBED_KEY not in source_tensors:
        raise PackagingError(f"Embedding tensor not found in source shard: {source_entry['shard']}")
    source_offsets = source_tensors[EMBED_KEY]["data_offsets"]
    source_hash = stream_payload_hash(source_shard, source_header["header_size"], *source_offsets)
    candidate_shard = by_file[embed["shard"]]
    candidate_hash = stream_payload_hash(text_dir / candidate_shard["file"],
                                         candidate_shard["header_size"], *embed["data_offsets"])
    if source_hash != candidate_hash:
        raise PackagingError("Embedding payload differs from source")
    return {"shards": shards, "bit_map": bit_map, "conversions": conversions,
            "body_trellis_bytes": body_trellis_bytes, "body_params": body_params,
            "embedding": {"key": EMBED_KEY, "dtype": embed["dtype"], "shape": embed["shape"],
                          "source_payload_sha256": source_hash,
                          "candidate_payload_sha256": candidate_hash, "match": True}}


def audit_vision_pilot(vision_dir, source_audit):
    vision_path = vision_dir / VISION_FILENAME
    if not vision_path.is_file():
        raise PackagingError(f"Vision pilot has no {VISION_FILENAME}: {vision_dir}")
    if vision_path.is_symlink():
        raise PackagingError(f"Vision file is a symlink: {vision_path}")
    expected_hashes = source_audit.get("vision_payload_sha256")
    if not isinstance(expected_hashes, dict) or len(expected_hashes) != EXPECTED_VISION:
        raise PackagingError("Source audit vision payload hashes are not 333 entries")
    if source_audit.get("vision_tensors") != EXPECTED_VISION:
        raise PackagingError("Source audit vision tensor count is not 333")
    shard = inspect_shard(vision_path)
    if shard["sha256"] != source_audit.get("vision_file_sha256"):
        raise PackagingError("Vision container hash does not match source audit")
    merged = {tensor["key"]: tensor for tensor in shard["tensors"]}
    missing = sorted(set(expected_hashes) - set(merged))
    unexpected = sorted(set(merged) - set(expected_hashes))
    if missing:
        raise PackagingError(f"Vision file is missing {len(missing)} tensors: {missing[:8]}")
    if unexpected:
        raise PackagingError(f"Vision file has {len(unexpected)} unexpected tensors: {unexpected[:8]}")
    audit_tensors = source_audit["tensors"]
    for key in sorted(expected_hashes):
        entry = audit_tensors.get(key)
        if entry is None:
            raise PackagingError(f"Vision tensor has no source audit entry: {key}")
        actual = merged[key]
        if actual["dtype"] != entry.get("dtype") or actual["shape"] != entry.get("shape"):
            raise PackagingError(f"Vision dtype/shape differs for {key}")
        payload_hash = stream_payload_hash(vision_path, shard["header_size"], *actual["data_offsets"])
        if payload_hash != expected_hashes[key]:
            raise PackagingError(f"Vision payload differs for {key}")
    return {"shard": shard, "hashes": expected_hashes}


def plan_source_metadata(source):
    copied, skipped = [], []
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.is_symlink():
            skipped.append(path.name)
            continue
        name = path.name
        if name in ("config.json", "model.safetensors.index.json") or name.startswith("."):
            skipped.append(name)
            continue
        if name.endswith(".safetensors") or name.endswith(IGNORED_METADATA_SUFFIXES):
            skipped.append(name)
            continue
        if name.endswith(SOURCE_METADATA_SUFFIXES) or name.startswith("README") or name.startswith("LICENSE"):
            copied.append(name)
        else:
            skipped.append(name)
    return copied, skipped


def plan_text_metadata(text_dir):
    kept, skipped = [], []
    shard_names = {path.name for path in collect_shards(text_dir, "text candidate")}
    for path in sorted(text_dir.iterdir()):
        if path.is_dir() or path.is_symlink():
            skipped.append(path.name)
            continue
        if not path.is_file():
            continue
        name = path.name
        if name in shard_names or name in TEXT_METADATA_FILES:
            if name in TEXT_METADATA_FILES:
                kept.append(name)
            continue
        skipped.append(name)
    return kept, skipped


def build_output_config(source_config, text_config):
    if not isinstance(source_config, dict) or not isinstance(text_config, dict):
        raise PackagingError("Source or text config is not a JSON object")
    quantization = text_config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "exl3":
        raise PackagingError("Text candidate config must identify quant_method=exl3")
    output_config = json.loads(json.dumps(source_config))
    text = output_config.get("text_config")
    if not isinstance(text, dict) or "mtp_num_hidden_layers" not in text:
        raise PackagingError("Source config text_config lacks mtp_num_hidden_layers")
    patch = {"text_config.mtp_num_hidden_layers": {"from": text["mtp_num_hidden_layers"], "to": 0}}
    text["mtp_num_hidden_layers"] = 0
    output_config["quantization_config"] = quantization
    patch["quantization_config"] = {"from": None, "to": "retained from text candidate"}
    if not isinstance(output_config.get("vision_config"), dict):
        raise PackagingError("Output config lost vision_config")
    return output_config, patch


def pick_vision_filename(text_basenames):
    candidate = VISION_FILENAME
    if candidate not in text_basenames:
        return candidate
    candidate = "vision-preserved.safetensors"
    counter = 2
    while candidate in text_basenames:
        candidate = f"vision-preserved-{counter}.safetensors"
        counter += 1
    return candidate


def write_json_new(path, payload):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def copy_file(source, destination):
    with open(source, "rb") as reader, open(destination, "xb") as writer:
        shutil.copyfileobj(reader, writer, length=HASH_CHUNK)


def package_text_vision(source, text_dir, vision_dir, source_audit, text_audit, vision_audit,
                        source_metadata, text_metadata, output):
    text_basenames = [shard["file"] for shard in text_audit["shards"]]
    vision_name = pick_vision_filename(set(text_basenames))
    for shard in text_audit["shards"]:
        copy_file(text_dir / shard["file"], output / shard["file"])
    copy_file(vision_dir / VISION_FILENAME, output / vision_name)
    for name in source_metadata:
        copy_file(source / name, output / name)
    for name in text_metadata:
        copy_file(text_dir / name, output / name)
    return vision_name


def build_index_and_config(output, text_audit, vision_audit, vision_name, output_config):
    weight_map, total_size = {}, 0
    for shard in text_audit["shards"]:
        for tensor in shard["tensors"]:
            weight_map[tensor["key"]] = shard["file"]
            total_size += tensor["bytes"]
    for tensor in vision_audit["shard"]["tensors"]:
        if tensor["key"] in weight_map:
            raise PackagingError(f"Vision/text tensor name collision: {tensor['key']}")
        weight_map[tensor["key"]] = vision_name
        total_size += tensor["bytes"]
    write_json_new(output / "model.safetensors.index.json",
                   {"metadata": {"total_size": total_size}, "weight_map": weight_map})
    write_json_new(output / "config.json", output_config)
    return weight_map, total_size


def reaudit_output(output, weight_map, total_size):
    shards = [inspect_shard(path) for path in collect_shards(output, "output candidate")]
    merged = merge_shard_tensors(shards, "output")
    if set(merged) != set(weight_map):
        raise PackagingError("Output shard coverage differs from generated index")
    for key, entry in merged.items():
        if entry["shard"] != weight_map[key]:
            raise PackagingError(f"Output index shard mismatch: {key}")
    actual_total = sum(entry["bytes"] for entry in merged.values())
    if actual_total != total_size:
        raise PackagingError("Output index total_size differs from actual payload bytes")
    if (output / INCOMPLETE_FILENAME).exists():
        raise PackagingError("Output unexpectedly contains an incomplete marker")
    return shards


def build_report(status, context):
    report = {"status": status, "privacy": "private - contains local filesystem paths"}
    report.update(context)
    return report


def run(source, text_dir, vision_dir, audit_path, mapping_path, output, report,
        allocation_path=None):
    base_context = {"inputs": {"source": str(source), "text_dir": str(text_dir),
                               "vision_dir": str(vision_dir), "source_audit": str(audit_path),
                               "mapping": str(mapping_path) if mapping_path else None,
                               "allocation": str(allocation_path) if allocation_path else None,
                               "output": str(output), "report": str(report)},
                    "mapping_used": mapping_path is not None,
                    "allocation_used": allocation_path is not None, "error": None}
    made_output = False
    try:
        stage = "check paths"
        check_paths(source, text_dir, vision_dir, audit_path, mapping_path, output, report,
                    allocation_path)
        stage = "load inputs"
        source_audit = read_json(audit_path)
        if not isinstance(source_audit, dict):
            raise PackagingError("Source audit is not a JSON object")
        mapping = read_json(mapping_path) if mapping_path else None
        if mapping is not None and not isinstance(mapping, dict):
            raise PackagingError("Mapping is not a JSON object")
        allocation_map = None
        allocation_sha256 = None
        if allocation_path is not None:
            allocation_map = read_json(allocation_path)
            if not isinstance(allocation_map, dict):
                raise PackagingError("Allocation is not a JSON object")
            allocation_sha256 = file_sha256(allocation_path)
        source_config = read_json(source / "config.json")
        text_config = read_json(text_dir / "config.json")
        if not (source / "model.safetensors.index.json").is_file():
            raise PackagingError("Source weight index is missing")
        check_source_config(source_config)
        stage = "audit text candidate"
        expected = build_expected_text(source_audit, mapping, allocation_map)
        text_audit = audit_text_candidate(text_dir, expected, source, source_audit)
        stage = "audit vision pilot"
        vision_audit = audit_vision_pilot(vision_dir, source_audit)
        if set(text_audit["bit_map"]) & set(vision_audit["hashes"]):
            raise PackagingError("Vision/text tensor name collision")
        stage = "plan metadata"
        source_metadata, source_skipped = plan_source_metadata(source)
        text_metadata, text_skipped = plan_text_metadata(text_dir)
        output_config, config_patch = build_output_config(source_config, text_config)
        stage = "copy package files"
        output.mkdir(parents=False, exist_ok=False)
        made_output = True
        vision_name = package_text_vision(source, text_dir, vision_dir, source_audit, text_audit,
                                          vision_audit, source_metadata, text_metadata, output)
        stage = "write index and config"
        weight_map, total_size = build_index_and_config(output, text_audit, vision_audit,
                                                       vision_name, output_config)
        stage = "re-audit output"
        output_shards = reaudit_output(output, weight_map, total_size)
        for shard in text_audit["shards"]:
            if file_sha256(output / shard["file"]) != shard["sha256"]:
                raise PackagingError(f"Copied text shard differs: {shard['file']}")
        if file_sha256(output / vision_name) != vision_audit["shard"]["sha256"]:
            raise PackagingError("Copied vision file differs")
        stage = "write report"
        output_files = []
        for path in sorted(output.iterdir()):
            if path.is_file():
                output_files.append({"file": path.name, "bytes": path.stat().st_size,
                                     "sha256": file_sha256(path)})
        total_package_bytes = sum(entry["bytes"] for entry in output_files)
        quantized_bytes = sum(tensor["bytes"] for shard in text_audit["shards"]
                              for tensor in shard["tensors"]
                              if tensor["key"] not in expected["passthrough"])
        passthrough_bytes = sum(tensor["bytes"] for shard in text_audit["shards"]
                                for tensor in shard["tensors"]
                                if tensor["key"] in expected["passthrough"])
        vision_bytes = vision_audit["shard"]["file_bytes"]
        metadata_bytes = sum(entry["bytes"] for entry in output_files
                             if entry["file"] in set(source_metadata) | set(text_metadata))
        config_index_bytes = sum(entry["bytes"] for entry in output_files
                                 if entry["file"] in ("config.json", "model.safetensors.index.json"))
        total_params = source_audit.get("source_parameters")
        if not isinstance(total_params, int) or total_params <= 0:
            total_params = None
        body_bpw = (text_audit["body_trellis_bytes"] * 8 / text_audit["body_params"]
                    if text_audit["body_params"] else None)
        if allocation_path is not None:
            rate_assumption = (
                "Quantized rates follow the frozen --allocation map (per-projection K1..K8, "
                "lm_head K6) with native mul1 marker 0x83DCD12D; packed geometry/rate checks "
                "and accounting use the frozen map, never inferred candidate bytes."
            )
        else:
            rate_assumption = "Body projections are K4 and lm_head is K6 with native mul1 marker 0x83DCD12D."
        context = dict(base_context)
        context.update({
            "allocation": {
                "file_sha256": allocation_sha256,
                "expected_bits": allocation_map,
            } if allocation_path is not None else None,
            "text": {
                "shards": [{"file": shard["file"], "bytes": shard["file_bytes"],
                            "sha256": shard["sha256"], "tensors": len(shard["tensors"])}
                           for shard in text_audit["shards"]],
                "quantized_projections": len(text_audit["bit_map"]),
                "body_projections": sum(1 for base in text_audit["bit_map"] if base != HEAD_BASE),
                "head_projections": 1 if HEAD_BASE in text_audit["bit_map"] else 0,
                "bit_map": text_audit["bit_map"],
                "body_trellis_stored_bits_per_weight": body_bpw,
                "body_quantized_parameters": text_audit["body_params"],
                "body_trellis_bytes": text_audit["body_trellis_bytes"],
                "passthrough_tensors": len(expected["passthrough"]),
                "dtype_conversions": text_audit["conversions"],
                "embedding": text_audit["embedding"],
            },
            "vision": {
                "source_file": VISION_FILENAME,
                "output_file": vision_name,
                "container_sha256": vision_audit["shard"]["sha256"],
                "tensors_verified": len(vision_audit["hashes"]),
                "payload_hashes_match": True,
                "payload_bytes": vision_audit["shard"]["payload_bytes"],
                "payload_sha256": vision_audit["hashes"],
            },
            "output": {
                "files": output_files,
                "total_package_bytes": total_package_bytes,
                "total_package_bits_per_weight": (total_package_bytes * 8 / total_params
                                                  if total_params else None),
                "total_parameters": total_params,
                "index_total_size": total_size,
                "components": {
                    "text_quantized_bytes": quantized_bytes,
                    "text_passthrough_bytes": passthrough_bytes,
                    "text_container_overhead_bytes": sum(shard["file_bytes"] - shard["payload_bytes"] for shard in text_audit["shards"]),
                    "vision_file_bytes": vision_bytes,
                    "metadata_bytes": metadata_bytes,
                    "config_index_bytes": config_index_bytes,
                },
            },
            "config": {
                "source_config_sha256": file_sha256(source / "config.json"),
                "source_index_sha256": file_sha256(source / "model.safetensors.index.json"),
                "generated_config_sha256": file_sha256(output / "config.json"),
                "patch": config_patch,
            },
            "metadata_copied_from_source": source_metadata,
            "metadata_skipped_from_source": source_skipped,
            "metadata_kept_from_text": text_metadata,
            "metadata_skipped_from_text": text_skipped,
            "assumptions": [
                "Text candidate stores one .trellis/.suh/.svh/.mul1 group per mapping qmap!=None linear.",
                rate_assumption,
                "Unquantized text tensors keep source names with F16 or BF16 dtype; only embedding must be BF16.",
                "No MTP tensors exist; any mtp.* tensor fails packaging.",
                "Vision weights are byte-preserved; no image runtime support is claimed.",
            ],
        })
        write_json_new(report, build_report("complete", context))
        return 0
    except (PackagingError, OSError) as error:
        marker = {"status": "incomplete", "stage": stage, "error": str(error)}
        try:
            if made_output and output.is_dir() and not (output / INCOMPLETE_FILENAME).exists():
                write_json_new(output / INCOMPLETE_FILENAME, marker)
        except OSError:
            pass
        context = dict(base_context)
        context.update({"stage": stage, "error": str(error)})
        try:
            if not report.exists():
                write_json_new(report, build_report("incomplete", context))
        except OSError as write_error:
            print(f"Cannot write report: {write_error}", file=sys.stderr)
        print(f"packaging failed at {stage}: {error}", file=sys.stderr)
        return 1


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original BF16 model directory")
    parser.add_argument("--text-dir", type=Path, required=True, help="Completed compiled text candidate")
    parser.add_argument("--vision-dir", type=Path, required=True, help="Step1 vision preservation pilot")
    parser.add_argument("--source-audit", type=Path, required=True, help="Step1 source-audit.json")
    parser.add_argument("--mapping", type=Path, default=None, help="Optional step1 mapping.json")
    parser.add_argument("--allocation", type=Path, default=None,
                        help="Optional frozen projection bitrate map (JSON base->K1..K8, lm_head K6)")
    parser.add_argument("--output", type=Path, required=True, help="New final candidate directory")
    parser.add_argument("--report", type=Path, required=True, help="New external report file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        resolved = {name: getattr(args, name).resolve() for name in
                    ("source", "text_dir", "vision_dir", "source_audit", "output", "report")}
        mapping = args.mapping.resolve() if args.mapping else None
        allocation_path = args.allocation.resolve() if args.allocation else None
    except OSError as error:
        print(f"Cannot resolve paths: {error}", file=sys.stderr)
        return 1
    return run(resolved["source"], resolved["text_dir"], resolved["vision_dir"],
               resolved["source_audit"], mapping, resolved["output"], resolved["report"],
               allocation_path)


if __name__ == "__main__":
    sys.exit(main())
