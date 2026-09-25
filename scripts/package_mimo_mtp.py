"""Derive a MiMo package with immutable-target donor MTP weights.

Stdlib-only CLI. Copies an existing target package (step-3 b50 smoke
candidate, mtp_num_hidden_layers=0, no mtp.* tensors) into a new directory,
adds Qwen3.5-9B donor MTP payload as BF16 plus license and provenance, flips
only text_config.mtp_num_hidden_layers 0->1, and extends the weight index.
Never modifies inputs. Fails closed before writing output whenever possible.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
               "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
EXPECTED_ARCH = "Qwen3_5ForConditionalGeneration"
DONOR_REPO = "Qwen/Qwen3.5-9B"
DONOR_FILES = ("mtp.safetensors", "config.json", "tokenizer.json", "LICENSE")
OUTPUT_DONOR_FILE = "mtp-donor-bf16.safetensors"
OUTPUT_LICENSE_FILE = "MTP-DONOR-LICENSE"
OUTPUT_PROVENANCE_FILE = "mtp-donor.json"
TARGET_CONFIG = "config.json"
TARGET_INDEX = "model.safetensors.index.json"
TARGET_TOKENIZER = "tokenizer.json"
HEADER_SIZE_CAP = 100 * 1024**2
HASH_CHUNK = 8 * 1024**2
SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
REV_RE = re.compile(r"^[0-9a-fA-F]{40}$")
MIMO_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
QWEN_TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
MIMO_EXTRA_TOKENS = dict(enumerate((
    "<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>",
    "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>",
), start=248070))

CANONICAL_MTP = (
    "mtp.fc.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
)
GEOMETRY_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "rms_norm_eps",
    "full_attention_interval",
    "layer_types",
    "attn_output_gate",
    "attention_bias",
    "hidden_act",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "rope_parameters",
)
REQUIRED_GEOMETRY = frozenset({
    "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "vocab_size",
    "rms_norm_eps", "hidden_act",
})
_MISSING = object()


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
        raise PackagingError(f"Invalid JSON in {Path(path).name}: {error}")


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


def within_or_equal(path, other):
    return path == other or other in path.parents


def is_safe_leaf(name):
    if not isinstance(name, str) or not name:
        return False
    if "\0" in name or "/" in name or "\\" in name or ":" in name:
        return False
    if name in (".", ".."):
        return False
    if any(ord(c) < 32 for c in name):
        return False
    return True


def check_paths(target, manifest_path, donor_dir, output, report):
    if not target.is_dir():
        raise PackagingError(f"--target is not a directory: {target}")
    if not donor_dir.is_dir():
        raise PackagingError(f"--donor-directory is not a directory: {donor_dir}")
    if not manifest_path.is_file():
        raise PackagingError(f"--target-manifest is not a file: {manifest_path}")
    pairs = (("--target", target, "--donor-directory", donor_dir),
             ("--target", target, "--output", output),
             ("--donor-directory", donor_dir, "--output", output))
    for aname, apath, bname, bpath in pairs:
        if within_or_equal(apath, bpath) or within_or_equal(bpath, apath):
            raise PackagingError(f"{aname} and {bname} must be separate, non-nested directories")
    for name, path in (("--target", target), ("--donor-directory", donor_dir)):
        if within_or_equal(report, path):
            raise PackagingError(f"--report must be outside {name}")
        if within_or_equal(output, path) or within_or_equal(path, output):
            raise PackagingError(f"--output must be separate from and not nested with {name}")
    if within_or_equal(report, output) or within_or_equal(output, report):
        raise PackagingError("--report must be outside the --output candidate")
    if within_or_equal(manifest_path, output):
        raise PackagingError("--target-manifest must be outside --output")
    if within_or_equal(report, donor_dir):
        raise PackagingError("--report must be outside --donor-directory")
    if os.path.lexists(report) or os.path.lexists(output):
        which = "--report" if os.path.lexists(report) else "--output"
        raise PackagingError(f"{which} already exists; refusing to overwrite")
    if not report.parent.is_dir():
        raise PackagingError(f"--report parent directory does not exist: {report.parent}")
    for path in (target, donor_dir):
        if path.is_symlink():
            raise PackagingError(f"Input directory is a symlink: {path}")
    if manifest_path.is_symlink():
        raise PackagingError("Input manifest is a symlink")


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


def merge_shard_tensors(shards, label):
    merged = {}
    for shard in shards:
        for tensor in shard["tensors"]:
            key = tensor["key"]
            if key in merged:
                raise PackagingError(f"Duplicate tensor across {label} shards: {key}")
            merged[key] = {"shard": shard["file"], **tensor}
    return merged


def _bf16_chunk_finite(chunk):
    if len(chunk) % 2:
        raise PackagingError("Odd-length BF16 payload chunk")
    if not chunk:
        return True
    high = chunk[1::2]
    if b"\x7f" not in high and b"\xff" not in high:
        return True
    low = chunk[0::2]
    for target in (b"\x7f", b"\xff"):
        start = 0
        while True:
            found = high.find(target, start)
            if found < 0:
                break
            if low[found] >= 0x80:
                return False
            start = found + 1
    return True


def check_bf16_finite(path, header_size, payload_size):
    with open(path, "rb") as stream:
        stream.seek(8 + header_size)
        remaining = payload_size
        while remaining:
            chunk = stream.read(min(HASH_CHUNK, remaining))
            if not chunk:
                raise PackagingError(f"Truncated payload in {Path(path).name}")
            if not _bf16_chunk_finite(chunk):
                raise PackagingError(f"Non-finite BF16 value in {Path(path).name}")
            remaining -= len(chunk)


def validate_target_manifest(path):
    data = read_json(path)
    if not isinstance(data, list) or not data:
        raise PackagingError("Target manifest must be a non-empty array")
    result = {}
    for entry in data:
        if not isinstance(entry, dict):
            raise PackagingError("Target manifest entry is not an object")
        fname, fbytes, fsha = entry.get("file"), entry.get("bytes"), entry.get("sha256")
        if not is_safe_leaf(fname):
            raise PackagingError(f"Unsafe target file name: {fname!r}")
        if fname in result:
            raise PackagingError(f"Duplicate target file name: {fname}")
        if type(fbytes) is not int or fbytes < 0:
            raise PackagingError(f"Malformed bytes for target file: {fname}")
        if not isinstance(fsha, str) or not SHA_RE.fullmatch(fsha):
            raise PackagingError(f"Malformed sha256 for target file: {fname}")
        result[fname] = {"bytes": fbytes, "sha256": fsha.lower()}
    return result


def validate_target_inventory(target, manifest):
    actual = {}
    for child in sorted(target.iterdir()):
        if child.is_symlink():
            raise PackagingError(f"Target contains a symlink: {child.name}")
        if child.is_dir():
            raise PackagingError(f"Target contains an unexpected subdirectory: {child.name}")
        if child.is_file():
            if not is_safe_leaf(child.name):
                raise PackagingError(f"Unsafe target file name: {child.name!r}")
            actual[child.name] = child
    missing = sorted(set(manifest) - set(actual))
    extra = sorted(set(actual) - set(manifest))
    if missing:
        raise PackagingError(f"Target is missing {len(missing)} files: {missing[:8]}")
    if extra:
        raise PackagingError(f"Target has {len(extra)} unexpected files: {extra[:8]}")
    for fname, path in actual.items():
        expected = manifest[fname]
        if path.stat().st_size != expected["bytes"]:
            raise PackagingError(f"Target file size mismatch: {fname}")
        if file_sha256(path) != expected["sha256"]:
            raise PackagingError(f"Target file hash mismatch: {fname}")
    return actual


def check_target_config(config):
    if not isinstance(config, dict):
        raise PackagingError("Target config is not a JSON object")
    archs = config.get("architectures")
    if not isinstance(archs, list) or EXPECTED_ARCH not in archs:
        raise PackagingError(f"Target architectures must include {EXPECTED_ARCH}")
    for arch in archs:
        if isinstance(arch, str) and "moe" in arch.lower():
            raise PackagingError(f"Target must be dense text, found MoE architecture: {arch}")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise PackagingError("Target config has no text_config object")
    mtp = text.get("mtp_num_hidden_layers")
    if type(mtp) is not int or mtp != 0:
        raise PackagingError(f"Target text_config.mtp_num_hidden_layers must be 0, found {mtp!r}")
    for key in ("num_experts", "moe_intermediate_size", "num_experts_per_tok"):
        if key in text:
            raise PackagingError(f"Target must be dense text, found MoE field: {key}")
    return text


def load_target_index(target):
    index = read_json(target / TARGET_INDEX)
    if not isinstance(index, dict):
        raise PackagingError("Target index is not a JSON object")
    metadata, weight_map = index.get("metadata"), index.get("weight_map")
    if not isinstance(metadata, dict) or type(metadata.get("total_size")) is not int:
        raise PackagingError("Target index metadata.total_size is malformed")
    if metadata["total_size"] < 0:
        raise PackagingError("Target index metadata.total_size is malformed")
    if not isinstance(weight_map, dict):
        raise PackagingError("Target index weight_map is malformed")
    for key, value in weight_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise PackagingError("Target index weight_map entries must be strings")
    return index


def validate_target_shards(target, index):
    shard_paths = []
    for path in sorted(target.iterdir()):
        if path.is_file() and path.suffix == ".safetensors":
            if path.is_symlink():
                raise PackagingError(f"Target contains a symlink: {path.name}")
            shard_paths.append(path)
    if not shard_paths:
        raise PackagingError("Target has no top-level safetensors shards")
    shards = [inspect_shard(path) for path in shard_paths]
    merged = merge_shard_tensors(shards, "target")
    mtp_keys = sorted(key for key in merged if key == "mtp" or key.startswith("mtp."))
    if mtp_keys:
        raise PackagingError(f"Target contains MTP tensors: {mtp_keys[:5]}")
    weight_map = index["weight_map"]
    missing = sorted(set(weight_map) - set(merged))
    unexpected = sorted(set(merged) - set(weight_map))
    if missing:
        raise PackagingError(f"Target index lists {len(missing)} missing tensors: {missing[:8]}")
    if unexpected:
        raise PackagingError(f"Target shards have {len(unexpected)} tensors missing from index: {unexpected[:8]}")
    for key, entry in merged.items():
        if entry["shard"] != weight_map[key]:
            raise PackagingError(f"Target index shard mismatch: {key}")
    actual_total = sum(entry["bytes"] for entry in merged.values())
    if actual_total != index["metadata"]["total_size"]:
        raise PackagingError("Target index total_size differs from actual payload bytes")
    return shards, merged, actual_total


def validate_donor_manifest(path):
    data = read_json(path)
    if not isinstance(data, dict):
        raise PackagingError("Donor manifest is not a JSON object")
    version = data.get("schema_version")
    if type(version) is not int or version != 1:
        raise PackagingError(f"Donor schema_version must be 1, found {version!r}")
    if data.get("complete") is not True:
        raise PackagingError("Donor manifest complete must be true")
    if data.get("repository") != DONOR_REPO:
        raise PackagingError(f"Donor repository must be {DONOR_REPO}")
    revision = data.get("revision")
    if not isinstance(revision, str) or not REV_RE.fullmatch(revision):
        raise PackagingError("Donor revision must be 40 hex characters")
    files = data.get("files")
    if not isinstance(files, dict) or set(files) != set(DONOR_FILES):
        raise PackagingError(f"Donor files must map exactly {sorted(DONOR_FILES)}")
    for fname in DONOR_FILES:
        entry = files.get(fname)
        if not isinstance(entry, dict):
            raise PackagingError(f"Donor file entry is malformed: {fname}")
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise PackagingError(f"Donor file bytes malformed: {fname}")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
            raise PackagingError(f"Donor file sha256 malformed: {fname}")
    tensors = data.get("tensors")
    if not isinstance(tensors, dict) or set(tensors) != set(CANONICAL_MTP):
        raise PackagingError("Donor tensors must map exactly the 15 canonical mtp.* names")
    for name in CANONICAL_MTP:
        entry = tensors.get(name)
        if not isinstance(entry, dict):
            raise PackagingError(f"Donor tensor entry is malformed: {name}")
        if entry.get("dtype") != "BF16":
            raise PackagingError(f"Donor manifest dtype must be BF16 for {name}")
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise PackagingError(f"Donor manifest shape malformed: {name}")
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise PackagingError(f"Donor manifest bytes malformed: {name}")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
            raise PackagingError(f"Donor manifest sha256 malformed: {name}")
    return data


def validate_donor_files(donor_dir, manifest):
    info = {}
    for fname in DONOR_FILES:
        path = donor_dir / fname
        if path.is_symlink() or not path.is_file():
            raise PackagingError(f"Donor file is missing: {fname}")
        expected = manifest["files"][fname]
        if path.stat().st_size != expected["bytes"]:
            raise PackagingError(f"Donor file size mismatch: {fname}")
        actual = file_sha256(path)
        if actual != expected["sha256"].lower():
            raise PackagingError(f"Donor file hash mismatch: {fname}")
        info[fname] = {"bytes": expected["bytes"], "sha256": actual}
    return info


def load_donor_text(donor_dir):
    config = read_json(donor_dir / "config.json")
    if not isinstance(config, dict):
        raise PackagingError("Donor config is not a JSON object")
    text = config.get("text_config")
    if not isinstance(text, dict):
        text = config
    mtp = text.get("mtp_num_hidden_layers")
    if type(mtp) is not int or mtp != 1:
        raise PackagingError(f"Donor MTP depth must be exactly 1, found {mtp!r}")
    return config, text


def _effective_head_dim(text, label):
    raw = text.get("head_dim", None)
    if raw is None:
        hidden = text.get("hidden_size")
        heads = text.get("num_attention_heads")
        if type(hidden) is not int or type(heads) is not int or hidden <= 0 or heads <= 0:
            raise PackagingError(f"{label} cannot derive head_dim without hidden_size/num_attention_heads")
        if hidden % heads:
            raise PackagingError(f"{label} hidden_size is not divisible by num_attention_heads")
        return hidden // heads
    if type(raw) is not int or raw <= 0:
        raise PackagingError(f"{label} head_dim must be a positive int")
    return raw


def compare_geometry(target_text, donor_text):
    for field in GEOMETRY_FIELDS:
        target_value = target_text.get(field, _MISSING)
        donor_value = donor_text.get(field, _MISSING)
        if target_value is _MISSING and donor_value is _MISSING:
            if field in REQUIRED_GEOMETRY:
                raise PackagingError(f"Geometry field is missing from target and donor: {field}")
            continue
        if target_value is _MISSING or donor_value is _MISSING:
            raise PackagingError(f"Geometry mismatch for {field}: present on one side only")
        if target_value != donor_value:
            raise PackagingError(f"Geometry mismatch for {field}: target={target_value!r} donor={donor_value!r}")
    target_head = _effective_head_dim(target_text, "Target")
    donor_head = _effective_head_dim(donor_text, "Donor")
    if target_head != donor_head:
        raise PackagingError(f"Geometry mismatch for head_dim: target={target_head!r} donor={donor_head!r}")
    target_dedicated = target_text.get("mtp_use_dedicated_embeddings", False)
    donor_dedicated = donor_text.get("mtp_use_dedicated_embeddings", False)
    if target_dedicated is not False:
        raise PackagingError("Target mtp_use_dedicated_embeddings must be false")
    if donor_dedicated is not False:
        raise PackagingError("Donor mtp_use_dedicated_embeddings must be false")
    for field in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"):
        value = target_text.get(field)
        if type(value) is not int or value <= 0:
            raise PackagingError(f"Geometry field must be a positive int: {field}")
    return target_head


def compare_tokenizers(target_path, donor_path):
    if not target_path.is_file():
        raise PackagingError("Target tokenizer.json is missing")
    target_tok = read_json(target_path)
    donor_tok = read_json(donor_path)
    if not isinstance(target_tok, dict) or not isinstance(donor_tok, dict):
        raise PackagingError("Tokenizer files must be JSON objects")
    # Most donors must match exactly. The one audited MiMo/Qwen pair differs
    # in preprocessing and seven target-only special tokens, but MTP consumes
    # target IDs through the shared target embedding/head, never a donor tokenizer.
    # Pin both complete files: this is not a generic ignore-tokenizer switch.
    if target_tok != donor_tok and (file_sha256(target_path), file_sha256(donor_path)) == (
            MIMO_TOKENIZER_SHA256, QWEN_TOKENIZER_SHA256):
        target_model, donor_model = target_tok['model'], donor_tok['model']
        if target_model['vocab'] != donor_model['vocab']:
            raise PackagingError("Tokenizer mismatch: shared vocabulary IDs differ")
        def merge_pairs(rows):
            pairs = [row.split(' ') if isinstance(row, str) else row for row in rows]
            if any(not isinstance(row, list) or len(row) != 2
                   or not all(isinstance(x, str) for x in row) for row in pairs):
                raise PackagingError("Tokenizer mismatch: invalid BPE merge pair")
            return pairs
        if merge_pairs(target_model['merges']) != merge_pairs(donor_model['merges']):
            raise PackagingError("Tokenizer mismatch: BPE merge order/content differs")
        target_added = {row['id']: row for row in target_tok['added_tokens']}
        donor_added = {row['id']: row for row in donor_tok['added_tokens']}
        if any(target_added.get(key) != row for key, row in donor_added.items()):
            raise PackagingError("Tokenizer mismatch: shared added-token definitions differ")
        extra = {key: row['content'] for key, row in target_added.items() if key not in donor_added}
        if extra != MIMO_EXTRA_TOKENS:
            raise PackagingError("Tokenizer mismatch: unexpected target-only tokens")
        return dict(mode="pinned_mimo_shared_target", json_semantically_equal=False,
                    shared_vocabulary_size=len(target_model['vocab']), shared_added_tokens=len(donor_added),
                    target_only_added_tokens=extra, normalized_merge_pairs_equal=True,
                    tokenizer_used="target", target_sha256=MIMO_TOKENIZER_SHA256,
                    donor_sha256=QWEN_TOKENIZER_SHA256,
                    limit="Donor preprocessing differs and is never executed by this MTP path; audio behavior unvalidated.")
    if target_tok != donor_tok:
        target_keys, donor_keys = set(target_tok), set(donor_tok)
        parts = []
        only_target = sorted(target_keys - donor_keys)
        only_donor = sorted(donor_keys - target_keys)
        differing = sorted(key for key in target_keys & donor_keys if target_tok[key] != donor_tok[key])
        if only_target:
            parts.append(f"only in target: {only_target[:5]}")
        if only_donor:
            parts.append(f"only in donor: {only_donor[:5]}")
        if differing:
            parts.append(f"differing: {differing[:8]}")
        detail = "; ".join(parts) if parts else "content differs"
        raise PackagingError(f"Tokenizer mismatch between target and donor: {detail}")
    return dict(mode="exact_json", json_semantically_equal=True, tokenizer_used="target")


def expected_mtp_shapes(text, head_dim):
    hidden = text["hidden_size"]
    intermediate = text["intermediate_size"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    return {
        "mtp.fc.weight": [hidden, 2 * hidden],
        "mtp.pre_fc_norm_hidden.weight": [hidden],
        "mtp.pre_fc_norm_embedding.weight": [hidden],
        "mtp.norm.weight": [hidden],
        "mtp.layers.0.input_layernorm.weight": [hidden],
        "mtp.layers.0.post_attention_layernorm.weight": [hidden],
        "mtp.layers.0.self_attn.q_proj.weight": [2 * heads * head_dim, hidden],
        "mtp.layers.0.self_attn.k_proj.weight": [kv_heads * head_dim, hidden],
        "mtp.layers.0.self_attn.v_proj.weight": [kv_heads * head_dim, hidden],
        "mtp.layers.0.self_attn.o_proj.weight": [hidden, heads * head_dim],
        "mtp.layers.0.self_attn.q_norm.weight": [head_dim],
        "mtp.layers.0.self_attn.k_norm.weight": [head_dim],
        "mtp.layers.0.mlp.gate_proj.weight": [intermediate, hidden],
        "mtp.layers.0.mlp.up_proj.weight": [intermediate, hidden],
        "mtp.layers.0.mlp.down_proj.weight": [hidden, intermediate],
    }


def validate_donor_tensors(donor_file, manifest_tensors, expected_shapes):
    shard = inspect_shard(donor_file)
    merged = {tensor["key"]: tensor for tensor in shard["tensors"]}
    missing = sorted(set(expected_shapes) - set(merged))
    unexpected = sorted(set(merged) - set(expected_shapes))
    if missing:
        raise PackagingError(f"Donor is missing {len(missing)} tensors: {missing[:8]}")
    if unexpected:
        raise PackagingError(f"Donor has {len(unexpected)} unexpected tensors: {unexpected[:8]}")
    hashes = {}
    total = 0
    for name in sorted(expected_shapes):
        entry = merged[name]
        shape = expected_shapes[name]
        if entry["dtype"] != "BF16":
            raise PackagingError(f"Donor tensor dtype must be BF16 for {name}: {entry['dtype']}")
        if entry["shape"] != shape:
            raise PackagingError(f"Donor shape mismatch for {name}: expected {shape} found {entry['shape']}")
        expected_bytes = math.prod(shape) * 2
        if entry["bytes"] != expected_bytes:
            raise PackagingError(f"Donor bytes mismatch for {name}")
        manifest_entry = manifest_tensors[name]
        if manifest_entry.get("shape") != shape or manifest_entry.get("bytes") != expected_bytes:
            raise PackagingError(f"Donor manifest shape/bytes mismatch for {name}")
        payload_hash = stream_payload_hash(donor_file, shard["header_size"], *entry["data_offsets"])
        if payload_hash != manifest_entry["sha256"].lower():
            raise PackagingError(f"Donor payload hash mismatch for {name}")
        hashes[name] = payload_hash
        total += entry["bytes"]
    check_bf16_finite(donor_file, shard["header_size"], shard["payload_bytes"])
    return shard, hashes, total


def check_collisions(target_files):
    folded = {name.casefold() for name in target_files}
    for name in (OUTPUT_DONOR_FILE, OUTPUT_LICENSE_FILE, OUTPUT_PROVENANCE_FILE):
        if name.casefold() in folded:
            raise PackagingError(f"Output name collision with target file: {name}")


def build_output_config(target_config):
    output_config = json.loads(json.dumps(target_config))
    text = output_config.get("text_config")
    if not isinstance(text, dict) or text.get("mtp_num_hidden_layers") != 0:
        raise PackagingError("Target config cannot supply the 0->1 MTP patch")
    text["mtp_num_hidden_layers"] = 1
    patch = {"text_config.mtp_num_hidden_layers": {"from": 0, "to": 1}}
    return output_config, patch


def build_output_index(target_index, expected_shapes, target_total, donor_total):
    old_map = target_index["weight_map"]
    for name in expected_shapes:
        if name in old_map:
            raise PackagingError(f"Donor/target tensor name collision: {name}")
    new_index = json.loads(json.dumps(target_index))
    new_map = new_index["weight_map"]
    for name in expected_shapes:
        new_map[name] = OUTPUT_DONOR_FILE
    new_total = target_total + donor_total
    new_index["metadata"]["total_size"] = new_total
    change = {"added_tensors": sorted(expected_shapes), "added_file": OUTPUT_DONOR_FILE,
              "previous_total_size": target_total, "new_total_size": new_total}
    return new_index, change


def build_provenance(donor_manifest, donor_files, donor_hashes, expected_shapes,
                     target_manifest, target_config_hash, target_index_hash, target_tokenizer_hash,
                     tokenizer_compatibility):
    files = {name: {"bytes": donor_files[name]["bytes"], "sha256": donor_files[name]["sha256"]}
             for name in DONOR_FILES}
    tensors = {name: {"dtype": "BF16", "shape": expected_shapes[name],
                      "bytes": math.prod(expected_shapes[name]) * 2, "sha256": donor_hashes[name]}
               for name in sorted(expected_shapes)}
    target_files = sorted(({"file": name, "bytes": entry["bytes"], "sha256": entry["sha256"]}
                           for name, entry in target_manifest.items()), key=lambda e: e["file"])
    return {
        "schema_version": 1,
        "donor": {"repository": DONOR_REPO, "revision": donor_manifest["revision"],
                  "schema_version": 1, "files": files, "tensors": tensors},
        "output": {"tensors_file": OUTPUT_DONOR_FILE, "license_file": OUTPUT_LICENSE_FILE,
                   "tensor_count": len(expected_shapes), "storage_dtype": "BF16"},
        "runtime": {"projections_load_dtype": "FP16", "norm_constant_bias": 1.0,
                    "norms_adjusted_in_storage": False,
                    "note": ("BF16 storage preserved byte-exact; runtime loads ordinary "
                             "projections as FP16 and applies Qwen constant_bias=1; no +1 "
                             "baked into stored norms, no cast/quantize.")},
        "sharing": {"mtp_use_dedicated_embeddings": False,
                    "embedding_source": "target", "lm_head_source": "target"},
        "tokenizer_compatibility": tokenizer_compatibility,
        "target_preservation": {"target_files": target_files, "config_sha256": target_config_hash,
                                "index_sha256": target_index_hash,
                                "tokenizer_sha256": target_tokenizer_hash},
        "claim_limits": [
            "CPU packaging/audit only; no runtime/GPU reload claimed.",
            "Target tensors and quantization metadata preserved byte-exact except recorded config/index deltas.",
            "Donor MTP payload stored as BF16; runtime precision (FP16 projections, constant_bias=1) recorded separately.",
            "Tokenizer compatibility mode recorded explicitly; target tokenizer_config/chat_template/generation_config preserved unchanged.",
            "Text geometry equality verified for listed fields; MTP depth changed 0->1 by design.",
        ],
    }


def write_json_new(path, payload):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def copy_file(source, destination):
    with open(source, "rb") as reader, open(destination, "xb") as writer:
        shutil.copyfileobj(reader, writer, length=HASH_CHUNK)


def reverify_output(output, target, target_manifest, target_config, target_index,
                    donor_files, expected_shapes):
    expected_files = set(target_manifest) | {OUTPUT_DONOR_FILE, OUTPUT_LICENSE_FILE,
                                             OUTPUT_PROVENANCE_FILE}
    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise PackagingError("Output file inventory differs from target plus donor additions")
    for path in output.iterdir():
        if path.is_symlink():
            raise PackagingError(f"Output contains a symlink: {path.name}")
    for fname, entry in target_manifest.items():
        if file_sha256(target / fname) != entry["sha256"]:
            raise PackagingError(f"Target input changed during packaging: {fname}")
        if fname in (TARGET_CONFIG, TARGET_INDEX):
            continue
        out_path = output / fname
        if not out_path.is_file():
            raise PackagingError(f"Output is missing preserved file: {fname}")
        if file_sha256(out_path) != entry["sha256"]:
            raise PackagingError(f"Copied target file differs: {fname}")
    output_config = read_json(output / TARGET_CONFIG)
    expected_config = json.loads(json.dumps(target_config))
    expected_config["text_config"]["mtp_num_hidden_layers"] = 1
    if output_config != expected_config:
        raise PackagingError("Output config differs beyond mtp_num_hidden_layers 0->1")
    output_index = read_json(output / TARGET_INDEX)
    if not isinstance(output_index, dict) or not isinstance(output_index.get("weight_map"), dict):
        raise PackagingError("Output index is malformed")
    old_map = target_index["weight_map"]
    new_map = output_index["weight_map"]
    if set(new_map) != set(old_map) | set(expected_shapes):
        raise PackagingError("Output index coverage differs from target plus 15 donor tensors")
    for key, value in old_map.items():
        if new_map.get(key) != value:
            raise PackagingError(f"Output index preserved entry changed: {key}")
    for name in expected_shapes:
        if new_map.get(name) != OUTPUT_DONOR_FILE:
            raise PackagingError(f"Output index donor shard mismatch: {name}")
    if set(output_index) != set(target_index):
        raise PackagingError("Output index top-level fields differ from target index")
    for key, value in target_index.items():
        if key in ("metadata", "weight_map"):
            continue
        if output_index.get(key) != value:
            raise PackagingError(f"Output index field changed: {key}")
    target_metadata = target_index.get("metadata", {})
    output_metadata = output_index.get("metadata", {})
    if not isinstance(output_metadata, dict) or set(output_metadata) != set(target_metadata):
        raise PackagingError("Output index metadata fields differ from target index")
    for key, value in target_metadata.items():
        if key == "total_size":
            continue
        if output_metadata.get(key) != value:
            raise PackagingError(f"Output index metadata changed: {key}")
    shards = [inspect_shard(path) for path in sorted(output.iterdir())
              if path.is_file() and path.suffix == ".safetensors"]
    merged = merge_shard_tensors(shards, "output")
    if set(merged) != set(new_map):
        raise PackagingError("Output shard coverage differs from generated index")
    for key, entry in merged.items():
        if entry["shard"] != new_map[key]:
            raise PackagingError(f"Output index shard mismatch: {key}")
    actual_total = sum(entry["bytes"] for entry in merged.values())
    if output_metadata.get("total_size") != actual_total:
        raise PackagingError("Output index total_size differs from actual payload bytes")
    if file_sha256(output / OUTPUT_DONOR_FILE) != donor_files["mtp.safetensors"]["sha256"]:
        raise PackagingError("Copied donor payload differs")
    if file_sha256(output / OUTPUT_LICENSE_FILE) != donor_files["LICENSE"]["sha256"]:
        raise PackagingError("Copied donor license differs")
    provenance = read_json(output / OUTPUT_PROVENANCE_FILE)
    if provenance.get("donor", {}).get("repository") != DONOR_REPO:
        raise PackagingError("Provenance donor repository mismatch")
    return shards, output_index, actual_total


def run(target, manifest_path, donor_dir, output, report):
    base_inputs = {"target": str(target), "target_manifest": str(manifest_path),
                   "donor_directory": str(donor_dir), "output": str(output), "report": str(report)}
    paths_validated = False
    stage = "check paths"
    try:
        check_paths(target, manifest_path, donor_dir, output, report)
        paths_validated = True
        stage = "validate target inventory"
        target_manifest = validate_target_manifest(manifest_path)
        validate_target_inventory(target, target_manifest)
        if TARGET_CONFIG not in target_manifest or TARGET_INDEX not in target_manifest:
            raise PackagingError("Target manifest must list config.json and model.safetensors.index.json")
        stage = "validate target package"
        target_config = read_json(target / TARGET_CONFIG)
        target_text = check_target_config(target_config)
        target_index = load_target_index(target)
        _, _, target_total = validate_target_shards(target, target_index)
        stage = "validate donor manifest"
        donor_manifest = validate_donor_manifest(donor_dir / "manifest.json")
        donor_files = validate_donor_files(donor_dir, donor_manifest)
        _, donor_text = load_donor_text(donor_dir)
        stage = "compare geometry and tokenizer"
        head_dim = compare_geometry(target_text, donor_text)
        tokenizer_compatibility = compare_tokenizers(target / TARGET_TOKENIZER, donor_dir / "tokenizer.json")
        expected_shapes = expected_mtp_shapes(target_text, head_dim)
        if set(expected_shapes) != set(CANONICAL_MTP):
            raise PackagingError("Internal canonical MTP inventory mismatch")
        stage = "validate donor tensors"
        _, donor_hashes, donor_total = validate_donor_tensors(
            donor_dir / "mtp.safetensors", donor_manifest["tensors"], expected_shapes)
        check_collisions(set(target_manifest))
        output_config, config_patch = build_output_config(target_config)
        output_index, index_change = build_output_index(target_index, expected_shapes,
                                                       target_total, donor_total)
        provenance = build_provenance(donor_manifest, donor_files, donor_hashes, expected_shapes,
                                      target_manifest, file_sha256(target / TARGET_CONFIG),
                                      file_sha256(target / TARGET_INDEX),
                                      file_sha256(target / TARGET_TOKENIZER), tokenizer_compatibility)
        stage = "copy package files"
        output.mkdir(parents=False, exist_ok=False)
        for fname in sorted(target_manifest):
            if fname in (TARGET_CONFIG, TARGET_INDEX):
                continue
            copy_file(target / fname, output / fname)
        copy_file(donor_dir / "mtp.safetensors", output / OUTPUT_DONOR_FILE)
        copy_file(donor_dir / "LICENSE", output / OUTPUT_LICENSE_FILE)
        stage = "write index, config and provenance"
        write_json_new(output / TARGET_CONFIG, output_config)
        write_json_new(output / TARGET_INDEX, output_index)
        write_json_new(output / OUTPUT_PROVENANCE_FILE, provenance)
        stage = "re-verify output"
        output_shards, verified_index, verified_total = reverify_output(
            output, target, target_manifest, target_config, target_index,
            donor_files, expected_shapes)
        stage = "write report"
        output_files = []
        for path in sorted(output.iterdir()):
            if path.is_file():
                output_files.append({"file": path.name, "bytes": path.stat().st_size,
                                     "sha256": file_sha256(path)})
        total_bytes = sum(entry["bytes"] for entry in output_files)
        payload_hashes = {name: donor_hashes[name] for name in sorted(donor_hashes)}
        context = {
            "inputs": base_inputs,
            "target": {"files": sorted(({"file": name, **entry}
                                        for name, entry in target_manifest.items()),
                                       key=lambda e: e["file"]),
                       "config_sha256": file_sha256(target / TARGET_CONFIG),
                       "index_sha256": file_sha256(target / TARGET_INDEX),
                       "tokenizer_sha256": file_sha256(target / TARGET_TOKENIZER),
                       "preserved": True, "tensor_count": len(verified_index["weight_map"]) - 15,
                       "payload_bytes": target_total},
            "donor": {"repository": DONOR_REPO, "revision": donor_manifest["revision"],
                      "files": {name: donor_files[name] for name in DONOR_FILES},
                      "tensors": {name: {"dtype": "BF16", "shape": expected_shapes[name],
                                         "bytes": math.prod(expected_shapes[name]) * 2,
                                         "sha256": payload_hashes[name]}
                                  for name in sorted(payload_hashes)},
                      "tensor_count": 15, "payload_bytes": donor_total},
            "output": {"files": output_files, "total_package_bytes": total_bytes,
                       "index_total_size": verified_total,
                       "shard_count": len(output_shards)},
            "config_change": config_patch,
            "tokenizer_compatibility": tokenizer_compatibility,
            "index_change": index_change,
            "precision": {"storage_dtype": "BF16", "runtime_projections_dtype": "FP16",
                          "norm_constant_bias": 1.0, "norms_adjusted_in_storage": False,
                          "note": ("BF16 storage preserved byte-exact; runtime loads ordinary "
                                   "projections as FP16 and applies constant_bias=1.")},
            "claim_limits": list(provenance["claim_limits"]),
            "error": None,
        }
        if os.path.lexists(report):
            raise PackagingError("--report already exists; refusing to overwrite")
        write_json_new(report, {"complete": True, "status": "complete", **context})
        return 0
    except (PackagingError, OSError) as error:
        context = {"complete": False, "status": "incomplete", "inputs": base_inputs,
                   "stage": stage, "error": str(error)}
        if paths_validated:
            try:
                if not os.path.lexists(report):
                    write_json_new(report, context)
            except OSError as write_error:
                print(f"Cannot write report: {write_error}", file=sys.stderr)
        print(f"packaging failed at {stage}: {error}", file=sys.stderr)
        return 1


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True,
                        help="Target package directory (mtp_num_hidden_layers=0, no mtp.* tensors)")
    parser.add_argument("--target-manifest", type=Path, required=True,
                        help="Step-3 b50 JSON array of {file, bytes, sha256} for exact target inventory")
    parser.add_argument("--donor-directory", type=Path, required=True,
                        help="Donor directory with manifest.json, mtp.safetensors, config.json, tokenizer.json, LICENSE")
    parser.add_argument("--output", type=Path, required=True,
                        help="New derived package directory (must not exist)")
    parser.add_argument("--report", type=Path, required=True,
                        help="New external JSON report file (must not exist, outside inputs/output)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        for name in ("target", "target_manifest", "donor_directory", "output", "report"):
            if getattr(args, name).is_symlink():
                raise PackagingError(f"--{name.replace('_', '-')} must not be a symlink")
        resolved = {name: getattr(args, name).resolve() for name in
                    ("target", "target_manifest", "donor_directory", "output", "report")}
    except (PackagingError, OSError) as error:
        print(f"Cannot resolve paths: {error}", file=sys.stderr)
        return 1
    return run(resolved["target"], resolved["target_manifest"], resolved["donor_directory"],
               resolved["output"], resolved["report"])


if __name__ == "__main__":
    sys.exit(main())
