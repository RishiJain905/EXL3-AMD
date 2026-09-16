"""CPU-only EXL3 storage audit; does not establish model completeness or fidelity."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
               "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
LIMIT = 10_000_000_000
TRELLIS_SUFFIX = ".trellis"
MARKER_SUFFIXES = (".mcg", ".mul1")
MARKER_DTYPE = "I32"
MIN_BITS_PER_WEIGHT = 1
MAX_BITS_PER_WEIGHT = 8


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def category(key):
    if key.startswith("mtp."):
        return "mtp"
    if key.startswith("model.visual."):
        return "unknown"
    if key.startswith(("model.", "lm_head.")):
        return "target"
    return "unknown"


def inspect_shard(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path.name}")
        header_size, = struct.unpack("<Q", prefix)
        if not 0 < header_size <= min(100 * 1024**2, size - 8):
            raise ValueError(f"Invalid header length: {path.name}")
        header = json.loads(stream.read(header_size), object_pairs_hook=unique_object)
        if not isinstance(header, dict):
            raise ValueError("Safetensors header must be an object")
        payload_size = size - 8 - header_size
        tensors = []
        for key, entry in header.items():
            if key == "__metadata__":
                if not isinstance(entry, dict) or any(not isinstance(v, str) for v in entry.values()):
                    raise ValueError("Safetensors metadata must contain string values")
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid tensor declaration: {key}")
            shape, dtype, offsets = entry.get("shape"), entry.get("dtype"), entry.get("data_offsets")
            if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(dtype, str) or dtype not in DTYPE_BYTES):
                raise ValueError(f"Invalid shape or unsupported dtype: {key}")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)):
                raise ValueError(f"Invalid offsets: {key}")
            begin, end = offsets
            expected = math.prod(shape) * DTYPE_BYTES[dtype]
            if not 0 <= begin <= end <= payload_size or end - begin != expected:
                raise ValueError(f"Out-of-bounds or incorrect payload size: {key}")
            tensors.append({"key": key, "shape": shape, "dtype": dtype,
                            "data_offsets": offsets, "bytes": expected, "category": category(key)})
        cursor = 0
        for tensor in sorted(tensors, key=lambda t: tuple(t["data_offsets"])):
            begin, end = tensor["data_offsets"]
            if begin != cursor:
                raise ValueError(f"Overlapping or uncovered payload: {tensor['key']}")
            cursor = end
        if cursor != payload_size:
            raise ValueError(f"Uncovered trailing payload: {path.name}")
        stream.seek(0)
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": path.name, "file_bytes": size, "header_and_prefix_bytes": 8 + header_size,
            "payload_bytes": payload_size, "sha256": sha, "tensors": tensors}


def summarize_codebooks(tensors):
    """Classify per-projection codebooks from paired trellis/marker metadata.

    Presence-faithful to LinearEXL3 loading (mcg/mul1 flags set when the
    sibling tensor exists); malformed markers and conflicting pairs are
    reported as issues without inferring decode correctness.
    """
    by_key = {tensor["key"]: tensor for tensor in tensors}
    trellis_keys = sorted(key for key in by_key if key.endswith(TRELLIS_SUFFIX))
    marker_keys = sorted(key for key in by_key if key.endswith(MARKER_SUFFIXES))
    counts = {"3inst": 0, "mcg": 0, "mul1": 0, "conflicting": 0}
    histogram, conflicting, malformed, issues = {}, [], [], []
    for key in trellis_keys:
        base = key[: -len(TRELLIS_SUFFIX)]
        has_mcg, has_mul1 = base + ".mcg" in by_key, base + ".mul1" in by_key
        if has_mcg and has_mul1:
            counts["conflicting"] += 1
            conflicting.append(base)
            issues.append(f"Conflicting codebook markers for {base}: both .mcg and .mul1 present")
        elif has_mcg:
            counts["mcg"] += 1
        elif has_mul1:
            counts["mul1"] += 1
        else:
            counts["3inst"] += 1
    orphans = []
    for key in marker_keys:
        entry = by_key[key]
        if entry["dtype"] != MARKER_DTYPE or entry["shape"] != []:
            reason = f"expected scalar {MARKER_DTYPE}, saw dtype={entry['dtype']} shape={entry['shape']}"
            malformed.append({"key": key, "dtype": entry["dtype"],
                            "shape": entry["shape"], "reason": reason})
            issues.append(f"Malformed codebook marker {key}: {reason}")
        if key[: key.rfind(".")] + TRELLIS_SUFFIX not in by_key:
            orphans.append(key)
            issues.append(f"Orphan codebook marker {key}: no paired .trellis tensor")
    for key in trellis_keys:
        entry = by_key[key]
        shape = entry["shape"]
        if entry["dtype"] != "I16" or len(shape) != 3 or min(shape) <= 0 or shape[-1] % 16:
            issues.append(f"Trellis dimensions invalid for bit-width inference: {key} "
                          f"dtype={entry['dtype']} shape={shape}")
            continue
        bits = shape[-1] // 16
        if not MIN_BITS_PER_WEIGHT <= bits <= MAX_BITS_PER_WEIGHT:
            issues.append(f"Trellis bit width outside {MIN_BITS_PER_WEIGHT}..{MAX_BITS_PER_WEIGHT}: "
                          f"{key} K={bits}")
            continue
        histogram[str(bits)] = histogram.get(str(bits), 0) + 1
    return {"scope": ("Codebook presence from paired .trellis/.mcg/.mul1 tensor metadata; "
                      "decode correctness and model fidelity are not established."),
            "projections_with_trellis": len(trellis_keys),
            "codebook_counts": counts,
            "packed_bit_width_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
            "orphan_markers": orphans,
            "conflicting_bases": conflicting,
            "malformed_markers": malformed,
            "issues": issues}


def audit(candidate, max_total_bytes=None):
    if (max_total_bytes is not None
            and (isinstance(max_total_bytes, bool) or not isinstance(max_total_bytes, int)
                 or max_total_bytes <= 0)):
        raise ValueError("max_total_bytes must be a positive integer")
    candidate = Path(candidate)
    config = json.loads((candidate / "config.json").read_text(), object_pairs_hook=unique_object)
    if config.get("quantization_config", {}).get("quant_method") != "exl3":
        raise ValueError("Candidate config must identify quant_method=exl3")
    files = sorted(candidate.rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("Candidate symlinks are not supported")
    files = [p for p in files if p.is_file()]
    shard_paths = [p for p in files if p.suffix == ".safetensors"]
    if not shard_paths:
        raise ValueError("Candidate has no safetensors shards")
    shards, seen = [], set()
    totals = dict(target=0, mtp=0, unknown=0)
    embedding_bytes = 0
    all_tensors = []
    for path in shard_paths:
        shard = inspect_shard(path)
        shard["file"] = path.relative_to(candidate).as_posix()
        for tensor in shard["tensors"]:
            key = tensor["key"]
            if key in seen:
                raise ValueError(f"Duplicate tensor across shards: {key}")
            seen.add(key)
            totals[tensor["category"]] += tensor["bytes"]
            if tensor["category"] == "target" and key.endswith(".embed_tokens.weight"):
                embedding_bytes += tensor["bytes"]
        shards.append(shard)
        all_tensors.extend(shard["tensors"])
    file_bytes = sum(p.stat().st_size for p in files)
    shard_bytes = sum(s["file_bytes"] for s in shards)
    overhead = file_bytes - sum(totals.values())
    return {"scope": "Storage accounting only; required-tensor completeness, runtime health and fidelity are not established.",
            "quant_method": "exl3", "tensor_payload_bytes": totals,
            "embedding_payload_bytes_in_target": embedding_bytes,
            "target_nonembedding_payload_bytes": totals["target"] - embedding_bytes,
            "combined_shard_file_bytes": shard_bytes, "combined_candidate_file_bytes": file_bytes,
            "shard_header_and_prefix_bytes": sum(s["header_and_prefix_bytes"] for s in shards),
            "nonshard_file_bytes": file_bytes - shard_bytes, "all_container_and_file_overhead_bytes": overhead,
            "conservative_target_plus_all_overhead_bytes": totals["target"] + overhead,
            "target_plus_unknown_plus_all_overhead_bytes": totals["target"] + totals["unknown"] + overhead,
            "decimal_target_limit_bytes": LIMIT,
            "target_payload_below_limit": totals["target"] < LIMIT,
            "target_plus_all_overhead_below_limit": totals["target"] + overhead < LIMIT,
            "overhead_assignment": "All shared shard headers and nonshard files charged to target; MTP payload excluded. Unknown payload reported separately.",
            "max_total_bytes": max_total_bytes,
            "combined_total_within_max_bytes": None if max_total_bytes is None else file_bytes <= max_total_bytes,
            "codebook_summary": summarize_codebooks(all_tensors),
            "files": [{"file": p.relative_to(candidate).as_posix(), "bytes": p.stat().st_size} for p in files],
            "shards": shards}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-total-bytes", type=int, default=None,
                        help="Optional ceiling for all model-directory files, including MTP and metadata")
    args = parser.parse_args()
    if args.max_total_bytes is not None and args.max_total_bytes <= 0:
        parser.error("--max-total-bytes must be a positive integer")
    candidate, output = args.candidate.resolve(), args.output.resolve()
    if output == candidate or candidate in output.parents:
        parser.error("output must be outside candidate")
    if output.exists():
        parser.error("output must be a new file")
    result = audit(candidate, max_total_bytes=args.max_total_bytes)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if result["combined_total_within_max_bytes"] is False:
        print(f"combined_candidate_file_bytes {result['combined_candidate_file_bytes']} "
              f"exceeds --max-total-bytes {args.max_total_bytes}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
