"""Strict, offline per-layer cache precision profiles; no Torch dependency.

Version 1 binds configuration bytes and global-attention geometry, not model
weights. Calibrate again for different weights even if their config matches.
"""
import hashlib
import json
from pathlib import Path
import re

MAX_BYTES = 1024 * 1024
FORMATS = frozenset(("q4", "q5", "q6", "q8", "aster5"))


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("Invalid cache policy fields")


def _validate(policy):
    _fields(policy, ("version", "config_sha256", "components"))
    if type(policy["version"]) is not int or policy["version"] != 1:
        raise ValueError("Unsupported cache policy version")
    if not isinstance(policy["config_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", policy["config_sha256"]):
        raise ValueError("Invalid cache policy config hash")
    _fields(policy["components"], ("target",), ("draft",))
    for entries in policy["components"].values():
        if not isinstance(entries, list) or not 1 <= len(entries) <= 512:
            raise ValueError("Invalid cache policy layer count")
        seen = set()
        for entry in entries:
            _fields(entry, ("layer_idx", "kv_heads", "head_dim", "k", "v"))
            for key in ("layer_idx", "kv_heads", "head_dim"):
                if type(entry[key]) is not int or not (0 if key == "layer_idx" else 1) <= entry[key] <= 65536:
                    raise ValueError("Invalid cache policy layer geometry")
            if entry["layer_idx"] in seen:
                raise ValueError("Duplicate cache policy layer")
            seen.add(entry["layer_idx"])
            k, v = entry["k"], entry["v"]
            if not isinstance(k, str) or not isinstance(v, str) or k not in FORMATS or v not in FORMATS:
                raise ValueError("Unsupported cache policy precision")
            if (k == "aster5") != (v == "aster5"):
                raise ValueError("Aster5 requires both K and V in each selected layer")
    return policy


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate cache policy JSON key")
        result[key] = value
    return result


def _nonfinite(unused):
    raise ValueError("Nonfinite cache policy JSON value")


def load_cache_policy(path):
    try:
        with Path(path).open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("Cache policy exceeds size limit")
        policy = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("Cannot read valid cache policy JSON") from None
    return _validate(policy)


def validate_cache_policy(policy, config_bytes, components):
    _validate(policy)
    if hashlib.sha256(config_bytes).hexdigest() != policy["config_sha256"]:
        raise ValueError("Cache policy model configuration mismatch")
    result = {}
    for name, descriptors in components.items():
        if name not in policy["components"]:
            raise ValueError("Cache policy lacks active component")
        expected = {d["layer_idx"]: (d["kv_heads"], d["head_dim"]) for d in descriptors}
        entries = policy["components"][name]
        actual = {d["layer_idx"]: (d["kv_heads"], d["head_dim"]) for d in entries}
        if len(expected) != len(descriptors) or actual != expected:
            raise ValueError("Cache policy layer coverage or geometry mismatch")
        result[name] = {d["layer_idx"]: (d["k"], d["v"]) for d in entries}
    return result


def policy_summary(policy):
    _validate(policy)
    raw = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return dict(policy, sha256=hashlib.sha256(raw).hexdigest())
