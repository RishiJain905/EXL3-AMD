"""Assemble a new uncommitted MiMo head checkpoint from exact tensor bytes.

Stdlib only. Streaming copies in <=8MiB chunks with bounded RAM.
Never overwrites/promotes/deletes/moves any existing checkpoint.
Output is a NEW directory with an explicit ``uncommitted`` marker.

Production geometry (fixed, no CLI overrides):
  state: tensor.0..127, F16 [1,2048,4096]
  original_input_ids: tensor.0..127, I64 [1,2048]
  rows 0..4: key "tensor", F16 [1,2048,248320]
Output state: tensor.0..127 numeric order, rows 0..4 from row files,
rows 5..127 byte-exact from original state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

COMPLETION_FILENAME = "assembly-completed.json"
MANIFEST_FILENAME = "rows-completed.json"
JOB_FILENAME = "job.json"
STATE_FILENAME = "state.safetensors"
TOKENS_FILENAME = "original_input_ids.safetensors"

HEADER_MAX_BYTES = 16 * 1024 * 1024
JSON_MAX_BYTES = 16 * 1024 * 1024
CHUNK_DEFAULT = 8 * 1024 * 1024

MANIFEST_SCHEMA = "mimo-head-postsave-v1"
MANIFEST_MODE = "recover"
OUTPUT_SCHEMA = "mimo-head-assembly-v1"

_U64 = struct.Struct("<Q")

_DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}

_CHECKPOINT_HASH_KEYS = (JOB_FILENAME, STATE_FILENAME, TOKENS_FILENAME)


class AssemblyError(Exception):
    pass


@dataclass(frozen=True)
class _Spec:
    num_rows: int = 128
    num_recovered: int = 5
    state_dtype: str = "F16"
    state_shape: tuple = (1, 2048, 4096)
    token_dtype: str = "I64"
    token_shape: tuple = (1, 2048)
    row_dtype: str = "F16"
    row_shape: tuple = (1, 2048, 248320)


_PROD_SPEC = _Spec()


@dataclass(frozen=True)
class _TensorEntry:
    dtype: str
    shape: tuple
    begin: int
    end: int


@dataclass(frozen=True)
class _SafetensorsInfo:
    entries: dict
    data_start: int
    file_size: int


def _prod(shape) -> int:
    p = 1
    for dim in shape:
        p *= int(dim)
    return int(p)


def _strict_json_loads(data: bytes, what: str):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssemblyError(f"{what}: invalid utf-8: {exc}") from exc

    def _reject_const(value: str):
        raise AssemblyError(f"{what}: invalid JSON constant {value!r}")

    def _pairs(pairs):
        obj = {}
        for key, val in pairs:
            if key in obj:
                raise AssemblyError(f"{what}: duplicate key {key!r}")
            obj[key] = val
        return obj

    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_reject_const,
        )
    except AssemblyError:
        raise
    except ValueError as exc:
        raise AssemblyError(f"{what}: invalid JSON: {exc}") from exc


def _read_bounded(path: Path, what: str, limit: int) -> bytes:
    if os.path.islink(path):
        raise AssemblyError(f"{what}: symlink not allowed: {path}")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise AssemblyError(f"{what}: cannot stat {path}: {exc}") from exc
    if size > limit:
        raise AssemblyError(f"{what}: file too large ({size} > {limit}): {path}")
    if size < 0:
        raise AssemblyError(f"{what}: invalid size for {path}")
    with open(path, "rb") as handle:
        data = handle.read()
    if data is None:
        raise AssemblyError(f"{what}: short read: {path}")
    if len(data) != size:
        raise AssemblyError(
            f"{what}: size changed during read ({len(data)} != {size}): {path}"
        )
    return data


def _load_json_document(path: Path, what: str):
    raw = _read_bounded(path, what, JSON_MAX_BYTES)
    obj = _strict_json_loads(raw, what)
    digest = hashlib.sha256(raw).hexdigest()
    return obj, digest, raw


def _is_hex64(value) -> bool:
    if type(value) is not str:
        return False
    if len(value) != 64:
        return False
    for ch in value:
        if ch not in "0123456789abcdefABCDEF":
            return False
    return True


def _normalize_hash(value, what: str) -> str:
    if not _is_hex64(value):
        raise AssemblyError(f"{what}: invalid sha256 hex (want 64 hex chars)")
    return str(value).lower()


def _sha256_file(path: Path, chunk: int, what: str) -> str:
    if os.path.islink(path):
        raise AssemblyError(f"{what}: symlink not allowed: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if block is None:
                raise AssemblyError(f"{what}: short read: {path}")
            if len(block) == 0:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_exact(handle, need: int, what: str) -> bytes:
    parts = []
    remaining = need
    while remaining > 0:
        block = handle.read(remaining)
        if block is None or len(block) == 0:
            raise AssemblyError(f"{what}: short read (want {need}, short by {remaining})")
        parts.append(block)
        remaining -= len(block)
    if len(parts) == 1:
        return parts[0]
    return b"".join(parts)


def _hash_range(path: Path, abs_off: int, nbytes: int, chunk: int, what: str) -> str:
    if os.path.islink(path):
        raise AssemblyError(f"{what}: symlink not allowed: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(abs_off, os.SEEK_SET)
        remaining = nbytes
        while remaining > 0:
            step = chunk if remaining > chunk else remaining
            block = _read_exact(handle, step, f"{what}: {path}")
            digest.update(block)
            remaining -= step
    return digest.hexdigest()


def _copy_range(src: Path, abs_off: int, nbytes: int, dst_handle, chunk: int, what: str) -> str:
    if os.path.islink(src):
        raise AssemblyError(f"{what}: symlink not allowed: {src}")
    digest = hashlib.sha256()
    with open(src, "rb") as handle:
        handle.seek(abs_off, os.SEEK_SET)
        remaining = nbytes
        while remaining > 0:
            step = chunk if remaining > chunk else remaining
            block = _read_exact(handle, step, f"{what}: {src}")
            written = dst_handle.write(block)
            if written is None or written != len(block):
                raise AssemblyError(
                    f"{what}: short write (wrote {written} of {len(block)})"
                )
            digest.update(block)
            remaining -= step
    return digest.hexdigest()


def _copy_file(src: Path, dst: Path, chunk: int, what: str) -> str:
    if os.path.islink(src):
        raise AssemblyError(f"{what}: symlink not allowed: {src}")
    digest = hashlib.sha256()
    with open(src, "rb") as reader, open(dst, "wb") as writer:
        while True:
            block = reader.read(chunk)
            if block is None:
                raise AssemblyError(f"{what}: short read: {src}")
            if len(block) == 0:
                break
            written = writer.write(block)
            if written is None or written != len(block):
                raise AssemblyError(
                    f"{what}: short write (wrote {written} of {len(block)}): {dst}"
                )
            digest.update(block)
    return digest.hexdigest()


def _resolved(path) -> Path:
    return Path(path).resolve()


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_dirs(checkpoint, rows_dir, output):
    for label, candidate in (
        ("checkpoint", checkpoint),
        ("rows", rows_dir),
        ("output", output),
    ):
        if os.path.islink(candidate):
            raise AssemblyError(f"{label}: symlink not allowed: {candidate}")
    ckpt = _resolved(checkpoint)
    rows = _resolved(rows_dir)
    out = _resolved(output)
    if ckpt == rows or _within(ckpt, rows) or _within(rows, ckpt):
        raise AssemblyError("checkpoint and rows directories must be separate and non-nested")
    if out == ckpt or _within(out, ckpt) or _within(ckpt, out):
        raise AssemblyError("output must be separate and non-nested vs checkpoint")
    if out == rows or _within(out, rows) or _within(rows, out):
        raise AssemblyError("output must be separate and non-nested vs rows")
    if not ckpt.is_dir():
        raise AssemblyError(f"checkpoint is not a directory: {checkpoint}")
    if not rows.is_dir():
        raise AssemblyError(f"rows is not a directory: {rows_dir}")
    if os.path.lexists(out):
        raise AssemblyError(f"output already exists (exclusive): {output}")
    # Re-check symlink form of resolved output (dangling link already covered by lexists).
    if os.path.islink(str(out)):
        raise AssemblyError(f"output: symlink not allowed: {output}")
    return ckpt, rows, out


def _read_safetensors_header(path: Path, what: str) -> _SafetensorsInfo:
    if os.path.islink(path):
        raise AssemblyError(f"{what}: symlink not allowed: {path}")
    with open(path, "rb") as handle:
        try:
            file_size = os.fstat(handle.fileno()).st_size
        except OSError as exc:
            raise AssemblyError(f"{what}: cannot stat {path}: {exc}") from exc
        prefix = handle.read(8)
        if prefix is None or len(prefix) != 8:
            raise AssemblyError(f"{what}: truncated header length: {path}")
        (header_len,) = _U64.unpack(prefix)
        if header_len > HEADER_MAX_BYTES:
            raise AssemblyError(
                f"{what}: header too large ({header_len} > {HEADER_MAX_BYTES}): {path}"
            )
        data_start = 8 + int(header_len)
        if data_start > file_size:
            raise AssemblyError(f"{what}: truncated header JSON: {path}")
        header_raw = _read_exact(handle, int(header_len), f"{what}: {path}")
    header = _strict_json_loads(header_raw, f"{what}: {path}")
    if type(header) is not dict:
        raise AssemblyError(f"{what}: header must be a JSON object: {path}")
    entries: dict[str, _TensorEntry] = {}
    for key, value in header.items():
        if key == "__metadata__":
            if type(value) is not dict:
                raise AssemblyError(f"{what}: __metadata__ must be an object: {path}")
            for mk, mv in value.items():
                if type(mk) is not str or type(mv) is not str:
                    raise AssemblyError(
                        f"{what}: __metadata__ must be str->str: {path}"
                    )
            continue
        if type(value) is not dict:
            raise AssemblyError(f"{what}: entry {key!r} must be an object: {path}")
        if set(value.keys()) != {"dtype", "shape", "data_offsets"}:
            raise AssemblyError(
                f"{what}: entry {key!r} must have exactly dtype/shape/data_offsets: {path}"
            )
        dtype = value["dtype"]
        shape = value["shape"]
        offsets = value["data_offsets"]
        if type(dtype) is not str or dtype not in _DTYPE_SIZES:
            raise AssemblyError(f"{what}: entry {key!r} has unknown dtype: {path}")
        if type(shape) is not list:
            raise AssemblyError(f"{what}: entry {key!r} has invalid shape: {path}")
        for dim in shape:
            if type(dim) is not int or dim < 0:
                raise AssemblyError(
                    f"{what}: entry {key!r} has invalid shape dim: {path}"
                )
        if type(offsets) is not list or len(offsets) != 2:
            raise AssemblyError(
                f"{what}: entry {key!r} has invalid data_offsets: {path}"
            )
        begin, end = offsets[0], offsets[1]
        if type(begin) is not int or type(end) is not int:
            raise AssemblyError(
                f"{what}: entry {key!r} has non-integer offsets: {path}"
            )
        if begin < 0 or end < begin:
            raise AssemblyError(
                f"{what}: entry {key!r} has invalid offset range: {path}"
            )
        expect = _prod(tuple(shape)) * _DTYPE_SIZES[dtype]
        if (end - begin) != expect:
            raise AssemblyError(
                f"{what}: entry {key!r} byte length {end - begin} != "
                f"shape/dtype {expect}: {path}"
            )
        entries[key] = _TensorEntry(
            dtype=dtype, shape=tuple(shape), begin=begin, end=end
        )
    ordered = sorted(entries.items(), key=lambda kv: (kv[1].begin, kv[1].end))
    expect_off = 0
    for key, info in ordered:
        if info.begin != expect_off:
            raise AssemblyError(
                f"{what}: noncontiguous offsets at {key!r} "
                f"(begin {info.begin} != {expect_off}): {path}"
            )
        expect_off = info.end
    data_len = file_size - data_start
    if expect_off != data_len:
        if expect_off > data_len:
            raise AssemblyError(f"{what}: truncated payloads: {path}")
        raise AssemblyError(f"{what}: trailer bytes after payloads: {path}")
    return _SafetensorsInfo(entries=entries, data_start=data_start, file_size=file_size)


def _expect_tensors(info: _SafetensorsInfo, names, dtype: str, shape: tuple, what: str, path: Path):
    actual = set(info.entries.keys())
    wanted = set(names)
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        raise AssemblyError(
            f"{what}: tensor keys mismatch (missing={missing} extra={extra}): {path}"
        )
    for name in names:
        entry = info.entries[name]
        if entry.dtype != dtype or entry.shape != tuple(shape):
            raise AssemblyError(
                f"{what}: {name!r} is {entry.dtype} {list(entry.shape)}, "
                f"want {dtype} {list(shape)}: {path}"
            )


def _validate_job(obj, path: Path):
    what = f"job: {path}"
    if type(obj) is not dict:
        raise AssemblyError(f"{what}: top level must be an object")
    if "next_module_idx" not in obj or "q_strategy" not in obj or "bad_rows" not in obj:
        raise AssemblyError(f"{what}: missing next_module_idx/q_strategy/bad_rows")
    nxt = obj["next_module_idx"]
    if type(nxt) is not int or nxt != 34:
        raise AssemblyError(f"{what}: next_module_idx must be 34")
    if obj["q_strategy"] is not None:
        raise AssemblyError(f"{what}: q_strategy must be null")
    bad = obj["bad_rows"]
    if type(bad) is not list or len(bad) != 0:
        raise AssemblyError(f"{what}: bad_rows must be []")


def _validate_manifest(obj, spec: _Spec, path: Path):
    what = f"manifest: {path}"
    if type(obj) is not dict:
        raise AssemblyError(f"{what}: top level must be an object")
    for key in ("schema", "mode", "complete", "checkpoint_sha256", "rows"):
        if key not in obj:
            raise AssemblyError(f"{what}: missing key {key!r}")
    if type(obj["schema"]) is not str or obj["schema"] != MANIFEST_SCHEMA:
        raise AssemblyError(f"{what}: schema must be {MANIFEST_SCHEMA!r}")
    if type(obj["mode"]) is not str or obj["mode"] != MANIFEST_MODE:
        raise AssemblyError(f"{what}: mode must be {MANIFEST_MODE!r}")
    if type(obj["complete"]) is not bool or obj["complete"] is not True:
        raise AssemblyError(f"{what}: complete must be true")
    hashes = obj["checkpoint_sha256"]
    if type(hashes) is not dict:
        raise AssemblyError(f"{what}: checkpoint_sha256 must be an object")
    if set(hashes.keys()) != set(_CHECKPOINT_HASH_KEYS):
        raise AssemblyError(
            f"{what}: checkpoint_sha256 must have exactly "
            f"{list(_CHECKPOINT_HASH_KEYS)}"
        )
    normalized = {}
    for key in _CHECKPOINT_HASH_KEYS:
        normalized[key] = _normalize_hash(hashes[key], f"{what}: checkpoint_sha256[{key}]")
    rows = obj["rows"]
    if type(rows) is not list:
        raise AssemblyError(f"{what}: rows must be a list")
    if len(rows) != spec.num_recovered:
        raise AssemblyError(
            f"{what}: rows must have exactly {spec.num_recovered} entries"
        )
    seen: dict[int, str] = {}
    normalized_rows: list[dict] = []
    for index, entry in enumerate(rows):
        ewhat = f"{what}: rows[{index}]"
        if type(entry) is not dict:
            raise AssemblyError(f"{ewhat}: must be an object")
        for key in ("row", "file", "sha256"):
            if key not in entry:
                raise AssemblyError(f"{ewhat}: missing key {key!r}")
        row = entry["row"]
        if type(row) is not int:
            raise AssemblyError(f"{ewhat}: row must be an integer (no bool)")
        if row < 0 or row >= spec.num_recovered:
            raise AssemblyError(f"{ewhat}: row {row} out of range")
        if row in seen:
            raise AssemblyError(f"{what}: duplicate row {row}")
        filename = entry["file"]
        if type(filename) is not str:
            raise AssemblyError(f"{ewhat}: file must be a string")
        if (
            "/" in filename
            or "\\" in filename
            or ".." in filename
            or filename.startswith("/")
            or (len(filename) >= 2 and filename[1] == ":")
        ):
            raise AssemblyError(f"{ewhat}: path traversal in file {filename!r}")
        want_name = f"row-{row:03d}.safetensors"
        if filename != want_name:
            raise AssemblyError(f"{ewhat}: unknown row file {filename!r}")
        digest = _normalize_hash(entry["sha256"], f"{ewhat}: sha256")
        seen[row] = filename
        normalized_rows.append({"row": row, "file": filename, "sha256": digest})
    if set(seen.keys()) != set(range(spec.num_recovered)):
        raise AssemblyError(f"{what}: rows must cover 0..{spec.num_recovered - 1} exactly")
    normalized_rows.sort(key=lambda item: item["row"])
    return normalized, normalized_rows


def _row_source_names(spec: _Spec):
    return [f"tensor.{idx}" for idx in range(spec.num_rows)]


def assemble_checkpoint(checkpoint, rows_dir, output, *, _spec=None, _chunk_size: int = CHUNK_DEFAULT):
    spec: _Spec = _PROD_SPEC if _spec is None else _spec
    if type(_chunk_size) is not int or _chunk_size <= 0 or _chunk_size > CHUNK_DEFAULT:
        raise AssemblyError(f"chunk size must be int in 1..{CHUNK_DEFAULT}")
    chunk = int(_chunk_size)

    ckpt, rows, out = _validate_dirs(checkpoint, rows_dir, output)

    job_path = ckpt / JOB_FILENAME
    state_path = ckpt / STATE_FILENAME
    tokens_path = ckpt / TOKENS_FILENAME
    manifest_path = rows / MANIFEST_FILENAME
    for label, path in (
        ("job", job_path),
        ("state", state_path),
        ("tokens", tokens_path),
        ("manifest", manifest_path),
    ):
        if os.path.islink(path):
            raise AssemblyError(f"{label}: symlink not allowed: {path}")
        if not path.is_file():
            raise AssemblyError(f"{label}: missing file: {path}")

    job_obj, job_digest, _job_raw = _load_json_document(job_path, "job")
    _validate_job(job_obj, job_path)

    manifest_obj, manifest_digest, _manifest_raw = _load_json_document(
        manifest_path, "manifest"
    )
    want_hashes, want_rows = _validate_manifest(manifest_obj, spec, manifest_path)

    # Full source hashes before any output is written.
    state_digest = _sha256_file(state_path, chunk, "state")
    tokens_digest = _sha256_file(tokens_path, chunk, "tokens")
    actual_hashes = {
        JOB_FILENAME: job_digest,
        STATE_FILENAME: state_digest,
        TOKENS_FILENAME: tokens_digest,
    }
    for key in _CHECKPOINT_HASH_KEYS:
        if actual_hashes[key].lower() != want_hashes[key].lower():
            raise AssemblyError(f"checkpoint hash mismatch for {key}")

    row_paths: dict[int, Path] = {}
    row_digests: dict[int, str] = {}
    for item in want_rows:
        row = item["row"]
        candidate = rows / item["file"]
        if os.path.islink(candidate):
            raise AssemblyError(f"row {row}: symlink not allowed: {candidate}")
        if not candidate.is_file():
            raise AssemblyError(f"row {row}: missing file: {candidate}")
        try:
            candidate.relative_to(rows)
        except ValueError as exc:
            raise AssemblyError(f"row {row}: escapes rows directory") from exc
        digest = _sha256_file(candidate, chunk, f"row {row}")
        if digest.lower() != item["sha256"].lower():
            raise AssemblyError(f"row {row}: source hash mismatch")
        row_paths[row] = candidate
        row_digests[row] = digest

    # Geometry before any output is written.
    state_info = _read_safetensors_header(state_path, "state")
    tokens_info = _read_safetensors_header(tokens_path, "tokens")
    names = _row_source_names(spec)
    _expect_tensors(state_info, names, spec.state_dtype, tuple(spec.state_shape), "state", state_path)
    _expect_tensors(tokens_info, names, spec.token_dtype, tuple(spec.token_shape), "tokens", tokens_path)
    row_infos: dict[int, _SafetensorsInfo] = {}
    for row in range(spec.num_recovered):
        info = _read_safetensors_header(row_paths[row], f"row {row}")
        _expect_tensors(info, ["tensor"], spec.row_dtype, tuple(spec.row_shape), f"row {row}", row_paths[row])
        row_infos[row] = info

    try:
        os.mkdir(out)
    except FileExistsError as exc:
        raise AssemblyError(f"output already exists (exclusive): {output}") from exc
    except OSError as exc:
        raise AssemblyError(f"cannot create output directory {output}: {exc}") from exc

    completion_path = out / COMPLETION_FILENAME
    out_state = out / STATE_FILENAME
    out_tokens = out / TOKENS_FILENAME
    out_job = out / JOB_FILENAME

    try:
        # Canonical output header in numeric row order.
        header: dict = {}
        offset = 0
        plan: list[dict] = []
        for idx in range(spec.num_rows):
            if idx < spec.num_recovered:
                dtype = spec.row_dtype
                shape = tuple(spec.row_shape)
                nbytes = _prod(shape) * _DTYPE_SIZES[dtype]
                src_path = row_paths[idx]
                src_info = row_infos[idx]
                src_entry = src_info.entries["tensor"]
                src_abs = src_info.data_start + src_entry.begin
                src_file = f"row-{idx:03d}.safetensors"
                src_full = row_digests[idx]
            else:
                dtype = spec.state_dtype
                shape = tuple(spec.state_shape)
                nbytes = _prod(shape) * _DTYPE_SIZES[dtype]
                src_path = state_path
                src_entry = state_info.entries[f"tensor.{idx}"]
                src_abs = state_info.data_start + src_entry.begin
                src_file = STATE_FILENAME
                src_full = state_digest
            if (src_entry.end - src_entry.begin) != nbytes:
                raise AssemblyError(f"row {idx}: source payload size mismatch")
            begin, end = offset, offset + nbytes
            header[f"tensor.{idx}"] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [begin, end],
            }
            plan.append(
                {
                    "row": idx,
                    "tensor": f"tensor.{idx}",
                    "dtype": dtype,
                    "shape": list(shape),
                    "src_path": src_path,
                    "src_abs": src_abs,
                    "nbytes": nbytes,
                    "src_file": src_file,
                    "src_full": src_full,
                    "out_begin": begin,
                    "out_end": end,
                }
            )
            offset = end

        header_raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(header_raw) > HEADER_MAX_BYTES:
            raise AssemblyError("output header too large")

        copy_hashes: dict[int, str] = {}
        try:
            with open(out_state, "wb") as writer:
                prefix = _U64.pack(len(header_raw))
                written = writer.write(prefix)
                if written is None or written != len(prefix):
                    raise AssemblyError("state: short write on header length")
                written = writer.write(header_raw)
                if written is None or written != len(header_raw):
                    raise AssemblyError("state: short write on header JSON")
                for item in plan:
                    digest = _copy_range(
                        item["src_path"],
                        item["src_abs"],
                        item["nbytes"],
                        writer,
                        chunk,
                        f"row {item['row']}",
                    )
                    copy_hashes[item["row"]] = digest
        except OSError as exc:
            raise AssemblyError(f"state: write failed: {exc}") from exc

        # Independent verification: re-stream every source and output payload.
        out_info = _read_safetensors_header(out_state, "output state")
        out_names = [f"tensor.{idx}" for idx in range(spec.num_rows)]
        ordered_keys = list(out_info.entries.keys())
        if ordered_keys != out_names:
            raise AssemblyError("output state: tensor keys/order mismatch")
        rows_record: list[dict] = []
        for item in plan:
            row = item["row"]
            out_entry = out_info.entries[item["tensor"]]
            if out_entry.dtype != item["dtype"] or out_entry.shape != tuple(item["shape"]):
                raise AssemblyError(f"row {row}: output geometry mismatch")
            if out_entry.begin != item["out_begin"] or out_entry.end != item["out_end"]:
                raise AssemblyError(f"row {row}: output offset mismatch")
            out_abs = out_info.data_start + out_entry.begin
            try:
                source_hash = _hash_range(
                    item["src_path"], item["src_abs"], item["nbytes"], chunk, f"row {row}"
                )
                output_hash = _hash_range(
                    out_state, out_abs, item["nbytes"], chunk, f"row {row}"
                )
            except OSError as exc:
                raise AssemblyError(f"row {row}: verification read failed: {exc}") from exc
            if source_hash != copy_hashes[row] or output_hash != copy_hashes[row]:
                raise AssemblyError(f"row {row}: payload hash mismatch")
            rows_record.append(
                {
                    "row": row,
                    "tensor": item["tensor"],
                    "dtype": item["dtype"],
                    "shape": list(item["shape"]),
                    "source_file": item["src_file"],
                    "source_file_sha256": item["src_full"],
                    "payload_sha256": source_hash,
                    "output_payload_sha256": output_hash,
                }
            )

        out_state_digest = _sha256_file(out_state, chunk, "output state")

        try:
            tokens_out_digest = _copy_file(tokens_path, out_tokens, chunk, "tokens")
        except OSError as exc:
            raise AssemblyError(f"tokens: copy failed: {exc}") from exc
        verify_tokens = _sha256_file(out_tokens, chunk, "output tokens")
        if tokens_out_digest != tokens_digest or verify_tokens != tokens_digest:
            raise AssemblyError("tokens: byte-for-byte copy mismatch")

        job_out = dict(job_obj)
        job_out["next_module_idx"] = 35
        try:
            with open(out_job, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(job_out, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            raise AssemblyError(f"job: write failed: {exc}") from exc
        out_job_digest = _sha256_file(out_job, chunk, "output job")

        record = {
            "schema": OUTPUT_SCHEMA,
            "uncommitted": True,
            "inputs": {"checkpoint": str(ckpt), "rows": str(rows)},
            "checkpoint_sha256": dict(actual_hashes),
            "manifest_sha256": manifest_digest,
            "outputs": {
                JOB_FILENAME: {"sha256": out_job_digest},
                STATE_FILENAME: {"sha256": out_state_digest},
                TOKENS_FILENAME: {"sha256": verify_tokens},
            },
            "rows": rows_record,
            "note": "Byte assembly only; no quality or quantization claim.",
        }
        try:
            with open(completion_path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            raise AssemblyError(f"completion: write failed: {exc}") from exc
        return record
    except Exception:
        try:
            if completion_path.exists() or os.path.lexists(completion_path):
                try:
                    os.unlink(completion_path)
                except OSError:
                    pass
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble a new uncommitted MiMo head checkpoint (streaming, bounded RAM)."
    )
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint directory")
    parser.add_argument("--rows", required=True, help="Recovered rows directory")
    parser.add_argument("--output", required=True, help="New output directory (must not exist)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        assemble_checkpoint(args.checkpoint, args.rows, args.output)
    except AssemblyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: I/O failure: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
