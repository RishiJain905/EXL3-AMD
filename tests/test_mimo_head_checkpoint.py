"""Tests for streaming MiMo head checkpoint assembler (tiny tempfile fixtures)."""
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "assemble_mimo_head_checkpoint.py"
_spec = importlib.util.spec_from_file_location("assemble_mimo_head_under_test", _SCRIPT)
asm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = asm
_spec.loader.exec_module(asm)

_TINY = asm._Spec(
    num_rows=128,
    num_recovered=5,
    state_dtype="F16",
    state_shape=(1, 2, 4),
    token_dtype="I64",
    token_shape=(1, 2),
    row_dtype="F16",
    row_shape=(1, 2, 8),
)
_SIZES = {"F16": 2, "I64": 8}
_U64 = struct.Struct("<Q")


def _payload(seed, nbytes):
    return bytes((seed + i) % 256 for i in range(nbytes))


def _nbytes(dtype, shape):
    n = 1
    for dim in shape:
        n *= dim
    return n * _SIZES[dtype]


def _write_st(path, entries):
    header = {}
    off = 0
    blobs = []
    for name, dtype, shape, raw in entries:
        if len(raw) != _nbytes(dtype, shape):
            raise AssertionError("bad fixture payload size")
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as handle:
        handle.write(_U64.pack(len(blob)))
        handle.write(blob)
        for raw in blobs:
            handle.write(raw)


def _file_sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _parse_header(path):
    with open(path, "rb") as handle:
        (hlen,) = _U64.unpack(handle.read(8))
        raw = handle.read(hlen)
        header = json.loads(raw.decode("utf-8"))
        data_start = 8 + hlen
    size = os.path.getsize(path)
    return header, data_start, size


def _read_payload(path, data_start, begin, end):
    with open(path, "rb") as handle:
        handle.seek(data_start + begin, os.SEEK_SET)
        data = handle.read(end - begin)
    if len(data) != end - begin:
        raise AssertionError("short fixture read")
    return data


def _make_fixture(root):
    root = Path(root)
    ckpt = root / "ckpt"
    rows = root / "rows"
    ckpt.mkdir()
    rows.mkdir()
    job = {"next_module_idx": 34, "q_strategy": None, "bad_rows": [],
           "run_id": "tiny-fixture", "note": "extra-preserved"}
    with open(ckpt / "job.json", "w", encoding="utf-8") as handle:
        json.dump(job, handle, indent=2)
    state_payloads = {}
    state_entries = []
    for idx in range(128):
        raw = _payload(idx + 1, _nbytes("F16", (1, 2, 4)))
        state_payloads[idx] = raw
        state_entries.append((f"tensor.{idx}", "F16", (1, 2, 4), raw))
    _write_st(ckpt / "state.safetensors", state_entries)
    token_payloads = {}
    token_entries = []
    for idx in range(128):
        raw = _payload(1000 + idx, _nbytes("I64", (1, 2)))
        token_payloads[idx] = raw
        token_entries.append((f"tensor.{idx}", "I64", (1, 2), raw))
    _write_st(ckpt / "original_input_ids.safetensors", token_entries)
    row_payloads = {}
    for row in range(5):
        raw = _payload(5000 + row, _nbytes("F16", (1, 2, 8)))
        row_payloads[row] = raw
        _write_st(rows / f"row-{row:03d}.safetensors",
                  [("tensor", "F16", (1, 2, 8), raw)])
    manifest = {
        "schema": "mimo-head-postsave-v1", "mode": "recover", "complete": True,
        "checkpoint_sha256": {
            "job.json": _file_sha(ckpt / "job.json"),
            "state.safetensors": _file_sha(ckpt / "state.safetensors"),
            "original_input_ids.safetensors": _file_sha(ckpt / "original_input_ids.safetensors"),
        },
        "rows": [{"row": r, "file": f"row-{r:03d}.safetensors",
                  "sha256": _file_sha(rows / f"row-{r:03d}.safetensors")} for r in range(5)],
        "diagnostic": "tiny-ok",
    }
    with open(rows / "rows-completed.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return {"ckpt": ckpt, "rows": rows, "job": job, "state": state_payloads,
            "tokens": token_payloads, "row": row_payloads}


def _input_hashes(fix):
    paths = [fix["ckpt"] / "job.json", fix["ckpt"] / "state.safetensors",
             fix["ckpt"] / "original_input_ids.safetensors",
             fix["rows"] / "rows-completed.json"]
    paths += [fix["rows"] / f"row-{r:03d}.safetensors" for r in range(5)]
    return {str(p): _file_sha(p) for p in paths}


def _assert_no_marker(test, out):
    test.assertFalse(os.path.lexists(os.path.join(str(out), asm.COMPLETION_FILENAME)),
                     "completion marker must be absent on failure")


class AssembleTests(unittest.TestCase):
    def test_success_preserves_all_rows_mixed_shapes_tokens_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            out = Path(tmp) / "out"
            before = _input_hashes(fix)
            record = asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(out),
                                             _spec=_TINY, _chunk_size=7)
            self.assertTrue(record["uncommitted"] is True)
            for name in ("job.json", "state.safetensors",
                         "original_input_ids.safetensors", asm.COMPLETION_FILENAME):
                self.assertTrue((out / name).is_file(), name)
            with open(out / "job.json", encoding="utf-8") as handle:
                job = json.load(handle)
            self.assertEqual(job["next_module_idx"], 35)
            self.assertIsNone(job["q_strategy"])
            self.assertEqual(job["bad_rows"], [])
            self.assertEqual(job["run_id"], "tiny-fixture")
            with open(fix["ckpt"] / "original_input_ids.safetensors", "rb") as a, \
                    open(out / "original_input_ids.safetensors", "rb") as b:
                self.assertEqual(a.read(), b.read())
            header, data_start, size = _parse_header(out / "state.safetensors")
            self.assertEqual(list(header.keys()), [f"tensor.{i}" for i in range(128)])
            for idx in range(128):
                entry = header[f"tensor.{idx}"]
                want_shape = [1, 2, 8] if idx < 5 else [1, 2, 4]
                self.assertEqual(entry["dtype"], "F16")
                self.assertEqual(entry["shape"], want_shape)
                begin, end = entry["data_offsets"]
                got = _read_payload(out / "state.safetensors", data_start, begin, end)
                want = fix["row"][idx] if idx < 5 else fix["state"][idx]
                self.assertEqual(got, want, f"row {idx}")
            self.assertEqual(len(record["rows"]), 128)
            for item in record["rows"]:
                row = item["row"]
                self.assertEqual(item["payload_sha256"], item["output_payload_sha256"])
                want = fix["row"][row] if row < 5 else fix["state"][row]
                self.assertEqual(item["payload_sha256"], hashlib.sha256(want).hexdigest())
            self.assertEqual(_input_hashes(fix), before)
            self.assertIn("no quality", record["note"].lower())
            self.assertEqual(record["outputs"]["original_input_ids.safetensors"]["sha256"],
                             before[str(fix["ckpt"] / "original_input_ids.safetensors")])

    def test_source_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            out = Path(tmp) / "out"
            with open(fix["ckpt"] / "state.safetensors", "r+b") as handle:
                header, data_start, _ = _parse_header(fix["ckpt"] / "state.safetensors")
                handle.seek(data_start + 3)
                handle.write(b"\xFF")
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(out),
                                        _spec=_TINY, _chunk_size=13)
            _assert_no_marker(self, out)
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            out = Path(tmp) / "out"
            with open(fix["rows"] / "row-002.safetensors", "r+b") as handle:
                header, data_start, _ = _parse_header(fix["rows"] / "row-002.safetensors")
                handle.seek(data_start + 1)
                handle.write(b"\x00")
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(out),
                                        _spec=_TINY, _chunk_size=13)
            _assert_no_marker(self, out)

    def test_malformed_geometry_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            out = Path(tmp) / "out"
            bad_entries = []
            for idx in range(128):
                if idx == 60:
                    raw = _payload(999, _nbytes("F16", (1, 2, 5)))
                    bad_entries.append((f"tensor.{idx}", "F16", (1, 2, 5), raw))
                else:
                    bad_entries.append((f"tensor.{idx}", "F16", (1, 2, 4), fix["state"][idx]))
            _write_st(fix["ckpt"] / "state.safetensors", bad_entries)
            manifest_path = fix["rows"] / "rows-completed.json"
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["checkpoint_sha256"]["state.safetensors"] = _file_sha(fix["ckpt"] / "state.safetensors")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(out),
                                        _spec=_TINY, _chunk_size=11)
            _assert_no_marker(self, out)

    def test_noncontiguous_offsets_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            out = Path(tmp) / "out"
            header, data_start, size = _parse_header(fix["ckpt"] / "state.safetensors")
            header["tensor.1"]["data_offsets"] = [999, 999 + 16]
            blob = json.dumps(header, separators=(",", ":")).encode()
            with open(fix["ckpt"] / "state.safetensors", "rb") as handle:
                handle.seek(data_start)
                payload = handle.read()
            with open(fix["ckpt"] / "state.safetensors", "wb") as handle:
                handle.write(_U64.pack(len(blob)))
                handle.write(blob)
                handle.write(payload)
            manifest_path = fix["rows"] / "rows-completed.json"
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["checkpoint_sha256"]["state.safetensors"] = _file_sha(fix["ckpt"] / "state.safetensors")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(out),
                                        _spec=_TINY, _chunk_size=9)
            _assert_no_marker(self, out)

    def test_truncated_payloads_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            out = Path(tmp) / "out"
            path = fix["ckpt"] / "state.safetensors"
            with open(path, "rb") as handle:
                data = handle.read()
            with open(path, "wb") as handle:
                handle.write(data[:-10])
            manifest_path = fix["rows"] / "rows-completed.json"
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["checkpoint_sha256"]["state.safetensors"] = _file_sha(path)
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(out),
                                        _spec=_TINY, _chunk_size=9)
            _assert_no_marker(self, out)

    def test_traversal_duplicate_and_bool_rows_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            manifest_path = fix["rows"] / "rows-completed.json"
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["rows"][0]["file"] = "../evil.safetensors"
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(Path(tmp) / "o1"),
                                        _spec=_TINY, _chunk_size=17)
            _assert_no_marker(self, Path(tmp) / "o1")
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            manifest_path = fix["rows"] / "rows-completed.json"
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["rows"][4] = dict(manifest["rows"][0])
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(Path(tmp) / "o2"),
                                        _spec=_TINY, _chunk_size=17)
            _assert_no_marker(self, Path(tmp) / "o2")
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            manifest_path = fix["rows"] / "rows-completed.json"
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["rows"][1]["row"] = True
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(Path(tmp) / "o3"),
                                        _spec=_TINY, _chunk_size=17)
            _assert_no_marker(self, Path(tmp) / "o3")

    def test_existing_and_nested_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            out = Path(tmp) / "out"
            out.mkdir()
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(out),
                                        _spec=_TINY, _chunk_size=23)
            _assert_no_marker(self, out)
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            nested = fix["ckpt"] / "nested-out"
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(nested),
                                        _spec=_TINY, _chunk_size=23)
            _assert_no_marker(self, nested)

    def test_write_failure_leaves_partial_without_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            out = Path(tmp) / "out"
            real_open = open

            def failing_open(file, mode="r", *args, **kwargs):
                name = str(file)
                if "w" in str(mode) and name.endswith("original_input_ids.safetensors") \
                        and str(out) in name:
                    raise OSError("injected write failure")
                return real_open(file, mode, *args, **kwargs)

            with mock.patch("builtins.open", side_effect=failing_open):
                with self.assertRaises((asm.AssemblyError, OSError)):
                    asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(out),
                                            _spec=_TINY, _chunk_size=7)
            self.assertTrue(out.is_dir())
            self.assertTrue((out / "state.safetensors").is_file())
            _assert_no_marker(self, out)

    def test_cli_has_no_shape_overrides_and_fails_closed(self):
        parser = asm.build_parser()
        dests = {action.dest for action in parser._actions}
        self.assertIn("checkpoint", dests)
        self.assertIn("rows", dests)
        self.assertIn("output", dests)
        for banned in ("shape", "row", "rows_count", "chunk", "dtype"):
            self.assertNotIn(banned, dests)
        with tempfile.TemporaryDirectory() as tmp:
            code = asm.main(["--checkpoint", str(Path(tmp) / "nope"),
                             "--rows", str(Path(tmp) / "nope2"),
                             "--output", str(Path(tmp) / "o")])
            self.assertEqual(code, 1)
            _assert_no_marker(self, Path(tmp) / "o")


    def test_duplicate_keys_and_nan_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            manifest_path = fix["rows"] / "rows-completed.json"
            with open(manifest_path, encoding="utf-8") as handle:
                raw = handle.read()
            dup = raw.replace('"row": 0', '"row": 0, "row": 0', 1)
            self.assertNotEqual(dup, raw)
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(dup)
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(Path(tmp) / "o4"),
                                        _spec=_TINY, _chunk_size=17)
            _assert_no_marker(self, Path(tmp) / "o4")
        with tempfile.TemporaryDirectory() as tmp:
            fix = _make_fixture(tmp)
            manifest_path = fix["rows"] / "rows-completed.json"
            with open(manifest_path, encoding="utf-8") as handle:
                raw = handle.read()
            nan = raw.replace('"diagnostic": "tiny-ok"', '"diagnostic": NaN', 1)
            self.assertNotEqual(nan, raw)
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(nan)
            with self.assertRaises(asm.AssemblyError):
                asm.assemble_checkpoint(str(fix["ckpt"]), str(fix["rows"]), str(Path(tmp) / "o5"),
                                        _spec=_TINY, _chunk_size=17)
            _assert_no_marker(self, Path(tmp) / "o5")


if __name__ == "__main__":
    unittest.main()
