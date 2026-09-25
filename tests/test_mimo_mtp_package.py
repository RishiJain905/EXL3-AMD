"""CPU-only checks for the MiMo MTP donor packager; tiny synthetic fixtures."""
import hashlib
import contextlib
import io
import importlib.util
import json
import math
import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("package_mimo_mtp_under_test",
                                               SCRIPTS / "package_mimo_mtp.py")
package = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(package)

HIDDEN = 4
HEADS = 2
HEAD_DIM = 2
KV_HEADS = 1
INTERMEDIATE = 6
LAYERS = 2
VOCAB = 32
TARGET_SHARD = "model-00001.safetensors"
REVISION = "a" * 40

EXPECTED_MTP_SHAPES = {
    "mtp.fc.weight": [4, 8],
    "mtp.pre_fc_norm_hidden.weight": [4],
    "mtp.pre_fc_norm_embedding.weight": [4],
    "mtp.norm.weight": [4],
    "mtp.layers.0.input_layernorm.weight": [4],
    "mtp.layers.0.post_attention_layernorm.weight": [4],
    "mtp.layers.0.self_attn.q_proj.weight": [8, 4],
    "mtp.layers.0.self_attn.k_proj.weight": [2, 4],
    "mtp.layers.0.self_attn.v_proj.weight": [2, 4],
    "mtp.layers.0.self_attn.o_proj.weight": [4, 4],
    "mtp.layers.0.self_attn.q_norm.weight": [2],
    "mtp.layers.0.self_attn.k_norm.weight": [2],
    "mtp.layers.0.mlp.gate_proj.weight": [6, 4],
    "mtp.layers.0.mlp.up_proj.weight": [6, 4],
    "mtp.layers.0.mlp.down_proj.weight": [4, 6],
}
TARGET_TENSORS = [
    ("model.embed_tokens.weight", "F32", [4, 4]),
    ("lm_head.weight", "F32", [4, 4]),
    ("model.visual.patch_embed.weight", "F32", [2, 2]),
]


def text_config(mtp_layers):
    return {
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-06,
        "full_attention_interval": 4,
        "layer_types": ["full_attention", "full_attention"],
        "attn_output_gate": True,
        "attention_bias": False,
        "hidden_act": "silu",
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 1,
        "linear_value_head_dim": 2,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "linear"},
        "head_dim": HEAD_DIM,
        "mtp_num_hidden_layers": mtp_layers,
        "mtp_use_dedicated_embeddings": False,
    }


def tokenizer_obj():
    return {
        "version": "1.0",
        "model": {"vocab": {"<unk>": 0, "hello": 1, "world": 2, "!": 3}},
        "added_tokens": [],
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def target_payload(dtype, shape, seed):
    count = math.prod(shape) * {"F32": 4, "BF16": 2, "F16": 2, "U8": 1}[dtype]
    return bytes((seed + i) % 251 for i in range(count))


def bf16_finite_payload(numel, seed):
    out = bytearray(2 * numel)
    for i in range(numel):
        out[2 * i] = (seed + i) % 256
        out[2 * i + 1] = 0x3F
    return bytes(out)


def write_safetensors(path, specs, metadata=None):
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    payload = bytearray()
    offset = 0
    for name, dtype, shape, raw in specs:
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        payload += raw
        offset += len(raw)
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(blob)))
        stream.write(blob)
        stream.write(payload)


def parse_safetensors_raw(path):
    data = Path(path).read_bytes()
    (header_size,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + header_size].decode("utf-8"))
    payload_base = 8 + header_size
    slices = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        slices[name] = data[payload_base + begin:payload_base + end]
    return header_size, header, slices, data[payload_base:]


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.target = self.root / "target"
        self.donor = self.root / "donor"
        self.target_manifest = self.root / "target-manifest.json"
        self.output = self.root / "output"
        self.report = self.root / "report.json"
        self.target.mkdir()
        self.donor.mkdir()

    def build_target(self):
        target_text = text_config(0)
        target_config = {"architectures": ["Qwen3_5ForConditionalGeneration"],
                         "model_type": "qwen3_5", "text_config": target_text}
        write_json(self.target / "config.json", target_config)
        tok = tokenizer_obj()
        write_json(self.target / "tokenizer.json", tok)
        (self.target / "generation_config.json").write_text(
            json.dumps({"do_sample": False}, indent=2) + "\n", encoding="utf-8")
        specs = []
        for index, (name, dtype, shape) in enumerate(TARGET_TENSORS):
            specs.append((name, dtype, shape, target_payload(dtype, shape, index + 1)))
        write_safetensors(self.target / TARGET_SHARD, specs, {"format": "test"})
        total = sum(len(raw) for _, _, _, raw in specs)
        index = {"metadata": {"total_size": total, "format": "test",
                              "extra_meta": "keep-me"},
                 "weight_map": {name: TARGET_SHARD for name, _, _ in TARGET_TENSORS},
                 "extra_top": "marker-top"}
        write_json(self.target / "model.safetensors.index.json", index)
        manifest = []
        for child in sorted(self.target.iterdir()):
            manifest.append({"file": child.name, "bytes": child.stat().st_size,
                             "sha256": file_sha256(child)})
        write_json(self.target_manifest, manifest)
        return {"config": target_config, "text": target_text, "tokenizer": tok,
                "index": index, "total": total, "manifest": manifest}

    def build_donor(self, target_tok, tokenizer_bytes=None):
        donor_text = text_config(1)
        donor_config = {"architectures": ["Qwen3_5ForConditionalGeneration"],
                        "text_config": donor_text}
        write_json(self.donor / "config.json", donor_config)
        if tokenizer_bytes is None:
            write_json(self.donor / "tokenizer.json", target_tok)
        else:
            (self.donor / "tokenizer.json").write_bytes(tokenizer_bytes)
        (self.donor / "LICENSE").write_text("Qwen license fixture\n", encoding="utf-8")
        specs = []
        payload_hashes = {}
        for index, name in enumerate(sorted(EXPECTED_MTP_SHAPES)):
            shape = EXPECTED_MTP_SHAPES[name]
            raw = bf16_finite_payload(math.prod(shape), index + 7)
            specs.append((name, "BF16", shape, raw))
            payload_hashes[name] = hashlib.sha256(raw).hexdigest()
        write_safetensors(self.donor / "mtp.safetensors", specs)
        files = {}
        for fname in ("mtp.safetensors", "config.json", "tokenizer.json", "LICENSE"):
            path = self.donor / fname
            files[fname] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        tensors = {}
        for name in sorted(EXPECTED_MTP_SHAPES):
            shape = EXPECTED_MTP_SHAPES[name]
            tensors[name] = {"dtype": "BF16", "shape": list(shape),
                             "bytes": math.prod(shape) * 2,
                             "sha256": payload_hashes[name]}
        manifest = {"schema_version": 1, "complete": True,
                    "repository": "Qwen/Qwen3.5-9B", "revision": REVISION,
                    "files": files, "tensors": tensors}
        write_json(self.donor / "manifest.json", manifest)
        total = sum(math.prod(s) * 2 for s in EXPECTED_MTP_SHAPES.values())
        return {"config": donor_config, "text": donor_text, "files": files,
                "tensors": tensors, "payload_hashes": payload_hashes,
                "total": total, "manifest": manifest}

    def run_packager(self, output=None, report=None):
        out = self.output if output is None else Path(output)
        rep = self.report if report is None else Path(report)
        return package.run(self.target.resolve(), self.target_manifest.resolve(),
                           self.donor.resolve(), out.resolve(), rep.resolve())

    def snapshot(self):
        snap = {}
        for directory in (self.target, self.donor):
            for child in sorted(directory.iterdir()):
                if child.is_file():
                    snap[str(child.resolve())] = file_sha256(child)
        snap[str(self.target_manifest.resolve())] = file_sha256(self.target_manifest)
        return snap


def refresh_target_entry(fix, fname):
    entries = json.loads(fix.target_manifest.read_text(encoding="utf-8"))
    for entry in entries:
        if entry["file"] == fname:
            path = fix.target / fname
            entry["bytes"] = path.stat().st_size
            entry["sha256"] = file_sha256(path)
    write_json(fix.target_manifest, entries)


def refresh_donor_file(fix, fname):
    manifest_path = fix.donor / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = fix.donor / fname
    manifest["files"][fname] = {"bytes": path.stat().st_size,
                                "sha256": file_sha256(path)}
    write_json(manifest_path, manifest)


def refresh_donor_tensor_hash(fix, name):
    _, _, slices, _ = parse_safetensors_raw(fix.donor / "mtp.safetensors")
    manifest_path = fix.donor / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensors"][name]["sha256"] = hashlib.sha256(slices[name]).hexdigest()
    write_json(manifest_path, manifest)


class TestMimoMtpPackage(unittest.TestCase):
    def test_success_preservation_and_exact_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            fix = Fixture(tmp)
            target = fix.build_target()
            donor = fix.build_donor(target["tokenizer"])
            before = fix.snapshot()
            self.assertEqual(fix.run_packager(), 0)
            report = json.loads(fix.report.read_text(encoding="utf-8"))
            self.assertTrue(report["complete"])
            self.assertEqual(report["status"], "complete")
            self.assertIsNone(report["error"])
            self.assertTrue(report["target"]["preserved"])
            # Target sources unchanged (config still mtp=0, index still original total).
            for key, value in before.items():
                self.assertEqual(file_sha256(Path(key)), value, key)
            target_config_after = json.loads((fix.target / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(target_config_after["text_config"]["mtp_num_hidden_layers"], 0)
            # Output inventory = target files + 3 additions.
            expected_files = {e["file"] for e in target["manifest"]} | {
                "mtp-donor-bf16.safetensors", "MTP-DONOR-LICENSE", "mtp-donor.json"}
            actual_files = {p.name for p in fix.output.iterdir() if p.is_file()}
            self.assertEqual(actual_files, expected_files)
            # Preserved files byte-exact.
            for entry in target["manifest"]:
                fname = entry["file"]
                if fname in ("config.json", "model.safetensors.index.json"):
                    continue
                self.assertEqual(file_sha256(fix.output / fname), entry["sha256"], fname)
            # Exact config delta: only mtp 0->1.
            output_config = json.loads((fix.output / "config.json").read_text(encoding="utf-8"))
            expected_config = json.loads(json.dumps(target["config"]))
            expected_config["text_config"]["mtp_num_hidden_layers"] = 1
            self.assertEqual(output_config, expected_config)
            self.assertEqual(report["config_change"],
                             {"text_config.mtp_num_hidden_layers": {"from": 0, "to": 1}})
            # Exact index delta: deep-copy + 15 entries + total_size only.
            output_index = json.loads((fix.output / "model.safetensors.index.json").read_text(encoding="utf-8"))
            self.assertEqual(output_index["extra_top"], "marker-top")
            self.assertEqual(output_index["metadata"]["extra_meta"], "keep-me")
            self.assertEqual(output_index["metadata"]["format"], "test")
            self.assertEqual(output_index["metadata"]["total_size"],
                             target["total"] + donor["total"])
            for key, value in target["index"]["weight_map"].items():
                self.assertEqual(output_index["weight_map"][key], value)
            added = sorted(set(output_index["weight_map"]) - set(target["index"]["weight_map"]))
            self.assertEqual(added, sorted(EXPECTED_MTP_SHAPES))
            for name in added:
                self.assertEqual(output_index["weight_map"][name], "mtp-donor-bf16.safetensors")
            self.assertEqual(report["index_change"]["previous_total_size"], target["total"])
            self.assertEqual(report["index_change"]["new_total_size"],
                             target["total"] + donor["total"])
            # Provenance and report donor hashes match fixture payload hashes.
            provenance = json.loads((fix.output / "mtp-donor.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["donor"]["revision"], REVISION)
            for name, sha in donor["payload_hashes"].items():
                self.assertEqual(provenance["donor"]["tensors"][name]["sha256"], sha)
                self.assertEqual(report["donor"]["tensors"][name]["sha256"], sha)
            _, _, copied, _ = parse_safetensors_raw(fix.output / "mtp-donor-bf16.safetensors")
            for name, raw in copied.items():
                self.assertEqual(hashlib.sha256(raw).hexdigest(), donor["payload_hashes"][name])


class TestMtpRejectionAndPreservation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fix = Fixture(self.temp.name)
        self.target = self.fix.build_target()
        self.donor = self.fix.build_donor(self.target['tokenizer'])

    def reject(self, message, *, output=None, report=None, path_error=False):
        before = self.fix.snapshot()
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            code = self.fix.run_packager(output=output, report=report)
        self.assertEqual(code, 1)
        self.assertIn(message.lower(), errors.getvalue().lower())
        self.assertEqual(self.fix.snapshot(), before)
        if not path_error:
            evidence = json.loads(self.fix.report.read_text())
            self.assertIs(evidence['complete'], False)
            self.assertFalse(self.fix.output.exists())

    def rewrite_donor(self, transform):
        path = self.fix.donor / 'mtp.safetensors'
        _, header, raw, _ = parse_safetensors_raw(path)
        specs = [(name, entry['dtype'], entry['shape'], raw[name])
                 for name, entry in header.items() if name != '__metadata__']
        write_safetensors(path, transform(specs))
        refresh_donor_file(self.fix, path.name)

    def test_tokenizer_formatting_and_key_order_are_accepted(self):
        tok = self.target['tokenizer']
        (self.fix.donor / 'tokenizer.json').write_text(json.dumps(tok, sort_keys=True, separators=(',', ':')))
        refresh_donor_file(self.fix, 'tokenizer.json')
        self.assertEqual(self.fix.run_packager(), 0)

    def test_changed_token_id_is_rejected(self):
        tok = tokenizer_obj()
        tok['model']['vocab']['hello'] = 19
        write_json(self.fix.donor / 'tokenizer.json', tok)
        refresh_donor_file(self.fix, 'tokenizer.json')
        self.reject('Tokenizer mismatch')

    def test_changed_geometry_is_rejected(self):
        config = self.donor['config']
        config['text_config']['head_dim'] = 4
        write_json(self.fix.donor / 'config.json', config)
        refresh_donor_file(self.fix, 'config.json')
        self.reject('Geometry mismatch for head_dim')

    def test_dedicated_embedding_donor_is_rejected(self):
        config = self.donor['config']
        config['text_config']['mtp_use_dedicated_embeddings'] = True
        write_json(self.fix.donor / 'config.json', config)
        refresh_donor_file(self.fix, 'config.json')
        self.reject('Donor mtp_use_dedicated_embeddings')

    def test_target_hash_tamper_is_rejected(self):
        path = self.fix.target / TARGET_SHARD
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        self.reject('Target file hash mismatch')

    def test_donor_hash_tamper_is_rejected(self):
        path = self.fix.donor / 'mtp.safetensors'
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        self.reject('Donor file hash mismatch')

    def test_tensor_digest_tamper_is_rejected(self):
        path = self.fix.donor / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['tensors']['mtp.fc.weight']['sha256'] = '0' * 64
        write_json(path, manifest)
        self.reject('Donor payload hash mismatch')

    def test_missing_tensor_is_rejected(self):
        self.rewrite_donor(lambda specs: specs[1:])
        self.reject('Donor is missing')

    def test_extra_tensor_is_rejected(self):
        self.rewrite_donor(lambda specs: specs + [('mtp.unexpected', 'BF16', [1], b'\x00\x3f')])
        self.reject('Donor has 1 unexpected')

    def test_wrong_shape_is_rejected(self):
        def changed(specs):
            name, dtype, shape, raw = specs[0]
            return [(name, dtype, [2, 16], raw)] + specs[1:]
        self.rewrite_donor(changed)
        self.reject('Donor shape mismatch')

    def test_wrong_dtype_is_rejected(self):
        self.rewrite_donor(lambda specs: [(specs[0][0], 'F16', specs[0][2], specs[0][3])] + specs[1:])
        self.reject('Donor tensor dtype must be BF16')

    def nonfinite(self, word):
        def changed(specs):
            name, dtype, shape, raw = specs[0]
            return [(name, dtype, shape, struct.pack('<H', word) + raw[2:])] + specs[1:]
        self.rewrite_donor(changed)
        refresh_donor_tensor_hash(self.fix, 'mtp.fc.weight')
        self.reject('Non-finite BF16')

    def test_infinity_is_rejected(self):
        self.nonfinite(0x7F80)

    def test_nan_is_rejected(self):
        self.nonfinite(0xFFC1)

    def test_overlapping_payload_is_rejected(self):
        path = self.fix.donor / 'mtp.safetensors'
        _, header, _, payload = parse_safetensors_raw(path)
        key = list(header)[1]
        begin, end = header[key]['data_offsets']
        header[key]['data_offsets'] = [begin - 2, end - 2]
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
        refresh_donor_file(self.fix, path.name)
        self.reject('Overlapping or uncovered payload')

    def test_unaccounted_trailing_payload_is_rejected(self):
        path = self.fix.donor / 'mtp.safetensors'
        path.write_bytes(path.read_bytes() + b'\x00\x00')
        refresh_donor_file(self.fix, path.name)
        self.reject('Uncovered trailing payload')

    def test_target_index_points_to_wrong_shard(self):
        path = self.fix.target / 'model.safetensors.index.json'
        obj = json.loads(path.read_text())
        obj['weight_map']['lm_head.weight'] = 'wrong.safetensors'
        write_json(path, obj)
        refresh_target_entry(self.fix, path.name)
        self.reject('Target index shard mismatch')

    def test_target_total_size_must_equal_tensor_payload(self):
        path = self.fix.target / 'model.safetensors.index.json'
        obj = json.loads(path.read_text())
        obj['metadata']['total_size'] += 8
        write_json(path, obj)
        refresh_target_entry(self.fix, path.name)
        self.reject('Target index total_size differs')

    def test_incomplete_donor_manifest_is_rejected(self):
        path = self.fix.donor / 'manifest.json'
        obj = json.loads(path.read_text())
        obj['complete'] = False
        write_json(path, obj)
        self.reject('complete must be true')

    def test_unsafe_manifest_file_name_is_rejected(self):
        entries = self.target['manifest']
        entries[0]['file'] = '../outside.json'
        write_json(self.fix.target_manifest, entries)
        self.reject('Unsafe target file name')

    def test_duplicate_manifest_name_is_rejected(self):
        entries = self.target['manifest']
        write_json(self.fix.target_manifest, entries + entries[:1])
        self.reject('Duplicate target file name')

    def test_existing_output_is_preserved(self):
        self.fix.output.mkdir()
        sentinel = self.fix.output / 'sentinel'
        sentinel.write_bytes(b'keep')
        self.reject('already exists', path_error=True)
        self.assertEqual(sentinel.read_bytes(), b'keep')
        self.assertFalse(self.fix.report.exists())

    def test_existing_report_is_preserved(self):
        self.fix.report.write_bytes(b'keep')
        self.reject('already exists', path_error=True)
        self.assertEqual(self.fix.report.read_bytes(), b'keep')
        self.assertFalse(self.fix.output.exists())

    def test_output_inside_target_is_rejected(self):
        dest = self.fix.target / 'new-output'
        self.reject('separate, non-nested', output=dest, path_error=True)
        self.assertFalse(dest.exists())
        self.assertFalse(self.fix.report.exists())

    def test_report_inside_target_creates_no_file(self):
        report = self.fix.target / 'unwanted-report.json'
        self.reject('--report must be outside', report=report, path_error=True)
        self.assertFalse(report.exists())
        self.assertFalse(self.fix.output.exists())

    def test_report_inside_donor_creates_no_file(self):
        report = self.fix.donor / 'unwanted-report.json'
        self.reject('--report must be outside', report=report, path_error=True)
        self.assertFalse(report.exists())

    def test_casefold_collision_is_rejected(self):
        name = 'MTP-DONOR.JSON'
        path = self.fix.target / name
        path.write_bytes(b'keep')
        entries = self.target['manifest'] + [dict(file=name, bytes=4, sha256=file_sha256(path))]
        write_json(self.fix.target_manifest, entries)
        self.reject('Output name collision')

    def test_failed_copy_preserves_partial_output_and_failure_report(self):
        before = self.fix.snapshot()
        original = package.copy_file
        calls = []
        def fail_second(source, dest):
            calls.append(dest)
            if len(calls) == 2:
                raise OSError('fixture disk failure')
            original(source, dest)
        with patch.object(package, 'copy_file', fail_second), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.fix.run_packager(), 1)
        self.assertEqual(self.fix.snapshot(), before)
        self.assertTrue(calls[0].is_file())
        report = json.loads(self.fix.report.read_text())
        self.assertIs(report['complete'], False)
        self.assertIn('fixture disk failure', report['error'])

    def test_source_config_change_during_copy_prevents_completion(self):
        original = package.copy_file
        def tamper_after_copy(source, dest):
            original(source, dest)
            (self.fix.target / 'config.json').write_text('{}')
        with patch.object(package, 'copy_file', tamper_after_copy), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.fix.run_packager(), 1)
        report = json.loads(self.fix.report.read_text())
        self.assertIs(report['complete'], False)
        self.assertIn('Target input changed during packaging: config.json', report['error'])

    def test_dangling_report_symlink_rejected_before_resolution(self):
        destination = self.fix.root / 'must-not-be-created'
        try:
            self.fix.report.symlink_to(destination)
        except OSError as error:
            self.skipTest(f'Host cannot create symlink: {error}')
        argv = ['--target', str(self.fix.target), '--target-manifest', str(self.fix.target_manifest),
                '--donor-directory', str(self.fix.donor), '--output', str(self.fix.output),
                '--report', str(self.fix.report)]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(package.main(argv), 1)
        self.assertTrue(self.fix.report.is_symlink())
        self.assertFalse(destination.exists())
        self.assertFalse(self.fix.output.exists())


class TestPinnedTokenizerCompatibility(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / 'target.json'
        self.donor = Path(self.temp.name) / 'donor.json'
        self.a = tokenizer_obj()
        self.a['added_tokens'] = [dict(id=248044, content='<eos>', special=True)]
        self.a['model']['merges'] = [['h', 'e']]
        self.b = json.loads(json.dumps(self.a))
        self.b['model']['merges'] = ['h e']
        self.a['added_tokens'] += [dict(id=k, content=v, special=True)
                                   for k, v in package.MIMO_EXTRA_TOKENS.items()]

    def check(self, *, pinned=True):
        write_json(self.target, self.a)
        write_json(self.donor, self.b)
        if not pinned:
            return package.compare_tokenizers(self.target, self.donor)
        with patch.object(package, 'MIMO_TOKENIZER_SHA256', file_sha256(self.target)), \
                patch.object(package, 'QWEN_TOKENIZER_SHA256', file_sha256(self.donor)):
            return package.compare_tokenizers(self.target, self.donor)

    def test_reviewed_pair_records_differences_without_claiming_json_equality(self):
        result = self.check()
        self.assertEqual(result['mode'], 'pinned_mimo_shared_target')
        self.assertFalse(result['json_semantically_equal'])
        self.assertEqual(result['tokenizer_used'], 'target')
        self.assertEqual(len(result['target_only_added_tokens']), 7)

    def test_unreviewed_pair_is_rejected_even_with_matching_shared_ids(self):
        with self.assertRaisesRegex(package.PackagingError, 'Tokenizer mismatch'):
            self.check(pinned=False)

    def test_changed_base_id_is_rejected(self):
        self.b['model']['vocab']['hello'] = 20
        with self.assertRaisesRegex(package.PackagingError, 'shared vocabulary IDs'):
            self.check()

    def test_changed_merges_are_rejected(self):
        self.b['model']['merges'] = ['h x']
        with self.assertRaisesRegex(package.PackagingError, 'BPE merge'):
            self.check()

    def test_changed_common_special_is_rejected(self):
        self.b['added_tokens'][0]['content'] = 'different'
        with self.assertRaisesRegex(package.PackagingError, 'shared added-token'):
            self.check()

    def test_unexpected_target_extra_is_rejected(self):
        self.a['added_tokens'][-1]['content'] = 'different'
        with self.assertRaisesRegex(package.PackagingError, 'unexpected target-only'):
            self.check()


if __name__ == "__main__":
    unittest.main()
