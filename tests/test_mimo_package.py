"""CPU-only checks for the MiMo EXL3 packaging/audit CLI; tiny synthetic fixtures."""
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("package_mimo_under_test",
                                               SCRIPTS / "package_mimo_exl3.py")
package = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(package)

IN = 16
BODY_OUT = 16
HEAD_OUT = 32
ATTN_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
MUL1 = struct.pack("<I", 0x83DCD12D)
DTYPE_BYTES = {"F16": 2, "BF16": 2, "I16": 2, "I32": 4}


def tensor_bytes(dtype, shape, seed):
    count = 1
    for dim in shape:
        count *= dim
    size = count * DTYPE_BYTES[dtype]
    return bytes((seed + i) % 251 for i in range(size))


def write_safetensors(path, tensors):
    header, payload, offset = {}, bytearray(), 0
    for index, (name, (dtype, shape)) in enumerate(tensors):
        raw = tensor_bytes(dtype, shape, index + 1)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        payload += raw
        offset += len(raw)
    blob = json.dumps(header).encode()
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(blob)))
        stream.write(blob)
        stream.write(payload)


def quantized_bases():
    bases = []
    for layer in range(32):
        prefix = f"model.language_model.layers.{layer}"
        bases += [f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj", f"{prefix}.mlp.down_proj"]
        if layer in ATTN_LAYERS:
            bases += [f"{prefix}.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        else:
            bases += [f"{prefix}.linear_attn.in_proj_qkv", f"{prefix}.linear_attn.in_proj_z",
                      f"{prefix}.linear_attn.out_proj"]
    bases.append("lm_head")
    assert len(bases) == 201
    return bases


def passthrough_keys():
    keys = ["model.language_model.embed_tokens.weight"]
    for layer in range(32):
        prefix = f"model.language_model.layers.{layer}"
        keys += [f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"]
        if layer not in ATTN_LAYERS:
            keys += [f"{prefix}.linear_attn.in_proj_a.weight",
                     f"{prefix}.linear_attn.in_proj_b.weight"]
    return keys


def vision_keys():
    return [f"model.visual.blocks.{index // 12}.tensor_{index % 12}.weight" for index in range(333)]


def dims_for(base):
    if base == "lm_head":
        return IN, HEAD_OUT, 6
    return IN, BODY_OUT, 4


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.source = self.root / "source"
        self.text = self.root / "text"
        self.vision = self.root / "vision"
        self.output = self.root / "output"
        self.report = self.root / "report.json"
        for path in (self.source, self.text, self.vision):
            path.mkdir(parents=True)
        self.bases = quantized_bases()
        self.pass_keys = passthrough_keys()
        self.vis_keys = vision_keys()
        self.build()

    def build(self):
        embed_shape = [HEAD_OUT, IN]
        write_safetensors(self.source / "model-00001-of-00001.safetensors",
                          [(self.pass_keys[0], ("BF16", embed_shape))])
        (self.source / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": {"hidden_size": 4096, "num_hidden_layers": 32,
                            "mtp_num_hidden_layers": 1, "vocab_size": 248320},
            "vision_config": {"hidden_size": 1152}, "tie_word_embeddings": False,
        }))
        (self.source / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 1024},
            "weight_map": {self.pass_keys[0]: "model-00001-of-00001.safetensors"}}))
        (self.source / "tokenizer.json").write_text('{"tok": true}')
        (self.source / "merges.txt").write_text("a b\n")
        (self.source / "chat_template.jinja").write_text("{{ x }}")
        (self.source / "README.md").write_text("readme")
        (self.source / ".gitattributes").write_text("*.safetensors filter=lfs\n")
        vision_tensors = [(key, ("BF16", [IN, IN] if i % 2 else [IN]))
                          for i, key in enumerate(self.vis_keys)]
        write_safetensors(self.vision / "vision.safetensors", vision_tensors)
        vision_hashes = {}
        vision_payload = (self.vision / "vision.safetensors").read_bytes()
        header_size, = struct.unpack("<Q", vision_payload[:8])
        header = json.loads(vision_payload[8:8 + header_size])
        for key in self.vis_keys:
            begin, end = header[key]["data_offsets"]
            chunk = vision_payload[8 + header_size + begin:8 + header_size + end]
            vision_hashes[key] = hashlib.sha256(chunk).hexdigest()
        tensors = {}
        for base in self.bases:
            in_features, out_features, _ = dims_for(base)
            key = "lm_head.weight" if base == "lm_head" else base + ".weight"
            tensors[key] = {"dtype": "BF16", "shape": [out_features, in_features],
                            "bytes": out_features * in_features * 2,
                            "parameters": out_features * in_features}
        for key in self.pass_keys:
            shape = embed_shape if key == self.pass_keys[0] else [IN]
            if key.endswith(("in_proj_a.weight", "in_proj_b.weight")):
                shape = [BODY_OUT, IN]
            tensors[key] = {"dtype": "BF16", "shape": shape, "bytes": 2,
                            "parameters": 1}
        tensors[self.pass_keys[0]]["shard"] = "model-00001-of-00001.safetensors"
        for i, key in enumerate(self.vis_keys):
            shape = [IN, IN] if i % 2 else [IN]
            tensors[key] = {"dtype": "BF16", "shape": shape, "bytes": 2, "parameters": 1}
        audit = {"tensors": tensors, "vision_tensors": 333,
                 "vision_payload_sha256": vision_hashes,
                 "vision_file_sha256": hashlib.sha256(vision_payload).hexdigest(),
                 "source_parameters": 1000000, "mtp_tensors": 0,
                 "source_files": {"config.json": {"bytes": 1, "sha256": "00"}}}
        self.audit_path = self.root / "source-audit.json"
        self.audit_path.write_text(json.dumps(audit))
        linears = []
        for base in self.bases:
            in_features, out_features, _ = dims_for(base)
            head = base == "lm_head"
            linears.append({"key": base, "component": "text",
                            "qmap": "block" if head else "block.mlp.input",
                            "qbits_key": "head_bits" if head else "bits",
                            "in_features": in_features, "out_features": out_features,
                            "source_shape": [out_features, in_features]})
        for layer in range(32):
            if layer in ATTN_LAYERS:
                continue
            for suffix in ("in_proj_a", "in_proj_b"):
                linears.append({"key": f"model.language_model.layers.{layer}.linear_attn.{suffix}",
                                "component": "text", "qmap": None, "qbits_key": "bits",
                                "in_features": IN, "out_features": BODY_OUT,
                                "source_shape": [BODY_OUT, IN]})
        self.mapping_path = self.root / "mapping.json"
        self.mapping_path.write_text(json.dumps({"architecture": "Qwen3_5ForConditionalGeneration",
                                                 "linears": linears}))
        self.write_text_candidate()

    def write_text_candidate(self, mutate=None):
        tensors = []
        for base in self.bases:
            in_features, out_features, bits = dims_for(base)
            tensors += [(f"{base}.trellis", ("I16", [in_features // 16, out_features // 16, bits * 16])),
                        (f"{base}.suh", ("F16", [in_features])),
                        (f"{base}.svh", ("F16", [out_features]))]
        for key in self.pass_keys:
            if key == self.pass_keys[0]:
                tensors.append((key, ("BF16", [HEAD_OUT, IN])))
            elif key.endswith(("in_proj_a.weight", "in_proj_b.weight")):
                tensors.append((key, ("F16", [BODY_OUT, IN])))
            else:
                tensors.append((key, ("F16", [IN])))
        first, second = tensors[:600], tensors[600:]
        write_safetensors(self.text / "model-00001-of-00002.safetensors", first)
        write_safetensors(self.text / "model-00002-of-00002.safetensors", second)
        self._append_markers(mutate)
        (self.text / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "quantization_config": {"quant_method": "exl3", "bits": 4, "head_bits": 6,
                                    "codebook": "mul1"}}))
        (self.text / "quantization_config.json").write_text('{"quant_method": "exl3"}')
        (self.text / "model.safetensors.index.json").write_text('{"stale": true}')
        # Force identical embedding payload bytes between source and text candidate.
        self._copy_embed_payload()

    def _append_markers(self, mutate):
        for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            path = self.text / shard
            blob = path.read_bytes()
            header_size, = struct.unpack("<Q", blob[:8])
            header = json.loads(blob[8:8 + header_size])
            payload = bytearray(blob[8 + header_size:])
            for base in self.bases:
                key = f"{base}.mul1"
                if key in header or f"{base}.trellis" not in header:
                    continue
                value = MUL1
                if mutate and mutate.get("marker") == base:
                    value = struct.pack("<I", 1)
                header[key] = {"dtype": "I32", "shape": [],
                               "data_offsets": [len(payload), len(payload) + 4]}
                payload += value
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(payload))

    def _copy_embed_payload(self):
        key = self.pass_keys[0]
        source_blob = (self.source / "model-00001-of-00001.safetensors").read_bytes()
        header_size, = struct.unpack("<Q", source_blob[:8])
        header = json.loads(source_blob[8:8 + header_size])
        begin, end = header[key]["data_offsets"]
        embed = source_blob[8 + header_size + begin:8 + header_size + end]
        for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            path = self.text / shard
            blob = bytearray(path.read_bytes())
            size, = struct.unpack("<Q", bytes(blob[:8]))
            shard_header = json.loads(bytes(blob[8:8 + size]))
            if key not in shard_header:
                continue
            sbegin, send = shard_header[key]["data_offsets"]
            assert send - sbegin == len(embed)
            blob[8 + size + sbegin:8 + size + send] = embed
            path.write_bytes(bytes(blob))

    def args(self, **overrides):
        params = {"source": str(self.source), "text_dir": str(self.text),
                  "vision_dir": str(self.vision), "source_audit": str(self.audit_path),
                  "mapping": str(self.mapping_path), "output": str(self.output),
                  "report": str(self.report)}
        params.update(overrides)
        argv = []
        for name in ("source", "text_dir", "vision_dir", "source_audit", "mapping",
                     "output", "report"):
            if params.get(name) is None:
                continue
            argv += [f"--{name.replace('_', '-')}", params[name]]
        return argv

    def run(self, **overrides):
        return package.main(self.args(**overrides))


class PackageMimoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fix = Fixture(self.tmp.name)

    def input_hashes(self):
        hashes = {}
        for root in (self.fix.source, self.fix.text, self.fix.vision):
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(self.fix.audit_path)] = hashlib.sha256(self.fix.audit_path.read_bytes()).hexdigest()
        return hashes

    def test_valid_package_end_to_end(self):
        before = self.input_hashes()
        self.assertEqual(self.fix.run(), 0)
        self.assertEqual(before, self.input_hashes())
        report = json.loads(self.fix.report.read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["text"]["quantized_projections"], 201)
        self.assertEqual(report["text"]["bit_map"]["lm_head"], 6)
        self.assertTrue(all(value == 4 for key, value in report["text"]["bit_map"].items()
                            if key != "lm_head"))
        self.assertEqual(report["text"]["body_trellis_stored_bits_per_weight"], 4.0)
        self.assertTrue(report["text"]["embedding"]["match"])
        self.assertEqual(report["vision"]["tensors_verified"], 333)
        self.assertTrue(report["vision"]["payload_hashes_match"])
        self.assertIsNone(report["error"])
        output_config = json.loads((self.fix.output / "config.json").read_text())
        self.assertEqual(output_config["text_config"]["mtp_num_hidden_layers"], 0)
        self.assertIn("vision_config", output_config)
        self.assertEqual(output_config["quantization_config"]["quant_method"], "exl3")
        index = json.loads((self.fix.output / "model.safetensors.index.json").read_text())
        self.assertEqual(len(index["weight_map"]), 804 + len(self.fix.pass_keys) + 333)
        self.assertEqual(report["output"]["index_total_size"], index["metadata"]["total_size"])
        self.assertEqual(sum(report["output"]["components"].values()), report["output"]["total_package_bytes"])
        self.assertFalse(output_config["tie_word_embeddings"])
        total = 0
        for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                      "vision.safetensors"):
            inspected = package.inspect_shard(self.fix.output / shard)
            total += sum(t["bytes"] for t in inspected["tensors"])
        self.assertEqual(index["metadata"]["total_size"], total)
        self.assertIn("tokenizer.json", [entry["file"] for entry in report["output"]["files"]])
        self.assertNotIn("model.safetensors.index.json", report["metadata_copied_from_source"])
        self.assertIn("quantization_config.json", report["metadata_kept_from_text"])
        self.assertIn(".gitattributes", report["metadata_skipped_from_source"])

    def test_valid_package_without_mapping(self):
        self.assertEqual(self.fix.run(mapping=None), 0)
        report = json.loads(self.fix.report.read_text())
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["mapping_used"])

    def test_duplicate_tensor_fails_before_writing(self):
        blob = (self.fix.text / "model-00001-of-00002.safetensors").read_bytes()
        header_size, = struct.unpack("<Q", blob[:8])
        header = json.loads(blob[8:8 + header_size])
        first_key = next(iter(header))
        other = self.fix.text / "model-00002-of-00002.safetensors"
        other_blob = other.read_bytes()
        other_size, = struct.unpack("<Q", other_blob[:8])
        other_header = json.loads(other_blob[8:8 + other_size])
        payload = other_blob[8 + other_size:]
        other_header[first_key] = {"dtype": "F16", "shape": [IN],
                                   "data_offsets": [len(payload), len(payload) + IN * 2]}
        raw = json.dumps(other_header).encode()
        other.write_bytes(struct.pack("<Q", len(raw)) + raw + payload + bytes(IN * 2))
        self.assertEqual(self.fix.run(), 1)
        self.assertFalse(self.fix.output.exists())
        report = json.loads(self.fix.report.read_text())
        self.assertEqual(report["status"], "incomplete")
        self.assertIn("Duplicate", report["error"])

    def test_missing_tensor_fails(self):
        path = self.fix.text / "model-00001-of-00002.safetensors"
        blob = path.read_bytes()
        header_size, = struct.unpack("<Q", blob[:8])
        header = json.loads(blob[8:8 + header_size])
        victim = next(key for key in header if key.endswith(".trellis"))
        begin, end = header.pop(victim)["data_offsets"]
        payload = blob[8 + header_size:]
        kept = payload[:begin] + payload[end:]
        shift = end - begin
        for entry in header.values():
            start, stop = entry["data_offsets"]
            if start >= end:
                entry["data_offsets"] = [start - shift, stop - shift]
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + kept)
        self.assertEqual(self.fix.run(), 1)
        report = json.loads(self.fix.report.read_text())
        self.assertIn("missing", report["error"])

    def test_extra_tensor_fails(self):
        path = self.fix.text / "model-00001-of-00002.safetensors"
        blob = path.read_bytes()
        header_size, = struct.unpack("<Q", blob[:8])
        header = json.loads(blob[8:8 + header_size])
        payload = blob[8 + header_size:]
        header["surprise.weight"] = {"dtype": "F16", "shape": [IN],
                                     "data_offsets": [len(payload), len(payload) + IN * 2]}
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload + bytes(IN * 2))
        self.assertEqual(self.fix.run(), 1)
        self.assertIn("unexpected", json.loads(self.fix.report.read_text())["error"])

    def test_mtp_tensor_rejected(self):
        path = self.fix.text / "model-00001-of-00002.safetensors"
        blob = path.read_bytes()
        header_size, = struct.unpack("<Q", blob[:8])
        header = json.loads(blob[8:8 + header_size])
        payload = blob[8 + header_size:]
        header["mtp.0.embed_tokens.weight"] = {"dtype": "F16", "shape": [IN],
                                               "data_offsets": [len(payload), len(payload) + IN * 2]}
        raw = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload + bytes(IN * 2))
        self.assertEqual(self.fix.run(), 1)
        self.assertIn("MTP", json.loads(self.fix.report.read_text())["error"])

    def test_bad_marker_fails(self):
        base = self.fix.bases[0]
        key = f"{base}.mul1"
        for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            path = self.fix.text / shard
            blob = bytearray(path.read_bytes())
            size, = struct.unpack("<Q", bytes(blob[:8]))
            header = json.loads(bytes(blob[8:8 + size]))
            if key not in header:
                continue
            begin, _ = header[key]["data_offsets"]
            blob[8 + size + begin:8 + size + begin + 4] = struct.pack("<I", 1)
            path.write_bytes(bytes(blob))
        self.assertEqual(self.fix.run(), 1)
        self.assertIn("marker", json.loads(self.fix.report.read_text())["error"])

    def test_bad_rate_fails(self):
        base = self.fix.bases[0]
        key = f"{base}.trellis"
        for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            path = self.fix.text / shard
            blob = path.read_bytes()
            size, = struct.unpack("<Q", blob[:8])
            header = json.loads(blob[8:8 + size])
            if key not in header:
                continue
            begin, end = header[key]["data_offsets"]
            payload = blob[8 + size:]
            grown = payload[:begin] + bytes(64) + payload[begin:]
            header[key]["shape"] = [1, 1, 96]
            header[key]["data_offsets"] = [begin, begin + 192]
            for name, entry in header.items():
                if name == key:
                    continue
                start, stop = entry["data_offsets"]
                if start >= end:
                    entry["data_offsets"] = [start + 64, stop + 64]
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + grown)
        self.assertEqual(self.fix.run(), 1)
        self.assertIn("geometry", json.loads(self.fix.report.read_text())["error"])

    def test_output_inside_source_refused(self):
        nested = self.fix.source / "nested-output"
        self.assertEqual(self.fix.run(output=str(nested)), 1)
        self.assertFalse(nested.exists())
        self.assertIn("nested", json.loads(self.fix.report.read_text())["error"])

    def test_no_overwrite(self):
        self.fix.output.mkdir()
        self.assertEqual(self.fix.run(), 1)
        self.assertIn("overwrite", json.loads(self.fix.report.read_text())["error"].lower())
        self.fix.output.rmdir()
        self.fix.report.write_text("taken")
        self.assertEqual(self.fix.run(), 1)
        self.assertEqual(self.fix.report.read_text(), "taken")

    def test_report_inside_output_refused(self):
        self.assertEqual(self.fix.run(report=str(self.fix.output / "report.json")), 1)
        self.assertFalse(self.fix.output.exists())

    def test_vision_tamper_fails(self):
        path = self.fix.vision / "vision.safetensors"
        blob = bytearray(path.read_bytes())
        blob[-1] ^= 1
        path.write_bytes(bytes(blob))
        self.assertEqual(self.fix.run(), 1)
        self.assertIn("Vision", json.loads(self.fix.report.read_text())["error"])

    def test_embedding_payload_mismatch_fails(self):
        key = self.fix.pass_keys[0]
        for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            path = self.fix.text / shard
            blob = bytearray(path.read_bytes())
            size, = struct.unpack("<Q", bytes(blob[:8]))
            header = json.loads(bytes(blob[8:8 + size]))
            if key not in header:
                continue
            begin, _ = header[key]["data_offsets"]
            blob[8 + size + begin] ^= 1
            path.write_bytes(bytes(blob))
        self.assertEqual(self.fix.run(), 1)
        self.assertIn("Embedding", json.loads(self.fix.report.read_text())["error"])

    def test_vision_filename_collision_renamed(self):
        target = self.fix.text / "vision.safetensors"
        (self.fix.text / "model-00002-of-00002.safetensors").rename(target)
        self.assertEqual(self.fix.run(), 0)
        report = json.loads(self.fix.report.read_text())
        self.assertEqual(report["vision"]["output_file"], "vision-preserved.safetensors")
        self.assertTrue((self.fix.output / "vision-preserved.safetensors").is_file())

    def test_partial_failure_preserved_with_marker(self):
        calls = []
        original = package.copy_file

        def flaky(source, destination):
            calls.append(str(destination))
            if len(calls) == 2:
                raise OSError("simulated disk failure")
            return original(source, destination)

        package.copy_file = flaky
        try:
            self.assertEqual(self.fix.run(), 1)
        finally:
            package.copy_file = original
        marker = json.loads((self.fix.output / "PACKAGING-INCOMPLETE.json").read_text())
        self.assertEqual(marker["status"], "incomplete")
        self.assertTrue((self.fix.output / "model-00001-of-00002.safetensors").is_file())
        report = json.loads(self.fix.report.read_text())
        self.assertEqual(report["status"], "incomplete")

    def test_help_needs_no_torch(self):
        self.assertNotIn("torch", sys.modules)
        with self.assertRaises(SystemExit) as raised:
            package.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertNotIn("torch", sys.modules)




if __name__ == "__main__":
    unittest.main()
