import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_exl3_candidate import DTYPE_BYTES, audit, inspect_shard


def write_shard(path, header, payload=b"\0" * 8):
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class CandidateAuditTests(unittest.TestCase):
    def test_exact_payload_container_and_category_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"quantization_config":{"quant_method":"exl3"}}')
            header = {key: {"dtype": "I16", "shape": [1], "data_offsets": [i * 2, i * 2 + 2]}
                      for i, key in enumerate(("model.language_model.embed_tokens.weight", "lm_head.trellis",
                                               "mtp.fc.trellis", "unrecognized.scale"))}
            shard = root / "model.safetensors"
            write_shard(shard, header)
            result = audit(root)
            self.assertEqual(result["tensor_payload_bytes"], {"target": 4, "mtp": 2, "unknown": 2})
            self.assertEqual(result["embedding_payload_bytes_in_target"], 2)
            self.assertEqual(result["combined_shard_file_bytes"], shard.stat().st_size)
            self.assertEqual(result["combined_candidate_file_bytes"], sum(p.stat().st_size for p in root.iterdir()))
            self.assertEqual(result["conservative_target_plus_all_overhead_bytes"], result["combined_candidate_file_bytes"] - 4)
            self.assertEqual(len(result["shards"][0]["sha256"]), 64)

    def test_malformed_payload_boundaries_and_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.safetensors"
            for dtype, shape, offsets in [("I64", [2], [0, 8]), ("U8", [9], [0, 9]),
                                           ("U8", [7], [1, 8]), ("unknown", [8], [0, 8]),
                                           ("U8", [-8], [0, 8])]:
                write_shard(path, {"x": {"dtype": dtype, "shape": shape, "data_offsets": offsets}})
                with self.assertRaises(ValueError):
                    inspect_shard(path)
            write_shard(path, {key: {"dtype": "I32", "shape": [1], "data_offsets": offsets}
                               for key, offsets in (("x", [0, 4]), ("y", [2, 6]))})
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                inspect_shard(path)

    def test_duplicate_shard_tensors_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"quantization_config":{"quant_method":"exl3"}}')
            for name in ("a", "b"):
                write_shard(root / f"{name}.safetensors", {"x": {"dtype": "I64", "shape": [], "data_offsets": [0, 8]}})
            with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
                audit(root)

    def test_existing_output_preserved_before_candidate_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out.json"
            output.write_text("retained")
            result = subprocess.run([sys.executable, str(SCRIPTS / "audit_exl3_candidate.py"),
                                     "--candidate", str(root / "absent"), "--output", str(output)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("new file", result.stderr)
            self.assertEqual(output.read_text(), "retained")


def write_tensors(path, tensors):
    header, offset = {}, 0
    for key, dtype, shape in tensors:
        size = math.prod(shape) * DTYPE_BYTES[dtype]
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    write_shard(path, header, b"\0" * offset)


class CodebookAuditTests(unittest.TestCase):
    def test_mixed_codebooks_malformed_orphan_conflicting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"quantization_config":{"quant_method":"exl3"}}')
            write_tensors(root / "model.safetensors", [
                ("model.layers.0.q_proj.trellis", "I16", [1, 1, 32]),
                ("model.layers.0.k_proj.trellis", "I16", [1, 1, 48]),
                ("model.layers.0.k_proj.mcg", "I32", []),
                ("model.layers.0.v_proj.trellis", "I16", [1, 1, 64]),
                ("model.layers.0.v_proj.mul1", "I32", []),
                ("model.layers.0.o_proj.trellis", "I16", [2, 1, 32]),
                ("model.layers.0.o_proj.mcg", "I32", []),
                ("model.layers.0.o_proj.mul1", "I32", []),
                ("model.layers.0.x_proj.mul1", "I32", []),
                ("model.layers.1.q_proj.trellis", "I16", [1, 1, 32]),
                ("model.layers.1.q_proj.mul1", "I16", [1]),
                ("model.layers.1.k_proj.trellis", "I16", [1, 1, 20]),
            ])
            summary = audit(root)["codebook_summary"]
            self.assertEqual(summary["projections_with_trellis"], 6)
            self.assertEqual(summary["codebook_counts"],
                             {"3inst": 2, "mcg": 1, "mul1": 2, "conflicting": 1})
            self.assertEqual(summary["packed_bit_width_histogram"], {"2": 3, "3": 1, "4": 1})
            self.assertEqual(summary["orphan_markers"], ["model.layers.0.x_proj.mul1"])
            self.assertEqual(summary["conflicting_bases"], ["model.layers.0.o_proj"])
            self.assertEqual([m["key"] for m in summary["malformed_markers"]],
                             ["model.layers.1.q_proj.mul1"])
            issues = "\n".join(summary["issues"])
            self.assertIn("Conflicting codebook markers", issues)
            self.assertIn("Orphan codebook marker", issues)
            self.assertIn("Malformed codebook marker", issues)
            self.assertIn("invalid for bit-width inference", issues)

    def test_max_total_bytes_includes_mtp_and_overhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"quantization_config":{"quant_method":"exl3"}}')
            write_tensors(root / "model.safetensors", [
                ("model.weight", "I16", [1]),
                ("mtp.fc.trellis", "I16", [1, 1, 32]),
            ])
            default = audit(root)
            self.assertIsNone(default["max_total_bytes"])
            self.assertIsNone(default["combined_total_within_max_bytes"])
            total = default["combined_candidate_file_bytes"]
            self.assertGreater(total, default["tensor_payload_bytes"]["target"])
            self.assertTrue(audit(root, max_total_bytes=total)["combined_total_within_max_bytes"])
            self.assertFalse(audit(root, max_total_bytes=total - 1)["combined_total_within_max_bytes"])
            self.assertFalse(audit(root, max_total_bytes=default["tensor_payload_bytes"]["target"])
                             ["combined_total_within_max_bytes"])
            for bad in (0, -1, "100", 1.5, True):
                with self.assertRaises(ValueError):
                    audit(root, max_total_bytes=bad)

    def test_cli_max_total_bytes_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "cand"
            candidate.mkdir()
            (candidate / "config.json").write_text('{"quantization_config":{"quant_method":"exl3"}}')
            write_tensors(candidate / "model.safetensors", [("model.weight", "I16", [1])])
            total = audit(candidate)["combined_candidate_file_bytes"]
            ok = subprocess.run([sys.executable, str(SCRIPTS / "audit_exl3_candidate.py"),
                                 "--candidate", str(candidate), "--output", str(root / "ok.json")],
                                capture_output=True, text=True, timeout=30)
            self.assertEqual(ok.returncode, 0)
            capped = subprocess.run([sys.executable, str(SCRIPTS / "audit_exl3_candidate.py"),
                                     "--candidate", str(candidate), "--output", str(root / "capped.json"),
                                     "--max-total-bytes", str(total - 1)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(capped.returncode, 1)
            self.assertIn("exceeds", capped.stderr)
            self.assertFalse(json.loads((root / "capped.json").read_text())
                             ["combined_total_within_max_bytes"])
            bad = subprocess.run([sys.executable, str(SCRIPTS / "audit_exl3_candidate.py"),
                                  "--candidate", str(candidate), "--output", str(root / "bad.json"),
                                  "--max-total-bytes", "0"],
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(bad.returncode, 2)
            self.assertFalse((root / "bad.json").exists())
