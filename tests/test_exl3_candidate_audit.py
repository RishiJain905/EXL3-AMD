import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from audit_exl3_candidate import audit, inspect_shard


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
