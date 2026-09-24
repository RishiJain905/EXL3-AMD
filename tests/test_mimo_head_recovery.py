"""CPU checks for the forward-only recovery input gates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/recover_mimo_head.py"
spec = importlib.util.spec_from_file_location("mimo_head_recovery", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecoveryGates(unittest.TestCase):
    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "input.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
                p.write_text(text)
                with self.assertRaises(ValueError): module.read_json(p)

    def test_committed_job_requires_exact_stage_and_zero_rejected_rows(self):
        valid = dict(next_module_idx=34, q_strategy=None, bad_rows=[])
        module.check_job(valid)
        for key, value in (("next_module_idx",35), ("next_module_idx",True),
                           ("q_strategy",{}), ("bad_rows",[0]), ("bad_rows",None)):
            with self.assertRaises(ValueError): module.check_job(dict(valid, **{key:value}))
        with self.assertRaises(ValueError): module.check_job(dict(valid, extra=True))

    def manifest(self):
        names = ["lm_head.safetensors", "model.language_model.embed_tokens.safetensors",
                 "model.language_model.norm.safetensors"]
        names += [f"model.language_model.layers.{i}.safetensors" for i in range(32)]
        return [dict(file=name, bytes=128, sha256="a"*64) for name in names]

    def test_packed_manifest_requires_every_module_once(self):
        manifest = self.manifest(); module.check_packed_manifest(manifest)
        for bad in (manifest[:-1], manifest+[manifest[0]], [manifest[0]]+manifest[:-1]):
            with self.assertRaises(ValueError): module.check_packed_manifest(bad)

    def test_packed_manifest_rejects_traversal_and_invalid_hashes(self):
        for key, value in (("file","../lm_head.safetensors"), ("file",12),
                           ("sha256","x"*64), ("sha256","a"*63),
                           ("bytes",True), ("bytes",-1), ("sha256",None)):
            manifest = self.manifest(); manifest[0][key] = value
            with self.assertRaises(ValueError): module.check_packed_manifest(manifest)

    def test_geometry_keeps_padded_head_and_original_dtype(self):
        fields = dict(key="lm_head", in_features=4096, out_features=248320,
                      num_slices=1, trim_padded_out=False, out_dtype=None, caps={"logits_output":True})
        module.check_geometry(types.SimpleNamespace(**fields))
        for key, value in (("out_features",248077), ("trim_padded_out",True),
                           ("out_dtype","float32"), ("num_slices",2), ("caps",{})):
            with self.assertRaises(ValueError):
                module.check_geometry(types.SimpleNamespace(**dict(fields, **{key:value})))

    def test_existing_completion_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "completion.json"
            module.write_new(p, {"complete":False})
            with self.assertRaises(FileExistsError): module.write_new(p, {"complete":True})
            self.assertEqual(json.loads(p.read_text()), {"complete":False})

    def test_execution_requires_both_permissions_and_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); config = root / "config.toml"; output = root / "output"
            argv = ["--config",str(config),"--work",str(root/"work"),"--recovery",str(root/"evidence"),
                    "--output",str(output),"--mode","probe"]
            for allow_inference, allow_probes, execute in ((False,True,True),(True,False,True),(True,True,False)):
                config.write_text(f"[execution]\nallow_local_inference={str(allow_inference).lower()}\n"
                                  f"allow_backend_probes={str(allow_probes).lower()}\n")
                with self.assertRaises(SystemExit) as error:
                    module.main(argv + (["--execute"] if execute else []))
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
