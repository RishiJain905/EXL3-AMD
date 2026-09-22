"""MTP-absent source adapter and estimate labeling; no native deps, no payloads."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_inspect_module():
    name = "inspect_exl3_source_under_test"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "inspect_exl3_source.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inspector = load_inspect_module()

TEXT_KEYS = (
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.input_layernorm.weight",
)
MTP_KEYS = ("mtp.fc.weight", "mtp.norm.weight")
VISION_KEYS = ("model.visual.blocks.0.attn.proj.weight", "model.visual.embed.weight")
FAKE_SHAPES = {
    TEXT_KEYS[0]: (32, 16),
    TEXT_KEYS[1]: (64, 16),
    TEXT_KEYS[2]: (16,),
    MTP_KEYS[0]: (16, 32),
    MTP_KEYS[1]: (16,),
}


class _FakeTensorCollection:
    def __init__(self, headers):
        self.file_headers = headers


class _FakeBaseConfig:
    def __init__(self, directory, text_cfg="text_config", text_model=None,
                 vision_model=None, mtp_model=None, **kwargs):
        self.architecture = "Qwen3_5ForConditionalGeneration"
        self.model_classes = ({"text": text_model} if text_model else {}) | (
            {"mtp": mtp_model} if mtp_model else {})
        raw = json.loads((Path(directory) / "config.json").read_text())
        text = raw[text_cfg] if text_cfg else raw
        self.vocab_size = text["vocab_size"]
        self.hidden_size = text["hidden_size"]
        self.mtp_num_hidden_layers = text.get("mtp_num_hidden_layers", 0)
        if self.mtp_num_hidden_layers == 0 and "mtp" in self.model_classes:
            del self.model_classes["mtp"]
        weight_map = json.loads((Path(directory) / "model.safetensors.index.json").read_text())["weight_map"]
        headers = {}
        for key in weight_map:
            canonical = key
            for old, new in self.get_tensor_name_fixes().items():
                if canonical.endswith(old):
                    canonical = canonical[: len(canonical) - len(old)] + new
            shape = list(FAKE_SHAPES.get(canonical, (2, 2)))
            size = 2
            for dim in shape:
                size *= dim
            headers[canonical] = {"shape": shape, "dtype": "BF16", "data_offsets": [0, size]}
        self.stc = _FakeTensorCollection({"model.safetensors": headers})

    def get_tensor_name_fixes(self):
        return {}


class _FakeVLConfig(_FakeBaseConfig):
    pass


class _FakeLinear:
    def __init__(self, key, in_features, out_features, qmap="block.mlp", qbits_key="body_bits"):
        self.key = key
        self.modules = []
        self.qmap = qmap
        self.qbits_key = qbits_key
        self.in_features = self.in_features_unpadded = in_features
        self.out_features = self.out_features_unpadded = out_features


class _FakeEmbedding:
    def __init__(self, key):
        self.key = key
        self.modules = []


class _FakeRMSNorm:
    def __init__(self, key, unweighted=False):
        self.key = key
        self.modules = []
        self.tensor_key = key + ".weight"
        self.unweighted = unweighted


class _FakeGatedDeltaNet:
    pass


class _FakeTextModel:
    def __init__(self, config, **kwargs):
        self.modules = [
            _FakeLinear("model.language_model.layers.0.mlp.up_proj", 16, 32),
            _FakeEmbedding("model.language_model.embed_tokens"),
            _FakeRMSNorm("model.language_model.layers.0.input_layernorm"),
        ]

    def __iter__(self):
        return iter(self.modules)


class _FakeMTPModel:
    def __init__(self, config, **kwargs):
        self.modules = [
            _FakeLinear("mtp.fc", 32, 16, qbits_key="mtp_bits"),
            _FakeRMSNorm("mtp.norm"),
        ]

    def __iter__(self):
        return iter(self.modules)


class _FakeModelNamespace:
    @staticmethod
    def from_config(config, component="text", **kwargs):
        assert component in config.model_classes, \
            f"{config.architecture} does not define a '{component}' component model"
        model = config.model_classes[component](config, **kwargs)
        model.component = component
        return model


def install_fake_exllamav3(test):
    previous = {name: sys.modules.get(name) for name in (
        "exllamav3", "exllamav3.architecture", "exllamav3.architecture.qwen3_5",
        "exllamav3.architecture.qwen3_5_mtp", "exllamav3.modules")}
    top = types.ModuleType("exllamav3")
    top.Model = _FakeModelNamespace
    arch = types.ModuleType("exllamav3.architecture")
    qwen = types.ModuleType("exllamav3.architecture.qwen3_5")
    qwen.Qwen3_5VLConfig = _FakeVLConfig
    qwen.Qwen3_5VLBaseConfig = _FakeBaseConfig
    qwen.Qwen3_5VLModel = _FakeTextModel
    qwen_mtp = types.ModuleType("exllamav3.architecture.qwen3_5_mtp")
    qwen_mtp.Qwen3_5MTPModel = _FakeMTPModel
    modules = types.ModuleType("exllamav3.modules")
    modules.Linear = _FakeLinear
    modules.Embedding = _FakeEmbedding
    modules.RMSNorm = _FakeRMSNorm
    modules.GatedDeltaNet = _FakeGatedDeltaNet
    sys.modules.update({
        "exllamav3": top, "exllamav3.architecture": arch,
        "exllamav3.architecture.qwen3_5": qwen,
        "exllamav3.architecture.qwen3_5_mtp": qwen_mtp,
        "exllamav3.modules": modules})

    def restore():
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    test.addCleanup(restore)


def write_fake_source(directory, *, mtp_keys=MTP_KEYS, mtp_declared=1):
    directory = Path(directory)
    weight_map = {key: "model.safetensors" for key in TEXT_KEYS + tuple(mtp_keys) + VISION_KEYS}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (directory / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"vocab_size": 64, "hidden_size": 16, "mtp_num_hidden_layers": mtp_declared},
    }))


class SourceInspectionTests(unittest.TestCase):
    def test_canonical_tensor_name_applies_suffix_fixes(self):
        self.assertEqual(inspector.canonical_tensor_name("a.b", {}), "a.b")
        fixes = {"language_model.language_model.mtp.fc.weight": "language_model.mtp.fc.weight"}
        self.assertEqual(
            inspector.canonical_tensor_name(
                "model.language_model.language_model.mtp.fc.weight", fixes),
            "model.language_model.mtp.fc.weight")
        self.assertEqual(
            inspector.canonical_tensor_name("model.language_model.layers.0.weight", fixes),
            "model.language_model.layers.0.weight")

    def test_is_mtp_tensor_name(self):
        for key in ("mtp", "mtp.fc.weight", "mtp.layers.0.mlp.up_proj.trellis",
                    "mtp.layers.0.mlp.up_proj.suh", "mtp.layers.0.mlp.up_proj.mul1",
                    "model.language_model.mtp.fc.weight"):
            self.assertTrue(inspector.is_mtp_tensor_name(key), key)
        for key in ("model.language_model.layers.0.mlp.up_proj.weight",
                    "model.visual.blocks.0.attn.proj.weight", "lm_head.weight"):
            self.assertFalse(inspector.is_mtp_tensor_name(key), key)

    def test_count_mtp_tensors_honors_source_map_canonicalization(self):
        from quantlab.methods.exl3.source_map import SOURCE_PREFIX, tensor_name_fixes
        legacy = SOURCE_PREFIX + "mtp.fc.weight"
        fixes = tensor_name_fixes([legacy, "mtp.norm.weight", TEXT_KEYS[0]])
        self.assertEqual(inspector.count_mtp_tensors([legacy, "mtp.norm.weight", TEXT_KEYS[0]], fixes), 2)
        self.assertEqual(inspector.count_mtp_tensors(list(TEXT_KEYS) + ["lm_head.weight"], {}), 0)

    def test_absent_mtp_suppressed_and_reported(self):
        install_fake_exllamav3(self)
        with tempfile.TemporaryDirectory() as tmp:
            write_fake_source(tmp, mtp_keys=())
            config, model, mtp, fix_count = inspector.construct_models(tmp)
            self.assertIsNone(mtp)
            self.assertNotIn("mtp", config.model_classes)
            self.assertEqual(config.mtp_num_hidden_layers, 1)
            self.assertEqual(config.mtp_source_tensors, 0)
            self.assertTrue(config.mtp_suppressed_missing_tensors)
            self.assertEqual(fix_count, 0)
            result = inspector.inspect(tmp)
            self.assertEqual(result["text_modules"], 3)
            self.assertEqual(result["mtp_modules"], 0)
            self.assertEqual(result["mtp_declared_layers"], 1)
            self.assertEqual(result["mtp_source_tensors"], 0)
            self.assertEqual(result["mtp_status"], inspector.MTP_STATUS_ABSENT_SUPPRESSED)
            self.assertEqual(result["missing_required_tensors"], [])
            self.assertEqual(result["shape_mismatches"], [])
            self.assertEqual(result["unconsumed_text_mtp_tensors"], [])
            self.assertEqual(result["excluded_vision_tensors"], 2)
            self.assertEqual([b["head_bits"] for b in result["size_estimates"]], [2, 3, 4])
            self.assertTrue(all(b["mtp_bits"] is None for b in result["size_estimates"]))
            self.assertIn("no draft recipe", result["estimate_scope"])

    def test_partial_mtp_preserved_so_missing_required_surfaces(self):
        install_fake_exllamav3(self)
        with tempfile.TemporaryDirectory() as tmp:
            write_fake_source(tmp, mtp_keys=("mtp.fc.weight",))
            config, model, mtp, _ = inspector.construct_models(tmp)
            self.assertIsNotNone(mtp)
            self.assertIn("mtp", config.model_classes)
            self.assertEqual(config.mtp_source_tensors, 1)
            self.assertFalse(config.mtp_suppressed_missing_tensors)
            result = inspector.inspect(tmp)
            self.assertEqual(result["mtp_modules"], 2)
            self.assertEqual(result["mtp_status"], inspector.MTP_STATUS_PRESENT)
            self.assertEqual(result["missing_required_tensors"], ["mtp.norm.weight"])
            self.assertTrue(all(b["mtp_bits"] == 4 for b in result["size_estimates"]))

    def test_explicit_mtp_component_rejected_when_suppressed(self):
        install_fake_exllamav3(self)
        with tempfile.TemporaryDirectory() as tmp:
            write_fake_source(tmp, mtp_keys=())
            config, _, _, _ = inspector.construct_models(tmp)
            with self.assertRaisesRegex(AssertionError, "mtp"):
                _FakeModelNamespace.from_config(config, component="mtp")

    def test_estimate_budgets_labeling_and_options(self):
        linears = [
            {"key": "body", "component": "text", "qmap": "q", "qbits_key": "body_bits",
             "in_features": 16, "out_features": 32, "source_shape": [32, 16]},
            {"key": "head", "component": "text", "qmap": "q", "qbits_key": "head_bits",
             "in_features": 8, "out_features": 8, "source_shape": [8, 8]},
        ]
        selected = {"body.weight": {"data_offsets": [0, 1024]},
                    "head.weight": {"data_offsets": [0, 128]},
                    "other.weight": {"data_offsets": [0, 100]}}
        (budget,) = inspector.estimate_budgets(
            linears, selected, body_bits=2, head_bits_options=(3,), mtp_present=False)
        self.assertEqual(budget["text_bits"], 2)
        self.assertEqual(budget["head_bits"], 3)
        self.assertIsNone(budget["mtp_bits"])
        self.assertEqual(budget["unchanged_source_tensor_bytes"], 100)
        self.assertEqual(budget["packed_trellis_bytes"], 128 + 24)
        self.assertEqual(budget["scale_and_codebook_marker_bytes"], 100 + 36)
        self.assertEqual(budget["padding_bytes_included_in_trellis"], 0)
        self.assertEqual(budget["derived_tensor_bytes"], 100 + 152 + 136)
        self.assertEqual(budget["quantized_linear_count"], 2)
        (present,) = inspector.estimate_budgets(
            linears, selected, head_bits_options=(3,), mtp_bits=5, mtp_present=True)
        self.assertEqual(present["mtp_bits"], 5)


if __name__ == "__main__":
    unittest.main()
