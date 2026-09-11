import unittest

from quantlab.methods.exl3.source_map import (
    RUNTIME_PREFIX,
    SOURCE_PREFIX,
    tensor_name_fixes,
)


class SourceMapTests(unittest.TestCase):
    def test_source_keys_match_adapter_without_changing_side_models(self):
        raw = SOURCE_PREFIX + "layers.0.linear_attn.in_proj_qkv.weight"
        unchanged = ["lm_head.weight", "mtp.fc.weight", "model.visual.norm.weight"]
        self.assertEqual(
            tensor_name_fixes([raw, *unchanged]),
            {raw: RUNTIME_PREFIX + "layers.0.linear_attn.in_proj_qkv.weight"},
        )

    def test_suffix_application_preserves_header_offsets(self):
        raw = SOURCE_PREFIX + "embed_tokens.weight"
        header = {raw: {"data_offsets": [12, 28], "shape": [2, 4], "dtype": "BF16"}}
        for old, new in tensor_name_fixes(header).items():
            header = {key[:-len(old)] + new if key.endswith(old) else key: value
                      for key, value in header.items()}
        self.assertEqual(header[RUNTIME_PREFIX + "embed_tokens.weight"]["data_offsets"], [12, 28])

    def test_canonical_input_needs_no_fixes(self):
        self.assertEqual(tensor_name_fixes([RUNTIME_PREFIX + "norm.weight"]), {})

    def test_vision_prefix_is_also_canonicalized(self):
        raw = "model.language_model.visual.blocks.0.attn.proj.weight"
        self.assertEqual(tensor_name_fixes([raw]), {raw: "model.visual.blocks.0.attn.proj.weight"})

    def test_collision_fails_in_either_order(self):
        keys = [SOURCE_PREFIX + "norm.weight", RUNTIME_PREFIX + "norm.weight"]
        for order in [keys, keys[::-1]]:
            with self.assertRaisesRegex(ValueError, "Duplicate tensor destination"):
                tensor_name_fixes(order)

    def test_duplicate_input_fails(self):
        with self.assertRaises(ValueError):
            tensor_name_fixes(["mtp.fc.weight", "mtp.fc.weight"])

    def test_empty_name_fails(self):
        with self.assertRaises(ValueError):
            tensor_name_fixes([""])


if __name__ == "__main__":
    unittest.main()
