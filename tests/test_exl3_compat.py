"""CPU-only dispatch tests with stub runtime classes; no GPU math is tested."""

import math
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from quantlab.methods.exl3.compat import install, prepare_loaded_module


class Tensor:
    def __init__(self, shape, dtype="half", contiguous=True):
        self.shape, self.dtype, self.contiguous = tuple(shape), dtype, contiguous
        self.writes = []

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def view(self, *shape):
        assert math.prod(shape) == self.numel()
        return Tensor(shape, self.dtype)

    def __getitem__(self, index):
        start, stop, step = index.indices(self.shape[0])
        return Tensor((len(range(start, stop, step)), *self.shape[1:]), self.dtype)

    def new_empty(self, shape):
        return Tensor(shape, self.dtype)

    def __setitem__(self, index, value):
        self.writes.append((index, value.shape, value.dtype))


class CompatTests(unittest.TestCase):
    def setUp(self):
        class Module:
            def __init__(self, *children):
                self.modules = children
                self.bc = object()
                self.multi_gu = [object(), object()]
                self.multi_kv = object()
                self.multi_qg = object()
                self.bc_split = True

            def __iter__(self):
                yield self
                for child in self.modules:
                    yield from child

        class Linear(Module):
            key = "projection"
            out_features = 5

            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, x, params, out_dtype=None):
                if "ovr" in params and self.key in params["ovr"]:
                    replacement = params["ovr"][self.key]
                    if replacement.inner is not self:
                        return replacement.forward(x, params, out_dtype)
                self.calls.append((x.shape, params, out_dtype))
                return Tensor((*x.shape[:-1], self.out_features), out_dtype or "half")

        self.Linear = Linear
        self.Attention = type("Attention", (Module,), {})
        self.GDN = type("GatedDeltaNet", (Module,), {})
        self.GatedMLP = type("GatedMLP", (Module,), {})
        self.MLP = type("MLP", (Module,), {})
        self.Module = Module
        specs = {
            "exllamav3.modules.quant.exl3": {"LinearEXL3": Linear},
            "exllamav3.modules.attn": {"Attention": self.Attention, "_bc_attn_enable": False},
            "exllamav3.modules.gated_delta_net": {"GatedDeltaNet": self.GDN},
            "exllamav3.modules.mlp": {"MLP": self.MLP, "GatedMLP": self.GatedMLP},
        }
        modules = {}
        for name, attrs in specs.items():
            modules[name] = ModuleType(name)
            modules[name].__dict__.update(attrs)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, modules).start()
        patch.dict(os.environ, {"EXL3_BC_ATTN": "0"}).start()
        self.config = SimpleNamespace(infer_params=SimpleNamespace(no_reconstruct=True))

    def test_opt_in_and_import_order_required(self):
        with patch.dict(os.environ, {"EXL3_BC_ATTN": "1"}):
            with self.assertRaises(RuntimeError):
                install(self.config)
        sys.modules["exllamav3.modules.attn"]._bc_attn_enable = True
        with self.assertRaises(RuntimeError):
            install(self.config)

    def test_rowwise_preserves_shape_dtype_and_params(self):
        install(self.config)
        for rows in (1, 2, 3):
            with self.subTest(rows=rows):
                layer = self.Linear()
                params = {"token": object()}
                result = layer.forward(Tensor((1, rows, 7)), params, "float")
                self.assertEqual(result.shape, (1, rows, 5))
                self.assertEqual(result.dtype, "float")
                self.assertEqual(len(layer.calls), rows)
                self.assertTrue(all(call[1] is params for call in layer.calls))
                self.assertNotIn("reconstruct", params)

    def test_prefill_and_explicit_reconstruct_are_single_calls(self):
        install(self.config)
        layer = self.Linear()
        params = {"token": object()}
        layer.forward(Tensor((2, 2, 7)), params)
        self.assertEqual(len(layer.calls), 1)
        self.assertIs(layer.calls[0][1]["token"], params["token"])
        self.assertTrue(layer.calls[0][1]["reconstruct"])
        self.assertNotIn("reconstruct", params)
        explicit = {"reconstruct": True}
        layer.forward(Tensor((3, 7)), explicit)
        self.assertIs(layer.calls[-1][1], explicit)

    def test_override_receives_original_shape_once(self):
        install(self.config)
        layer = self.Linear()
        calls = []
        replacement = SimpleNamespace(inner=object())
        replacement.forward = lambda *args: calls.append(args) or "override"
        x = Tensor((1, 3, 7))
        params = {"ovr": {layer.key: replacement}}
        self.assertEqual(layer.forward(x, params), "override")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], x)
        self.assertEqual(layer.calls, [])

    def test_repeated_install_does_not_nest_and_strides_fail(self):
        install(self.config)
        forward = self.Linear.forward
        install(self.config)
        self.assertIs(self.Linear.forward, forward)
        self.assertFalse(self.config.infer_params.no_reconstruct)
        self.assertFalse(self.config.infer_params.use_mgemm(2, 128, mul1=False))
        self.assertEqual(os.environ["EXL3_GEMV"], "2")
        with self.assertRaises(AssertionError):
            self.Linear().forward(Tensor((2, 7), contiguous=False), {})

    def test_prepare_clears_bypasses_and_preserves_linear_handles(self):
        linear = self.Linear()
        handle = linear.bc
        mlp, gdn, attention = self.GatedMLP(linear), self.GDN(), self.Attention()
        root = self.Module(mlp, gdn, attention)
        root_handle = root.bc
        prepare_loaded_module(root)
        self.assertIsNone(mlp.bc)
        self.assertEqual(mlp.multi_gu, [None, None])
        self.assertIsNone(gdn.bc)
        self.assertFalse(gdn.bc_split)
        self.assertIsNone(attention.multi_kv)
        self.assertIsNone(attention.multi_qg)
        self.assertIs(linear.bc, handle)
        self.assertIs(root.bc, root_handle)
        mlp.bc = object()  # A later deferred load can recreate the bypass.
        prepare_loaded_module(root)
        self.assertIsNone(mlp.bc)

    def test_native_smallm_is_explicit_and_keeps_unsupported_fallback(self):
        install(self.config, native_smallm=True)
        for bits, mcg, mul1, width, expected in (
                (2, False, False, 128, 1), (3, False, False, 256, 1),
                (4, False, False, 128, 1), (5, False, False, 128, 3),
                (2, True, False, 128, 3), (2, False, True, 128, 3),
                (2, False, False, 127, 3)):
            layer = self.Linear()
            layer.K, layer.mcg, layer.mul1 = bits, mcg, mul1
            layer.in_features, layer.out_features = width, 256
            result = layer.forward(Tensor((1, 3, width)), {}, "float")
            self.assertEqual(result.shape, (1, 3, 256))
            self.assertEqual(len(layer.calls), expected)
        install(self.config)
        layer = self.Linear()
        layer.K, layer.mcg, layer.mul1 = 2, False, False
        layer.in_features, layer.out_features = 128, 256
        layer.forward(Tensor((2, 128)), {})
        self.assertEqual(len(layer.calls), 2)

    def supported_layer(self, bits=2, width=128):
        layer = self.Linear()
        layer.K, layer.mcg, layer.mul1 = bits, False, False
        layer.in_features, layer.out_features = width, 256
        return layer

    def test_mul1_requires_verified_capability_and_survives_reinstall(self):
        install(self.config, native_smallm=True, native_smallm_max_rows=9)
        layer = self.supported_layer()
        layer.mul1 = True
        layer.forward(Tensor((1, 9, 128)), {})
        self.assertEqual(len(layer.calls), 9)
        install(self.config, native_smallm=True, native_smallm_max_rows=9,
                native_smallm_codebooks=(0, 2))
        for bits in (2, 3, 4):
            for rows in range(2, 10):
                layer = self.supported_layer(bits)
                layer.mul1 = True
                layer.forward(Tensor((1, rows, 128)), {})
                self.assertEqual(len(layer.calls), 1)
                self.assertNotIn('reconstruct', layer.calls[0][1])
        for bits, mcg, width in ((5, False, 128), (2, True, 128), (2, False, 127)):
            layer = self.supported_layer(bits, width)
            layer.mul1, layer.mcg = True, mcg
            layer.forward(Tensor((1, 3, width)), {})
            self.assertEqual(len(layer.calls), 3)
        install(self.config, native_smallm=True, native_smallm_max_rows=9)
        layer = self.supported_layer()
        layer.mul1 = True
        layer.forward(Tensor((1, 3, 128)), {})
        self.assertEqual(len(layer.calls), 3)

    def test_invalid_codebook_capabilities_rejected(self):
        for codebooks in ((), (2,), (0, 1, 2), (False,), (0, 2.0)):
            with self.assertRaises(ValueError):
                install(self.config, native_smallm=True, native_smallm_codebooks=codebooks)

    def test_native_smallm_max_rows_validated_and_exported(self):
        for bad in (2, 4, 6, 7, 8, 10, 0, "5", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                install(self.config, native_smallm=True, native_smallm_max_rows=bad)
        install(self.config)
        self.assertEqual(os.environ["EXL3_SMALLM"], "0")
        self.assertEqual(os.environ["EXL3_SMALLM_MAX_ROWS"], "3")
        install(self.config, native_smallm=True, native_smallm_max_rows=5)
        self.assertEqual(os.environ["EXL3_SMALLM"], "1")
        self.assertEqual(os.environ["EXL3_SMALLM_MAX_ROWS"], "5")
        install(self.config, native_smallm=True, native_smallm_max_rows=9)
        self.assertEqual(os.environ["EXL3_SMALLM"], "1")
        self.assertEqual(os.environ["EXL3_SMALLM_MAX_ROWS"], "9")

    def test_native_max5_supported_rows_fuse_and_rows6_reconstructs(self):
        install(self.config, native_smallm=True, native_smallm_max_rows=5)
        for bits in (2, 3, 4):
            for rows in (4, 5):
                with self.subTest(bits=bits, rows=rows):
                    layer = self.supported_layer(bits)
                    result = layer.forward(Tensor((1, rows, 128)), {}, "float")
                    self.assertEqual(result.shape, (1, rows, 256))
                    self.assertEqual(len(layer.calls), 1)
                    self.assertNotIn("reconstruct", layer.calls[0][1])
        layer = self.supported_layer(2)
        params = {"token": object()}
        result = layer.forward(Tensor((1, 6, 128)), params)
        self.assertEqual(result.shape, (1, 6, 256))
        self.assertEqual(len(layer.calls), 1)
        self.assertTrue(layer.calls[0][1]["reconstruct"])
        self.assertNotIn("reconstruct", params)
    def test_native_max9_supported_rows_fuse_and_rows10_reconstructs(self):
        install(self.config, native_smallm=True, native_smallm_max_rows=9)
        for bits in (2, 3, 4):
            for rows in (6, 7, 8, 9):
                with self.subTest(bits=bits, rows=rows):
                    layer = self.supported_layer(bits)
                    result = layer.forward(Tensor((1, rows, 128)), {}, "float")
                    self.assertEqual(result.shape, (1, rows, 256))
                    self.assertEqual(len(layer.calls), 1)
                    self.assertNotIn("reconstruct", layer.calls[0][1])
        layer = self.supported_layer(2)
        params = {"token": object()}
        result = layer.forward(Tensor((1, 10, 128)), params)
        self.assertEqual(result.shape, (1, 10, 256))
        self.assertEqual(len(layer.calls), 1)
        self.assertTrue(layer.calls[0][1]["reconstruct"])
        self.assertNotIn("reconstruct", params)
        for bits, mcg, width in ((5, False, 128), (2, True, 128), (2, False, 127)):
            with self.subTest(bits=bits, mcg=mcg, width=width):
                layer = self.supported_layer(2)
                layer.K, layer.mcg = bits, mcg
                layer.in_features = width
                result = layer.forward(Tensor((1, 9, width)), {})
                self.assertEqual(result.shape, (1, 9, 256))
                self.assertEqual(len(layer.calls), 9)

    def test_opt_outs_and_fallbacks_hold_at_new_row_counts(self):
        install(self.config, native_smallm=False, native_smallm_max_rows=5)
        self.assertEqual(os.environ["EXL3_SMALLM_MAX_ROWS"], "5")
        for rows in (4, 5):
            with self.subTest(rows=rows):
                layer = self.supported_layer(3)
                params = {"token": object()}
                layer.forward(Tensor((1, rows, 128)), params)
                self.assertEqual(len(layer.calls), 1)
                self.assertTrue(layer.calls[0][1]["reconstruct"])
                self.assertNotIn("reconstruct", params)
        install(self.config, native_smallm=True, native_smallm_max_rows=5)
        for bits, mcg, mul1, width in ((5, False, False, 128), (2, True, False, 128),
                                       (2, False, False, 127)):
            with self.subTest(bits=bits, mcg=mcg, width=width):
                layer = self.supported_layer(2)
                layer.K, layer.mcg, layer.mul1 = bits, mcg, mul1
                layer.in_features = width
                params = {}
                result = layer.forward(Tensor((1, 4, width)), params)
                self.assertEqual(result.shape, (1, 4, 256))
                self.assertEqual(len(layer.calls), 4)
                self.assertNotIn("reconstruct", params)
        layer = self.supported_layer(2)
        layer.K = 5
        layer.forward(Tensor((1, 5, 128)), {})
        self.assertEqual(len(layer.calls), 5)
        explicit = {"reconstruct": True}
        layer = self.supported_layer(2)
        layer.forward(Tensor((1, 5, 128)), explicit)
        self.assertEqual(len(layer.calls), 1)
        self.assertIs(layer.calls[0][1], explicit)
        calls = []
        replacement = SimpleNamespace(inner=object())
        replacement.forward = lambda *args: calls.append(args) or "override"
        layer = self.supported_layer(2)
        x = Tensor((1, 5, 128))
        self.assertEqual(layer.forward(x, {"ovr": {layer.key: replacement}}), "override")
        self.assertEqual(len(calls), 1)
        self.assertEqual(layer.calls, [])

    def test_repeated_install_switches_max_rows_without_nesting(self):
        install(self.config, native_smallm=True, native_smallm_max_rows=3)
        forward = self.Linear.forward
        layer = self.supported_layer(2)
        layer.forward(Tensor((1, 4, 128)), {})
        self.assertEqual(len(layer.calls), 1)
        self.assertTrue(layer.calls[0][1]["reconstruct"])
        install(self.config, native_smallm=True, native_smallm_max_rows=5)
        self.assertIs(self.Linear.forward, forward)
        self.assertEqual(os.environ["EXL3_SMALLM_MAX_ROWS"], "5")
        layer = self.supported_layer(2)
        layer.forward(Tensor((1, 4, 128)), {})
        self.assertEqual(len(layer.calls), 1)
        self.assertNotIn("reconstruct", layer.calls[0][1])
        install(self.config)
        self.assertIs(self.Linear.forward, forward)
        self.assertEqual(os.environ["EXL3_SMALLM"], "0")
        self.assertEqual(os.environ["EXL3_SMALLM_MAX_ROWS"], "3")
        layer = self.supported_layer(2)
        layer.forward(Tensor((2, 128)), {})
        self.assertEqual(len(layer.calls), 2)


if __name__ == "__main__":
    unittest.main()
