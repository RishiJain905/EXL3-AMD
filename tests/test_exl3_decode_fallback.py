"""CPU-only small-M graph admission tests with stub runtime classes; no GPU math is tested."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from quantlab.methods.exl3.decode_diagnostics import prepare_decode_fusions


class FakeTensor:
    def __init__(self, rows):
        self.shape = (1, rows, 8)

    def numel(self):
        return self.shape[1] * 8


class GraphFallbackTests(unittest.TestCase):
    def setUp(self):
        class LinearEXL3:
            """Isinstance stand-in for exllamav3's LinearEXL3."""

        class FakeInner(LinearEXL3):
            def __init__(self, mul1, codebooks=(0,), K=4):
                self.mul1 = mul1
                self._quantlab_smallm_codebooks = codebooks
                self.K = K
                self.mcg = False
                self.in_features = 128
                self.out_features = 256

        class GatedDeltaNet:
            key = "gdn"

        class GatedMLP:
            key = "gated-mlp"

        class MLP:
            key = "mlp"

        class Attention:
            key = "attention"

        class FakeGDN(GatedDeltaNet):
            def __init__(self, inners, fail=False):
                self._inners = list(inners)
                self._fail = fail
                self.bc = "native-bc"
                self.bc_split = True
                self.calls = []

            def __iter__(self):
                for inner in self._inners:
                    yield SimpleNamespace(inner=inner)

            def forward(self, x, params, marker=None):
                self.calls.append((self.bc, self.bc_split, marker))
                if self._fail:
                    raise ValueError("test failure")
                return marker

        class Outer:
            def __init__(self, *children):
                self._children = children

            def __iter__(self):
                return iter(self._children)

        self.FakeInner = FakeInner
        self.FakeGDN = FakeGDN
        self.Outer = Outer
        specs = {
            "exllamav3.modules.quant.exl3": {"LinearEXL3": LinearEXL3},
            "exllamav3.modules.attn": {"Attention": Attention},
            "exllamav3.modules.gated_delta_net": {"GatedDeltaNet": GatedDeltaNet},
            "exllamav3.modules.mlp": {"MLP": MLP, "GatedMLP": GatedMLP},
        }
        modules = {}
        for name, attrs in specs.items():
            modules[name] = ModuleType(name)
            modules[name].__dict__.update(attrs)
        runtime_patch = patch.dict(sys.modules, modules)
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)

    def prepare(self, gdn, mode="gdn", **kwargs):
        records = prepare_decode_fusions(
            self.Outer(gdn), mode, native_smallm_graph=True,
            native_smallm_max_rows=3, **kwargs)
        self.assertEqual(len(records), 1)
        return records[0]

    def test_old_binary_mul1_declines_multirow_but_keeps_single_row(self):
        gdn = self.FakeGDN([self.FakeInner(True, codebooks=(0,))])
        record = self.prepare(gdn)
        self.assertEqual(record["allowed_rows"], [1])
        marker = object()
        self.assertIs(gdn.forward(FakeTensor(1), {}, marker), marker)
        self.assertEqual(gdn.calls, [("native-bc", True, marker)])
        self.assertIsNone(gdn.bc)
        self.assertFalse(gdn.bc_split)
        gdn.forward(FakeTensor(2), {})
        gdn.forward(FakeTensor(3), {})
        self.assertEqual(gdn.calls[1:], [(None, False, None), (None, False, None)])
        self.assertIsNone(gdn.bc)
        self.assertFalse(gdn.bc_split)
        self.assertEqual(
            (record["fused_calls"], record["fallback_calls"]), (1, 2))

    def test_new_binary_mul1_admits_multirow(self):
        gdn = self.FakeGDN([self.FakeInner(True, codebooks=(0, 2))])
        record = self.prepare(gdn)
        self.assertEqual(record["allowed_rows"], [1, 2, 3])
        for rows in (1, 2, 3):
            gdn.forward(FakeTensor(rows), {})
            self.assertIsNone(gdn.bc)
            self.assertFalse(gdn.bc_split)
        self.assertEqual(
            [call[:2] for call in gdn.calls],
            [("native-bc", True)] * 3)
        gdn.forward(FakeTensor(4), {})
        self.assertEqual(gdn.calls[3][:2], (None, False))
        self.assertEqual(
            (record["fused_calls"], record["fallback_calls"]), (3, 1))

    def test_mixed_projections_decline_multirow(self):
        gdn = self.FakeGDN([
            self.FakeInner(False, codebooks=(0,)),
            self.FakeInner(True, codebooks=(0,)),
        ])
        record = self.prepare(gdn)
        self.assertEqual(record["allowed_rows"], [1])
        gdn.forward(FakeTensor(1), {})
        gdn.forward(FakeTensor(2), {})
        self.assertEqual(
            [call[:2] for call in gdn.calls],
            [("native-bc", True), (None, False)])
        self.assertEqual(
            (record["fused_calls"], record["fallback_calls"]), (1, 1))

    def test_no_packed_projections_declines_multirow(self):
        gdn = self.FakeGDN([])
        record = self.prepare(gdn)
        self.assertEqual(record["allowed_rows"], [1])
        gdn.forward(FakeTensor(1), {})
        gdn.forward(FakeTensor(2), {})
        self.assertEqual(
            (record["fused_calls"], record["fallback_calls"]), (1, 1))

    def test_exception_restores_disabled_handles(self):
        gdn = self.FakeGDN(
            [self.FakeInner(True, codebooks=(0, 2))], fail=True)
        record = self.prepare(gdn)
        self.assertEqual(record["allowed_rows"], [1, 2, 3])
        with self.assertRaises(ValueError):
            gdn.forward(FakeTensor(2), {})
        self.assertEqual(gdn.calls, [("native-bc", True, None)])
        self.assertIsNone(gdn.bc)
        self.assertFalse(gdn.bc_split)

    def test_mgemv_graph_mode_still_rejected(self):
        gdn = self.FakeGDN([self.FakeInner(True, codebooks=(0, 2))])
        with self.assertRaisesRegex(ValueError, "MultiLinear"):
            prepare_decode_fusions(
                self.Outer(gdn), "gdn-mlp-mgemv",
                native_smallm_graph=True, native_smallm_max_rows=3)


if __name__ == "__main__":
    unittest.main()
