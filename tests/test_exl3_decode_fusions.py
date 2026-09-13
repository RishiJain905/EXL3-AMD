import unittest
from types import SimpleNamespace

from quantlab.methods.exl3.decode_diagnostics import guard_single_row


class FakeTensor:
    def __init__(self, rows):
        self.shape = (1, rows, 8)
    def numel(self):
        return self.shape[1] * 8


class DecodeFusionBoundaryTests(unittest.TestCase):
    def make_module(self, fail=False):
        module = SimpleNamespace(bc=None, bc_split=False)
        calls = []
        def forward(x, params, marker=None):
            calls.append((module.bc, module.bc_split, marker))
            if fail:
                raise ValueError('test failure')
            return marker
        module.forward = forward
        counters = dict(fused_calls=0, fallback_calls=0)
        guard_single_row(module, dict(bc='native', bc_split=True), counters)
        return module, calls, counters

    def test_one_row_uses_native_and_restores_handles(self):
        module, calls, counters = self.make_module()
        marker = object()
        self.assertIs(module.forward(FakeTensor(1), {}, marker), marker)
        self.assertEqual(calls, [('native', True, marker)])
        self.assertIsNone(module.bc)
        self.assertFalse(module.bc_split)
        self.assertEqual(counters['fused_calls'], 1)

    def test_multirow_and_special_paths_preserve_fallback(self):
        for rows, params in [(2, {}), (3, {}), (256, {}), (1, {'reconstruct': True}), (1, {'ovr': {}})]:
            module, calls, counters = self.make_module()
            module.forward(FakeTensor(rows), params)
            self.assertEqual(calls, [(None, False, None)])
            self.assertEqual(counters['fallback_calls'], 1)

    def test_exception_restores_disabled_state(self):
        module, calls, counters = self.make_module(fail=True)
        with self.assertRaises(ValueError):
            module.forward(FakeTensor(1), {})
        self.assertEqual(calls, [('native', True, None)])
        self.assertIsNone(module.bc)
        self.assertFalse(module.bc_split)

    def test_projection_list_is_exposed_only_for_one_row(self):
        disabled = [None]
        native = [object()]
        module = SimpleNamespace(bc=None, multi_gu=disabled)
        module.forward = lambda x, params: module.multi_gu
        counters = dict(fused_calls=0, fallback_calls=0)
        guard_single_row(module, dict(bc='native', multi_gu=native), counters)
        self.assertIs(module.forward(FakeTensor(1), {}), native)
        self.assertIs(module.multi_gu, disabled)
        self.assertIs(module.forward(FakeTensor(3), {}), disabled)

    def test_explicit_smallm_rows_do_not_enable_prefill_or_overrides(self):
        module = SimpleNamespace(bc=None)
        module.forward = lambda x, params: module.bc
        counters = dict(fused_calls=0, fallback_calls=0)
        guard_single_row(module, dict(bc='native'), counters, allowed_rows=(1, 2, 3))
        for rows in (1, 2, 3):
            self.assertEqual(module.forward(FakeTensor(rows), {}), 'native')
            self.assertIsNone(module.bc)
        for rows, params in ((4, {}), (256, {}), (3, {'ovr': {}}), (2, {'reconstruct': True})):
            self.assertIsNone(module.forward(FakeTensor(rows), params))
        self.assertEqual(counters, dict(fused_calls=3, fallback_calls=4))

    def test_smallm_rows5_fuses_and_rows6_restores_fallback(self):
        module = SimpleNamespace(bc=None)
        module.forward = lambda x, params: module.bc
        counters = dict(fused_calls=0, fallback_calls=0)
        guard_single_row(module, dict(bc='native'), counters, allowed_rows=(1, 2, 3, 4, 5))
        self.assertEqual(module.forward(FakeTensor(5), {}), 'native')
        self.assertIsNone(module.bc)
        self.assertIsNone(module.forward(FakeTensor(6), {}))
        self.assertIsNone(module.bc)
        self.assertEqual(counters, dict(fused_calls=1, fallback_calls=1))
    def test_smallm_rows9_fuses_and_rows10_restores_fallback(self):
        module = SimpleNamespace(bc=None)
        module.forward = lambda x, params: module.bc
        counters = dict(fused_calls=0, fallback_calls=0)
        guard_single_row(module, dict(bc='native'), counters,
                         allowed_rows=(1, 2, 3, 4, 5, 6, 7, 8, 9))
        self.assertEqual(module.forward(FakeTensor(9), {}), 'native')
        self.assertIsNone(module.bc)
        self.assertIsNone(module.forward(FakeTensor(10), {}))
        self.assertIsNone(module.bc)
        self.assertEqual(counters, dict(fused_calls=1, fallback_calls=1))


if __name__ == '__main__':
    unittest.main()
