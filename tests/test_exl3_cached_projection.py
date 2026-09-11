"""CPU-only tests for the optional MTP projection cache; no torch, GPU, or model files.

Selection, deduplication, and budget checks run against narrow fake
``torch``/``exllamav3`` modules. Every selected inner is sized to exceed the
1.5 GiB cache budget on its own, so the pre-allocation RuntimeError proves
selection (via its exact byte total) without emulating any GPU arithmetic.
Rejection happens before reconstruct/allocation: fake inners raise if
``get_inner_weight_tensor`` runs, and the fake ``torch`` module raises on any
attribute access.
"""

import contextlib
import re
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from quantlab.methods.exl3.cached_projection import cache_mtp_projections


_HUGE_IN = 65536
_HUGE_OUT = 16384
_HUGE_BYTES = 2147483648  # _HUGE_IN * _HUGE_OUT * 2, pinned contract literal
_BUDGET_BYTES = 1610612736  # 1536 * 1024 * 1024, pinned contract literal


class LinearEXL3:
    """Isinstance stand-in for exllamav3's LinearEXL3."""


class FakeInner(LinearEXL3):
    """Fake projection inner; reconstruct/forward must never run in these tests."""

    def __init__(self, key=None, in_features=1024, out_features=1024):
        self.key = key
        self.in_features = in_features
        self.out_features = out_features
        self.forward = object()  # sentinel; identity-checked, never called
        self.reconstruct_calls = 0

    def get_inner_weight_tensor(self):
        self.reconstruct_calls += 1
        raise AssertionError("get_inner_weight_tensor must not run before budget check")


class Wrapper:
    """Fake parent module exposing an inner projection."""

    def __init__(self, inner=None, key=None):
        self.inner = inner
        self.key = key


class _NoTorch(ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"torch.{name} must not be touched before budget check")


@contextlib.contextmanager
def _fake_runtime():
    """Install narrow torch/exllamav3 fakes; no host dependency."""
    ext_mod = ModuleType("exllamav3.ext.exllamav3_ext")
    ext_pkg = ModuleType("exllamav3.ext")
    ext_pkg.__path__ = []
    ext_pkg.exllamav3_ext = ext_mod
    exl3_mod = ModuleType("exllamav3.modules.quant.exl3")
    exl3_mod.LinearEXL3 = LinearEXL3
    quant_pkg = ModuleType("exllamav3.modules.quant")
    quant_pkg.__path__ = []
    quant_pkg.exl3 = exl3_mod
    mods_pkg = ModuleType("exllamav3.modules")
    mods_pkg.__path__ = []
    mods_pkg.quant = quant_pkg
    root_pkg = ModuleType("exllamav3")
    root_pkg.__path__ = []
    root_pkg.ext = ext_pkg
    root_pkg.modules = mods_pkg
    fakes = {
        "torch": _NoTorch("torch"),
        "exllamav3": root_pkg,
        "exllamav3.ext": ext_pkg,
        "exllamav3.ext.exllamav3_ext": ext_mod,
        "exllamav3.modules": mods_pkg,
        "exllamav3.modules.quant": quant_pkg,
        "exllamav3.modules.quant.exl3": exl3_mod,
    }
    with patch.dict(sys.modules, fakes):
        yield


def _huge(key=None):
    return FakeInner(key=key, in_features=_HUGE_IN, out_features=_HUGE_OUT)


class ValidationTests(unittest.TestCase):
    def test_none_draft_rejected_before_imports(self):
        with self.assertRaisesRegex(ValueError, "must not be None"):
            cache_mtp_projections(None, "fc")

    def test_invalid_mode_rejected_before_imports(self):
        with self.assertRaisesRegex(ValueError, "invalid mode"):
            cache_mtp_projections(object(), "everything")

    def test_valid_modes_pass_validation(self):
        for mode in ("fc", "attention", "mlp", "all"):
            with self.subTest(mode=mode), _fake_runtime():
                with self.assertRaisesRegex(ValueError, "matched no"):
                    cache_mtp_projections([], mode)


class SelectionTests(unittest.TestCase):
    def test_unsliced_wide_projection_rejected_before_allocation(self):
        wide = FakeInner(key="mtp.fc", out_features=65536)
        with _fake_runtime(), self.assertRaisesRegex(ValueError, "out_features"):
            cache_mtp_projections([wide], "fc")
        self.assertEqual(wide.reconstruct_calls, 0)

    def _assert_rejected_before_reconstruct(self, draft, mode, expected_total, inners):
        forwards = {id(inner): inner.forward for inner in inners}
        with _fake_runtime():
            with self.assertRaises(RuntimeError) as caught:
                cache_mtp_projections(draft, mode)
        message = str(caught.exception)
        match = re.search(r"(\d+) cached weight bytes", message)
        self.assertIsNotNone(match, message)
        self.assertEqual(int(match.group(1)), expected_total)
        self.assertIn(str(_BUDGET_BYTES), message)
        for inner in inners:
            self.assertEqual(inner.reconstruct_calls, 0, inner.key)
            self.assertIs(inner.forward, forwards[id(inner)], inner.key)

    def test_no_matches_rejected(self):
        decoy = FakeInner(key="encoder.block.0")
        junk = [object(), Wrapper(), Wrapper(inner="not-a-linear"),
                Wrapper(inner=None), Wrapper(inner=decoy)]
        with _fake_runtime():
            with self.assertRaisesRegex(ValueError, "matched no"):
                cache_mtp_projections(junk, "mlp")
        self.assertEqual(decoy.reconstruct_calls, 0)

    def test_single_non_iterable_draft_without_match(self):
        with _fake_runtime():
            with self.assertRaisesRegex(ValueError, "matched no"):
                cache_mtp_projections(Wrapper(inner=FakeInner(key="mtp.fc")), "mlp")

    def test_fc_selects_exact_key_and_dedupes_shared_inner(self):
        fc = _huge("mtp.fc")
        backup = FakeInner(key="mtp.fc.backup")
        attn = FakeInner(key="mtp.layers.0.self_attn.q_proj")
        draft = [fc, Wrapper(inner=fc), Wrapper(inner=fc, key="ignored"),
                 backup, attn, object(), Wrapper(), Wrapper(inner=42)]
        self._assert_rejected_before_reconstruct(draft, "fc", _HUGE_BYTES, [fc, backup, attn])

    def test_key_falls_back_to_wrapper(self):
        inner = _huge()
        draft = [Wrapper(inner=inner, key="mtp.fc"), object()]
        self._assert_rejected_before_reconstruct(draft, "fc", _HUGE_BYTES, [inner])

    def test_attention_selects_dotted_self_attn(self):
        attn = _huge("mtp.layers.3.self_attn.o_proj")
        plain = FakeInner(key="mtp.self_attn_free")
        draft = [Wrapper(inner=attn), plain, object()]
        self._assert_rejected_before_reconstruct(draft, "attention", _HUGE_BYTES, [attn, plain])

    def test_mlp_selects_dotted_mlp(self):
        mlp = _huge("mtp.layers.3.mlp.gate_proj")
        fc = FakeInner(key="mtp.fc")
        draft = [mlp, Wrapper(inner=fc), object()]
        self._assert_rejected_before_reconstruct(draft, "mlp", _HUGE_BYTES, [mlp, fc])

    def test_all_matches_keyless_inner(self):
        keyless = _huge()
        self._assert_rejected_before_reconstruct([Wrapper(inner=keyless)], "all",
                                                 _HUGE_BYTES, [keyless])

    def test_none_key_matches_nothing_except_all(self):
        keyless = FakeInner()
        draft = [Wrapper(inner=keyless)]
        for mode in ("fc", "attention", "mlp"):
            with self.subTest(mode=mode), _fake_runtime():
                with self.assertRaisesRegex(ValueError, "matched no"):
                    cache_mtp_projections(draft, mode)
        self.assertEqual(keyless.reconstruct_calls, 0)

    def test_oversized_total_sums_distinct_selected_only(self):
        first = _huge("mtp.layers.0.mlp.gate_proj")
        second = _huge("mtp.layers.1.mlp.up_proj")
        draft = [first, Wrapper(inner=first), second, Wrapper(inner="junk"), object()]
        self._assert_rejected_before_reconstruct(draft, "mlp", 4294967296, [first, second])


if __name__ == "__main__":
    unittest.main()
