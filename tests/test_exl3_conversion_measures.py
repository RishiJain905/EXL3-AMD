"""CPU-only bounded conversion-error metrics; no GPU, backend, or weights."""
import importlib.util
import math
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def _load_measures():
    path = Path(__file__).resolve().parents[1] / "vendor/rocm-exl3/exllamav3/util/measures.py"
    spec = importlib.util.spec_from_file_location("exl3_measures_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measures = _load_measures() if torch is not None else None

EPS = 1e-8
RFN_REL = 1e-6
RFN_ABS = 1e-8
COS_ABS = 1e-6
SQ_ABS = 1e-4


def _reference(x, ref, eps=EPS):
    xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
    rf = ref.reshape(-1, ref.shape[-1]).to(torch.float32)
    err = (torch.linalg.norm(xf - rf, "fro") / torch.linalg.norm(rf, "fro")).item()
    a_flat = xf.view(xf.shape[0], -1)
    b_flat = rf.view(rf.shape[0], -1)
    signal = torch.sum(b_flat ** 2, dim=1)
    noise = torch.sum((a_flat - b_flat) ** 2, dim=1) + eps
    sq = (10.0 * torch.log10(signal / noise)).mean().item()
    cos = 1.0 - F.cosine_similarity(a_flat, b_flat, dim=1, eps=eps).mean().item()
    return err, cos, sq


def _same_nonfinite(a, b):
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
        return True
    return None


@unittest.skipIf(torch is None, "CPU Torch required")
class ConversionMeasuresTests(unittest.TestCase):
    def assertTripletClose(self, got, exp):
        rfn_same = _same_nonfinite(got[0], exp[0])
        if rfn_same is None:
            self.assertTrue(
                math.isclose(got[0], exp[0], rel_tol=RFN_REL, abs_tol=RFN_ABS),
                f"rfn {got[0]} != {exp[0]}",
            )
        cos_same = _same_nonfinite(got[1], exp[1])
        if cos_same is None:
            self.assertTrue(
                math.isclose(got[1], exp[1], rel_tol=0.0, abs_tol=COS_ABS),
                f"cos {got[1]} != {exp[1]}",
            )
        sq_same = _same_nonfinite(got[2], exp[2])
        if sq_same is None:
            self.assertTrue(
                math.isclose(got[2], exp[2], rel_tol=0.0, abs_tol=SQ_ABS),
                f"sqnr {got[2]} != {exp[2]}",
            )
        for v in got:
            self.assertIsInstance(v, float)

    def test_fp32_matches_reference_single_and_chunked(self):
        torch.manual_seed(0)
        x = torch.randn(7, 5)
        ref = torch.randn(7, 5)
        exp = _reference(x, ref)
        self.assertTripletClose(measures.state_error(x, ref), exp)
        # 12 elements // 5 cols = 2 rows/chunk -> 2,2,2,1 uneven tail.
        self.assertTripletClose(measures.state_error(x, ref, max_elements=12), exp)
        x3 = torch.randn(2, 3, 4)
        ref3 = torch.randn(2, 3, 4)
        exp3 = _reference(x3, ref3)
        self.assertTripletClose(measures.state_error(x3, ref3, max_elements=10), exp3)

    def test_fp16_matches_reference_chunked(self):
        torch.manual_seed(1)
        x = torch.randn(9, 6).to(torch.float16)
        ref = torch.randn(9, 6).to(torch.float16)
        exp = _reference(x, ref)
        # 14 // 6 = 2 rows/chunk -> 2,2,2,2,1 uneven tail.
        self.assertTripletClose(measures.state_error(x, ref, max_elements=14), exp)

    def test_zero_identical_small_norm(self):
        torch.manual_seed(2)
        zeros = torch.zeros(4, 8)
        exp_zz = _reference(zeros, zeros)
        got_zz = measures.state_error(zeros, zeros, max_elements=16)
        self.assertTrue(math.isnan(got_zz[0]) and math.isnan(exp_zz[0]))
        self.assertTripletClose(got_zz, exp_zz)
        self.assertEqual(got_zz[1], 1.0)
        self.assertEqual(got_zz[2], float("-inf"))

        x = torch.randn(5, 7)
        ref = x.clone()
        self.assertTripletClose(
            measures.state_error(x, ref, max_elements=15), _reference(x, ref)
        )
        self.assertEqual(measures.state_error(x, ref)[0], 0.0)

        xnz = torch.randn(3, 6)
        zref = torch.zeros(3, 6)
        self.assertEqual(measures.state_error(xnz, zref, max_elements=12)[0], float("inf"))
        self.assertTripletClose(
            measures.state_error(xnz, zref, max_elements=12), _reference(xnz, zref)
        )

        tiny_x = torch.randn(3, 4) * 1e-10
        tiny_ref = torch.randn(3, 4) * 1e-10
        self.assertTripletClose(
            measures.state_error(tiny_x, tiny_ref, max_elements=8),
            _reference(tiny_x, tiny_ref),
        )

    def test_no_mutation_and_strided(self):
        torch.manual_seed(4)
        x = torch.randn(6, 8)
        ref = torch.randn(6, 8)
        xc, rc = x.clone(), ref.clone()
        measures.state_error(x, ref, max_elements=20)
        self.assertTrue(torch.equal(x, xc) and torch.equal(ref, rc))

        base_x = torch.randn(10, 6)
        base_ref = torch.randn(10, 6)
        base_xc, base_refc = base_x.clone(), base_ref.clone()
        sx, sr = base_x[::2], base_ref[::2]
        self.assertFalse(sx.is_contiguous())
        self.assertTripletClose(
            measures.state_error(sx, sr, max_elements=14), _reference(sx, sr)
        )
        self.assertTrue(torch.equal(base_x, base_xc) and torch.equal(base_ref, base_refc))

        tx = torch.randn(6, 8).t()
        tr = torch.randn(6, 8).t()
        self.assertFalse(tx.is_contiguous())
        self.assertTripletClose(
            measures.state_error(tx, tr, max_elements=14), _reference(tx, tr)
        )

    def test_chunked_dispatch_bounded(self):
        torch.manual_seed(5)
        x = torch.randn(8, 4)
        ref = torch.randn(8, 4)
        exp = _reference(x, ref)
        with (
            patch.object(measures.F, "cosine_similarity", wraps=F.cosine_similarity) as mock_cos,
            patch.object(torch.linalg, "norm", wraps=torch.linalg.norm) as mock_norm,
        ):
            got = measures.state_error(x, ref, max_elements=8)
        self.assertTripletClose(got, exp)
        # 8 elements // 4 cols = 2 rows/chunk over 8 rows -> 4 bounded calls.
        self.assertEqual(mock_cos.call_count, 4)
        total_rows = 0
        for call in mock_cos.call_args_list:
            a, b = call.args[:2]
            self.assertLessEqual(a.shape[0], 2)
            self.assertLessEqual(b.shape[0], 2)
            total_rows += a.shape[0]
        self.assertEqual(total_rows, 8)
        # No Frobenius-norm call may see the full tensor; chunked path avoids it entirely.
        full_numel = x.numel()
        for call in mock_norm.call_args_list:
            first = call.args[0] if call.args else None
            if first is not None and hasattr(first, "numel"):
                self.assertLess(first.numel(), full_numel)

    def test_rejects_unbounded_fallback_inputs(self):
        x = torch.ones(4, 8)
        with self.assertRaises(ValueError):
            measures.state_error(x, torch.ones(1, 8))
        for budget in (0, -1, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                measures.state_error(x, x, max_elements=budget)

    def test_profiler_float_temporaries_bounded(self):
        try:
            from torch.profiler import ProfilerActivity, profile
        except ImportError:
            self.skipTest("torch.profiler unavailable")
        torch.manual_seed(6)
        n_rows, n_cols = 8, 16
        x = torch.randn(n_rows, n_cols).to(torch.float16)
        ref = torch.randn(n_rows, n_cols).to(torch.float16)
        full_float_bytes = n_rows * n_cols * 4
        with profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True,
        ) as prof:
            measures.state_error(x, ref, max_elements=32)
        mems = []
        # Inspect individual allocations, not sums grouped across every chunk.
        for event in prof.events():
            value = getattr(event, "self_cpu_memory_usage", None)
            if isinstance(value, (int, float)) and value > 0:
                mems.append(value)
        if not mems:
            self.skipTest("CPU profiler reported no memory data")
        self.assertLess(
            max(mems),
            full_float_bytes,
            f"max CPU alloc {max(mems)} should stay below full-tensor {full_float_bytes}",
        )


if __name__ == "__main__":
    unittest.main()
