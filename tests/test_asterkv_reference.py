"""Tests for AsterKV-5 NumPy CPU reference; no GPU/torch."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels" / "exl3"))
import asterkv_reference as ref


def sylvester_h32():
    h = np.array([[(-1) ** ((r & c).bit_count()) for c in range(32)]
                  for r in range(32)], dtype=np.float64)
    return h / np.sqrt(32.0)


class HadamardTests(unittest.TestCase):
    def test_matches_sylvester(self):
        got = ref.hadamard_rotate(np.eye(32, dtype=np.float64))
        self.assertTrue(np.allclose(got, sylvester_h32(), atol=1e-12))

    def test_inverse_roundtrip(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal((2, 3, 64))
        xi = ref.hadamard_inverse(ref.hadamard_rotate(x))
        self.assertTrue(np.allclose(xi, x, atol=1e-11))
        xr = ref.hadamard_rotate(ref.hadamard_inverse(x))
        self.assertTrue(np.allclose(xr, x, atol=1e-11))

    def test_zero_and_shape(self):
        z = np.zeros((1, 32))
        self.assertTrue(np.allclose(ref.hadamard_rotate(z), 0.0))
        with self.assertRaises(ValueError):
            ref.hadamard_rotate(np.zeros((30,)))
        with self.assertRaises(ValueError):
            ref.hadamard_rotate(np.full((32,), np.inf))
        with self.assertRaises(ValueError):
            ref.hadamard_inverse(np.full((2, 33), 0.0))


class UniformTests(unittest.TestCase):
    def test_matches_native_midpoint(self):
        rng = np.random.default_rng(21)
        x = rng.standard_normal((4, 64))
        for bits in (4, 5, 6, 8):
            got = ref.uniform_roundtrip(x, bits=bits).astype(np.float64)
            h = sylvester_h32()
            y = x.reshape(-1, 32) @ h.T
            s = np.abs(y).max(axis=1, keepdims=True) + 1e-10
            m = float(1 << (bits - 1))
            q = np.clip(np.floor(y / s * m + m), 0, (1 << bits) - 1)
            sh = s.astype(np.float16).astype(np.float64)
            exp = ((q - (m - 0.5)) / m * sh @ h.T).reshape(x.shape)
            self.assertTrue(np.allclose(got, exp.astype(np.float16), atol=1e-3),
                            f"bits={bits}")

    def test_zero_and_validation(self):
        z = np.zeros((2, 32))
        r = ref.uniform_roundtrip(z, bits=5)
        self.assertEqual(r.dtype, np.float16)
        self.assertTrue(np.all(np.isfinite(r.astype(np.float64))))
        self.assertTrue(np.allclose(r.astype(np.float64), 0.0, atol=1e-6))
        with self.assertRaises(ValueError):
            ref.uniform_roundtrip(z, bits=7)
        with self.assertRaises(ValueError):
            ref.uniform_roundtrip(np.zeros((31,)), bits=5)
        with self.assertRaises(ValueError):
            ref.uniform_roundtrip(np.full((32,), np.nan), bits=5)

    def test_fp16_scale_accounting(self):
        rng = np.random.default_rng(22)
        x = rng.standard_normal((8, 32)) * 2.0
        got = ref.uniform_roundtrip(x, bits=5).astype(np.float64)
        h = sylvester_h32()
        y = x.reshape(-1, 32) @ h.T
        s = np.abs(y).max(axis=1, keepdims=True) + 1e-10
        m = 16.0
        q = np.clip(np.floor(y / s * m + m), 0, 31)
        sh16 = s.astype(np.float16).astype(np.float64)
        exp16 = ((q - 15.5) / m * sh16 @ h.T).reshape(x.shape)
        exp32 = ((q - 15.5) / m * s @ h.T).reshape(x.shape)
        self.assertTrue(np.allclose(got, exp16.astype(np.float16), atol=1e-3))
        # FP16 rounding must matter somewhere on this data (else test is weak).
        self.assertGreater(np.abs(exp16 - exp32).max(), 0.0)


class FitTests(unittest.TestCase):
    def _calib(self, seed=0, n=6000):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, 32))
        y = ref.hadamard_rotate(x)
        s = np.abs(y).max(axis=1, keepdims=True) + 1e-10
        return (y / s).reshape(-1)

    def test_symmetric_monotonic_5bit(self):
        c = ref.fit_symmetric_codebook(self._calib(), bits=5)
        self.assertEqual(c.shape, (32,))
        self.assertTrue(np.all(np.isfinite(c)))
        self.assertTrue(np.all(np.diff(c) > 0))
        self.assertTrue(np.allclose(c, -c[::-1], atol=0))
        self.assertEqual(float(c[0]), -1.0)
        self.assertEqual(float(c[-1]), 1.0)
        self.assertTrue(np.all(c >= -1.0) and np.all(c <= 1.0))

    def test_4bit_and_unpinned(self):
        c4 = ref.fit_symmetric_codebook(self._calib(), bits=4)
        self.assertEqual(c4.shape, (16,))
        self.assertTrue(np.all(np.diff(c4) > 0))
        self.assertTrue(np.allclose(c4, -c4[::-1], atol=0))
        cu = ref.fit_symmetric_codebook(self._calib(), bits=5, pin_endpoints=False)
        self.assertEqual(cu.shape, (32,))
        self.assertTrue(np.all(np.diff(cu) > 0))
        self.assertTrue(np.allclose(cu, -cu[::-1], atol=0))
        self.assertLess(float(cu[-1]), 1.0)

    def test_deterministic_and_degenerate(self):
        s = self._calib()
        a = ref.fit_symmetric_codebook(s, bits=5)
        b = ref.fit_symmetric_codebook(s, bits=5)
        self.assertTrue(np.array_equal(a, b))
        z = ref.fit_symmetric_codebook(np.zeros(256), bits=5)
        self.assertEqual(z.shape, (32,))
        self.assertTrue(np.all(np.diff(z) > 0))
        self.assertTrue(np.allclose(z, -z[::-1], atol=0))

    def test_validation(self):
        with self.assertRaises(ValueError):
            ref.fit_symmetric_codebook(np.array([]), bits=5)
        with self.assertRaises(ValueError):
            ref.fit_symmetric_codebook(np.array([0.0, 1.5]), bits=5)
        with self.assertRaises(ValueError):
            ref.fit_symmetric_codebook(np.array([0.0, np.inf]), bits=5)
        with self.assertRaises(ValueError):
            ref.fit_symmetric_codebook(np.zeros(64), bits=6)


class NonuniformTests(unittest.TestCase):
    def _centroids(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((3000, 32))
        y = ref.hadamard_rotate(x)
        s = np.abs(y).max(axis=1, keepdims=True) + 1e-10
        return ref.fit_symmetric_codebook((y / s).reshape(-1), bits=5)

    def test_nearest_assignment_and_scales(self):
        rng = np.random.default_rng(31)
        x = rng.standard_normal((4, 64))
        c = self._centroids()
        recon, codes, scales = ref.nonuniform_roundtrip(x, c)
        self.assertEqual(recon.dtype, np.float16)
        self.assertEqual(codes.dtype, np.uint8)
        self.assertEqual(scales.dtype, np.float16)
        self.assertEqual(codes.shape, x.shape)
        self.assertEqual(scales.shape, (4, 2))
        self.assertTrue(int(codes.min()) >= 0 and int(codes.max()) < 32)
        # Independent scalar nearest check on a few entries.
        h = sylvester_h32()
        y = (x.reshape(-1, 32) @ h.T).reshape(x.shape)
        s = np.abs(y.reshape(-1, 32)).max(axis=1).reshape(4, 2)
        sh = s.astype(np.float16).astype(np.float64)
        for r in (0, 3):
            for g in (0, 1):
                sc = float(s[r, g])
                for j in range(0, 32, 7):
                    v = float(y[r, g * 32 + j]) / sc
                    best = min(range(32), key=lambda k: abs(v - float(c[k])))
                    self.assertEqual(int(codes[r, g * 32 + j]), best)
                self.assertAlmostEqual(float(scales[r, g]), float(sh[r, g]), places=6)

    def test_scalar_small_group_decode(self):
        # One 32-group scalar reference with explicit Python loops.
        h = sylvester_h32().tolist()
        xv = [0.5 * ((i % 7) - 3) + 0.125 * ((i % 3) - 1) for i in range(32)]
        c = ref.fit_symmetric_codebook(
            np.array(xv * 64, dtype=np.float64) / 2.0, bits=5)
        cl = [float(v) for v in c.tolist()]
        y = [sum(xv[k] * h[j][k] for k in range(32)) for j in range(32)]
        s = max(abs(v) for v in y) + 1e-10
        sh = float(np.float16(s))
        codes = [min(range(32), key=lambda k: abs(y[j] / s - cl[k])) for j in range(32)]
        yh = [cl[k] * sh for k in codes]
        xh = [sum(yh[k] * h[j][k] for k in range(32)) for j in range(32)]
        recon, got_codes, got_scales = ref.nonuniform_roundtrip(
            np.array([xv], dtype=np.float64), c)
        self.assertEqual(list(map(int, got_codes.reshape(-1).tolist())), codes)
        self.assertAlmostEqual(float(got_scales.reshape(-1)[0]), sh, places=6)
        self.assertTrue(np.allclose(recon.astype(np.float64).reshape(-1),
                                    np.array(xh, dtype=np.float16), atol=1e-3))

    def test_zero_and_validation(self):
        c = self._centroids()
        z = np.zeros((1, 32))
        recon, codes, scales = ref.nonuniform_roundtrip(z, c)
        self.assertTrue(np.all(np.isfinite(recon.astype(np.float64))))
        self.assertTrue(np.allclose(recon.astype(np.float64), 0.0, atol=1e-6))
        self.assertEqual(scales.dtype, np.float16)
        with self.assertRaises(ValueError):
            ref.nonuniform_roundtrip(np.zeros((31,)), c)
        with self.assertRaises(ValueError):
            ref.nonuniform_roundtrip(np.full((32,), np.nan), c)
        with self.assertRaises(ValueError):
            ref.nonuniform_roundtrip(z, np.array([[0.0, 1.0]]))
        with self.assertRaises(ValueError):
            ref.nonuniform_roundtrip(z, np.array([0.0, 0.0, 1.0]))
        with self.assertRaises(ValueError):
            ref.nonuniform_roundtrip(z, np.array([1.0, 0.0]))
        with self.assertRaises(ValueError):
            ref.nonuniform_roundtrip(z, np.array([0.0, np.inf]))

    def test_refine_keeps_scale_count(self):
        rng = np.random.default_rng(33)
        x = rng.standard_normal((2, 64))
        c = self._centroids()
        r0, _, s0 = ref.nonuniform_roundtrip(x, c, refine_scale=False)
        r1, _, s1 = ref.nonuniform_roundtrip(x, c, refine_scale=True)
        self.assertEqual(s0.shape, s1.shape)
        self.assertEqual(s1.dtype, np.float16)
        self.assertTrue(np.all(np.isfinite(r1.astype(np.float64))))


class ValidationSplitTests(unittest.TestCase):
    def test_fitted_beats_uniform_unseen(self):
        rng_c = np.random.default_rng(100)
        xc = rng_c.standard_normal((4000, 32))
        yc = ref.hadamard_rotate(xc)
        sc = np.abs(yc).max(axis=1, keepdims=True) + 1e-10
        c = ref.fit_symmetric_codebook((yc / sc).reshape(-1), bits=5)
        rng_v = np.random.default_rng(200)  # unseen split
        xv = rng_v.standard_normal((2000, 64))
        ru = ref.uniform_roundtrip(xv, bits=5).astype(np.float64)
        rn, _, _ = ref.nonuniform_roundtrip(xv, c)
        rn = rn.astype(np.float64)
        mse_u = float(np.mean((ru - xv) ** 2))
        mse_n = float(np.mean((rn - xv) ** 2))
        self.assertLess(mse_n, mse_u * 0.98,
                        f"nonuniform {mse_n:.6g} vs uniform {mse_u:.6g}")


if __name__ == "__main__":
    unittest.main()
