"""AsterKV-5 experimental CPU reference (NumPy, no GPU).

Hadamard rotation and scalar Lloyd-Max quantization are inherited ideas,
see TurboQuant (arXiv:2504.19874). This module is an experimental reference,
not a claim of novelty nor a GPU implementation.

Format: normalized H32 rotation, per-32 absmax scale (+1e-10) stored as FP16,
uniform midpoint grid (native) or symmetric fitted centroids (experimental).
Codes here are unpacked reference indices only, not native bit-packing.
"""
from __future__ import annotations

import numpy as np

GROUP = 32
EPS = 1e-10
HIST_BINS = 4096


def hadamard_matrix32() -> np.ndarray:
    """Return normalized 32x32 Sylvester Hadamard matrix (float64)."""
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < GROUP:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(float(GROUP))


def _check_groups(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] % GROUP != 0:
        raise ValueError("last dim must be a multiple of 32")
    if a.size and not np.all(np.isfinite(a)):
        raise ValueError("values must be finite")
    return a


def hadamard_rotate(values) -> np.ndarray:
    """Rotate final axis by normalized H32 per 32-group. Returns float64."""
    a = _check_groups(values)
    h = hadamard_matrix32()
    flat = a.reshape(-1, GROUP)
    return (flat @ h.T).reshape(a.shape)


def hadamard_inverse(values) -> np.ndarray:
    """Inverse H32 rotation (H is symmetric orthogonal). Returns float64."""
    a = _check_groups(values)
    h = hadamard_matrix32()
    flat = a.reshape(-1, GROUP)
    return (flat @ h.T).reshape(a.shape)


def uniform_roundtrip(values, bits: int = 5) -> np.ndarray:
    """Uniform midpoint roundtrip matching native grid. Returns FP16 recon.

    Error magnitude is illustrative only, not a model-quality promise.
    """
    if bits not in (4, 5, 6, 8):
        raise ValueError("bits must be one of 4/5/6/8")
    a = _check_groups(values)
    h = hadamard_matrix32()
    orig = a.shape
    y = (a.reshape(-1, GROUP) @ h.T)
    s = np.abs(y).max(axis=1, keepdims=True) + EPS
    m = float(1 << (bits - 1))
    q = np.floor(y / s * m + m)
    q = np.clip(q, 0, (1 << bits) - 1)
    sh = s.astype(np.float16).astype(np.float64)
    yh = (q - (m - 0.5)) / m * sh
    xh = (yh @ h.T).reshape(orig)
    return xh.astype(np.float16)


def _symmetric_fallback(n: int, pin: bool) -> np.ndarray:
    kh = n // 2
    if pin:
        pos = np.linspace(1.0 / (n - 1), 1.0, kh, dtype=np.float64)
    else:
        pos = (np.arange(kh, dtype=np.float64) + 0.5) / kh
    return np.concatenate([-pos[::-1], pos])


def fit_symmetric_codebook(normalized_samples, bits: int = 5,
                           iterations: int = 30,
                           pin_endpoints: bool = True) -> np.ndarray:
    """Fit symmetric Lloyd-Max centroids on magnitudes in [-1,1].

    Fits Kh=K/2 positive centroids via 4096-bin histogram, mirrors for
    symmetry. Caller supplies calibration samples only (no eval mixing).
    Returns float64 centroids, strictly increasing, symmetric.
    """
    if bits not in (4, 5):
        raise ValueError("bits must be 4 or 5")
    if iterations is None or int(iterations) < 1:
        raise ValueError("iterations must be >= 1")
    iterations = int(iterations)
    s = np.asarray(normalized_samples, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("empty calibration samples")
    if not np.all(np.isfinite(s)):
        raise ValueError("samples must be finite")
    if np.any(s < -1.0) or np.any(s > 1.0):
        raise ValueError("samples must be in [-1, 1]")
    n = 1 << bits
    kh = n // 2
    fallback = _symmetric_fallback(n, bool(pin_endpoints))
    mags = np.abs(s)
    if np.all(mags == 0.0):
        return fallback
    counts, _ = np.histogram(mags, bins=HIST_BINS, range=(0.0, 1.0))
    centers = (np.arange(HIST_BINS, dtype=np.float64) + 0.5) / HIST_BINS
    pos = fallback[kh:].copy()
    for _ in range(iterations):
        thr = (pos[:-1] + pos[1:]) / 2.0
        idx = np.searchsorted(thr, centers)
        new = pos.copy()
        for k in range(kh):
            if pin_endpoints and k == kh - 1:
                continue
            m = idx == k
            w = counts[m].sum()
            if w > 0:
                new[k] = (counts[m] * centers[m]).sum() / float(w)
        if pin_endpoints:
            free = np.sort(new[:-1])
            if np.any(np.diff(free) <= 0) or np.any(free <= 0) or np.any(free >= 1.0):
                return fallback
            pos[:-1] = free
            pos[-1] = 1.0
        else:
            free = np.sort(new)
            if np.any(np.diff(free) <= 0) or np.any(free <= 0) or np.any(free > 1.0):
                return fallback
            pos = free
    full = np.concatenate([-pos[::-1], pos])
    if full.shape != (n,) or not np.all(np.isfinite(full)):
        return fallback
    if np.any(np.diff(full) <= 0):
        return fallback
    return full


def _check_centroids(centroids) -> np.ndarray:
    c = np.asarray(centroids, dtype=np.float64)
    if c.ndim != 1 or c.size not in (16, 32):
        raise ValueError("centroids must be a 1D four- or five-bit grid (16 or 32 values)")
    if not np.all(np.isfinite(c)):
        raise ValueError("centroids must be finite")
    if np.any(c < -1.0) or np.any(c > 1.0):
        raise ValueError("centroids must be in [-1, 1]")
    if np.any(np.diff(c) <= 0):
        raise ValueError("centroids must be strictly increasing")
    return c


def nonuniform_roundtrip(values, centroids, refine_scale: bool = False):
    """Nonuniform H32+absmax roundtrip with nearest-centroid codes.

    Returns (recon FP16 same shape, codes uint8 same shape unpacked,
    scales FP16 shape leading+(n_groups,)). Codes are unpacked indices,
    not native packing. refine_scale runs one LS scale step per group
    then re-quantizes with the stored FP16 scale; scale count unchanged.
    """
    a = _check_groups(values)
    c = _check_centroids(centroids)
    h = hadamard_matrix32()
    orig = a.shape
    ng = orig[-1] // GROUP
    lead = orig[:-1]
    y2 = (a.reshape(-1, GROUP) @ h.T).reshape(-1, GROUP)
    s0 = np.abs(y2).max(axis=1, keepdims=True) + EPS
    n0 = y2 / s0
    thresholds = (c[:-1] + c[1:]) * 0.5
    codes0 = np.searchsorted(thresholds, n0, side="left")
    d0 = c[codes0]
    if not refine_scale:
        sh = s0.astype(np.float16).astype(np.float64)
        yh = d0 * sh
        codes = codes0.reshape(orig).astype(np.uint8)
        scales = sh.reshape(-1).reshape(lead + (ng,)).astype(np.float16)
    else:
        num = (y2 * d0).sum(axis=1, keepdims=True)
        den = (d0 * d0).sum(axis=1, keepdims=True)
        sr = np.where(den > 0, num / np.maximum(den, 1e-300), s0)
        sr = np.where(np.isfinite(sr) & (sr > 0), sr, s0)
        sh = sr.astype(np.float16).astype(np.float64)
        safe = np.where(sh == 0.0, 1.0, sh)
        n1 = np.where(sh == 0.0, 0.0, y2 / safe)
        codes1 = np.searchsorted(thresholds, n1, side="left")
        d1 = c[codes1]
        yh = d1 * sh
        codes = codes1.reshape(orig).astype(np.uint8)
        scales = sh.reshape(-1).reshape(lead + (ng,)).astype(np.float16)
    xh = (yh.reshape(-1, GROUP) @ h.T).reshape(orig)
    return xh.astype(np.float16), codes, scales
