"""Independent CPU reader for native EXL3 cb=0/2, K=2/3/4 packed weights.

Format reference: https://github.com/CarouselAether/rocm_exl3 at
550dcfed786ad7bffa08b7a6b2a216fc474cbbb5, quant/pack.cu,
quant/codebook.cuh, rocm/quant/exl3_dq_rdna.hip.h and
rocm/quant/reconstruct_rdna.hip (beneath exllamav3/exllamav3_ext).
Upstream is MIT licensed, Copyright (c) 2025 Turboderp. This independent
NumPy implementation uses bit streams rather than the native funnel shifts.
It does not load upstream code or invoke an accelerator.

SHA256 of inspected files (including any local changes):
pack.cu: 9cba0f372ac96c204d59c4b18669f9202b3191a5b9aff9d508a2c3dc6d1da3df
codebook.cuh: 3558f224d5af14ea7357fd6d1da13aa473182cdea88b210304e90587f6556dc9
exl3_dq_rdna.hip.h: 3b07c671a08d2d258167755f0390c20bebac60be281d1f15531c485cde8c5cab
reconstruct_rdna.hip: 316c837090f5c9c27dfa6e06bf48bb8983758406b787a2e6d5dcd39894364f4b
"""

import numpy as np


DECODE_TILE_BATCH = 256


def decode_trellis(trellis, K, *, codebook=0):
    """Return decoded float32 weights in (input, output) orientation.

    ``trellis`` must be an int16 ndarray of shape (input/16, output/16,
    16*K). Codebooks 0 (default) and 2 (mul1) are supported.
    Codebook arithmetic rounds to FP16, exactly as the native decoder does.
    Intermediate arrays cover at most DECODE_TILE_BATCH tiles; only the
    returned float32 matrix scales with the full decoded weight size.
    """
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or K not in (2, 3, 4):
        raise ValueError("K must be integer 2, 3 or 4")
    if isinstance(codebook, (bool, np.bool_)) or not isinstance(codebook, (int, np.integer)) or codebook not in (0, 2):
        raise ValueError("codebook must be integer 0 or 2")
    packed = np.asarray(trellis)
    if packed.dtype.kind != "i" or packed.dtype.itemsize != 2:
        raise ValueError("trellis must have int16 dtype")
    if packed.ndim != 3 or min(packed.shape) == 0 or packed.shape[2] != 16 * K:
        raise ValueError("trellis must have nonempty shape (input/16, output/16, 16*K)")
    # Each state is the circular 16-bit window ending at the current K-bit symbol.
    indices = (np.arange(256)[:, None] * K + K - 16 + np.arange(16)) % (256 * K)
    shifts = np.arange(15, -1, -1)
    # Each lane holds eight entries in two 2x2 fragments of the 16x16 tile.
    index = np.arange(256)
    lane, value = index // 8, index % 8
    row = (lane % 4) * 2 + value % 2 + ((value // 2) % 2) * 8
    col = (lane // 8) * 2 + (lane // 4) % 2 + (value // 4) * 8
    output = np.empty((packed.shape[0] * 16, packed.shape[1] * 16), dtype=np.float32)
    tile_count = packed.shape[0] * packed.shape[1]
    for start in range(0, tile_count, DECODE_TILE_BATCH):
        tile_ids = np.arange(start, min(start + DECODE_TILE_BATCH, tile_count))
        tile_rows, tile_cols = tile_ids // packed.shape[1], tile_ids % packed.shape[1]
        # Index before converting: even noncontiguous inputs need no full copy.
        batch = packed[tile_rows, tile_cols].astype(np.uint16)
        # Native SWAP16 reverses each pair of words, not bytes within words.
        words = batch.reshape(len(batch), -1, 2)[..., ::-1].reshape(batch.shape)
        bits = ((words[..., None] >> shifts) & 1).reshape(len(batch), -1)
        states = np.sum(bits[:, indices] << shifts, axis=-1, dtype=np.uint64)
        if codebook == 0:
            mixed = ((states * 89226354 + 64248484) & 0x8FFF8FFF) ^ 0x3B603B60
            low = (mixed & 65535).astype(np.uint16).view(np.float16).astype(np.float32)
            high = (mixed >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
            values = (low + high).astype(np.float16).astype(np.float32)
        else:
            product = (states * 0x83DCD12D) & 0xFFFFFFFF
            byte_sum = sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
            # 0x6400 + sum represents the exact integer 1024 + sum in half.
            # Evaluate the half FMA exactly before its one final half rounding.
            inv, bias = np.array([0x1EEE, 0xC931], dtype=np.uint16).view(np.float16).astype(np.float64)
            values = ((1024 + byte_sum).astype(np.float64) * inv + bias).astype(np.float16).astype(np.float32)
        output[tile_rows[:, None] * 16 + row, tile_cols[:, None] * 16 + col] = values
    return output


def _hadamard128_last(values):
    result = values.copy().reshape(*values.shape[:-1], -1, 128)
    for stride in (1, 2, 4, 8, 16, 32, 64):
        blocks = result.reshape(*result.shape[:-1], -1, 2, stride)
        left, right = blocks[..., 0, :].copy(), blocks[..., 1, :].copy()
        blocks[..., 0, :] = left + right
        blocks[..., 1, :] = left - right
    result *= np.float32(1 / np.sqrt(128))
    return result.reshape(values.shape)


def reconstruct(trellis, K, suh, svh, *, codebook=0):
    """Return diag(suh) H128 W_hat H128 diag(svh), shape (input, output).

    Both dimensions must be multiples of 128. Transforms and scaling use
    float32: this is a mathematical oracle, not an emulation of native
    FP16 butterfly rounding or the packed GEMM's accumulation order.
    """
    weights = decode_trellis(trellis, K, codebook=codebook)
    if any(size % 128 for size in weights.shape):
        raise ValueError("reconstruction dimensions must be multiples of 128")
    scales = []
    for name, scale, size in (("suh", suh, weights.shape[0]), ("svh", svh, weights.shape[1])):
        raw = np.asarray(scale)
        if raw.shape != (size,) or raw.dtype.kind not in "fiu" or not np.all(np.isfinite(raw)):
            raise ValueError(f"{name} must be a finite real vector of length {size}")
        converted = raw.astype(np.float32)
        if not np.all(np.isfinite(converted)):
            raise ValueError(f"{name} must be representable in float32")
        scales.append(converted)
    weights = _hadamard128_last(_hadamard128_last(weights.T).T)
    return weights * scales[0][:, None] * scales[1][None, :]
