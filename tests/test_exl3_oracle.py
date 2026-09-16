"""CPU-only native-format fixtures; no upstream pack/unpack helpers."""

import struct
import unittest

import numpy as np

from quantlab.methods.exl3.oracle import decode_trellis, reconstruct


def scalar_codebook(state, codebook=0):
    if codebook == 2:
        # Integer byte sum plus exact dyadic constants from the format. The
        # product and addition are exact in Python double; round only at store.
        product = (int(state) * 0x83DCD12D) & 0xFFFFFFFF
        value = 1024 + sum(product.to_bytes(4, 'little'))
        return struct.unpack('<e', struct.pack('<e', value * (887 / 131072) - 10.3828125))[0]
    mixed = ((int(state) * 89226354 + 64248484) & 0x8FFF8FFF) ^ 0x3B603B60
    a = struct.unpack("<e", struct.pack("<H", mixed & 65535))[0]
    b = struct.unpack("<e", struct.pack("<H", mixed >> 16))[0]
    return struct.unpack("<e", struct.pack("<e", a + b))[0]


def fixture(symbols, bits):
    """Hand serialize symbols using text bits and native little-endian uint32s."""
    stream = "".join(format(int(s), f"0{bits}b") for s in symbols)
    data = b"".join(struct.pack("<I", int(stream[i:i + 32], 2)) for i in range(0, len(stream), 32))
    return np.frombuffer(data, dtype="<i2")


def expected_tile(symbols, bits, codebook=0):
    # Settle the rolling state over one full circuit before collecting outputs.
    state = 0
    for symbol in symbols:
        state = ((state << bits) | int(symbol)) & 65535
    decoded = []
    for symbol in symbols:
        state = ((state << bits) | int(symbol)) & 65535
        decoded.append(scalar_codebook(state, codebook))
    result = np.empty((16, 16), dtype=np.float32)
    for row in range(16):
        for col in range(16):
            lane = ((col % 8) // 2) * 8 + (col % 2) * 4 + (row % 8) // 2
            offset = (col // 8) * 4 + (row // 8) * 2 + row % 2
            result[row, col] = decoded[lane * 8 + offset]
    return result


class Exl3OracleTests(unittest.TestCase):
    def test_mul1_circular_tiles_match_scalar_reference(self):
        rng = np.random.default_rng(917)
        for bits in (2, 3, 4):
            symbols = rng.integers(0, 1 << bits, 256)
            symbols[0], symbols[-1] = 0, (1 << bits) - 1
            packed = fixture(symbols, bits).reshape(1, 1, 16 * bits)
            actual = decode_trellis(packed, bits, codebook=2)
            np.testing.assert_array_equal(actual, expected_tile(symbols, bits, 2))
            self.assertFalse(np.array_equal(actual, decode_trellis(packed, bits)))
        zeros = np.zeros((1, 1, 32), dtype=np.int16)
        np.testing.assert_array_equal(decode_trellis(zeros, 2, codebook=2),
                                      np.full((16, 16), scalar_codebook(0, 2), dtype=np.float32))

    def test_mul1_reconstruction_matches_explicit_hadamard(self):
        rng = np.random.default_rng(719)
        packed = rng.integers(-32768, 32768, (8, 8, 48), dtype=np.int16)
        h = np.array([[(-1) ** ((r & c).bit_count()) for c in range(128)] for r in range(128)]) / np.sqrt(128)
        su = np.linspace(-1, 1, 128).astype(np.float32)
        sv = np.linspace(1, -2, 128).astype(np.float32)
        expected = (h @ decode_trellis(packed, 3, codebook=2) @ h) * su[:, None] * sv[None, :]
        np.testing.assert_allclose(reconstruct(packed, 3, su, sv, codebook=2), expected, atol=2e-6, rtol=1e-5)

    def test_invalid_codebooks(self):
        for codebook in (True, False, 2.0, 1, 3, None, 'mul1'):
            with self.assertRaises(ValueError):
                decode_trellis(np.zeros((1, 1, 32), dtype=np.int16), 2, codebook=codebook)

    def test_chunk_boundary_with_independent_tiles(self):
        # 289 tiles cross the 256-tile batch boundary in the middle of a row,
        # and exercise a partial final batch. Every tile has its own stream.
        rng = np.random.default_rng(2519)
        for bits in (2, 3, 4):
            packed = np.empty((17, 17, 16 * bits), dtype=np.int16)
            expected = np.empty((272, 272), dtype=np.float32)
            for row in range(17):
                for col in range(17):
                    symbols = rng.integers(0, 1 << bits, 256)
                    packed[row, col] = fixture(symbols, bits)
                    expected[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = expected_tile(symbols, bits)
            np.testing.assert_array_equal(decode_trellis(packed, bits), expected)
            # A reversed input must not force full-array materialization or
            # change how tile boundaries are mapped onto output rows.
            reversed_expected = expected.reshape(17, 16, 17, 16)[::-1, :, ::-1, :].reshape(272, 272)
            np.testing.assert_array_equal(decode_trellis(packed[::-1, ::-1], bits), reversed_expected)

    def test_nontrivial_tiles_and_circular_boundaries(self):
        rng = np.random.default_rng(32019)
        for bits in (2, 3, 4):
            packed = np.empty((2, 3, 16 * bits), dtype=np.int16)
            expected = np.empty((32, 48), dtype=np.float32)
            for row in range(2):
                for col in range(3):
                    symbols = rng.integers(0, 1 << bits, 256)
                    symbols[0], symbols[-1] = 1, (1 << bits) - 1
                    packed[row, col] = fixture(symbols, bits)
                    expected[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = expected_tile(symbols, bits)
            actual = decode_trellis(packed, bits)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(decode_trellis(packed[:, ::-1], bits), expected.reshape(32, 3, 16)[:, ::-1].reshape(32, 48))

    def test_fixed_boundary_fixture(self):
        # A final symbol of 3 and all other symbols zero: first state is 12,
        # and the second is 48; wrapping must stay within this tile.
        symbols = np.zeros(256, dtype=int)
        symbols[-1] = 3
        packed = fixture(symbols, 2).reshape(1, 1, 32)
        self.assertEqual(int(packed[0, 0, -2]), 3)
        self.assertEqual(int(packed[0, 0, -1]), 0)
        actual = decode_trellis(packed, 2)
        self.assertEqual(actual[0, 0], scalar_codebook(12))
        self.assertEqual(actual[1, 0], scalar_codebook(48))
        self.assertEqual(actual[15, 15], scalar_codebook(3))

    def test_k4_fixed_native_word_layout(self):
        symbols = np.tile(np.arange(16), 16)
        packed = fixture(symbols, 4).reshape(1, 1, 64)
        # Symbol stream 01234567 89abcdef is stored as little-endian uint32s.
        np.testing.assert_array_equal(packed.view(np.uint16)[0, 0, :4], [0x4567, 0x0123, 0xCDEF, 0x89AB])
        actual = decode_trellis(packed, 4)
        self.assertEqual(actual[0, 0], scalar_codebook(0xDEF0))
        self.assertEqual(actual[1, 0], scalar_codebook(0xEF01))
        self.assertEqual(actual[15, 15], scalar_codebook(0xCDEF))
        np.testing.assert_array_equal(actual, expected_tile(symbols, 4))

    def test_reconstruction_matches_explicit_hadamard_and_scales(self):
        rng = np.random.default_rng(71)
        packed = rng.integers(-32768, 32768, (8, 16, 48), dtype=np.int16)
        decoded = decode_trellis(packed, 3)
        # Sylvester entry is (-1)**parity(row & column), independently of butterflies.
        h = np.array([[(-1) ** ((r & c).bit_count()) for c in range(128)] for r in range(128)], dtype=np.float64) / np.sqrt(128)
        su = np.linspace(-2, 3, 128, dtype=np.float32)
        sv = np.linspace(4, -1, 256, dtype=np.float32)
        expected = np.concatenate([h @ decoded[:, start:start + 128] @ h for start in (0, 128)], axis=1)
        expected *= su[:, None] * sv[None, :]
        actual = reconstruct(packed, 3, su, sv)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(actual.dtype, np.float32)

    def test_malformed_inputs(self):
        valid = np.zeros((8, 8, 32), dtype=np.int16)
        for bits in (True, 2.0, 1, 5, None):
            with self.assertRaises(ValueError):
                decode_trellis(valid, bits)
        for packed in (valid.astype(np.uint16), valid.astype(np.float32), valid[0], valid[..., :-1], valid[:0]):
            with self.assertRaises(ValueError):
                decode_trellis(packed, 2)
        with self.assertRaises(ValueError):
            reconstruct(valid[:1], 2, np.ones(16), np.ones(128))
        for scale in (np.ones(127), np.ones((128, 1)), np.full(128, np.nan), np.full(128, np.inf), np.ones(128, dtype=complex)):
            with self.assertRaises(ValueError):
                reconstruct(valid, 2, scale, np.ones(128))
            with self.assertRaises(ValueError):
                reconstruct(valid, 2, np.ones(128), scale)


if __name__ == "__main__":
    unittest.main()
