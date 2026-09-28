#pragma once

#include "../rdna_wmma.hip.h"
#include "codebook_rdna.hip.h"

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift)
{
     uint64_t merged = ((uint64_t)a << 32) | (uint64_t) b;
     return (uint32_t)(merged >> shift);

    // Conditional funnel shift is somehow no longer faster
    // if (shift < 32) return __funnelshift_r(b, a, shift);
    // return a >> (shift - 32);
}

template <int bits, int cb>
__device__ __forceinline__ half dq(const uint32_t* ptr, int t_offset)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;  // bit index, start of word0
    int b1 = b0 + 16;                                   // bit index, end of word0
    int i0 = b0 / 32;                                   // uint32 containing first bit of word0
    int i1 = (b1 - 1) / 32;                             // uint32 containing last bit of word0, may be == i0
    int s0 = (i1 + 1) * 32 - b1;                        // shift value to align word1 to 32-bit boundary

    // Load 32 or 64 bits containing word0
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i1 % (bits * 256 / 32)];

    // Shift into place
    uint32_t w0 = __funnelshift_r(b, a, s0) & 0xffff;
    return decode_3inst<cb>(w0);
}

template <int bits, int cb>
__device__ __forceinline__ half2 dq2(const uint32_t* ptr, int t_offset)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;  // bit index, start of word0
    int b1 = b0 + 16;                                   // bit index, end of word0
    int i0 = b0 / 32;                                   // uint32 containing first bit of word0
    int i1 = (b1 - 1) / 32;                             // uint32 containing last bit of word0, may be == i0
    int s0 = (i1 + 1) * 32 - b1;                        // shift value to align word1 to 32-bit boundary

    // Load 32 or 64 bits containing word0
    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i1 % (bits * 256 / 32)];

    // Shift into place
    uint32_t w1 = __funnelshift_r(b, a, s0)        & 0xffff;
    uint32_t w0 = __funnelshift_r(b, a, s0 + bits) & 0xffff;
    return decode_3inst_2<cb>(w0, w1);
}

template <int bits, int cb>
__device__ __forceinline__ void dq4(const uint32_t* ptr, int t_offset, FragB& frag)
{
    int b0 = (t_offset + 257) * bits - 16;      // start of first word
    int b1 = b0 + 3 * bits;                     // start of last word
    int b2 = b1 + 16;                           // end of last word
    int i0 = b0 / 32;                           // uint32 containing first bit of first word
    int i2 = (b2 - 1) / 32;                     // uint32 containing last bit of last word, may be == i0
    int s2 = (i2 + 1) * 32 - b2;                // shift value to align last word to 32-bit boundary

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w3 = fshift(b, a, s2)            & 0xffff;
    uint32_t w2 = fshift(b, a, s2 + bits)     & 0xffff;
    uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
    uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
    half2 d0d1 = decode_3inst_2<cb>(w0, w1);
    half2 d2d3 = decode_3inst_2<cb>(w2, w3);
    frag[0] = d0d1;
    frag[1] = d2d3;
}

template <int bits, int cb>
__device__ __forceinline__ void dq2x2(const uint32_t* ptr, int t_offset, FragB& frag)
{
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        int b0 = (t_offset + 2 * i + 257) * bits - 16;  // start of first word
        int b1 = b0 + 1 * bits;                         // start of last word
        int b2 = b1 + 16;                               // end of last word
        int i0 = b0 / 32;                               // uint32 containing first bit of first word
        int i2 = (b2 - 1) / 32;                         // uint32 containing last bit of last word, may be == i0
        int s2 = (i2 + 1) * 32 - b2;                    // shift value to align last word to 32-bit boundary

        uint32_t a = ptr[i0 % (bits * 256 / 32)];
        uint32_t b = ptr[i2 % (bits * 256 / 32)];
        uint32_t w1 = fshift(b, a, s2)        & 0xffff;
        uint32_t w0 = fshift(b, a, s2 + bits) & 0xffff;
        half2 d0d1 = decode_3inst_2<cb>(w0, w1);
        frag[i] = d0d1;
    }
}

template <int bits, int cb, int align>
__device__ __forceinline__ void dq8(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    int b1 = (t_offset + 257) * bits;               // end of first word
    int b0 = b1 - 16;                               // start of first word
    int b2 = b1 + bits * 7;
    int i0 = b0 / 32;                               // uint32 containing first bit of word0
    int i2 = (b2 - 1) / 32;                         // uint32 containing last bit of word0, may be == i0
    int s2 = (i2 + 1) * 32 - b2;                    // shift value to align last word to 32-bit boundary

    uint32_t a = ptr[i0 % (bits * 256 / 32)];
    uint32_t b = ptr[i2 % (bits * 256 / 32)];
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    if constexpr (align == 1)
    {
        w7 = fshift(b, a, s2);
        w6 = fshift(b, a, s2 + bits);
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = fshift(b, a, s2 + bits * 3);
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = fshift(b, a, s2 + bits * 5);
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = fshift(b, a, s2 + bits * 7);
    }
    if constexpr (align == 2)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = w1 >> bits;
    }
    if constexpr (align == 4)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    if constexpr (align == 8)
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = w4 >> bits;
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    half2 d0d1 = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);
    half2 d2d3 = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);
    half2 d4d5 = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);
    half2 d6d7 = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);
    frag0[0] = d0d1;
    frag0[1] = d2d3;
    frag1[0] = d4d5;
    frag1[1] = d6d7;
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 3;
    i0 = (i1 + 31) & 31;
    a = ptr[i0];
    b = ptr[i1];
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_2bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 4;
    i0 = (i1 + 15) & 15;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_1bit(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 5;
    i0 = (i1 + 7) & 7;
    a = ptr[i0];
    b = ptr[i1];
    b = fshift(b, a, ((~t_offset) & 24));
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 1);
    BFE16_IMM(w5, b, 2);
    BFE16_IMM(w4, b, 3);
    BFE16_IMM(w3, b, 4);
    BFE16_IMM(w2, b, 5);
    BFE16_IMM(w1, b, 6);
    BFE16_IMM(w0, b, 7);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

// Aligned eight-value unpackers for 5 and 6 bits. Every in-tree caller passes
// idx = lane*8 (GEMV/small-M/paired-dot/prefill/reconstruct host audit,
// September 2026), so the eight words span at most three trellis words with
// lane-closed-form shifts. This replaces two generic dq4 calls (four
// constant-modulo word indices, four loads, eight multi-instruction 64-bit
// funnel shifts) with one modulo, three loads, and single-instruction
// funnels plus chained extracts. Extraction order and decode inputs are
// bit-exact against 2x dq4 for all 32 lanes (job-local dq8_check.py).
template <int cb>
__device__ __forceinline__ void dq8_aligned_5bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t q = ((uint32_t) t_offset >> 3) & 3u; // (lane & 3): shift/pair selector
    int b0 = (t_offset + 257) * 5 - 16;
    int i0 = (b0 >> 5) % 40;
    int i1 = i0 + 1; if (i1 == 40) i1 = 0;
    int i2 = i1 + 1; if (i2 == 40) i2 = 0;
    uint32_t w_lo = ptr[i0], w_mid = ptr[i1], w_hi = ptr[i2];
    // First four words always sit in (w_lo, w_mid); 3*5+16 = 31 bits fit one
    // 32-bit funnel output, so one funnel plus chained extracts suffices.
    uint32_t f0, f1, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(f0, w_mid, w_lo, (12u - (q << 3)) & 31u);
    w3 = f0 & 0xffff;
    BFE16_IMM(w2, f0, 5);
    BFE16_IMM(w1, f0, 10);
    BFE16_IMM(w0, f0, 15);
    // Last four words sit in (w_mid, w_hi) for q < 2, else (w_lo, w_mid).
    uint32_t lo1 = (q < 2) ? w_hi : w_mid;
    uint32_t hi1 = (q < 2) ? w_mid : w_lo;
    FSHF_IMM(f1, lo1, hi1, (24u - (q << 3)) & 31u);
    w7 = f1 & 0xffff;
    BFE16_IMM(w6, f1, 5);
    BFE16_IMM(w5, f1, 10);
    BFE16_IMM(w4, f1, 15);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int cb>
__device__ __forceinline__ void dq8_aligned_6bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t p = ((uint32_t) t_offset >> 3) & 1u; // lane parity: shift/pair selector
    int b0 = (t_offset + 257) * 6 - 16;
    int i0 = (b0 >> 5) % 48;
    int i1 = i0 + 1; if (i1 == 48) i1 = 0;
    int i2 = i1 + 1; if (i2 == 48) i2 = 0;
    uint32_t w_lo = ptr[i0], w_mid = ptr[i1], w_hi = ptr[i2];
    // 3*6+16 = 34 bits exceed one 32-bit funnel output, so each group needs
    // two funnels (align-2 style). First group always sits in (w_lo, w_mid);
    // on odd lanes its second pair (shifts 36/42) lies in w_lo alone past
    // the funnel window and is extracted with direct shifts.
    uint32_t f0, fb, fc, fd, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(f0, w_mid, w_lo, p ? 24u : 8u);
    w3 = f0 & 0xffff;
    BFE16_IMM(w2, f0, 6);
    FSHF_IMM(fb, w_mid, w_lo, 20);
    w1 = p ? ((w_lo >> 4) & 0xffff) : (fb & 0xffff);
    w0 = p ? ((w_lo >> 10) & 0xffff) : (((fb >> 6) & 0xffff));
    // Last group sits in (w_mid, w_hi) on even lanes, (w_lo, w_mid) on odd.
    uint32_t lo1 = p ? w_mid : w_hi;
    uint32_t hi1 = p ? w_lo : w_mid;
    uint32_t s1 = p ? 0u : 16u;
    FSHF_IMM(fc, lo1, hi1, s1);
    FSHF_IMM(fd, lo1, hi1, s1 + 12);
    w7 = fc & 0xffff;
    BFE16_IMM(w6, fc, 6);
    w5 = fd & 0xffff;
    BFE16_IMM(w4, fd, 6);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}


template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits_bfe64(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    int i1 = t_offset / 8;
    int i0 = (i1 + 31) % 32;
    uint32_t a = ptr[i0];
    uint32_t b = ptr[i1];
    uint32_t w7 = bfe64(b, a, 0, 16);
    uint32_t w6 = bfe64(b, a, 4, 16);
    uint32_t w5 = bfe64(b, a, 8, 16);
    uint32_t w4 = bfe64(b, a, 12, 16);
    uint32_t w3 = bfe64(b, a, 16, 16);
    uint32_t w2 = bfe64(b, a, 20, 16);
    uint32_t w1 = bfe64(b, a, 24, 16);
    uint32_t w0 = bfe64(b, a, 28, 16);
    frag0[0] = decode_3inst_2<cb>(w0, w1);
    frag0[1] = decode_3inst_2<cb>(w2, w3);
    frag1[0] = decode_3inst_2<cb>(w4, w5);
    frag1[1] = decode_3inst_2<cb>(w6, w7);
}

template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch(const uint32_t* ptr, int idx, FragB& frag0, FragB& frag1)
{
    if constexpr (bits == 1)
    {
        dq8_aligned_1bit<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 2)
    {
        dq8_aligned_2bits<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 3)
    {
        dq8<bits, cb, 4>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 4)
    {
        dq8_aligned_4bits<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 5)
    {
        // All callers pass idx = lane*8; see dq8_aligned_5bits note.
        dq8_aligned_5bits<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 6)
    {
        // All callers pass idx = lane*8; see dq8_aligned_6bits note.
        dq8_aligned_6bits<cb>(ptr, idx, frag0, frag1);
    }
    else if constexpr (bits == 7)
    {
        dq2x2<bits, cb>(ptr, idx, frag0);
        dq2x2<bits, cb>(ptr, idx + 4, frag1);
    }
    else if constexpr (bits == 8)
    {
        dq4<bits, cb>(ptr, idx, frag0);
        dq4<bits, cb>(ptr, idx + 4, frag1);
    }
}