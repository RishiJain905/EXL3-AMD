// Native FP32-output GEMM: fp16 A/B -> fp32 C via RDNA WMMA.
//
// Target: the model projections hgemm runs with fp32 output (fp16-output path
// is fast but NOT an acceptable substitute). Non-cooperative, no atomics, no
// persistent buffers; fp32 accumulators stay in registers.
//
// Provenance: built only on the validated rdna_wmma.hip.h primitives
// (load_matrix_a / load_matrix_b / mma_sync / store_matrix_c*). Operand order
// (B, A, C) and the C-fragment lane layout are handled inside those helpers.

#include "hgemm_wmma.hip.h"

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cstring>
#include <limits>

#include "rdna_wmma.hip.h"
#include "util.cuh"   // half4, cuda_check (parent dir is on the .hip include path)

namespace {

// Two tile shapes share the templated kernel below; host selects by M.
// Serial 128x128x64 covers M < 512; prefetch 256x64x32 covers M >= 512.
// Both shapes launch 512 threads; grids derive from the selected shape.
constexpr int kSerialTileM = 128;
constexpr int kSerialTileN = 128;
constexpr int kSerialTileK = 64;
constexpr int kPrefTileM = 256;
constexpr int kPrefTileN = 64;
constexpr int kPrefTileK = 32;
constexpr int64_t kPrefMinM = 512;

// Zero half4 without braced half2 init (ambiguous overloads on HIP).
__device__ __forceinline__ half4 half4_zero()
{
    half z = __float2half(0.0f);
    half2 z2 = __halves2half2(z, z);
    return half4(z2, z2);
}

// Output-tiled GEMM, K looped in TILE_K slices.
//
// Thread map: one 32x32 output quadrant per wave32, so
// NTHREADS = (TILE_M/32) * (TILE_N/32) * 32. Each wave holds
// four 16x16x16 WMMA accumulators (2x2) covering its quadrant in fp32.
//
// Per K tile the block runs TILE_K/16 WMMA sub-steps from padded LDS.
// PREFETCH=false stages the current tile into LDS each iteration (serial
// load, barrier, compute, trailing barrier). PREFETCH=true stages tile 0
// once, then each iteration issues the next tile's global half4 loads into
// registers before computing the current tile, and barrier/commit/barrier
// publishes it when a next tile exists. M-tail rows zero-fill on load or
// at prefetch and are skipped on store; K/N are host-gated to full tiles
// (see hgemm_wmma_try).
template <int TILE_M, int TILE_N, int TILE_K, bool PREFETCH>
__global__ __launch_bounds__((TILE_M / 32) * (TILE_N / 32) * 32)
__attribute__((amdgpu_flat_work_group_size((TILE_M / 32) * (TILE_N / 32) * 32, (TILE_M / 32) * (TILE_N / 32) * 32)))
void hgemm_wmma_tile(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, int ldc)
{
    static_assert(TILE_M % 32 == 0 && TILE_N % 32 == 0,
        "tile must be a whole number of 32x32 wave quadrants");
    static_assert(TILE_K % 16 == 0 && TILE_K > 0,
        "K tile must be a whole number of WMMA_K steps");
    static_assert(128 % TILE_K == 0 && 128 % TILE_N == 0,
        "tile must divide the host K/N gate (multiples of 128) into full tiles");

    constexpr int WAVE_M = TILE_M / 32;
    constexpr int WAVE_N = TILE_N / 32;
    constexpr int NWAVES = WAVE_M * WAVE_N;
    constexpr int NTHREADS = NWAVES * 32;
    // A fragments load a 32-byte vector, so every A row must stay aligned.
    // B fragments gather scalars; padding breaks up LDS bank conflicts.
    constexpr int SA = TILE_K + 16;
    constexpr int SB = TILE_N + 8;
    static_assert(((size_t)TILE_M * SA + (size_t)TILE_K * SB) * sizeof(half) <= 48 * 1024,
        "A/B LDS tiles must fit comfortably in per-workgroup LDS");
    // half4-vectorized cooperative loads: tile element counts must be whole
    // half4 slots (follows from the K-step assert, stated for the reader).
    static_assert((TILE_M * TILE_K) % 4 == 0 && (TILE_K * TILE_N) % 4 == 0,
        "tiles must vectorize into half4");
    constexpr int A4 = TILE_M * TILE_K / 4;
    constexpr int B4 = TILE_K * TILE_N / 4;
    // Fixed per-thread half4 slots feeding the register prefetch arrays
    // (serial 128x128x64: 4 A + 4 B per thread; prefetch 256x64x32:
    // 4 A + 1 B per thread). Divisibility is required so every thread owns
    // the same fixed slots; the unrolled loops below then index the
    // register arrays with constants.
    constexpr int kLoadsA4 = A4 / NTHREADS;
    constexpr int kLoadsB4 = B4 / NTHREADS;
    static_assert(A4 % NTHREADS == 0,
        "A half4 slots must split evenly for fixed prefetch registers");
    static_assert(B4 % NTHREADS == 0,
        "B half4 slots must split evenly for fixed prefetch registers");

    const int tid = threadIdx.x;   // host launches exactly NTHREADS threads
    const int wave = tid >> 5;
    const int wave_m = wave / WAVE_N;
    const int wave_n = wave % WAVE_N;
    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    __shared__ __align__(32) half sh_a[TILE_M * SA];
    __shared__ __align__(32) half sh_b[TILE_K * SB];

    // Four partial and four total FP32 accumulators per wave.
    // Lane layout is the rdna_wmma C mapping: lane L holds rows (L % 16),
    // columns [i*2 + (L >= 16)] for i in 0..7.
    WmmaFragC acc[2][2];
    WmmaFragC total[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            acc[i][j].clear();
            total[i][j].clear();
        }

    const int k_tiles = K / TILE_K;   // exact: host gates K % 128 == 0

    // Serial stages every tile; prefetch stages tile 0 and then consumes the
    // tile published by the previous iteration. All publication/load guards
    // are uniform across the block. PREFETCH=false removes the next-tile work.
    for (int kt = 0; kt < k_tiles; ++kt)
    {
        const int k0 = kt * TILE_K;
        if (!PREFETCH || kt == 0)
        {
            // Stage current tile (k0). Addressing, alignment and guards:
            // A base 8-byte aligned (host gate), row pitch K*2 a multiple
            // of 256, col*2 a multiple of 8; LDS base 32-byte aligned,
            // (row*SA + col) a multiple of 4 (SA % 4 == 0, col % 4 == 0).
            // M-tail rows zero-fill.
            for (int i = tid; i < A4; i += NTHREADS)
            {
                const int e = i * 4;
                const int row = e / TILE_K;
                const int col = e % TILE_K;
                const int grow = m0 + row;
                half4 v = half4_zero();
                if (grow < M)
                    v = *(const half4*)(A + (size_t)grow * K + k0 + col);
                *(half4*)(sh_a + (size_t)row * SA + col) = v;
            }
            // B tile, half4-vectorized. In-range by the host K/N multiple-of-128
            // gate; relaxing that gate requires the same zero-fill guard as A.
            for (int i = tid; i < B4; i += NTHREADS)
            {
                const int e = i * 4;
                const int row = e / TILE_N;
                const int col = e % TILE_N;
                *(half4*)(sh_b + (size_t)row * SB + col) =
                    *(const half4*)(B + (size_t)(k0 + row) * N + n0 + col);
            }
            __syncthreads();
        }

        const bool has_next = PREFETCH && (kt + 1 < k_tiles);
        half4 pref_a[kLoadsA4];
        half4 pref_b[kLoadsB4];

        // Prefetch NEXT tile globals into registers (no LDS touch, so no
        // barrier needed around this). Same addresses as the staged loads
        // with k0_next in place of k0; A M-tail zero-fills here exactly.
        // Dead code when PREFETCH=false (has_next is always false).
        if (has_next)
        {
            const int k0_next = (kt + 1) * TILE_K;
            #pragma unroll
            for (int j = 0; j < kLoadsA4; ++j)
            {
                const int i = tid + j * NTHREADS;
                const int e = i * 4;
                const int row = e / TILE_K;
                const int col = e % TILE_K;
                const int grow = m0 + row;
                half4 v = half4_zero();
                if (grow < M)
                    v = *(const half4*)(A + (size_t)grow * K + k0_next + col);
                pref_a[j] = v;
            }
            #pragma unroll
            for (int j = 0; j < kLoadsB4; ++j)
            {
                const int i = tid + j * NTHREADS;
                const int e = i * 4;
                const int row = e / TILE_N;
                const int col = e % TILE_N;
                pref_b[j] =
                    *(const half4*)(B + (size_t)(k0_next + row) * N + n0 + col);
            }
        }

        #pragma unroll
        for (int s = 0; s < TILE_K / 16; ++s)
        {
            WmmaFragA fa[2];
            WmmaFragB fb[2];
            #pragma unroll
            for (int mr = 0; mr < 2; ++mr)
                rdna_wmma::load_matrix_a(fa[mr],
                    sh_a + (size_t)(wave_m * 32 + mr * 16) * SA + s * 16, SA);
            #pragma unroll
            for (int nc = 0; nc < 2; ++nc)
                rdna_wmma::load_matrix_b(fb[nc],
                    sh_b + (size_t)s * 16 * SB + wave_n * 32 + nc * 16, SB);
            // All 32 lanes execute every WMMA; operand order (B, A, C) is
            // handled inside mma_sync. Tail rows carry zeros and are skipped
            // on store, so no lane diverges here.
            #pragma unroll
            for (int mr = 0; mr < 2; ++mr)
                #pragma unroll
                for (int nc = 0; nc < 2; ++nc)
                    rdna_wmma::mma_sync(acc[mr][nc], fa[mr], fb[nc]);
        }
        // Bound WMMA accumulation chains. Folding 256 K elements with normal
        // FP32 addition avoids the measured long-K drift on gfx1101.
        if ((kt + 1) % (256 / TILE_K) == 0 || kt + 1 == k_tiles)
        {
            #pragma unroll
            for (int mr = 0; mr < 2; ++mr)
                #pragma unroll
                for (int nc = 0; nc < 2; ++nc)
                {
                    total[mr][nc].data += acc[mr][nc].data;
                    acc[mr][nc].clear();
                }
        }

        // Publish prefetched tile for the next iteration. First barrier: all
        // LDS reads of the current tile are done. Second barrier: the commit
        // is visible to the whole block. Skipped on the last tile (nothing
        // prefetched, stores do not use LDS). Dead code when PREFETCH=false.
        if (has_next)
        {
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < kLoadsA4; ++j)
            {
                const int i = tid + j * NTHREADS;
                const int e = i * 4;
                const int row = e / TILE_K;
                const int col = e % TILE_K;
                *(half4*)(sh_a + (size_t)row * SA + col) = pref_a[j];
            }
            #pragma unroll
            for (int j = 0; j < kLoadsB4; ++j)
            {
                const int i = tid + j * NTHREADS;
                const int e = i * 4;
                const int row = e / TILE_N;
                const int col = e % TILE_N;
                *(half4*)(sh_b + (size_t)row * SB + col) = pref_b[j];
            }
            __syncthreads();
        }
        if constexpr (!PREFETCH)
            __syncthreads();
    }

    // FP32 stores. Rows guarded for the M tail (uniform per wave, so the
    // full/partial choice does not diverge); columns are full tiles by the
    // host N gate. For a future N-tail shape, route the partial-column case
    // through store_matrix_c_checked with the matching valid_cols.
    #pragma unroll
    for (int mr = 0; mr < 2; ++mr)
    {
        const int r = m0 + wave_m * 32 + mr * 16;
        const int valid = M - r;
        #pragma unroll
        for (int nc = 0; nc < 2; ++nc)
        {
            if (valid > 0)
            {
                float* dst = C + (size_t)r * ldc + n0 + wave_n * 32 + nc * 16;
                if (valid >= 16)
                    rdna_wmma::store_matrix_c(dst, total[mr][nc], ldc);
                else
                    rdna_wmma::store_matrix_c_checked(dst, total[mr][nc], ldc, valid, 16);
            }
        }
    }
}

// True when [storage_offset, storage_offset + span_elems) * esz fits inside
// the storage, with overflow-safe arithmetic. Guards strided C slices whose
// numel does not cover the (M-1)*ldc+N span.
static bool storage_covers(const at::Tensor& t, int64_t span_elems, int64_t esz)
{
    if (span_elems < 0 || esz <= 0) return false;
    int64_t off = t.storage_offset();
    if (off < 0) return false;
    uint64_t nbytes = (uint64_t)t.storage().nbytes();
    uint64_t span = (uint64_t)span_elems;
    uint64_t e = (uint64_t)esz;
    uint64_t o = (uint64_t)off;
    if (span != 0 && e > UINT64_MAX / span) return false;
    uint64_t need = span * e;
    if (o != 0 && e > UINT64_MAX / o) return false;
    uint64_t offb = o * e;
    if (need > UINT64_MAX - offb) return false;
    return offb + need <= nbytes;
}

} // namespace

bool hgemm_wmma_try(at::Tensor a, at::Tensor b, at::Tensor c, cudaStream_t stream)
{
    // Strict guards: anything unexpected returns false with no launch and no
    // output mutation. The common hgemm validation handles malformed calls.
    if (!a.defined() || !b.defined() || !c.defined()) return false;
    if (!a.device().is_cuda() || !b.device().is_cuda() || !c.device().is_cuda()) return false;
    if (a.get_device() != b.get_device() || a.get_device() != c.get_device()) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf || c.dtype() != at::kFloat) return false;
    if (a.dim() < 1 || b.dim() != 2 || c.dim() < 1) return false;

    const int64_t K64 = a.size(-1);
    const int64_t N64 = b.size(1);
    if (K64 <= 0 || N64 <= 0) return false;
    if (b.size(0) != K64) return false;
    if (a.numel() % K64 != 0) return false;
    const int64_t M64 = a.numel() / K64;

    // A: contiguous inner dim plus row-major flattening (contiguous any-dim,
    // or a 2D row slice with stride (K, 1)).
    if (a.stride(-1) != 1) return false;
    if (!a.is_contiguous() && !(a.dim() == 2 && a.stride(0) == K64)) return false;
    // B: row-major K x N.
    if (b.stride(0) != N64 || b.stride(1) != 1) return false;

    // C: contiguous (any dim, matching M*N elements) or 2D with contiguous
    // columns and row stride >= N (output slices). Never resized.
    int64_t ldc64;
    if (c.dim() == 1)
    {
        if (!c.is_contiguous()) return false;
        ldc64 = N64;
    }
    else
    {
        if (c.stride(-1) != 1) return false;
        ldc64 = c.stride(-2);
        if (ldc64 < N64) return false;
        if (!c.is_contiguous())
        {
            if (c.dim() != 2) return false;
            if (c.size(0) != M64 || c.size(1) != N64) return false;
        }
    }

    // Integer range before any multiplication or kernel pointer arithmetic.
    constexpr int64_t IMAX = (int64_t)std::numeric_limits<int>::max();
    if (M64 > IMAX || K64 > IMAX || N64 > IMAX || ldc64 > IMAX) return false;
    if (c.is_contiguous() && c.numel() != M64 * N64) return false;  // M64*N64 fits int64 here

    // Shape gate: prefill-sized rows, K/N positive multiples of 128 (hence
    // also multiples of either tile width: full K tiles, full N tiles,
    // M tail only).
    if (M64 < 64) return false;
    if (K64 % 128 != 0 || N64 % 128 != 0) return false;

    // Tile choice by M: serial 128x128x64 below 512 rows, prefetch
    // 256x64x32 at or above. Both shapes divide the K/N gate into full
    // tiles and launch 512 threads; grids derive from the selected shape.
    const bool use_prefetch = (M64 >= kPrefMinM);
    const int64_t selTileM = use_prefetch ? kPrefTileM : kSerialTileM;
    const int64_t selTileN = use_prefetch ? kPrefTileN : kSerialTileN;

    // Grid fits dim3 (y/x limits 65535).
    const int64_t m_tiles = (M64 + selTileM - 1) / selTileM;
    const int64_t n_tiles = N64 / selTileN;   // exact by the N gate
    if (m_tiles > 65535 || n_tiles > 65535 || n_tiles < 1) return false;

    // Vector-load alignment: half4 global loads need 8-byte A/B bases
    // (row pitches and tile offsets are then provably 8-byte aligned, see
    // the kernel); C stores are scalar floats, 4-byte alignment suffices.
    if (reinterpret_cast<uintptr_t>(a.data_ptr()) % 8 != 0) return false;
    if (reinterpret_cast<uintptr_t>(b.data_ptr()) % 8 != 0) return false;
    if (reinterpret_cast<uintptr_t>(c.data_ptr()) % 4 != 0) return false;

    // Storage coverage for the exact spans the kernel touches.
    if (!storage_covers(a, M64 * K64, 2)) return false;
    if (!storage_covers(b, K64 * N64, 2)) return false;
    if (!storage_covers(c, (M64 - 1) * ldc64 + N64, 4)) return false;

    // Explicit WMMA targets; gfx12 uses the shared fragment adapter. This is
    // an opt-in path. Hardware validation is recorded separately per target.
    // The gcnArchName field exists only on ROCm; elsewhere unsupported.
#if defined(USE_ROCM)
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, a.get_device()) != cudaSuccess) return false;
    if (std::strncmp(prop.gcnArchName, "gfx1101", 7) != 0 &&
        std::strncmp(prop.gcnArchName, "gfx1200", 7) != 0 &&
        std::strncmp(prop.gcnArchName, "gfx1201", 7) != 0) return false;
    if (prop.gcnArchName[7] != '\0' && prop.gcnArchName[7] != ':') return false;
#else
    return false;
#endif

    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    constexpr int kSerialThreads = (kSerialTileM / 32) * (kSerialTileN / 32) * 32;
    constexpr int kPrefThreads = (kPrefTileM / 32) * (kPrefTileN / 32) * 32;
    static_assert(kSerialThreads == 512 && kPrefThreads == 512,
        "both tile shapes launch 512 threads");
    constexpr int NTHREADS = 512;
    dim3 grid((unsigned)n_tiles, (unsigned)m_tiles);

    if (use_prefetch)
        hgemm_wmma_tile<kPrefTileM, kPrefTileN, kPrefTileK, true><<<grid, NTHREADS, 0, stream>>>(
            static_cast<const half*>(a.data_ptr()),
            static_cast<const half*>(b.data_ptr()),
            static_cast<float*>(c.data_ptr()),
            (int)M64, (int)N64, (int)K64, (int)ldc64);
    else
        hgemm_wmma_tile<kSerialTileM, kSerialTileN, kSerialTileK, false><<<grid, NTHREADS, 0, stream>>>(
            static_cast<const half*>(a.data_ptr()),
            static_cast<const half*>(b.data_ptr()),
            static_cast<float*>(c.data_ptr()),
            (int)M64, (int)N64, (int)K64, (int)ldc64);
    cuda_check(cudaPeekAtLastError());
    return true;
}
