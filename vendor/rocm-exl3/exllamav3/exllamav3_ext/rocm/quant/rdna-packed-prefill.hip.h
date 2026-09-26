#pragma once
// Tiled packed prefill (rows 65..4096) noncooperative WMMA path over packed weights.
// 128x64x32 output/K tile with 256 threads (eight wave32 quadrants, four 16x16
// accumulators per wave). Each packed 16x16 weight tile is decoded once per
// block into padded LDS and reused across 128 M rows, instead of decoding per
// 16-row tile. M tails masked; K/N are host-gated to multiples of 128.
// Borrows the proven dq_dispatch unswizzle from rdna-smallm-mid.hip.h and the
// dense hgemm_wmma.hip accumulation scheme (fold FP32 partials every 256 K).
// No full FP16 weight reconstruction, cooperative launch, atomics, global
// software barriers, JIT allocation or autotuning.
// Derived from ExLlamaV3 / the CarouselAether ROCm fork. MIT, Copyright (c)
// 2025 Turboderp. License retained in rdna-smallm-LICENSE.txt.
// Included after the small-M transforms, WMMA core and warp-selection helper.

extern "C" __attribute__((visibility("default")))
int quantlab_exl3_packed_prefill_abi() { return 1; }

__device__ __forceinline__ static half4 exl3_prefill_half4_zero()
{
    half z = __float2half(0.0f);
    half2 z2 = __halves2half2(z, z);
    return half4(z2, z2);
}

template <int BITS, int CB, bool FP32>
static __global__ __launch_bounds__(256)
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void exl3_packed_prefill_wmma(const half* __restrict__ A, const uint16_t* __restrict__ B,
                    void* __restrict__ C, int size_m, int size_k, int size_n)
{
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 64;
    constexpr int TILE_K = 32;
    constexpr int WAVE_N = 2;
    constexpr int SA = TILE_K + 16;
    constexpr int SB = TILE_N + 8;
    constexpr int A4 = TILE_M * TILE_K / 4;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int wave_m = wave / WAVE_N;
    const int wave_n = wave % WAVE_N;
    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    __shared__ __align__(32) half sh_a[TILE_M * SA];
    __shared__ __align__(32) half sh_b[TILE_K * SB];

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

    const int k_tiles = size_k / TILE_K;
    const int n_tiles = size_n / 16;

    for (int kt = 0; kt < k_tiles; ++kt)
    {
        const int k0 = kt * TILE_K;

        // Stage the dense A tile (half4-vectorized, M-tail zero-fill). K is a
        // full tile by the host gate. LDS base is 32-byte aligned and
        // (row*SA+col) is a multiple of 4 (SA % 4 == 0, col % 4 == 0).
        for (int i = tid; i < A4; i += 256)
        {
            const int e = i * 4;
            const int row = e / TILE_K;
            const int col = e % TILE_K;
            const int grow = m0 + row;
            half4 v = exl3_prefill_half4_zero();
            if (grow < size_m)
                v = *(const half4*)(A + (size_t)grow * size_k + k0 + col);
            *(half4*)(sh_a + (size_t)row * SA + col) = v;
        }

        // Decode the 2x4 weight tiles covering this K slice into sh_b. Wave w
        // owns tile (ki = w>>2, ni = w&3); the proven shuffle/unswizzle from
        // exl3_gemm_inner_rdna.hip.h writes it row-major at stride SB. All
        // lanes call __shfl_down; only !(lane & 4) store.
        {
            const int ki = wave >> 2;
            const int ni = wave & 3;
            const int k_tile = k0 / 16 + ki;
            const int tile_n = n0 / 16 + ni;
            const uint32_t* packed = (const uint32_t*)
                (B + (k_tile * n_tiles + tile_n) * (16 * BITS));
            FragB frag0, frag1;
            dq_dispatch<BITS, CB>(packed, lane << 3, frag0, frag1);

            half2 n0l = __shfl_down(frag0[0], 4, 32);
            half2 n1l = __shfl_down(frag0[1], 4, 32);
            half2 n2l = __shfl_down(frag1[0], 4, 32);
            half2 n3l = __shfl_down(frag1[1], 4, 32);

            if (!(lane & 4))
            {
                half* dst = sh_b + (size_t)ki * 16 * SB + ni * 16;
                const int r0 = (lane % 4) * 2;
                const int r1 = r0 + 1;
                const int r2 = r0 + 8;
                const int r3 = r0 + 9;
                const int c0 = (lane / 8) * 2;
                const int c1 = c0 + 8;

                dst[r0 * SB + c0]     = __low2half (frag0[0]);
                dst[r0 * SB + c0 + 1] = __low2half (n0l);
                dst[r1 * SB + c0]     = __high2half(frag0[0]);
                dst[r1 * SB + c0 + 1] = __high2half(n0l);

                dst[r2 * SB + c0]     = __low2half (frag0[1]);
                dst[r2 * SB + c0 + 1] = __low2half (n1l);
                dst[r3 * SB + c0]     = __high2half(frag0[1]);
                dst[r3 * SB + c0 + 1] = __high2half(n1l);

                dst[r0 * SB + c1]     = __low2half (frag1[0]);
                dst[r0 * SB + c1 + 1] = __low2half (n2l);
                dst[r1 * SB + c1]     = __high2half(frag1[0]);
                dst[r1 * SB + c1 + 1] = __high2half(n2l);

                dst[r2 * SB + c1]     = __low2half (frag1[1]);
                dst[r2 * SB + c1 + 1] = __low2half (n3l);
                dst[r3 * SB + c1]     = __high2half(frag1[1]);
                dst[r3 * SB + c1 + 1] = __high2half(n3l);
            }
        }

        // sh_b is block-shared (each wave reads tiles decoded by other waves),
        // so the whole block barriers before compute. No mem_fence needed.
        __syncthreads();

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
            // All 32 lanes always execute WMMA.
            #pragma unroll
            for (int mr = 0; mr < 2; ++mr)
                #pragma unroll
                for (int nc = 0; nc < 2; ++nc)
                    rdna_wmma::mma_sync(acc[mr][nc], fa[mr], fb[nc]);
        }
        // Bound WMMA accumulation chains: fold 256 K elements with normal FP32
        // addition (TILE_K=32, so every 8 slices), as in hgemm_wmma.hip.
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
        if (kt + 1 < k_tiles)
            __syncthreads();
    }

    // Stores with M-tail masking; N is a full tile by the host gate.
    #pragma unroll
    for (int mr = 0; mr < 2; ++mr)
    {
        const int r = m0 + wave_m * 32 + mr * 16;
        const int valid = size_m - r;
        #pragma unroll
        for (int nc = 0; nc < 2; ++nc)
        {
            if (valid > 0)
            {
                if constexpr (FP32)
                {
                    float* dst = (float*)C + (size_t)r * size_n + n0 + wave_n * 32 + nc * 16;
                    if (valid >= 16)
                        rdna_wmma::store_matrix_c(dst, total[mr][nc], size_n);
                    else
                        rdna_wmma::store_matrix_c_checked(dst, total[mr][nc], size_n, valid, 16);
                }
                else
                {
                    half* dst = (half*)C + (size_t)r * size_n + n0 + wave_n * 32 + nc * 16;
                    if (valid >= 16)
                        rdna_wmma::store_matrix_c_half(dst, total[mr][nc], size_n);
                    else
                        rdna_wmma::store_matrix_c_half_checked(dst, total[mr][nc], size_n, valid, 16);
                }
            }
        }
    }
}

template <int BITS, int CB, bool FP32>
static void exl3_packed_prefill_launch(const half* a, const uint16_t* b, void* c,
    int m, int k, int n, const half* suh, half* ah, const half* svh, cudaStream_t stream)
{
    hipLaunchKernelGGL((exl3_smallm_had<false, false>), dim3((k / 128 + 7) / 8, 1, m),
        dim3(256), 0, stream, a, ah, suh, k, nullptr);
    int m_tiles = (m + 127) / 128;
    hipLaunchKernelGGL((exl3_packed_prefill_wmma<BITS, CB, FP32>),
        dim3(n / 64, m_tiles), dim3(256), 0, stream, ah, b, c, m, k, n);
    hipLaunchKernelGGL((exl3_smallm_had<true, FP32>), dim3((n / 128 + 7) / 8, 1, m),
        dim3(256), 0, stream, c, c, svh, n, nullptr);
}

static bool exl3_packed_prefill_try(const half* a, const uint16_t* b, void* c,
    int m, int k, int n, int bits, int cb, bool fp32,
    const half* suh, half* ah, const half* svh, cudaStream_t stream)
{
    const char* flag = std::getenv("EXL3_PACKED_PREFILL");
    if (!flag || atoi(flag) != 1) return false;
    if (m < 65 || m > 4096 || k <= 0 || n <= 0 || k % 128 || n % 128) return false;
    if (cb != 0 && cb != 2) return false;
    if (cb == 0 && (bits < 2 || bits > 4)) return false;
    if (cb == 2 && (bits < 2 || bits > 6)) return false;
    if (!suh || !ah || !svh) return false;
    // half4 global loads need 8-byte bases (row pitches and tile offsets are
    // then provably aligned); scalar C stores need 4-byte alignment.
    if ((reinterpret_cast<uintptr_t>(ah) % 8) != 0) return false;
    if ((reinterpret_cast<uintptr_t>(c) % 4) != 0) return false;
    (void) a; (void) b;
    #define PACKED_PREFILL_TYPED(K, C) \
        if (fp32) exl3_packed_prefill_launch<K, C, true>(a,b,c,m,k,n,suh,ah,svh,stream); \
        else exl3_packed_prefill_launch<K, C, false>(a,b,c,m,k,n,suh,ah,svh,stream)
    #define PACKED_PREFILL_CB(K) \
        switch (cb) { \
            case 0: PACKED_PREFILL_TYPED(K, 0); break; \
            case 2: PACKED_PREFILL_TYPED(K, 2); break; \
        }
    switch (bits)
    {
        case 2: PACKED_PREFILL_CB(2); break;
        case 3: PACKED_PREFILL_CB(3); break;
        case 4: PACKED_PREFILL_CB(4); break;
        case 5: PACKED_PREFILL_TYPED(5, 2); break;
        case 6: PACKED_PREFILL_TYPED(6, 2); break;
    }
    #undef PACKED_PREFILL_CB
    #undef PACKED_PREFILL_TYPED
    return true;
}
