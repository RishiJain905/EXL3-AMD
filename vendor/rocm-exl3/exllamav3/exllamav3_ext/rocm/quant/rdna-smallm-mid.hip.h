#pragma once
// Packed mid-M (rows 10..64) noncooperative WMMA path over packed weights.
// Reuses the proven small-M decoding, LDS unswizzle and WMMA fragment layout
// (see rdna-smallm-wmma.hip.h); M is runtime with 16-row blockIdx.y tiles and
// masked A loads / C stores, so wide batches cost at most 4 passes over the
// weights. No full FP16 weight reconstruction, no cooperative launch, no
// cross-block reduction: bounded in-block split-K with WARPS 4/8/16 only.
// Derived from ExLlamaV3 / the CarouselAether ROCm fork. MIT, Copyright (c)
// 2025 Turboderp. License retained in rdna-smallm-LICENSE.txt.
// Included after the small-M transforms, WMMA core and warp-selection helper.

extern "C" __attribute__((visibility("default")))
int quantlab_exl3_packed_mid_abi() { return 1; }

template <int BITS, int CB, bool FP32, int WARPS>
static __global__ __launch_bounds__(WARPS * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS * 32, WARPS * 32)))
void exl3_packed_mid_wmma(const half* __restrict__ A, const uint16_t* __restrict__ B,
                    void* __restrict__ C, int size_m, int size_k, int size_n)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int tile_n = blockIdx.x;
    int row_base = blockIdx.y * 16;
    int n_tiles = size_n / 16;
    int k_tiles = size_k / 16;
    int chunk = (k_tiles + WARPS - 1) / WARPS;
    int begin = warp * chunk;
    int end = begin + chunk < k_tiles ? begin + chunk : k_tiles;

    // Same staging as the small-M WMMA kernel: warp-private dequant tile with
    // stride-18 LDS padding plus the block-wide split-K reduction buffer.
    __shared__ half decoded[WARPS][16 * 18];
    __shared__ float partial[WARPS][16 * 16];

    WmmaFragC acc;
    acc.clear();

    for (int tile_k = begin; tile_k < end; ++tile_k)
    {
        const uint32_t* packed = (const uint32_t*)
            (B + (tile_k * n_tiles + tile_n) * (16 * BITS));
        FragB frag0, frag1;
        dq_dispatch<BITS, CB>(packed, lane << 3, frag0, frag1);

        // Proven shuffle/unswizzle mapping from exl3_gemm_inner_rdna.hip.h:
        // lanes with bit 2 clear combine their own values with lane+4 and
        // write the 16x16 block row-major. All lanes call __shfl_down; only
        // !(lane & 4) store.
        half* B_lds = decoded[warp];
        half2 n0 = __shfl_down(frag0[0], 4, 32);
        half2 n1 = __shfl_down(frag0[1], 4, 32);
        half2 n2 = __shfl_down(frag1[0], 4, 32);
        half2 n3 = __shfl_down(frag1[1], 4, 32);

        if (!(lane & 4))
        {
            constexpr int S = 18;
            const int r0 = (lane % 4) * 2;
            const int r1 = r0 + 1;
            const int r2 = r0 + 8;
            const int r3 = r0 + 9;
            const int c0 = (lane / 8) * 2;
            const int c1 = c0 + 8;

            B_lds[r0 * S + c0]     = __low2half (frag0[0]);
            B_lds[r0 * S + c0 + 1] = __low2half (n0);
            B_lds[r1 * S + c0]     = __high2half(frag0[0]);
            B_lds[r1 * S + c0 + 1] = __high2half(n0);

            B_lds[r2 * S + c0]     = __low2half (frag0[1]);
            B_lds[r2 * S + c0 + 1] = __low2half (n1);
            B_lds[r3 * S + c0]     = __high2half(frag0[1]);
            B_lds[r3 * S + c0 + 1] = __high2half(n1);

            B_lds[r0 * S + c1]     = __low2half (frag1[0]);
            B_lds[r0 * S + c1 + 1] = __low2half (n2);
            B_lds[r1 * S + c1]     = __high2half(frag1[0]);
            B_lds[r1 * S + c1 + 1] = __high2half(n2);

            B_lds[r2 * S + c1]     = __low2half (frag1[1]);
            B_lds[r2 * S + c1 + 1] = __low2half (n3);
            B_lds[r3 * S + c1]     = __high2half(frag1[1]);
            B_lds[r3 * S + c1 + 1] = __high2half(n3);
        }

        // B_lds is warp-private, so ordering only has to hold within the wave:
        // drain LDS before the B-fragment load and after it (before the next
        // tile's stores). No block barrier inside the K loop.
        mem_fence();

        WmmaFragA a;
        int a_row = row_base + (lane % 16);
        if (a_row < size_m)
            a.data = *((const half16_t*)(A + a_row * size_k + tile_k * 16));
        else
            a.clear();

        WmmaFragB b;
        rdna_wmma::load_matrix_b(b, B_lds, 18);
        mem_fence();

        // All 32 lanes always execute WMMA.
        rdna_wmma::mma_sync(acc, a, b);
    }

    // Spill the accumulator to the block-wide reduction buffer in C-fragment
    // layout: row = lane % 16, col = 2*i + (lane >= 16).
    {
        int row = lane % 16;
        int col_base = (lane >= 16) ? 1 : 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            partial[warp][row * 16 + i * 2 + col_base] = acc[i];
    }

    __syncthreads();

    // warp0 reduces across WARPS in increasing order and stores. One lane per
    // (row, even/odd column set); rows past size_m stay idle so partial tail
    // tiles (m = 10/17/33/63) never write out of bounds.
    if (warp == 0)
    {
        int row = lane % 16;
        int grow = row_base + row;
        if (grow < size_m)
        {
            int col_base = (lane >= 16) ? 1 : 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                int col = i * 2 + col_base;
                float value = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) value += partial[w][row * 16 + col];
                int index = grow * size_n + tile_n * 16 + col;
                if constexpr (FP32) ((float*) C)[index] = value;
                else ((half*) C)[index] = __float2half(value);
            }
        }
    }
}

template <int BITS, int CB, bool FP32>
static void exl3_packed_mid_launch(const half* a, const uint16_t* b, void* c,
    int m, int k, int n, const half* suh, half* ah, const half* svh, cudaStream_t stream)
{
    hipLaunchKernelGGL((exl3_smallm_had<false, false>), dim3((k / 128 + 7) / 8, 1, m),
        dim3(256), 0, stream, a, ah, suh, k, nullptr);
    // This path only instantiates split-K WARPS 4/8/16. The clamp keeps any
    // explicit global/head override via the shared selector; heuristic 1
    // (split-K disabled or very wide n) falls back to the shape-aware
    // 4/8/16 rule instead of a missing single-warp kernel.
    int warps = exl3_smallm_warps(k, n, m);
    if (warps != 4 && warps != 8 && warps != 16)
        warps = exl3_gemv_splitk_warps(k / 16, n / 16, 1);
    int m_tiles = (m + 15) / 16;
    #define PACKED_MID_DOT(W) hipLaunchKernelGGL((exl3_packed_mid_wmma<BITS, CB, FP32, W>), \
        dim3(n / 16, m_tiles), dim3(W * 32), 0, stream, ah, b, c, m, k, n)
    switch (warps)
    {
        case 4: PACKED_MID_DOT(4); break;
        case 8: PACKED_MID_DOT(8); break;
        case 16: PACKED_MID_DOT(16); break;
    }
    #undef PACKED_MID_DOT
    hipLaunchKernelGGL((exl3_smallm_had<true, FP32>), dim3((n / 128 + 7) / 8, 1, m),
        dim3(256), 0, stream, c, c, svh, n, nullptr);
}

static bool exl3_packed_mid_try(const half* a, const uint16_t* b, void* c,
    int m, int k, int n, int bits, int cb, bool fp32,
    const half* suh, half* ah, const half* svh, cudaStream_t stream)
{
    const char* flag = std::getenv("EXL3_PACKED_MID");
    if (!flag || atoi(flag) != 1) return false;
    if (m < 10 || m > 64 || k <= 0 || n <= 0 || k % 128 || n % 128) return false;
    if (cb != 0 && cb != 2) return false;
    if (cb == 0 && (bits < 2 || bits > 4)) return false;
    if (cb == 2 && (bits < 2 || bits > 6)) return false;
    if (!suh || !ah || !svh) return false;
    #define PACKED_MID_TYPED(K, C) \
        if (fp32) exl3_packed_mid_launch<K, C, true>(a,b,c,m,k,n,suh,ah,svh,stream); \
        else exl3_packed_mid_launch<K, C, false>(a,b,c,m,k,n,suh,ah,svh,stream)
    #define PACKED_MID_CB(K) \
        switch (cb) { \
            case 0: PACKED_MID_TYPED(K, 0); break; \
            case 2: PACKED_MID_TYPED(K, 2); break; \
        }
    switch (bits)
    {
        case 2: PACKED_MID_CB(2); break;
        case 3: PACKED_MID_CB(3); break;
        case 4: PACKED_MID_CB(4); break;
        case 5: PACKED_MID_TYPED(5, 2); break;
        case 6: PACKED_MID_TYPED(6, 2); break;
    }
    #undef PACKED_MID_CB
    #undef PACKED_MID_TYPED
    return true;
}
