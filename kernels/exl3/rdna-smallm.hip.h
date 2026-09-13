#pragma once
// Experimental noncooperative m=2/3 extension of the pinned RDNA GEMV core.
// Derived from ExLlamaV3 / its ROCm fork. MIT, Copyright (c) 2025 Turboderp.
// The upstream license is retained in rdna-smallm-LICENSE.txt.
// Included after the original GEMV helpers and host warp-selection functions.

template <bool OUTPUT, bool FP32, bool GRAPH = false>
static __global__ __launch_bounds__(256)
void exl3_smallm_had(const void* input, void* output, const half* scales, int size,
                    const Exl3GemvGraphParams* pb)
{
    if constexpr (GRAPH) { input = pb->C; output = pb->C; scales = pb->svh; }
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp >= size / 128) return;
    int offset = warp * 128;
    // The shared Hadamard helper itself uses blockIdx.y in its scale index.
    // Keep y=0 and put independent input rows on z instead.
    int row_offset = blockIdx.z * size + offset;
    if constexpr (FP32)
        had_ff_r_128_inner<false, true>((const float*) input + row_offset,
            (float*) output + row_offset, scales + offset, 0.088388347648f);
    else
        had_hf_r_128_inner<!OUTPUT, OUTPUT>((const half*) input + row_offset,
            (half*) output + row_offset, scales + offset, 0.088388347648f);
}

template <int M, int BITS, bool FP32, int WARPS, bool GRAPH = false>
static __global__ __launch_bounds__(WARPS * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS * 32, WARPS * 32)))
void exl3_smallm_dot(const half* __restrict__ A, const uint16_t* __restrict__ B,
                    void* __restrict__ C, int size_k, int size_n,
                    const Exl3GemvGraphParams* pb)
{
    if constexpr (GRAPH) { A = pb->A_had; B = pb->B; C = pb->C; }
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int tile_n = blockIdx.x;
    int n_tiles = size_n / 16;
    int k_tiles = size_k / 16;
    int chunk = (k_tiles + WARPS - 1) / WARPS;
    int begin = warp * chunk;
    int end = begin + chunk < k_tiles ? begin + chunk : k_tiles;
    int r0 = (lane & 3) * 2;
    float acc_a[M] = {};
    float acc_b[M] = {};

    for (int tile_k = begin; tile_k < end; ++tile_k)
    {
        const uint32_t* packed = (const uint32_t*)
            (B + (tile_k * n_tiles + tile_n) * (16 * BITS));
        FragB frag0, frag1;
        dq_dispatch<BITS, 0>(packed, lane << 3, frag0, frag1);
        // Decode once, then use the same fragments for every input row.
        #pragma unroll
        for (int row = 0; row < M; ++row)
        {
            const half2* a = (const half2*) (A + row * size_k + tile_k * 16);
            half2 a01 = a[r0 >> 1];
            half2 a89 = a[(r0 >> 1) + 4];
            acc_a[row] = __builtin_amdgcn_fdot2(a01, frag0[0], acc_a[row], false);
            acc_a[row] = __builtin_amdgcn_fdot2(a89, frag0[1], acc_a[row], false);
            acc_b[row] = __builtin_amdgcn_fdot2(a01, frag1[0], acc_b[row], false);
            acc_b[row] = __builtin_amdgcn_fdot2(a89, frag1[1], acc_b[row], false);
        }
    }

    __shared__ float partial[M][WARPS][16];
    #pragma unroll
    for (int row = 0; row < M; ++row)
    {
        float a = acc_a[row], b = acc_b[row];
        a += __shfl_xor(a, 1, 32);
        a += __shfl_xor(a, 2, 32);
        b += __shfl_xor(b, 1, 32);
        b += __shfl_xor(b, 2, 32);
        float va = __shfl(a, (lane & 7) * 4, 32);
        float vb = __shfl(b, (lane & 7) * 4, 32);
        if (lane < 16) partial[row][warp][lane] = (lane & 8) ? vb : va;
    }
    __syncthreads();
    if (warp == 0 && lane < 16)
    {
        #pragma unroll
        for (int row = 0; row < M; ++row)
        {
            float value = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) value += partial[row][w][lane];
            int index = row * size_n + tile_n * 16 + lane;
            if constexpr (FP32) ((float*) C)[index] = value;
            else ((half*) C)[index] = __float2half(value);
        }
    }
}

template <int M, int BITS, bool FP32>
static void exl3_smallm_launch_typed(const half* a, const uint16_t* b, void* c,
    int k, int n, const half* suh, half* ah, const half* svh, cudaStream_t stream)
{
    hipLaunchKernelGGL((exl3_smallm_had<false, false>), dim3((k / 128 + 7) / 8, 1, M),
        dim3(256), 0, stream, a, ah, suh, k, nullptr);
    int warps = exl3_gemv_splitk_enabled() && n / 16 <= EXL3_GEMV_SPLITK_MAX_TILES
        ? exl3_gemv_splitk_warps(k / 16, n / 16, 1) : 1;
    #define SMALLM_DOT(W) hipLaunchKernelGGL((exl3_smallm_dot<M, BITS, FP32, W>), \
        dim3(n / 16), dim3(W * 32), 0, stream, ah, b, c, k, n, nullptr)
    switch (warps)
    {
        case 1: SMALLM_DOT(1); break;
        case 4: SMALLM_DOT(4); break;
        case 8: SMALLM_DOT(8); break;
        case 16: SMALLM_DOT(16); break;
    }
    #undef SMALLM_DOT
    hipLaunchKernelGGL((exl3_smallm_had<true, FP32>), dim3((n / 128 + 7) / 8, 1, M),
        dim3(256), 0, stream, c, c, svh, n, nullptr);
}

static bool exl3_smallm_try_launch(const half* a, const uint16_t* b, void* c,
    int m, int k, int n, int bits, int cb, bool fp32,
    const half* suh, half* ah, const half* svh, cudaStream_t stream)
{
    const char* flag = std::getenv("EXL3_SMALLM");
    if (!flag || atoi(flag) != 1 || (m != 2 && m != 3) || cb != 0 ||
        bits < 2 || bits > 4 || k % 128 || n % 128 || !suh || !ah || !svh)
        return false;
    #define SMALLM_TYPED(M, K) \
        if (fp32) exl3_smallm_launch_typed<M, K, true>(a,b,c,k,n,suh,ah,svh,stream); \
        else exl3_smallm_launch_typed<M, K, false>(a,b,c,k,n,suh,ah,svh,stream)
    #define SMALLM_BITS(M) \
        switch (bits) { \
            case 2: SMALLM_TYPED(M, 2); break; \
            case 3: SMALLM_TYPED(M, 3); break; \
            case 4: SMALLM_TYPED(M, 4); break; \
        }
    if (m == 2) { SMALLM_BITS(2); }
    else { SMALLM_BITS(3); }
    #undef SMALLM_BITS
    #undef SMALLM_TYPED
    return true;
}
