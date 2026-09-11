#pragma once
// Graph extension using the pinned GEMV six-site pointer republish contract.
// Derived from ExLlamaV3. MIT, Copyright (c) 2025 Turboderp.
// Include after exl3_gemv_graph_param_block is defined.

static __global__ __launch_bounds__(256)
void exl3_smallm_graph_in(const half* A, const uint16_t* B, void* C,
    const half* suh, half* A_had, const half* svh,
    Exl3GemvGraphParams* pb, int k)
{
    if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    {
        pb->B = B; pb->C = C; pb->A_had = A_had; pb->svh = svh;
    }
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp >= k / 128) return;
    int offset = warp * 128;
    int row_offset = blockIdx.z * k + offset;
    had_hf_r_128_inner<true, false>(A + row_offset, A_had + row_offset,
                                  suh + offset, 0.088388347648f);
}

template <int M, int BITS, bool FP32>
static void exl3_smallm_graph_typed(const Exl3GemvGraphParams* pb,
    int k, int n, cudaStream_t stream)
{
    int warps = exl3_gemv_splitk_enabled() && n / 16 <= EXL3_GEMV_SPLITK_MAX_TILES
        ? exl3_gemv_splitk_warps(k / 16, n / 16, 1) : 1;
    #define SMALLM_GRAPH_DOT(W) \
        hipLaunchKernelGGL((exl3_smallm_dot<M, BITS, FP32, W, true>), \
            dim3(n / 16), dim3(W * 32), 0, stream, nullptr, nullptr, nullptr, k, n, pb)
    switch (warps)
    {
        case 1: SMALLM_GRAPH_DOT(1); break;
        case 4: SMALLM_GRAPH_DOT(4); break;
        case 8: SMALLM_GRAPH_DOT(8); break;
        case 16: SMALLM_GRAPH_DOT(16); break;
    }
    #undef SMALLM_GRAPH_DOT
    hipLaunchKernelGGL((exl3_smallm_had<true, FP32, true>),
        dim3((n / 128 + 7) / 8, 1, M), dim3(256), 0, stream,
        nullptr, nullptr, nullptr, n, pb);
}

static bool exl3_smallm_graph_try(const half* a, const uint16_t* b, void* c,
    const half* suh, half* ah, const half* svh, int m, int k, int n,
    int bits, int cb, bool fp32, int device, cudaStream_t stream, Graph* graph)
{
    const char* smallm = std::getenv("EXL3_SMALLM");
    const char* enabled = std::getenv("EXL3_SMALLM_GRAPH");
    if (!smallm || atoi(smallm) != 1 || !enabled || atoi(enabled) != 1 ||
        (m != 2 && m != 3) || bits < 2 || bits > 4 || cb != 0 ||
        k % 128 || n % 128 || !suh || !ah || !svh || !graph ||
        exl3_gemv_env_mode() != 2 || !exl3_gemv_graph_enabled()) return false;
    // The ordinary eager pass prewarms this storage. Never allocate in capture.
    auto* pb = exl3_gemv_graph_param_block(device, false);
    if (!pb) return false;
    hipLaunchKernelGGL(exl3_smallm_graph_in, dim3((k / 128 + 7) / 8, 1, m),
        dim3(256), 0, stream, a, b, c, suh, ah, svh, pb, k);
    void* kernel = (void*) exl3_smallm_graph_in;
    graph->record_param(kernel, GP_gemm_A, 0);
    graph->record_param(kernel, GP_gemm_B_trellis, 1);
    graph->record_param(kernel, GP_gemm_C, 2);
    graph->record_param(kernel, GP_gemm_B_suh, 3);
    graph->record_param(kernel, GP_gemm_A_had, 4);
    graph->record_param(kernel, GP_gemm_B_svh, 5);
    graph->record_param(kernel, GP_end, 0);
    #define SMALLM_GRAPH_TYPED(M, K) \
        if (fp32) exl3_smallm_graph_typed<M, K, true>(pb,k,n,stream); \
        else exl3_smallm_graph_typed<M, K, false>(pb,k,n,stream)
    #define SMALLM_GRAPH_BITS(M) \
        switch (bits) { \
            case 2: SMALLM_GRAPH_TYPED(M, 2); break; \
            case 3: SMALLM_GRAPH_TYPED(M, 3); break; \
            case 4: SMALLM_GRAPH_TYPED(M, 4); break; \
        }
    if (m == 2) { SMALLM_GRAPH_BITS(2); }
    else { SMALLM_GRAPH_BITS(3); }
    #undef SMALLM_GRAPH_BITS
    #undef SMALLM_GRAPH_TYPED
    return true;
}
