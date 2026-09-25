#pragma once
// Bounded mul1 K5/K6 verification extension of the existing small-M dot kernel.
// Derived from ExLlamaV3 / the CarouselAether ROCm fork. MIT, Copyright (c) 2025
// Turboderp. License retained in rdna-smallm-LICENSE.txt.
// Included after the small-M transforms, dot core and warp-selection helper.

extern "C" __attribute__((visibility("default")))
int quantlab_exl3_smallm_highbit_abi() { return 1; }

template <int M, int BITS, bool FP32>
static void exl3_smallm_highbit_launch(const half* a, const uint16_t* b, void* c,
    int k, int n, const half* suh, half* ah, const half* svh, cudaStream_t stream)
{
    hipLaunchKernelGGL((exl3_smallm_had<false, false>), dim3((k / 128 + 7) / 8, 1, M),
        dim3(256), 0, stream, a, ah, suh, k, nullptr);
    #define HIGHBIT_DOT(W) hipLaunchKernelGGL((exl3_smallm_dot<M, BITS, 2, FP32, W>), \
        dim3(n / 16), dim3(W * 32), 0, stream, ah, b, c, k, n, nullptr)
    switch (exl3_smallm_warps(k, n))
    {
        case 1: HIGHBIT_DOT(1); break;
        case 4: HIGHBIT_DOT(4); break;
        case 8: HIGHBIT_DOT(8); break;
        case 16: HIGHBIT_DOT(16); break;
    }
    #undef HIGHBIT_DOT
    hipLaunchKernelGGL((exl3_smallm_had<true, FP32>), dim3((n / 128 + 7) / 8, 1, M),
        dim3(256), 0, stream, c, c, svh, n, nullptr);
}

static bool exl3_smallm_highbit_try(const half* a, const uint16_t* b, void* c,
    int m, int k, int n, int bits, int cb, bool fp32,
    const half* suh, half* ah, const half* svh, cudaStream_t stream)
{
    const char* flag = std::getenv("EXL3_SMALLM");
    if (!flag || atoi(flag) != 1 || cb != 2 || (bits != 5 && bits != 6) ||
        (m != 2 && m != 3 && m != 5) || k <= 0 || n <= 0 || k % 128 || n % 128 ||
        !suh || !ah || !svh || exl3_smallm_use_wmma() || exl3_smallm_use_register_b())
        return false;
    #define HIGHBIT_TYPED(M, K) \
        if (fp32) exl3_smallm_highbit_launch<M, K, true>(a,b,c,k,n,suh,ah,svh,stream); \
        else exl3_smallm_highbit_launch<M, K, false>(a,b,c,k,n,suh,ah,svh,stream)
    #define HIGHBIT_BITS(M) \
        if (bits == 5) { HIGHBIT_TYPED(M, 5); } else { HIGHBIT_TYPED(M, 6); }
    switch (m)
    {
        case 2: HIGHBIT_BITS(2); break;
        case 3: HIGHBIT_BITS(3); break;
        case 5: HIGHBIT_BITS(5); break;
    }
    #undef HIGHBIT_BITS
    #undef HIGHBIT_TYPED
    return true;
}
