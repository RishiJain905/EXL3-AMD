// Narrow dense GEMM for decode-sized inputs: C = A @ B with FP32 accumulation.
//
// Targets, both measured on gfx1101 in docs/DECODE-PROFILING.md:
// - FP16 GatedDeltaNet in_proj_a/in_proj_b (hidden -> value heads, e.g. 4096 -> 32,
//   FP32 output) behind extension.hgemm. BLAS runs these at ~80-90 us per call
//   because one tiny tile walks the whole K range.
// - BF16 MTP draft projections (narrow_gemm_bf16), where the BLAS GEMV path
//   reaches well under the streaming rate at one or two rows.
//
// Row-major A (M x K) and B (K x N), 1-8 rows (1-4 for BF16), N a multiple of 8. A block covers
// VECS 8-column vectors (the last block may be partial) with THREADS threads: thread t
// owns vector t % VECS and the K slice k = t / VECS, stepping by THREADS / VECS. Narrow
// outputs use 1024 threads to spread K; wide ones use the largest block size whose
// grid still fits the device in one pass, avoiding a nearly empty tail wave.
// Partial sums are reduced in a fixed order (lane xor-shuffles, then warps in
// order through LDS), so results are deterministic; the order differs from BLAS.
// No atomics, workspace or persistent state; safe under stream capture.

#include "narrow_gemm.hip.h"

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "util.cuh"

extern "C" __attribute__((visibility("default")))
int quantlab_exl3_narrow_gemm_abi() { return 1; }

namespace {

enum class Out { F32, F16, BF16 };

template <bool BF16>
__device__ __forceinline__ float to_float(uint16_t v)
{
    if constexpr (BF16) return __uint_as_float((uint32_t) v << 16);
    else return __half2float(__ushort_as_half(v));
}

__device__ __forceinline__ uint16_t to_bf16(float v)
{
    // Round to nearest even, as the BLAS BF16 epilogue does; NaN stays NaN.
    uint32_t u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

template <int M, int VECS, int THREADS, bool BF16, Out OUT>
__global__ __launch_bounds__(THREADS)
void narrow_gemm_kernel(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,
                        int size_k, int size_n, int ldc)
{
    constexpr int kWarps = THREADS / 32;
    constexpr int SLICES = THREADS / VECS;
    constexpr int COLS = VECS * 8;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int vec = t % VECS;
    const int col = blockIdx.x * COLS + vec * 8;
    const bool active = col < size_n;  // size_n % 8 == 0: vectors are whole or absent

    float acc[M][8];
    #pragma unroll
    for (int m = 0; m < M; ++m)
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[m][j] = 0.0f;

    #pragma unroll 4
    for (int k = active ? t / VECS : size_k; k < size_k; k += SLICES)
    {
        uint4 raw = *reinterpret_cast<const uint4*>(B + (size_t) k * size_n + col);
        const uint16_t* w16 = reinterpret_cast<const uint16_t*>(&raw);
        float w[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = to_float<BF16>(w16[j]);
        #pragma unroll
        for (int m = 0; m < M; ++m)
        {
            float a = to_float<BF16>(A[(size_t) m * size_k + k]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc[m][j] = fmaf(a, w[j], acc[m][j]);
        }
    }

    // Lanes sharing a column vector differ in the lane bits above log2(VECS).
    #pragma unroll
    for (int m = 0; m < M; ++m)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            float v = acc[m][j];
            #pragma unroll
            for (int s = VECS; s < 32; s <<= 1) v += __shfl_xor(v, s, 32);
            acc[m][j] = v;
        }

    __shared__ float partial[kWarps][M][COLS];
    if (lane < VECS)
        #pragma unroll
        for (int m = 0; m < M; ++m)
            #pragma unroll
            for (int j = 0; j < 8; ++j) partial[warp][m][lane * 8 + j] = acc[m][j];
    __syncthreads();

    for (int i = t; i < M * COLS; i += THREADS)
    {
        const int m = i / COLS;
        const int c = i % COLS;
        if (blockIdx.x * COLS + c >= size_n) continue;
        float value = 0.0f;
        #pragma unroll 8
        for (int w = 0; w < kWarps; ++w) value += partial[w][m][c];
        const size_t index = (size_t) m * ldc + blockIdx.x * COLS + c;
        if constexpr (OUT == Out::F32) static_cast<float*>(C)[index] = value;
        else if constexpr (OUT == Out::F16) static_cast<half*>(C)[index] = __float2half(value);
        else static_cast<uint16_t*>(C)[index] = to_bf16(value);
    }
}

// Instantiates rows 1..MAX_M; the LDS partials grow with M * VECS * THREADS / 32.
template <int VECS, int THREADS, int MAX_M, bool BF16, Out OUT>
void launch_threads(int m, const uint16_t* a, const uint16_t* b, void* c, int k, int n, int ldc, cudaStream_t stream)
{
    constexpr int COLS = VECS * 8;
    dim3 grid((n + COLS - 1) / COLS);
    switch (m)
    {
        #define NARROW_CASE(M) case M: if constexpr (M <= MAX_M)             narrow_gemm_kernel<M, VECS, THREADS, BF16, OUT><<<grid, THREADS, 0, stream>>>(a, b, c, k, n, ldc); break;
        NARROW_CASE(1) NARROW_CASE(2) NARROW_CASE(3) NARROW_CASE(4)
        NARROW_CASE(5) NARROW_CASE(6) NARROW_CASE(7) NARROW_CASE(8)
        #undef NARROW_CASE
    }
}

template <int VECS, int MAX_M, bool BF16, Out OUT>
void launch(int m, const uint16_t* a, const uint16_t* b, void* c, int k, int n, int ldc, cudaStream_t stream)
{
    // One pass of resident blocks holds compute units x 1024 threads.
    const int blocks = (n + VECS * 8 - 1) / (VECS * 8);
    const int units = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    if (blocks * 1024 <= units * 1024) launch_threads<VECS, 1024, MAX_M, BF16, OUT>(m, a, b, c, k, n, ldc, stream);
    else if (blocks * 512 <= units * 1024) launch_threads<VECS, 512, MAX_M, BF16, OUT>(m, a, b, c, k, n, ldc, stream);
    else launch_threads<VECS, 256, MAX_M, BF16, OUT>(m, a, b, c, k, n, ldc, stream);
}

bool enabled()
{
    // EXL3_NARROW_GEMM=0 (the --no-narrow-gemm diagnostic) or unset declines.
    const char* flag = std::getenv("EXL3_NARROW_GEMM");
    return flag && std::atoi(flag) == 1;
}

// Shared shape/layout guard. Returns M or 0 when the request is unsupported.
int64_t eligible_rows(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c,
                      at::ScalarType input, int64_t max_m, int64_t max_n, int64_t& ldc64)
{
    if (!a.defined() || !b.defined() || !c.defined()) return 0;
    if (!a.is_cuda() || a.device() != b.device() || a.device() != c.device()) return 0;
    if (a.scalar_type() != input || b.scalar_type() != input) return 0;
    if (b.dim() != 2 || c.dim() < 1 || !a.is_contiguous() || !b.is_contiguous()) return 0;
    const int64_t K64 = a.size(-1);
    const int64_t N64 = b.size(1);
    if (K64 <= 0 || b.size(0) != K64 || a.numel() % K64) return 0;
    const int64_t M64 = a.numel() / K64;
    if (M64 < 1 || M64 > max_m || N64 < 8 || N64 > max_n || N64 % 8) return 0;
    if (K64 > std::numeric_limits<int>::max() / 8) return 0;
    ldc64 = N64;
    if (c.dim() >= 2)
    {
        if (c.stride(-1) != 1) return 0;
        ldc64 = c.stride(-2);
        if (ldc64 < N64 || (!c.is_contiguous() && (c.dim() != 2 || c.size(0) != M64))) return 0;
    }
    else if (!c.is_contiguous())
        return 0;
    if (c.numel() < (M64 - 1) * ldc64 + N64 || ldc64 > std::numeric_limits<int>::max()) return 0;
    // 16-byte B vectors: base and row pitch (N % 8 == 0) keep every load aligned.
    if (reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 != 0) return 0;
    return M64;
}

}  // namespace

bool narrow_gemm_try(at::Tensor a, at::Tensor b, at::Tensor c, cudaStream_t stream)
{
    if (!enabled()) return false;
    if (c.dtype() != at::kFloat && c.dtype() != at::kHalf) return false;
    int64_t ldc64;
    // Measured FP16 envelope: the GatedDeltaNet gate projections; wider outputs keep BLAS.
    const int64_t m = eligible_rows(a, b, c, at::kHalf, 8, 128, ldc64);
    if (!m) return false;
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    const auto* a_ptr = static_cast<const uint16_t*>(a.data_ptr());
    const auto* b_ptr = static_cast<const uint16_t*>(b.data_ptr());
    const int k = (int) a.size(-1), n = (int) b.size(1), ldc = (int) ldc64;
    if (c.dtype() == at::kFloat) launch<4, 8, false, Out::F32>((int) m, a_ptr, b_ptr, c.data_ptr(), k, n, ldc, stream);
    else launch<4, 8, false, Out::F16>((int) m, a_ptr, b_ptr, c.data_ptr(), k, n, ldc, stream);
    cuda_check(cudaPeekAtLastError());
    return true;
}

void narrow_gemm_bf16(const at::Tensor& a, const at::Tensor& b, at::Tensor& c)
{
    TORCH_CHECK(enabled(), "narrow_gemm_bf16 requires EXL3_NARROW_GEMM=1");
    TORCH_CHECK(c.scalar_type() == at::kBFloat16, "narrow_gemm_bf16 writes a BF16 output");
    int64_t ldc64;
    const int64_t m = eligible_rows(a, b, c, at::kBFloat16, 4, 65536, ldc64);
    TORCH_CHECK(m, "narrow_gemm_bf16: unsupported shape, layout or dtype");
    at::assert_no_overlap(c, a);
    at::assert_no_overlap(c, b);
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch<8, 4, true, Out::BF16>((int) m, static_cast<const uint16_t*>(a.data_ptr()),
        static_cast<const uint16_t*>(b.data_ptr()), c.data_ptr(), (int) a.size(-1), (int) b.size(1), (int) ldc64, stream);
    cuda_check(cudaPeekAtLastError());
}
