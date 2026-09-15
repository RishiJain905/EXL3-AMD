#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>
#ifdef USE_ROCM
#include "rocm/hgemm_wmma.hip.h"
#include <cstdlib>
#include <cstring>
#endif

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float32 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;

#ifdef USE_ROCM
static bool hgemm_use_wmma()
{
    static const bool enabled = [] {
        const char* value = std::getenv("EXL3_HGEMM_IMPL");
        if (!value || !*value || std::strcmp(value, "blas") == 0) return false;
        TORCH_CHECK(std::strcmp(value, "wmma") == 0,
                    "EXL3_HGEMM_IMPL must be blas or wmma");
        return true;
    }();
    return enabled;
}

extern "C" __attribute__((visibility("default")))
int quantlab_exl3_hgemm_abi() { return 1; }
#endif

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream
)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "hgemm tensors must be on GPU");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(),
                "hgemm tensors must be on the same device");
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK(a.dim() >= 1, "a must have at least one dimension");
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");
    TORCH_CHECK(c.dim() == 2 || c.is_contiguous(), "batched c must be contiguous");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int64_t k64 = a.size(-1);
    TORCH_CHECK(k64 > 0, "hgemm K must be positive");
    int64_t m64 = a.numel() / k64;
    int64_t n64 = b.size(-1);
    TORCH_CHECK(k64 <= std::numeric_limits<int>::max() &&
                m64 <= std::numeric_limits<int>::max() &&
                n64 <= std::numeric_limits<int>::max(), "hgemm dimensions exceed native range");
    TORCH_CHECK(c.numel() >= m64 * n64, "hgemm output is too small");
    int size_k = (int) k64;
    int size_m = (int) m64;
    int size_n = (int) n64;
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    if (size_m == 0 || size_n == 0) return;
    at::assert_no_overlap(c, a);
    at::assert_no_overlap(c, b);
#ifdef USE_ROCM
    if (hgemm_use_wmma() && hgemm_wmma_try(a, b, c, stream)) return;
#endif

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_16F, size_n,
                 a_ptr, CUDA_R_16F, size_k,
        &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    hgemm_gemmex_impl(a, b, c, stream);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}
