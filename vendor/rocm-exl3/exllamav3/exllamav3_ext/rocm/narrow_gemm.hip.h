#pragma once
//
// Narrow dense GEMM for decode-sized inputs (ROCm only).
//
// Declarations only; the kernels live in the ROCm-only translation unit
// narrow_gemm.hip. Both entries accumulate in FP32 with a fixed (deterministic)
// order that differs from BLAS, and require EXL3_NARROW_GEMM=1, which the Python
// capability probe sets for binaries exporting quantlab_exl3_narrow_gemm_abi.
//

#include <ATen/Tensor.h>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

// FP16 A (M x K) @ B (K x N) -> FP32 or FP16 C for 1-8 rows and 8-128 output
// columns (multiples of 8): Qwen3.5/MiMo GatedDeltaNet in_proj_a/in_proj_b at
// decode and MTP verification. hgemm.cu calls it ahead of BLAS. Returns false
// without launching when unsupported; true after an async launch on `stream`.
bool narrow_gemm_try(at::Tensor a, at::Tensor b, at::Tensor c, cudaStream_t stream);

// BF16 A (M x K) @ B (K x N) -> BF16 C (round to nearest even) for 1-4 rows and
// N a multiple of 8 up to 65536: the BF16 MTP draft projections. Explicit entry;
// throws on unsupported requests instead of falling back.
void narrow_gemm_bf16(const at::Tensor& a, const at::Tensor& b, at::Tensor& c);
