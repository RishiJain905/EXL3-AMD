#pragma once
//
// Optional FP32-output GEMM for gfx1101, using RDNA WMMA.
//
// Declaration only. The definition lives in the ROCm-only translation unit
// hgemm_wmma.hip, which uses RDNA WMMA builtins and a gcnArchName device gate.
// The hgemm.cu dispatch guards the call with USE_ROCM so CUDA
// builds never reference the symbol.
//

#include <ATen/Tensor.h>

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime_api.h>
#endif

// Computes C = A @ B (row-major, fp16 A/B, fp32 accumulate AND output) with a
// native non-cooperative WMMA kernel when shape/layout/arch are supported.
//
// A is flattened to M x K, B is K x N; C holds M x N with row stride >= N.
// Returns false when unsupported: no launch is issued and C is untouched.
// Returns true after launching on `stream` (async; usual stream ordering,
// no sync, no graph capture, no persistent state).
bool hgemm_wmma_try(at::Tensor a, at::Tensor b, at::Tensor c, cudaStream_t stream);
