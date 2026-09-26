#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>

// QTIP-style small-m GEMV path (see exl3_gemv_kernel.cuh). Launched from exl3_gemm when the
// heuristic applies; also exposed directly for testing. Same kernel arguments as
// exl3_gemm_kernel, so graph recording/patching is identical.

// Try to dispatch a GEMM call to the GEMV kernel. Returns false (launching nothing) if the
// call is not eligible. On success *launched_kernel receives the kernel pointer for graph
// recording. `force` bypasses the shape heuristic but not the hard constraints.
bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    bool has_su_sv,
    int device,
    cudaStream_t stream,
    void** launched_kernel,
    bool force
);

// Direct entry point (testing): errors if the call is not hard-eligible
void exl3_gemv
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
);

// Paired gate/up projection plus SwiGLU (ROCm-only, defined in
// rocm/quant/rdna-mlp-pair.hip.h). Returns None; writes input_scratch,
// projection_scratch and output. Throws on unsupported requests.
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
void exl3_mlp_gate_up
(
    const at::Tensor& x,
    const at::Tensor& gate_trellis,
    const at::Tensor& up_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& up_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_svh,
    at::Tensor& input_scratch,
    at::Tensor& projection_scratch,
    at::Tensor& output,
    int64_t bits,
    bool mul1
);
// Repacked-layout head projection prototype (ROCm-only, defined in
// rocm/quant/rdna-head-tiled.hip.h). Explicit entry; the only consumer of
// [N/128,K/16,8,16*BITS] bytes. Output Half or Float. Throws on unsupported
// requests.
void exl3_head_repacked
(
    const at::Tensor& x,
    const at::Tensor& packed,
    const at::Tensor& suh,
    at::Tensor& input_scratch,
    const at::Tensor& svh,
    at::Tensor& output,
    int64_t bits,
    int64_t codebook
);
#endif
