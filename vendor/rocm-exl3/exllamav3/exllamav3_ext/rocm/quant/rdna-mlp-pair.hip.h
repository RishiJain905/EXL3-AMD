#pragma once
#include <ATen/MemoryOverlap.h>
// Paired gate/up projection and SwiGLU: three launches replacing seven.
// Joint independent gate/up input Hadamards, joint packed gate/up dot
// projection, joint output Hadamards plus SwiGLU. Down projection stays
// separate. FP16 gate/up projection outputs (actual Qwen3.5/MiMo GatedMLP uses
// interm_dtype=torch.half); FP32 intermediates are NOT supported.
// M in (1,2,3,5), identical gate/up bit/codebook format, cb0 bits 2/3/4 or cb2
// bits 2..6, positive K,N multiples of 128. No biases, other activation,
// act_limit, or tensor parallelism.
// Numerics preserve the separate path exactly: FP16 rounding after dot
// (__float2half), after output Hadamard (half post-scale), and half2
// SiLU/multiply (activation_kernels.cuh). Split-K schedule and increasing-order
// reduction match exl3_smallm_dot; the M=1 accumulation is the GEMV direct
// core (exl3_gemv_dot_tile_direct), which is the same per-tile fdot2 sequence
// and quad reduction as small-M M=1, verified against the actual code rather
// than assumed. No persistent allocation; fixed supplied buffers capture/replay
// under external graphs. Native Graph pointer patching is outside this entry.
// Derived from ExLlamaV3 / the CarouselAether ROCm fork. MIT, Copyright (c)
// 2025 Turboderp. License retained in rdna-smallm-LICENSE.txt.
// Included after the small-M transforms, dot core and warp-selection helper.

extern "C" __attribute__((visibility("default")))
int quantlab_exl3_mlp_pair_abi() { return 1; }

// Joint input Hadamards: x (M,K) -> input_scratch (2,M,K) with distinct gate/up
// sign vectors. Grid z selects the row (y stays 0: the shared helper indexes
// scales as blockIdx.y*32+t, so rows must live on z).
template <int M>
static __global__ __launch_bounds__(256)
void exl3_mlp_pair_had_in(const half* __restrict__ x, half* __restrict__ out,
    const half* __restrict__ gate_suh, const half* __restrict__ up_suh, int size_k)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp >= size_k / 128) return;
    int offset = warp * 128;
    int row = blockIdx.z;
    const half* row_in = x + (size_t)row * size_k + offset;
    had_hf_r_128_inner<true, false>(row_in,
        out + (size_t)row * size_k + offset, gate_suh + offset, 0.088388347648f);
    had_hf_r_128_inner<true, false>(row_in,
        out + (size_t)(M + row) * size_k + offset, up_suh + offset, 0.088388347648f);
}

// Joint packed dot: input_scratch (2,M,K) -> projection_scratch (2,M,N) in FP16.
// Grid y selects the matrix (0 gate, 1 up); grid x selects the N tile. The
// per-matrix math is exl3_smallm_dot with FP32=false verbatim: same chunk
// schedule, same fdot2 order, same quad reduction and broadcast, same
// increasing-order split-K reduction through LDS, same __float2half store.
template <int M, int BITS, int CB, int WARPS>
static __global__ __launch_bounds__(WARPS * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS * 32, WARPS * 32)))
void exl3_mlp_pair_dot(const half* __restrict__ A,
    const uint16_t* __restrict__ gate_B, const uint16_t* __restrict__ up_B,
    half* __restrict__ C, int size_k, int size_n)
{
    const int mat = blockIdx.y;
    const half* mat_A = mat == 0 ? A : A + (size_t)M * size_k;
    const uint16_t* mat_B = mat == 0 ? gate_B : up_B;
    half* mat_C = mat == 0 ? C : C + (size_t)M * size_n;
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

    // Keep accumulation order while overlapping tiles for narrow high-bit rows.
    // Low-bit formats and wider rows retain the lower register footprint.
    #pragma clang loop unroll_count(M <= 3 && BITS >= 4 && BITS <= 6 ? 2 : 1)
    for (int tile_k = begin; tile_k < end; ++tile_k)
    {
        const uint32_t* packed = (const uint32_t*)
            (mat_B + (tile_k * n_tiles + tile_n) * (16 * BITS));
        FragB frag0, frag1;
        dq_dispatch<BITS, CB>(packed, lane << 3, frag0, frag1);
        #pragma unroll
        for (int row = 0; row < M; ++row)
        {
            const half2* a = (const half2*) (mat_A + row * size_k + tile_k * 16);
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
            mat_C[row * size_n + tile_n * 16 + lane] = __float2half(value);
        }
    }
}

// Joint output Hadamards plus SwiGLU: projection_scratch (2,M,N) -> output
// (M,N). Inlines the had_hf 4-element plus warp-shuffle math (no pre-scale)
// and the half2 SiLU from activation_kernels.cuh, with explicit scale indexing
// (grid y stays 0; scales are offset by warp*128 and indexed by lane t, exactly
// as the helper would). Register intermediates are bitwise identical to the
// separate global roundtrip (half store/load is exact). SiLU only, no limit.
template <int M>
static __global__ __launch_bounds__(256)
void exl3_mlp_pair_had_out(const half* __restrict__ proj, half* __restrict__ out,
    const half* __restrict__ gate_svh, const half* __restrict__ up_svh, int size_n)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp >= size_n / 128) return;
    int offset = warp * 128;
    int row = blockIdx.z;
    int t = threadIdx.x & 31;

    const half* g_in = proj + (size_t)row * size_n + offset;
    const half* u_in = proj + (size_t)(M + row) * size_n + offset;
    half* dst = out + (size_t)row * size_n + offset;

    half4 vg = ((const half4*) g_in)[t];
    half4 vu = ((const half4*) u_in)[t];

    auto had = [&](half4 v)
    {
        float v0 = __half2float(__low2half(v.x));
        float v1 = __half2float(__high2half(v.x));
        float v2 = __half2float(__low2half(v.y));
        float v3 = __half2float(__high2half(v.y));
        float s0 = v0 + v1;
        float d0 = v0 - v1;
        float s1 = v2 + v3;
        float d1 = v2 - v3;
        float h0 = s0 + s1;
        float h1 = d0 + d1;
        float h2 = s0 - s1;
        float h3 = d0 - d1;
        shuffle_had_f4x32(h0, h1, h2, h3, t);
        v.x = __floats2half2_rn(h0 * 0.088388347648f, h1 * 0.088388347648f);
        v.y = __floats2half2_rn(h2 * 0.088388347648f, h3 * 0.088388347648f);
        return v;
    };
    vg = had(vg);
    vu = had(vu);

    half4 sg = ((const half4*) (gate_svh + offset))[t];
    half4 su = ((const half4*) (up_svh + offset))[t];
    vg.x = __hmul2(vg.x, sg.x);
    vg.y = __hmul2(vg.y, sg.y);
    vu.x = __hmul2(vu.x, su.x);
    vu.y = __hmul2(vu.y, su.y);

    auto silu = [&](const half2& x)
    {
        half2 one = __float2half2_rn(1.0f);
        half2 neg_x = __hneg2(x);
        half2 e = h2exp(neg_x);
        half2 sum = __hadd2(one, e);
        half2 r = h2rcp(sum);
        return __hmul2(x, r);
    };
    vg.x = silu(vg.x);
    vg.y = silu(vg.y);
    vg.x = __hmul2(vg.x, vu.x);
    vg.y = __hmul2(vg.y, vu.y);

    ((half4*) dst)[t] = vg;
}

template <int M>
static void exl3_mlp_pair_launch_typed(const half* x,
    const uint16_t* gate_b, const uint16_t* up_b,
    const half* gate_suh, const half* up_suh,
    const half* gate_svh, const half* up_svh,
    half* in_scratch, half* proj_scratch, half* out,
    int k, int n, int bits, int cb, cudaStream_t stream)
{
    hipLaunchKernelGGL((exl3_mlp_pair_had_in<M>), dim3((k / 128 + 7) / 8, 1, M),
        dim3(256), 0, stream, x, in_scratch, gate_suh, up_suh, k);
    int warps = exl3_smallm_warps(k, n, M);
    #define MLP_PAIR_DOT(W, K, C) \
        hipLaunchKernelGGL((exl3_mlp_pair_dot<M, K, C, W>), dim3(n / 16, 2), \
            dim3(W * 32), 0, stream, in_scratch, gate_b, up_b, proj_scratch, k, n)
    #define MLP_PAIR_WARPS(K, C) \
        switch (warps) { \
            case 1: MLP_PAIR_DOT(1, K, C); break; \
            case 4: MLP_PAIR_DOT(4, K, C); break; \
            case 8: MLP_PAIR_DOT(8, K, C); break; \
            case 16: MLP_PAIR_DOT(16, K, C); break; \
        }
    #define MLP_PAIR_CB(K) \
        switch (cb) { \
            case 0: MLP_PAIR_WARPS(K, 0); break; \
            case 2: MLP_PAIR_WARPS(K, 2); break; \
        }
    switch (bits)
    {
        case 2: MLP_PAIR_CB(2); break;
        case 3: MLP_PAIR_CB(3); break;
        case 4: MLP_PAIR_CB(4); break;
        case 5: MLP_PAIR_WARPS(5, 2); break;
        case 6: MLP_PAIR_WARPS(6, 2); break;
    }
    #undef MLP_PAIR_CB
    #undef MLP_PAIR_WARPS
    #undef MLP_PAIR_DOT
    hipLaunchKernelGGL((exl3_mlp_pair_had_out<M>), dim3((n / 128 + 7) / 8, 1, M),
        dim3(256), 0, stream, proj_scratch, out, gate_svh, up_svh, n);
}

// Guarded host entry: three launches, fixed supplied buffers, no allocation.
// Throws (TORCH_CHECK) on any unsupported request; never falls back.
static void exl3_mlp_gate_up_check_overlap(const at::Tensor& w,
    const at::Tensor* others[], int n_others)
{
    for (int i = 0; i < n_others; ++i)
        at::assert_no_overlap(w, *others[i]);
}

void exl3_mlp_gate_up(const at::Tensor& x, const at::Tensor& gate_trellis,
    const at::Tensor& up_trellis, const at::Tensor& gate_suh, const at::Tensor& up_suh,
    const at::Tensor& gate_svh, const at::Tensor& up_svh, at::Tensor& input_scratch,
    at::Tensor& projection_scratch, at::Tensor& output, int64_t bits, bool mul1)
{
    TORCH_CHECK(x.defined() && gate_trellis.defined() && up_trellis.defined(), "missing input tensor");
    TORCH_CHECK(gate_suh.defined() && up_suh.defined() && gate_svh.defined() && up_svh.defined(),
        "missing scale tensor");
    TORCH_CHECK(input_scratch.defined() && projection_scratch.defined() && output.defined(),
        "missing scratch/output tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA");
    const at::Tensor* all[] = { &x, &gate_trellis, &up_trellis, &gate_suh, &up_suh,
        &gate_svh, &up_svh, &input_scratch, &projection_scratch, &output };
    for (int i = 1; i < 10; ++i)
        TORCH_CHECK(all[i]->device() == x.device(), "all tensors must be on the same GPU");
    for (int i = 0; i < 10; ++i)
        TORCH_CHECK(all[i]->is_contiguous(), "all tensors must be contiguous");
    // Contiguous views can still start at an unaligned storage offset.
    for (int i = 0; i < 10; ++i)
        TORCH_CHECK(reinterpret_cast<uintptr_t>(all[i]->data_ptr()) % 8 == 0,
                    "all tensors must have 8-byte aligned data");
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(gate_trellis, kShort);
    TORCH_CHECK_DTYPE(up_trellis, kShort);
    TORCH_CHECK_DTYPE(gate_suh, kHalf);
    TORCH_CHECK_DTYPE(up_suh, kHalf);
    TORCH_CHECK_DTYPE(gate_svh, kHalf);
    TORCH_CHECK_DTYPE(up_svh, kHalf);
    TORCH_CHECK_DTYPE(input_scratch, kHalf);
    TORCH_CHECK_DTYPE(projection_scratch, kHalf);
    TORCH_CHECK_DTYPE(output, kHalf);

    TORCH_CHECK_DIM(x, 2);
    TORCH_CHECK_DIM(gate_trellis, 3);
    TORCH_CHECK_DIM(up_trellis, 3);
    TORCH_CHECK_DIM(gate_suh, 1);
    TORCH_CHECK_DIM(up_suh, 1);
    TORCH_CHECK_DIM(gate_svh, 1);
    TORCH_CHECK_DIM(up_svh, 1);
    TORCH_CHECK_DIM(input_scratch, 3);
    TORCH_CHECK_DIM(projection_scratch, 3);
    TORCH_CHECK_DIM(output, 2);

    int64_t m = x.size(0);
    int64_t k = x.size(1);
    TORCH_CHECK(m == 1 || m == 2 || m == 3 || m == 5, "M must be one of (1,2,3,5)");
    TORCH_CHECK(k > 0 && k % 128 == 0, "K must be a positive multiple of 128");
    TORCH_CHECK(output.size(0) == m, "output rows must match x");
    int64_t n = output.size(1);
    TORCH_CHECK(n > 0 && n % 128 == 0, "N must be a positive multiple of 128");
    int cb = mul1 ? 2 : 0;
    if (cb == 0)
    {
        TORCH_CHECK(bits >= 2 && bits <= 4, "cb0 supports bits 2/3/4");
    }
    else
    {
        TORCH_CHECK(bits >= 2 && bits <= 6, "cb2 supports bits 2..6");
    }
    TORCH_CHECK(gate_trellis.size(0) == k / 16 && gate_trellis.size(1) == n / 16 &&
        gate_trellis.size(2) == 16 * bits, "gate_trellis must be (K/16,N/16,16*bits)");
    TORCH_CHECK(up_trellis.size(0) == k / 16 && up_trellis.size(1) == n / 16 &&
        up_trellis.size(2) == 16 * bits, "up_trellis must be (K/16,N/16,16*bits)");
    TORCH_CHECK(gate_suh.size(0) == k && up_suh.size(0) == k, "input scales must have K elements");
    TORCH_CHECK(gate_svh.size(0) == n && up_svh.size(0) == n, "output scales must have N elements");
    TORCH_CHECK(input_scratch.size(0) == 2 && input_scratch.size(1) == m && input_scratch.size(2) == k,
        "input_scratch must be (2,M,K)");
    TORCH_CHECK(projection_scratch.size(0) == 2 && projection_scratch.size(1) == m &&
        projection_scratch.size(2) == n, "projection_scratch must be (2,M,N)");
    TORCH_CHECK(m <= 5 && k <= 65536 && n <= 65536, "M/K/N out of supported integer range");

    // Written buffers must not overlap any other tensor's storage.
    const at::Tensor* others_in[] = { &x, &gate_trellis, &up_trellis, &gate_suh, &up_suh,
        &gate_svh, &up_svh, &projection_scratch, &output };
    const at::Tensor* others_proj[] = { &x, &gate_trellis, &up_trellis, &gate_suh, &up_suh,
        &gate_svh, &up_svh, &input_scratch, &output };
    const at::Tensor* others_out[] = { &x, &gate_trellis, &up_trellis, &gate_suh, &up_suh,
        &gate_svh, &up_svh, &input_scratch, &projection_scratch };
    exl3_mlp_gate_up_check_overlap(input_scratch, others_in, 9);
    exl3_mlp_gate_up_check_overlap(projection_scratch, others_proj, 9);
    exl3_mlp_gate_up_check_overlap(output, others_out, 9);

    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const half* x_ptr = (const half*) x.data_ptr();
    const uint16_t* gate_b = (const uint16_t*) gate_trellis.data_ptr();
    const uint16_t* up_b = (const uint16_t*) up_trellis.data_ptr();
    const half* gate_suh_ptr = (const half*) gate_suh.data_ptr();
    const half* up_suh_ptr = (const half*) up_suh.data_ptr();
    const half* gate_svh_ptr = (const half*) gate_svh.data_ptr();
    const half* up_svh_ptr = (const half*) up_svh.data_ptr();
    half* in_ptr = (half*) input_scratch.data_ptr();
    half* proj_ptr = (half*) projection_scratch.data_ptr();
    half* out_ptr = (half*) output.data_ptr();
    int ki = (int) k;
    int ni = (int) n;
    int bi = (int) bits;

    switch (m)
    {
        case 1: exl3_mlp_pair_launch_typed<1>(x_ptr, gate_b, up_b, gate_suh_ptr, up_suh_ptr,
            gate_svh_ptr, up_svh_ptr, in_ptr, proj_ptr, out_ptr, ki, ni, bi, cb, stream); break;
        case 2: exl3_mlp_pair_launch_typed<2>(x_ptr, gate_b, up_b, gate_suh_ptr, up_suh_ptr,
            gate_svh_ptr, up_svh_ptr, in_ptr, proj_ptr, out_ptr, ki, ni, bi, cb, stream); break;
        case 3: exl3_mlp_pair_launch_typed<3>(x_ptr, gate_b, up_b, gate_suh_ptr, up_suh_ptr,
            gate_svh_ptr, up_svh_ptr, in_ptr, proj_ptr, out_ptr, ki, ni, bi, cb, stream); break;
        case 5: exl3_mlp_pair_launch_typed<5>(x_ptr, gate_b, up_b, gate_suh_ptr, up_suh_ptr,
            gate_svh_ptr, up_svh_ptr, in_ptr, proj_ptr, out_ptr, ki, ni, bi, cb, stream); break;
    }
    cuda_check(cudaPeekAtLastError());
}
