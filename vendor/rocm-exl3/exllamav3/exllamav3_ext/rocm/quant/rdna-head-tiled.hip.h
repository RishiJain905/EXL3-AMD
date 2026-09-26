#pragma once
// Grouped wide-output packed head projection prototype (experimental).
// Two launches: existing input Hadamard, then a paired dot + output-Hadamard
// kernel where each 256-thread block owns one complete 128-column group.
// Eight wave32 each compute an independent 16-column tile with the whole-K
// loop and fdot2 order of the single-warp wide-output head, decoding each
// weight fragment once per K tile and reusing it across M rows. Dot outputs
// are staged in LDS (M x 128 floats, or M x 128 FP16-rounded halves on the
// repacked FP16 path), synchronized once, then one warp per row runs the
// output Hadamard via the existing had_ff/had_hf_r_128_inner helper matching
// the output dtype. Raw layout is FP32-only, gated by EXL3_HEAD_TILED=1;
// absent stays OFF.
// The REPACKED=true specialization reads a transposed trellis tile order,
// [N/128,K/16,8,16*BITS], via the explicit exl3_head_repacked entry only;
// ordinary dispatch never sees repacked bytes (no environment flag). The
// repacked entry accepts FP32 or FP16 outputs; FP16 rounds each dot output
// to half before the output Hadamard, exactly like the native dot kernels.
// Derived from ExLlamaV3 / the CarouselAether ROCm fork. MIT, Copyright (c)
// 2025 Turboderp. License retained in rdna-smallm-LICENSE.txt.
// Included after rdna-smallm.hip.h and rdna-mlp-pair.hip.h.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <type_traits>
#include <ATen/MemoryOverlap.h>

extern "C" __attribute__((visibility("default")))
int quantlab_exl3_head_tiled_abi() { return 1; }

static std::atomic<uint64_t> exl3_head_tiled_call_count{0};
extern "C" __attribute__((visibility("default")))
uint64_t quantlab_exl3_head_tiled_calls() { return exl3_head_tiled_call_count.load(); }

extern "C" __attribute__((visibility("default")))
int quantlab_exl3_head_repacked_abi() { return 3; }

static std::atomic<uint64_t> exl3_head_repacked_call_count{0};
extern "C" __attribute__((visibility("default")))
uint64_t quantlab_exl3_head_repacked_calls() { return exl3_head_repacked_call_count.load(); }

static bool exl3_head_tiled_device(int device)
{
    if (device < 0 || device >= 64) return false;
    static std::once_flag once[64];
    static bool supported[64] = {};
    std::call_once(once[device], [device]() {
        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, device) == hipSuccess)
            supported[device] = std::strncmp(props.gcnArchName, "gfx1101", 7) == 0
                && (props.gcnArchName[7] == '\0' || props.gcnArchName[7] == ':');
    });
    return supported[device];
}

// Paired dot + output Hadamard. Grid: (N/128) blocks of 256 threads.
// A_had is (M,K) rotated input, B is (K/16,N/16,16*BITS) trellis
// ([N/128,K/16,8,16*BITS] when REPACKED), C is (M,N) output (float when
// FP32, half otherwise), svh is (N) output scales. The FP16 path rounds
// each dot output via __float2half before staging, exactly like the native
// dot kernels, then runs the existing had_hf_r_128_inner helper.
template <int M, int BITS, int CB, bool REPACKED = false, bool FP32 = true>
static __global__ __launch_bounds__(256)
__attribute__((amdgpu_flat_work_group_size(256, 256)))
void exl3_head_tiled_dot_had(
    const half* __restrict__ A_had,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    int size_k,
    int size_n,
    const half* __restrict__ svh)
{
    // One 128-element LDS row per input row. FP32 rows are 32 float4
    // (16-byte row alignment for the helper's float4 loads); FP16 rows are
    // 128 halves with explicit 16-byte alignment for the helper's half4
    // loads (rows are 512/256 bytes, so every row stays aligned).
    using TileElem = std::conditional_t<FP32, float4, half>;
    static constexpr int TILE_COLS = FP32 ? 32 : 128;
    __shared__ alignas(16) TileElem tile[M][TILE_COLS];

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int group = blockIdx.x;
    int n_tiles = size_n / 16;
    int k_tiles = size_k / 16;
    int tile_n = group * 8 + warp;
    int r0 = (lane & 3) * 2;

    float acc_a[M] = {};
    float acc_b[M] = {};

    // Repacked M<=3 benefits from overlapping tile loads. Larger row batches
    // regressed with that unroll factor; ordinary layout retains the short loop.
    // The fdot2 accumulator dependency order is unchanged.
    #pragma clang loop unroll_count(REPACKED && M <= 3 ? 4 : 1)
    for (int tile_k = 0; tile_k < k_tiles; ++tile_k)
    {
        size_t b_idx;
        if constexpr (REPACKED)
        {
            // Transposed tile order: B is [N/128,K/16,8,16*BITS].
            (void) n_tiles;
            (void) tile_n;
            b_idx = (((size_t) group * (size_t) k_tiles + (size_t) tile_k) * 8
                + (size_t) warp) * (size_t) (16 * BITS);
        }
        else
        {
            b_idx = ((size_t) tile_k * (size_t) n_tiles + (size_t) tile_n)
                * (size_t) (16 * BITS);
        }
        const uint32_t* packed = (const uint32_t*) (B + b_idx);
        FragB frag0, frag1;
        dq_dispatch<BITS, CB>(packed, lane << 3, frag0, frag1);
        #pragma unroll
        for (int row = 0; row < M; ++row)
        {
            const half2* a = (const half2*)
                (A_had + (size_t) row * (size_t) size_k + (size_t) tile_k * 16);
            half2 a01 = a[r0 >> 1];
            half2 a89 = a[(r0 >> 1) + 4];
            acc_a[row] = __builtin_amdgcn_fdot2(a01, frag0[0], acc_a[row], false);
            acc_a[row] = __builtin_amdgcn_fdot2(a89, frag0[1], acc_a[row], false);
            acc_b[row] = __builtin_amdgcn_fdot2(a01, frag1[0], acc_b[row], false);
            acc_b[row] = __builtin_amdgcn_fdot2(a89, frag1[1], acc_b[row], false);
        }
    }

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
        if (lane < 16)
        {
            float v = (lane & 8) ? vb : va;
            if constexpr (M > 1)
            {
                // Preserve the WARPS=1 small-M final addition from zero.
                float value = 0.0f;
                value += v;
                v = value;
            }
            if constexpr (FP32)
                ((float*) tile[row])[warp * 16 + lane] = v;
            else
                ((half*) tile[row])[warp * 16 + lane] = __float2half(v);
        }
    }
    __syncthreads();

    // One warp per row runs the 128-wide output Hadamard. Grid y stays 0 so
    // the helper's blockIdx.y scale index is t; the row is explicit in the
    // shared-input and global-output pointers.
    if (warp < M)
    {
        int row = warp;
        if constexpr (FP32)
        {
            const float* in = (const float*) tile[row];
            float* out = (float*) C + (size_t) row * (size_t) size_n + (size_t) group * 128;
            had_ff_r_128_inner<false, true>(
                in, out, svh + (size_t) group * 128, 0.088388347648f);
        }
        else
        {
            const half* in = (const half*) tile[row];
            half* out = (half*) C + (size_t) row * (size_t) size_n + (size_t) group * 128;
            had_hf_r_128_inner<false, true>(
                in, out, svh + (size_t) group * 128, 0.088388347648f);
        }
    }
}

template <int M, bool REPACKED = false, bool FP32 = true>
static void exl3_head_tiled_launch_typed(const half* a, const uint16_t* b,
    void* c, int k, int n, int bits, int cb,
    const half* suh, half* ah, const half* svh, cudaStream_t stream)
{
    hipLaunchKernelGGL((exl3_smallm_had<false, false>),
        dim3((k / 128 + 7) / 8, 1, M),
        dim3(256), 0, stream, a, ah, suh, k, nullptr);
    #define HEAD_TILED_DOT(BITS, CB) \
        hipLaunchKernelGGL((exl3_head_tiled_dot_had<M, BITS, CB, REPACKED, FP32>), dim3(n / 128), \
            dim3(256), 0, stream, ah, b, c, k, n, svh)
    #define HEAD_TILED_CB(BITS) \
        switch (cb) { \
            case 0: HEAD_TILED_DOT(BITS, 0); break; \
            case 2: HEAD_TILED_DOT(BITS, 2); break; \
        }
    switch (bits)
    {
        case 2: HEAD_TILED_CB(2); break;
        case 3: HEAD_TILED_CB(3); break;
        case 4: HEAD_TILED_CB(4); break;
        case 5: HEAD_TILED_DOT(5, 2); break;
        case 6: HEAD_TILED_DOT(6, 2); break;
    }
    #undef HEAD_TILED_CB
    #undef HEAD_TILED_DOT
}

// Guarded host entry: two launches, fixed supplied buffers, no allocation.
// Returns true when the request was launched, false to fall through.
static bool exl3_head_tiled_try(const half* a, const uint16_t* b, void* c,
    int m, int k, int n, int bits, int cb, bool fp32,
    const half* suh, half* ah, const half* svh, int device, cudaStream_t stream)
{
    const char* flag = std::getenv("EXL3_HEAD_TILED");
    if (!flag || atoi(flag) != 1) return false;
    if (m != 1 && m != 2 && m != 3 && m != 5) return false;
    if (k <= 0 || k % 128 || k > 65536) return false;
    // Boundary n == 32768 (n/16 == EXL3_GEMV_SPLITK_MAX_TILES) stays on the old
    // path: its split-K reduction differs there.
    if (n / 16 <= EXL3_GEMV_SPLITK_MAX_TILES || n > 1048576 || n % 128) return false;
    if (!fp32) return false;
    if (cb == 0)
    {
        if (bits < 2 || bits > 4) return false;
    }
    else if (cb == 2)
    {
        if (bits < 2 || bits > 6) return false;
    }
    else return false;
    if (!a || !b || !c || !suh || !ah || !svh) return false;
    auto aligned8 = [](const void* p)
    {
        return (reinterpret_cast<uintptr_t>(p) & 7u) == 0;
    };
    if (!aligned8(a) || !aligned8(b) || (reinterpret_cast<uintptr_t>(c) & 15u) ||
        !aligned8(suh) || !aligned8(ah) || !aligned8(svh))
        return false;
    // Diagnostic overrides preserve the old path.
    if (std::getenv("EXL3_SMALLM_HEAD_WARPS")) return false;
    if (exl3_gemv_lds_core()) return false;
    if (exl3_smallm_use_wmma() || exl3_smallm_use_register_b()) return false;
    if (!exl3_gemv_splitk_enabled()) return false;
    if (std::getenv("EXL3_GEMV_SPLITK_WARPS")) return false;
    if (!exl3_head_tiled_device(device)) return false;

    switch (m)
    {
        case 1: exl3_head_tiled_launch_typed<1>(a,b,c,k,n,bits,cb,suh,ah,svh,stream); break;
        case 2: exl3_head_tiled_launch_typed<2>(a,b,c,k,n,bits,cb,suh,ah,svh,stream); break;
        case 3: exl3_head_tiled_launch_typed<3>(a,b,c,k,n,bits,cb,suh,ah,svh,stream); break;
        case 5: exl3_head_tiled_launch_typed<5>(a,b,c,k,n,bits,cb,suh,ah,svh,stream); break;
        default: return false;
    }
    exl3_head_tiled_call_count.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Explicit repacked-layout head entry (experimental): the ONLY consumer of
// [N/128,K/16,8,16*BITS] bytes. Output is Half or Float (inferred from the
// output tensor dtype). Two launches (input Hadamard, grouped dot +
// output Hadamard), fixed supplied buffers, no allocation. Throws
// (TORCH_CHECK) on any unsupported request; never falls back.
static void exl3_head_repacked_check_overlap(const at::Tensor& w,
    const at::Tensor* others[], int n_others)
{
    for (int i = 0; i < n_others; ++i)
        at::assert_no_overlap(w, *others[i]);
}

void exl3_head_repacked(const at::Tensor& x, const at::Tensor& packed,
    const at::Tensor& suh, at::Tensor& input_scratch, const at::Tensor& svh,
    at::Tensor& output, int64_t bits, int64_t codebook)
{
    TORCH_CHECK(x.defined() && packed.defined() && suh.defined(), "missing input tensor");
    TORCH_CHECK(input_scratch.defined() && svh.defined() && output.defined(),
        "missing scratch/scale/output tensor");
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA");
    const at::Tensor* all[] = { &x, &packed, &suh, &input_scratch, &svh, &output };
    for (int i = 1; i < 6; ++i)
        TORCH_CHECK(all[i]->device() == x.device(), "all tensors must be on the same GPU");
    for (int i = 0; i < 6; ++i)
        TORCH_CHECK(all[i]->is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
        "output must have 16-byte aligned data");
    // Contiguous views can still start at an unaligned storage offset.
    const at::Tensor* align8[] = { &x, &packed, &suh, &input_scratch, &svh };
    for (int i = 0; i < 5; ++i)
        TORCH_CHECK(reinterpret_cast<uintptr_t>(align8[i]->data_ptr()) % 8 == 0,
                    "x/packed/suh/input_scratch/svh must have 8-byte aligned data");
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(packed, kShort);
    TORCH_CHECK_DTYPE(suh, kHalf);
    TORCH_CHECK_DTYPE(input_scratch, kHalf);
    TORCH_CHECK_DTYPE(svh, kHalf);
    TORCH_CHECK_FLOAT_HALF(output);
    bool out_fp32 = output.dtype() == at::kFloat;

    TORCH_CHECK_DIM(x, 2);
    TORCH_CHECK_DIM(packed, 4);
    TORCH_CHECK_DIM(suh, 1);
    TORCH_CHECK_DIM(input_scratch, 2);
    TORCH_CHECK_DIM(svh, 1);
    TORCH_CHECK_DIM(output, 2);

    int64_t m = x.size(0);
    int64_t k = x.size(1);
    TORCH_CHECK(m == 1 || m == 2 || m == 3 || m == 5, "M must be one of (1,2,3,5)");
    TORCH_CHECK(k > 0 && k % 128 == 0 && k <= 65536,
        "K must be a positive multiple of 128, at most 65536");
    TORCH_CHECK(output.size(0) == m, "output rows must match x");
    int64_t n = output.size(1);
    TORCH_CHECK(n > 32768 && n % 128 == 0 && n <= 1048576,
        "N must be a multiple of 128 in (32768,1048576]");
    if (codebook == 0)
    {
        TORCH_CHECK(bits >= 2 && bits <= 4, "cb0 supports bits 2/3/4");
    }
    else if (codebook == 2)
    {
        TORCH_CHECK(bits >= 2 && bits <= 6, "cb2 supports bits 2..6");
    }
    else
    {
        TORCH_CHECK(false, "codebook must be 0 or 2");
    }
    TORCH_CHECK(packed.size(0) == n / 128 && packed.size(1) == k / 16 &&
        packed.size(2) == 8 && packed.size(3) == 16 * bits,
        "packed must be (N/128,K/16,8,16*bits)");
    TORCH_CHECK(suh.size(0) == k, "suh must have K elements");
    TORCH_CHECK(svh.size(0) == n, "svh must have N elements");
    TORCH_CHECK(input_scratch.size(0) == m && input_scratch.size(1) == k,
        "input_scratch must be (M,K)");
    TORCH_CHECK(m <= 5 && k <= 65536 && n <= 1048576, "M/K/N out of supported integer range");

    // Written buffers must not overlap any other tensor's storage.
    const at::Tensor* others_in[] = { &x, &packed, &suh, &svh, &output };
    const at::Tensor* others_out[] = { &x, &packed, &suh, &input_scratch, &svh };
    exl3_head_repacked_check_overlap(input_scratch, others_in, 5);
    exl3_head_repacked_check_overlap(output, others_out, 5);

    TORCH_CHECK(exl3_head_tiled_device(x.device().index()), "exl3_head_repacked requires gfx1101");

    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const half* x_ptr = (const half*) x.data_ptr();
    const uint16_t* b_ptr = (const uint16_t*) packed.data_ptr();
    const half* suh_ptr = (const half*) suh.data_ptr();
    half* ah_ptr = (half*) input_scratch.data_ptr();
    const half* svh_ptr = (const half*) svh.data_ptr();
    void* out_ptr = output.data_ptr();
    int ki = (int) k;
    int ni = (int) n;
    int bi = (int) bits;
    int cb = (int) codebook;

    switch (m)
    {
        case 1:
            if (out_fp32) exl3_head_tiled_launch_typed<1, true, true>(x_ptr, b_ptr, out_ptr, ki, ni, bi, cb,
                suh_ptr, ah_ptr, svh_ptr, stream);
            else exl3_head_tiled_launch_typed<1, true, false>(x_ptr, b_ptr, out_ptr, ki, ni, bi, cb,
                suh_ptr, ah_ptr, svh_ptr, stream);
            break;
        case 2:
            if (out_fp32) exl3_head_tiled_launch_typed<2, true, true>(x_ptr, b_ptr, out_ptr, ki, ni, bi, cb,
                suh_ptr, ah_ptr, svh_ptr, stream);
            else exl3_head_tiled_launch_typed<2, true, false>(x_ptr, b_ptr, out_ptr, ki, ni, bi, cb,
                suh_ptr, ah_ptr, svh_ptr, stream);
            break;
        case 3:
            if (out_fp32) exl3_head_tiled_launch_typed<3, true, true>(x_ptr, b_ptr, out_ptr, ki, ni, bi, cb,
                suh_ptr, ah_ptr, svh_ptr, stream);
            else exl3_head_tiled_launch_typed<3, true, false>(x_ptr, b_ptr, out_ptr, ki, ni, bi, cb,
                suh_ptr, ah_ptr, svh_ptr, stream);
            break;
        case 5:
            if (out_fp32) exl3_head_tiled_launch_typed<5, true, true>(x_ptr, b_ptr, out_ptr, ki, ni, bi, cb,
                suh_ptr, ah_ptr, svh_ptr, stream);
            else exl3_head_tiled_launch_typed<5, true, false>(x_ptr, b_ptr, out_ptr, ki, ni, bi, cb,
                suh_ptr, ah_ptr, svh_ptr, stream);
            break;
    }
    cuda_check(cudaPeekAtLastError());
    exl3_head_repacked_call_count.fetch_add(1, std::memory_order_relaxed);
}
