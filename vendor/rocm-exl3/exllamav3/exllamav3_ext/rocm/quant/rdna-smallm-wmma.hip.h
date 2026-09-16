#pragma once
// Experimental noncooperative WMMA alternative to exl3_smallm_dot (M = 2..9,
// K tiles of 16, BITS = 2/3/4). No Hadamard/scaling inside; host dispatch applies the
// existing separate input/output transforms.
// Derived from Turboderp ExLlamaV3 and the CarouselAether ROCm fork's WMMA
// primitives / dequantization mapping; MIT license retained in the sibling
// rdna-smallm-LICENSE.txt.
// Included after rdna-smallm.hip.h; reuses Exl3GemvGraphParams, FragB,
// dq_dispatch, WmmaFragA/B/C and rdna_wmma helpers already included
// transitively. No includes needed.

template <int M, int BITS, int CB, bool FP32, int WARPS, bool GRAPH = false, bool REGISTER_B = false>
static __global__ __launch_bounds__(WARPS * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS * 32, WARPS * 32)))
void exl3_smallm_wmma(const half* __restrict__ A, const uint16_t* __restrict__ B,
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

    // Warp-private dequant staging (stride 18 pads the 16-wide tile against
    // LDS bank conflicts on the strided B-fragment load) plus the block-wide
    // split-K reduction buffer. Both statically sized; WARPS is a
    // compile-time constant. REGISTER_B builds the B fragment from registers
    // and never touches decoded (sized 1, unused).
    __shared__ half decoded[REGISTER_B ? 1 : WARPS][16 * 18];
    __shared__ float partial[WARPS][16 * 16];

    WmmaFragC acc;
    acc.clear();

    for (int tile_k = begin; tile_k < end; ++tile_k)
    {
        const uint32_t* packed = (const uint32_t*)
            (B + (tile_k * n_tiles + tile_n) * (16 * BITS));
        FragB frag0, frag1;
        dq_dispatch<BITS, CB>(packed, lane << 3, frag0, frag1);

        if constexpr (REGISTER_B)
        {
            // Register-only B fragment: the same 16 values
            // rdna_wmma::load_matrix_b would gather from the unswizzled LDS
            // tile (lane L holds column L % 16, rows 0..15), pulled straight
            // from the dequant fragments with wave shuffles. No LDS staging
            // and no mem_fence; A loading, FP32 accumulation and the split-K
            // reduction below are unchanged.
            //
            // Inverse of the unswizzle in exl3_gemm_inner_rdna.hip.h (layout
            // also documented in exl3_gemv_kernel_rdna.hip.h): lane S holds
            //   frag0[0] = (B[r0][cA], B[r0+1][cA])      r0 = (S % 4) * 2
            //   frag0[1] = (B[r0+8][cA], B[r0+9][cA])    cA = (S/8)*2+((S>>2)&1)
            //   frag1[0] = (B[r0][cB], B[r0+1][cB])      cB = cA + 8
            //   frag1[1] = (B[r0+8][cB], B[r0+9][cB])
            // so weight (row, col) lives in lane
            //   src = ((col%8)/2)*8 + (col%2)*4 + (row%8)/2,
            // in frag0 (col < 8) or frag1 (col >= 8), element 0 (row < 8) or
            // 1 (row >= 8), low half for even rows, high half for odd rows.
            // Both frags are shuffled first and the destination lane selects
            // afterwards: selecting before the shuffle would apply the
            // *source* lane's column to the frag0/frag1 choice. All lanes
            // execute every shuffle and the WMMA.
            WmmaFragB b;
            {
                int col = lane % 16;
                bool use_frag1 = col >= 8;
                _Float16* bd = (_Float16*)&b.data;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    // Pair j covers rows 2*j, 2*j+1: (j % 4) is ((row%8)/2)
                    // for both rows, and (j / 4) is the frag element.
                    int src = ((col % 8) / 2) * 8 + (col % 2) * 4 + (j % 4);
                    half2 s0 = __shfl(frag0[j / 4], src, 32);
                    half2 s1 = __shfl(frag1[j / 4], src, 32);
                    half2 got = use_frag1 ? s1 : s0;
                    bd[2 * j]     = (_Float16)__half2float(__low2half(got));
                    bd[2 * j + 1] = (_Float16)__half2float(__high2half(got));
                }
            }

            WmmaFragA a;
            int a_row = lane % 16;
            if (a_row < M)
                a.data = *((const half16_t*)(A + a_row * size_k + tile_k * 16));
            else
                a.clear();

            // All 32 lanes always execute WMMA.
            rdna_wmma::mma_sync(acc, a, b);
        }
        else
        {
            // Proven shuffle/unswizzle mapping from exl3_gemm_inner_rdna.hip.h:
            // lanes with bit 2 clear combine their own values with lane+4 and
            // write the 16x16 block row-major. All lanes call __shfl_down; only
            // !(lane & 4) store.
            half* B_lds = decoded[warp];
            half2 n0 = __shfl_down(frag0[0], 4, 32);
            half2 n1 = __shfl_down(frag0[1], 4, 32);
            half2 n2 = __shfl_down(frag1[0], 4, 32);
            half2 n3 = __shfl_down(frag1[1], 4, 32);

            if (!(lane & 4))
            {
                constexpr int S = 18;
                const int r0 = (lane % 4) * 2;
                const int r1 = r0 + 1;
                const int r2 = r0 + 8;
                const int r3 = r0 + 9;
                const int c0 = (lane / 8) * 2;
                const int c1 = c0 + 8;

                B_lds[r0 * S + c0]     = __low2half (frag0[0]);
                B_lds[r0 * S + c0 + 1] = __low2half (n0);
                B_lds[r1 * S + c0]     = __high2half(frag0[0]);
                B_lds[r1 * S + c0 + 1] = __high2half(n0);

                B_lds[r2 * S + c0]     = __low2half (frag0[1]);
                B_lds[r2 * S + c0 + 1] = __low2half (n1);
                B_lds[r3 * S + c0]     = __high2half(frag0[1]);
                B_lds[r3 * S + c0 + 1] = __high2half(n1);

                B_lds[r0 * S + c1]     = __low2half (frag1[0]);
                B_lds[r0 * S + c1 + 1] = __low2half (n2);
                B_lds[r1 * S + c1]     = __high2half(frag1[0]);
                B_lds[r1 * S + c1 + 1] = __high2half(n2);

                B_lds[r2 * S + c1]     = __low2half (frag1[1]);
                B_lds[r2 * S + c1 + 1] = __low2half (n3);
                B_lds[r3 * S + c1]     = __high2half(frag1[1]);
                B_lds[r3 * S + c1 + 1] = __high2half(n3);
            }

            // B_lds is warp-private, so ordering only has to hold within the wave:
            // drain LDS before the B-fragment load and after it (before the next
            // tile's stores). No block barrier inside the K loop.
            mem_fence();

            WmmaFragA a;
            int a_row = lane % 16;
            if (a_row < M)
                a.data = *((const half16_t*)(A + a_row * size_k + tile_k * 16));
            else
                a.clear();

            WmmaFragB b;
            rdna_wmma::load_matrix_b(b, B_lds, 18);
            mem_fence();

            // All 32 lanes always execute WMMA.
            rdna_wmma::mma_sync(acc, a, b);
        }
    }

    // Spill the accumulator to the block-wide reduction buffer in C-fragment
    // layout: row = lane % 16, col = 2*i + (lane >= 16).
    {
        int row = lane % 16;
        int col_base = (lane >= 16) ? 1 : 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            partial[warp][row * 16 + i * 2 + col_base] = acc[i];
    }

    __syncthreads();

    // warp0 reduces across WARPS in increasing order and stores. One lane per
    // (row, even/odd column set); lanes with row >= M stay idle.
    if (warp == 0)
    {
        int row = lane % 16;
        if (row < M)
        {
            int col_base = (lane >= 16) ? 1 : 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                int col = i * 2 + col_base;
                float value = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) value += partial[w][row * 16 + col];
                int index = row * size_n + tile_n * 16 + col;
                if constexpr (FP32) ((float*) C)[index] = value;
                else ((half*) C)[index] = __float2half(value);
            }
        }
    }
}
