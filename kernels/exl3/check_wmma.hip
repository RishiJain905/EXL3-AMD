// Standalone HIP primitive checks; no Torch or model required.
// Compile for each target. Execute only under an acquired GPU lease/timeout.
// See docs/GPU-COMPATIBILITY.md. All references below are scalar CPU math.
#include "../../vendor/rocm-exl3/exllamav3/exllamav3_ext/rocm/rdna_wmma.hip.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define HIP_CHECK(call) do { hipError_t error = (call); if (error != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(error)); \
    std::exit(1); } } while (0)

template <typename T> struct Buffer {
    T* data;
    explicit Buffer(size_t count) { HIP_CHECK(hipMalloc(&data, count * sizeof(T))); }
    ~Buffer() { (void)hipFree(data); }
    void upload(const std::vector<T>& value) {
        HIP_CHECK(hipMemcpy(data, value.data(), value.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    std::vector<T> download(size_t count) {
        std::vector<T> value(count);
        HIP_CHECK(hipMemcpy(value.data(), data, count * sizeof(T), hipMemcpyDeviceToHost));
        return value;
    }
};

template <bool BF16>
__global__ void floating(const half* a, const half* b, const __hip_bfloat16* ab,
                         const __hip_bfloat16* bb, const float* initial, float* out) {
    WmmaFragC c;
    c.clear();
    rdna_wmma::load_accumulate_c(c, initial, 16);
    if constexpr (BF16) {
        WmmaFragA_bf16 af;
        WmmaFragB_bf16 bf;
        rdna_wmma::load_matrix_a_bf16(af, ab, 16);
        rdna_wmma::load_matrix_b_bf16(bf, bb, 16);
        for (int repeat = 0; repeat < 2; ++repeat) rdna_wmma::mma_sync_bf16(c, af, bf);
    } else {
        WmmaFragA af;
        WmmaFragB bf;
        rdna_wmma::load_matrix_a(af, a, 16);
        rdna_wmma::load_matrix_b(bf, b, 16);
        for (int repeat = 0; repeat < 2; ++repeat) rdna_wmma::mma_sync(c, af, bf);
    }
    rdna_wmma::store_matrix_c(out, c, 16);
}

template <bool OPSEL>
__global__ void half_accumulate(const half* a, const half* b, const float* initial,
                                half* out, int* preserved) {
    int lane = threadIdx.x & 31;
    WmmaFragA af;
    WmmaFragB bf;
    WmmaFragC_f16 c;
    rdna_wmma::load_matrix_a(af, a, 16);
    rdna_wmma::load_matrix_b(bf, b, 16);
    for (int i = 0; i < 8; ++i) {
        c.data[2 * i + OPSEL] = (_Float16)initial[(lane % 16) * 16 + 2 * i + lane / 16];
        c.data[2 * i + !OPSEL] = (_Float16)(11 + i);
    }
    for (int repeat = 0; repeat < 2; ++repeat) rdna_wmma::mma_sync_f16<OPSEL>(c, af, bf);
    rdna_wmma::store_matrix_c_f16<OPSEL>(out, c, 16);
    int okay = 1;
    for (int i = 0; i < 8; ++i) okay &= c.data[2 * i + !OPSEL] == (_Float16)(11 + i);
    preserved[lane] = okay;
}

template <bool SIGN_A, bool SIGN_B, bool CLAMP>
__global__ void integer(const int8_t* a, const int8_t* b, const int* initial, int* out) {
    int lane = threadIdx.x & 31;
    WmmaFragA_i8 af;
    WmmaFragB_i8 bf;
    WmmaFragC_i32 c;
    rdna_wmma::load_matrix_a_i8(af, a, 16);
    rdna_wmma::load_matrix_b_i8(bf, b, 16);
    for (int i = 0; i < 8; ++i) c[i] = initial[(lane % 16) * 16 + 2 * i + lane / 16];
    rdna_wmma::mma_sync_i8<SIGN_A, SIGN_B, CLAMP>(c, af, bf);
    rdna_wmma::store_matrix_c_i32(out, c, 16);
}

template <typename T> void compare(const std::vector<T>& actual,
                                   const std::vector<double>& expected, const char* label) {
    for (size_t i = 0; i < expected.size(); ++i) {
        double value = static_cast<double>(actual[i]);
        if (!std::isfinite(value) || std::abs(value - expected[i]) > 0.001) {
            std::fprintf(stderr, "%s[%zu]: got %.10g, expected %.10g\n", label, i, value, expected[i]);
            std::exit(1);
        }
    }
    std::printf("PASS %s (%zu elements)\n", label, expected.size());
}

// Exercise the actual gfx12 lane adapters on gfx11 too. Test each direction
// against independently populated matrix coordinates: an incorrect pair of
// inverse permutations must not pass merely because their roundtrip is identity.
__global__ void layouts(int* out) {
    int lane = threadIdx.x & 31;
    int row = lane % 16;
    WmmaFragC cf;
    WmmaFragC_i32 ci;
    half8_t ch, nh;
    float8_t nf;
    int32x8_t ni;
    half16_t af;
    bf16x16_t ab;
    int32x4_t ai;
    for (int i = 0; i < 16; ++i) {
        ((unsigned short*)&af)[i] = 0x3800 + row * 16 + i;
        ((unsigned short*)&ab)[i] = 0x3800 + row * 16 + i;
        ((uint8_t*)&ai)[i] = row * 16 + i;
    }
    for (int i = 0; i < 8; ++i) {
        int external = row * 16 + 2 * i + lane / 16;
        int native = row * 16 + 8 * (lane / 16) + i;
        cf[i] = __int_as_float(0x4b000001 + external);
        ci[i] = 0x4b000001 + external;
        ((unsigned short*)&ch)[i] = 0x3800 + external;
        nf[i] = __int_as_float(0x4b000001 + native);
        ni[i] = 0x4b000001 + native;
        ((unsigned short*)&nh)[i] = 0x3800 + native;
    }
    auto f_to = rdna_wmma::rdna4_c_to_native_f32(cf);
    auto f_from = rdna_wmma::rdna4_c_from_native_f32(nf);
    auto i_to = rdna_wmma::rdna4_c_to_native_i32(ci);
    auto i_from = rdna_wmma::rdna4_c_from_native_i32(ni);
    auto h_to = rdna_wmma::rdna4_c_to_native_f16(ch);
    auto h_from = rdna_wmma::rdna4_c_from_native_f16(nh);
    auto f_narrow = rdna_wmma::rdna4_narrow_f16(af);
    auto b_narrow = rdna_wmma::rdna4_narrow_bf16(ab);
    auto i_narrow = rdna_wmma::rdna4_narrow_i8(ai);
    for (int i = 0; i < 8; ++i) {
        out[(0 * 32 + lane) * 8 + i] = __float_as_int(f_to[i]);
        out[(1 * 32 + lane) * 8 + i] = __float_as_int(f_from[i]);
        out[(2 * 32 + lane) * 8 + i] = i_to[i];
        out[(3 * 32 + lane) * 8 + i] = i_from[i];
        out[(4 * 32 + lane) * 8 + i] = ((unsigned short*)&h_to)[i];
        out[(5 * 32 + lane) * 8 + i] = ((unsigned short*)&h_from)[i];
        out[(6 * 32 + lane) * 8 + i] = ((unsigned short*)&f_narrow)[i];
        out[(7 * 32 + lane) * 8 + i] = ((unsigned short*)&b_narrow)[i];
        out[(8 * 32 + lane) * 8 + i] = ((uint8_t*)&i_narrow)[i];
    }
}

void check_layouts() {
    constexpr int count = 9 * 32 * 8;
    Buffer<int> out(count);
    hipLaunchKernelGGL(layouts, dim3(1), dim3(32), 0, 0, out.data);
    HIP_CHECK(hipGetLastError());
    std::vector<double> expected(count);
    for (int test = 0; test < 9; ++test) for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 8; ++i) {
            int col = (test < 6 && test % 2) ? 2 * i + lane / 16 : i + 8 * (lane / 16);
            int offset = (lane % 16) * 16 + col;
            int base = test < 4 ? 0x4b000001 : (test < 8 ? 0x3800 : 0);
            expected[(test * 32 + lane) * 8 + i] = base + offset;
        }
    compare(out.download(count), expected, "gfx12 input/accumulator layout (all lanes and registers)");
}

template <bool SIGN_A, bool SIGN_B, bool CLAMP = false> void check_integer(bool saturate = false) {
    std::vector<int8_t> a(256), b(256);
    std::vector<int> initial(256);
    for (int i = 0; i < 256; ++i) {
        a[i] = saturate ? 120 : static_cast<int8_t>((i * 37 + 17) & 255);
        b[i] = saturate ? 120 : static_cast<int8_t>((i * 53 + 91) & 255);
        initial[i] = saturate ? std::numeric_limits<int>::max() - 10 : 17000001 + i * 19;
    }
    std::vector<double> expected(256);
    for (int r = 0; r < 16; ++r) for (int col = 0; col < 16; ++col) {
        int64_t value = initial[r * 16 + col];
        for (int k = 0; k < 16; ++k) {
            int av = SIGN_A ? int(a[r * 16 + k]) : int(static_cast<uint8_t>(a[r * 16 + k]));
            int bv = SIGN_B ? int(b[k * 16 + col]) : int(static_cast<uint8_t>(b[k * 16 + col]));
            value += av * bv;
        }
        if (CLAMP && value > std::numeric_limits<int>::max()) value = std::numeric_limits<int>::max();
        expected[r * 16 + col] = double(value);
    }
    Buffer<int8_t> ad(256), bd(256);
    Buffer<int> cd(256), out(256);
    ad.upload(a); bd.upload(b); cd.upload(initial);
    hipLaunchKernelGGL((integer<SIGN_A, SIGN_B, CLAMP>), dim3(1), dim3(32), 0, 0,
                      ad.data, bd.data, cd.data, out.data);
    HIP_CHECK(hipGetLastError());
    char label[96];
    std::snprintf(label, sizeof(label), "int8 signed-A=%d signed-B=%d clamp=%d saturate=%d",
                  SIGN_A, SIGN_B, CLAMP, saturate);
    compare(out.download(256), expected, label);
}

int main() {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::printf("WMMA checks on %s (%s)\n", props.name, props.gcnArchName);
    check_layouts();
    std::vector<half> a(256), b(256);
    std::vector<__hip_bfloat16> ab(256), bb(256);
    std::vector<float> initial(256);
    for (int r = 0; r < 16; ++r) for (int col = 0; col < 16; ++col) {
        int i = r * 16 + col;
        float av = ((r * 7 + col * 3) % 13 - 6) * 0.25f;
        float bv = ((r * 5 + col * 11) % 17 - 8) * 0.5f;
        a[i] = __float2half(av); b[i] = __float2half(bv);
        ab[i] = __hip_bfloat16(av); bb[i] = __hip_bfloat16(bv);
        initial[i] = (i % 19 - 9) * 0.125f;
    }
    std::vector<double> expected(256);
    for (int r = 0; r < 16; ++r) for (int col = 0; col < 16; ++col) {
        double value = initial[r * 16 + col];
        for (int k = 0; k < 16; ++k) value += 2 * double(__half2float(a[r * 16 + k])) * __half2float(b[k * 16 + col]);
        expected[r * 16 + col] = value;
    }
    Buffer<half> ad(256), bd(256), hout(256);
    Buffer<__hip_bfloat16> abd(256), bbd(256);
    Buffer<float> cd(256), out(256);
    Buffer<int> preserved(32);
    ad.upload(a); bd.upload(b); abd.upload(ab); bbd.upload(bb); cd.upload(initial);
    hipLaunchKernelGGL((floating<false>), dim3(1), dim3(32), 0, 0,
                      ad.data, bd.data, abd.data, bbd.data, cd.data, out.data);
    HIP_CHECK(hipGetLastError());
    compare(out.download(256), expected, "f16->f32 repeated nonzero accumulate");
    hipLaunchKernelGGL((floating<true>), dim3(1), dim3(32), 0, 0,
                      ad.data, bd.data, abd.data, bbd.data, cd.data, out.data);
    HIP_CHECK(hipGetLastError());
    compare(out.download(256), expected, "bf16->f32 repeated nonzero accumulate");
    #define CHECK_HALF(OP) do { \
        hipLaunchKernelGGL((half_accumulate<OP>), dim3(1), dim3(32), 0, 0, \
                          ad.data, bd.data, cd.data, hout.data, preserved.data); \
        HIP_CHECK(hipGetLastError()); \
        compare(hout.download(256), expected, "f16 accumulate opsel=" #OP); \
        compare(preserved.download(32), std::vector<double>(32, 1), "other half preserved"); \
    } while (0)
    CHECK_HALF(false);
    CHECK_HALF(true);
    check_integer<false, false>();
    check_integer<false, true>();
    check_integer<true, false>();
    check_integer<true, true>();
    check_integer<false, false, true>();
    check_integer<false, true, true>();
    check_integer<true, false, true>();
    check_integer<true, true, true>();
    check_integer<true, true, true>(true);
    HIP_CHECK(hipDeviceSynchronize());
    std::puts("All WMMA primitive checks passed");
}
