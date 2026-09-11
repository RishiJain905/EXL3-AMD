// Bounded HIP cooperative-launch diagnostic; no EXL3 or model dependency.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstring>

namespace cg = cooperative_groups;
constexpr int count = 1024;

__global__ void cooperative_probe(unsigned* scratch, unsigned* output)
{
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        scratch[i] = static_cast<unsigned>(i);
    cg::this_grid().sync();
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        output[i] = scratch[(i + 511) % count] + 3;
}

static bool checked(hipError_t result, const char* operation)
{
    if (result == hipSuccess) return true;
    std::fprintf(stderr, "%s: %s (%d)\n", operation,
                 hipGetErrorName(result), static_cast<int>(result));
    return false;
}

int main(int argc, char** argv)
{
    if (argc != 2 || std::strcmp(argv[1], "--execute") != 0)
    {
        std::fprintf(stderr, "Usage: cooperative_probe --execute\n");
        return 2;
    }
    hipDeviceProp_t prop{};
    if (!checked(hipGetDeviceProperties(&prop, 0), "device properties")) return 1;
    std::printf("{\"stage\":\"properties\",\"cooperativeLaunch\":%d,"
                "\"sharedMemPerBlock\":%zu,\"multiProcessorCount\":%d}\n",
                prop.cooperativeLaunch, prop.sharedMemPerBlock,
                prop.multiProcessorCount);
    std::fflush(stdout);

    unsigned* allocation = nullptr;
    if (!checked(hipMalloc(reinterpret_cast<void**>(&allocation),
                           2 * count * sizeof(unsigned)), "allocate")) return 1;
    unsigned* output = allocation + count;
    void* args[] = {&allocation, &output};
    const hipError_t launch = hipLaunchCooperativeKernel(
        reinterpret_cast<void*>(cooperative_probe), dim3(1), dim3(256), args, 0, nullptr);
    bool ok = checked(launch, "cooperative launch");
    if (ok) ok = checked(hipDeviceSynchronize(), "synchronize");
    unsigned host[count]{};
    if (ok) ok = checked(hipMemcpy(host, output, sizeof(host), hipMemcpyDeviceToHost), "copy");
    int mismatches = 0;
    if (ok)
    {
        for (int i = 0; i < count; ++i)
            mismatches += host[i] != static_cast<unsigned>((i + 511) % count + 3);
        std::printf("{\"stage\":\"result\",\"values\":%d,\"mismatches\":%d,"
                    "\"allocation_bytes\":%zu}\n", count, mismatches,
                    2 * count * sizeof(unsigned));
    }
    const bool released = checked(hipFree(allocation), "free");
    return ok && released && mismatches == 0 ? 0 : 1;
}
