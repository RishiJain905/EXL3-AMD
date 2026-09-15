# EXL3 kernel development

Sources, patches and operator validators supporting [the vendored runtime](../../vendor/rocm-exl3/). The vendored tree is the active build source. Historical patch files are references and should not be blindly reapplied.

Developer tools include packed projection/small-M validation and conversion repairs. Some require explicit tensors, configurations or build outputs; they are not an installer or model download.

EXL3 decoding and base kernels originate with Turboderp/ExLlamaV3; RDNA/HIP primitives and the direct port originate with CarouselAether. Local changes reuse those components. See [UPSTREAMS.md](../../docs/UPSTREAMS.md), [LICENSE.upstream](LICENSE.upstream), and [rdna-smallm-LICENSE.txt](rdna-smallm-LICENSE.txt).

Use isolated matching extensions and known fixtures for operator checks. Keep outputs and private paths outside Git. CPU tests do not validate GPU kernels or full models. [Build instructions](../../docs/BUILD.md).

`check_hgemm.py` exposes `run_checks(torch, extension)` for the optional FP32-output
prefill GEMM. The caller must already hold the GPU lease and supply a verified,
loaded extension; the validator neither loads nor builds a binary. It checks
numerical references, dispatch fallbacks, output canaries, invalid arguments,
stream behavior and graph replay, including both sides of the 512-row WMMA
dispatch boundary. [Initial measurements](../../docs/GPU-PERFORMANCE.md) and
[full CLI follow-up](../../docs/RUNTIME-PERFORMANCE.md).
