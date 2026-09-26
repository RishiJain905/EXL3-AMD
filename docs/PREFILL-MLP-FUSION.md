# MiMo prefill and paired MLP work

September 26, 2026. This follows the [packed MLP / 10–64-row work](PACKED-MLP-PERFORMANCE.md)
and implements roadmap items 3 and 4. Scope is large-prefill projections and
paired gate/up projections with fused output Hadamards and SwiGLU. Attention,
KV policy, vocabulary head, sampling, and a whole-model persistent scheduler
are outside this change.

The qualified implementation is now primary. No optimization flags are
required. The local registration selects the fresh 110-unit gfx1101 build
with SHA-256 `a00edbf45f5e70431510c3a28db70447e85a47574de26f4f86f7004c9f89fed9`.
Its 331 native/build source fingerprints match the reviewed source.

## Measurement setup

- RX 7800 XT, gfx1101, 16 GiB, WSL Ubuntu. Cooperative launch is unavailable.
- Python 3.12, Torch 2.13.0+rocm7.2, ROCm SDK 7.2.4, Triton 3.7.1.
- Frozen MiMo 9B K5/H6 mul1 model with BF16 MTP donor; 32 target blocks,
  hidden width 4096, MLP intermediate width 12288. Model files were unchanged.
- MTP depth 2, rowwise target verification, decode fusions off, FP16 KV,
  4096 context, 256-token prefill chunks, CPU embeddings, no prompt reuse,
  allocator fraction 0.65. Each prompt receives a discarded warmup and three
  measured requests. Short/code prompts emit 128 tokens; long logs emit 32.
- Timing version 2 excludes the complete first emitting iteration from decode
  throughput. Reported input counts include the held last token; actual prefill
  processes one fewer token.
- BF16-MTP comparisons use the frozen Step4 evaluator/source at `b502d37`,
  injecting this checkout's compatibility, capability, and paired-MLP modules.
  This does not merge the separate Step4 work or establish BF16-MTP support
  in this checkout's public launcher.

The primary reference extension before this change has SHA-256
`006ac53097629cdbdd4237050a5fec7feeef9a74af7a8c6a83cb0325361135f1`.

## Fresh baseline

Medians in tokens/s. BLAS and WMMA run in separate processes because the native
dense-GEMM selector is cached at first use. All output token IDs match between
these two baselines (2,880 measured tokens).

| Prompt / input tokens | BLAS prefill | WMMA prefill | BLAS decode | WMMA decode |
| --- | ---: | ---: | ---: | ---: |
| SQL / 78 | 494.00 | 573.65 | 70.83 | 70.89 |
| Binary search / 93 | 592.15 | 683.91 | 79.54 | 78.83 |
| Short greeting / 41 | 572.19 | 563.15 | 76.94 | 74.70 |
| Repeated log / 284 | 716.10 | 985.04 | 70.63 | 72.17 |
| Repeated log / 1244 | 865.74 | 1336.31 | 70.66 | 68.09 |
| Repeated log / 2924 | 882.83 | 1377.89 | 70.91 | 73.08 |

The 41-token prompt already uses packed-mid. Differences there and in decode
are run variability, not evidence that dense prefill WMMA speeds those paths.
The new packed-prefill implementation must also be compared with the faster
WMMA baseline, not just BLAS.

## Implementations

Automatic selection combines the measured winners:

- Packed-mid retains its existing 10–64-row envelope.
- Tiled packed prefill handles 65–128 rows with FP16 or FP32 output, and
  129–512 rows with FP32 output. Output width is limited to 32768 in the
  automatic policy; vocabulary-head work is excluded from this change.
- Larger projections retain reconstruction. `--prefill-gemm auto` now selects
  verified FP32-output WMMA where supported; FP16 output and unsupported
  shapes use BLAS. Binaries without the capability retain BLAS.
- Eligible MLP gate/up projections use the paired entry automatically. Bias,
  transforms, custom fusions, and other unsupported cases retain fallback.

`--no-packed-prefill` and `--no-mlp-pair` are diagnostic controls; explicit
enabling requires the corresponding verified ABI. The ordinary defaults
automatically fall back on legacy binaries and reject unknown ABI versions.
`--prefill-gemm blas` reproduces the previous dense selection. Explicit small-M
WMMA or GEMV LDS diagnostics keep their separate projection arithmetic.

`rdna-packed-prefill.hip.h` decodes packed weights into a shared-memory tile
and feeds WMMA directly. A block computes 128 rows by 64 output columns in
32-element K steps, with 256 threads. Decoded weights are reused across the
128-row tile. FP32 partials are folded every 256 K elements to bound long-chain
WMMA error. Input and output Hadamards remain separate. No complete FP16
weight matrix, cooperative launch, or cross-block software barrier is used.

Packed-prefill ABI 1 supports rows 65–4096, positive K/N divisible by 128,
default codebook K2/K3/K4 and mul1 K2–K6, FP16 input, and FP16/FP32 output.
The Python dispatcher bypasses the upstream 144-row reconstruction threshold
only when the verified capability and shape policy admit the request.

`rdna-mlp-pair.hip.h` supplies `exl3_mlp_gate_up`. Three launches replace the
seven gate/up/activation launches: joint input Hadamards with distinct scale
vectors, paired packed projections, then output Hadamards plus SwiGLU.
The down projection retains its ordinary dispatch. Projection results round
to FP16 before the output Hadamards; post-scaling and activation retain the
existing half2 arithmetic. This is required by the actual MiMo intermediates.

Paired-MLP ABI 1 supports rows 1/2/3/5, the same eight packed formats, matching
gate/up geometry, and FP16 intermediates. The host validates tensor shapes,
dtypes, alignment, devices, contiguity, and buffer overlap. The runtime keeps
fallbacks for biases, LoRA, changed projections, overridden forwards,
nontrivial scaling/trimming, capture parameters, activation limits, other
activations, tensor parallelism, and sliced MLPs. Existing custom/native MLP
fusion wrappers retain their behavior. External graph capture uses fixed
buffers; native graph pointer patching is outside this entry point.

## Validation and evidence

The final policy passed 108 measured requests / 8,640 generated tokens across
256- and 1024-token prefill chunks. Every output token matched its control at
the same chunk size. Each treatment was bracketed by controls in one process.

For 256-token chunks, the following medians compare the original fresh BLAS
baseline with the final primary selection. The final column isolates the
additional prefill gain against the mean of the two WMMA control medians.

| Input tokens | Original prefill tok/s | Primary prefill tok/s | Gain over WMMA controls |
| ---: | ---: | ---: | ---: |
| 78 | 494.00 | 610.54 | 6.95% |
| 93 | 592.15 | 734.34 | 8.75% |
| 41 | 572.19 | 568.19 | 0.03% |
| 284 | 716.10 | 1030.36 | 3.72% |
| 1244 | 865.74 | 1386.28 | 4.17% |
| 2924 | 882.83 | 1401.85 | 2.51% |

The two long prompts gain approximately 60% and 59% against the original
BLAS baseline. At 1024-token chunks they reach 1685.56 and 1765.57 tok/s;
the narrow packed policy is essentially neutral against dense WMMA there
(-0.07% and +0.19%), as intended by the crossover rule.

Decode improvement is small: final 256-chunk medians are 70.52/78.85/76.74
tok/s for SQL/binary-search/greeting, versus 70.59/78.22/75.88 for bracketed
WMMA controls. Across all six prompts, differences range from -0.41% to
+2.50%; with 1024-token chunks, -0.61% to +2.67%. Treat this as roughly flat
to a modest gain, not a major decode-speed improvement. The paired entry
reduces launch count and improves warm complete-MLP operator time by about
2–4%; evicted-weight gains are smaller. External graph replay did not show
a useful consistent advantage, so no additional runtime graph cache was added.

The wide prototype was rejected as an automatic policy: packing FP16 outputs
above 128 rows and FP32 outputs at 1024+ rows lost to the dense alternatives.
Both wide model runs also changed the 2924-token prompt's output despite
passing numerical operator tolerances. The final policy restores exact token
agreement. It retains the wider native ABI for correctness testing without
selecting those losing shapes automatically.

The SQL prompt's MTP acceptance schedule can differ after the FP16 prefill
change while final token IDs remain identical (71 versus 72 accepted draft
tokens in one 256-chunk comparison). Do not attribute every small decode-rate
difference solely to kernel execution time.

The new suites passed 232 packed-prefill and 143 paired-MLP checks. Existing
suites passed 408 small-M, 48 high-bit, 929 packed-mid and 56 dense-WMMA checks:
**1,816 GPU checks total**. Both CPU suites passed all 542 tests: Windows had
25 dependency/platform skips; Linux had 10 Windows-only skips.

A normal public CLI launch, with no `--config` or optimization flags, resolved
WMMA, packed-mid, packed-prefill and paired-MLP capabilities automatically.
All 125 generated target-only tokens matched a separate disabled-control run.
The 32 eligible target MLPs recorded 5,056 paired calls, including warmup.
The original GDN default was retained for these public-interface checks.

Live loopback HTTP checks also passed with no optimization flags: a 74-token
prompt exercised 73 prefill rows, streaming and non-streaming returned the
same 112 output tokens, and the stream delivered its completion marker.
Both requests completed without failures. The monitor's stop control produced
a confirmed clean shutdown and launcher exit zero. An initial harness trial
sent SIGTERM directly: the server shut down cleanly, but the monitor correctly
did not classify it as a requested stop. That trial is retained separately;
only the harness shutdown method changed for the successful repeat.

Existing ROCm `SharedSignalPool` teardown warnings appeared in the standalone
GPU checks. No resource guard stopped the accepted measurements. This work
does not claim to fix that pre-existing teardown issue.

The supervised validator has new `packed-prefill` and `mlp-pair` suites.
Prefill checks independent CPU reconstruction across formats, row/tile tails,
long K reductions, buffer guards, streams and changed-input graph replay.
Paired-MLP checks require exact agreement with the separate native operations,
compare each projection with independent CPU weights, and reject malformed
buffers before launch. Real layer-0 gate/up/down tensors supply additional
operator and complete-MLP timing cases, both with warm caches and after a
256 MiB eviction-buffer update. HIP-event intervals include submission gaps;
these are not memory-bandwidth utilization measurements.

Private evidence is under `artifacts/prefill-pair-20260926/`: before snapshots,
frozen baseline records, the model treatment matrix, operator cases,
qualification logs, and CPU test logs. OMP 18.1.16 used the configured
`@default` model for the bounded native implementation; metadata, actual exit
status and logs are under `.codex/.omp-jobs/prefill-pair-20260926/`.
The coordinator reviewed the implementation, added alignment guards, fixed a
compile-time macro issue, and owns integration and hardware qualification.

No result here establishes 27B performance, RDNA4 hardware correctness,
arbitrary-model quality, or 90–95% sustained GPU memory bandwidth.
