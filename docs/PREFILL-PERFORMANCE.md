# Storage and prefill measurements

## Method

This study uses the same local dense 27B Qwen-family EXL3 bytes and verified
native extension on gfx1101. Model copies were SHA-256 checked file by file.
The normal permission gates, GPU lease, Windows adapter telemetry, WSL resource
monitor, and offline loading remain active. No native kernel was changed or
rebuilt for these measurements.

Load profiles use 32768 context, Q8 K/V, and integrated MTP depth 6 with adaptive
confidence 0.6. The total readiness timer starts at model verification; the
weight timer covers both draft and target `Model.load` calls. Host caches and
runtime initialization are not controlled, so these are observed startup
profiles rather than guaranteed cold-start numbers.

| Model location | Both weight loads | Verification to readiness |
| --- | ---: | ---: |
| Windows filesystem on USB HDD | 252.81 s | 324.80 s |
| WSL Linux filesystem, virtual disk on the same USB HDD | 134.41 s | 195.29 s |
| Windows filesystem on NVMe SSD | 17.73 s | 164.52 s |

The NVMe placement reduced weight-loading time by about 14.3x in this comparison.
The inference interpreter, packages, and SDK still reside in the HDD-backed WSL
installation. Import/runtime initialization varied substantially between runs;
placing weights on NVMe does not remove that remaining startup cost. Original
model files and the existing WSL installation were preserved.

A second NVMe run loaded both weight components in 17.76 seconds and reached
readiness in 80.06 seconds. The stable weight-load time and variable overall
startup time are why storage and initialization are reported separately.
Jinja renders the conversation template; it does not accelerate weight I/O.

## Prefill protocol

The baseline serving file was frozen before implementation. It creates a fresh
generator for every request, so repeated test inputs do not receive cross-request
prefix-cache credit. The study uses exact 1024, 4096, 8192, and 16384 token prefixes
of deterministic synthetic Python service code. Input-token hashes verify that
each chunk setting sees identical tokens. Output is limited to eight tokens;
prefill timing excludes the first decode iteration. This measures ingestion,
not answer quality or decode throughput.

Chunk sizes 256, 512, 1024, and 2048 are compared in one loaded process. Each
input/chunk pair runs twice; first-use and repeated results remain separate.
Peak allocation is reset before each request. Short prompts, first-use kernel
initialization, long occupied context, and reused prefixes must not be combined
into one throughput claim.

### Measured chunk sweep

The following are second-trial compute rates in tokens/second, with no prefix
reuse. All 32 requests completed without engine failures. Input hashes and the
eight output token IDs agreed across chunks and trials at each input length.

| Input tokens | Chunk 256 | Chunk 512 | Chunk 1024 | Chunk 2048 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 264.85 | 275.60 | 285.99 | 286.18 |
| 4096 | 264.63 | 285.69 | 289.57 | 292.27 |
| 8192 | 261.11 | 281.60 | 284.90 | 288.04 |
| 16384 | 252.00 | 272.08 | 275.00 | 277.93 |

Serve now defaults to chunk 1024. It improved 4K and 16K compute throughput by
about 9%, while 2048 added only about 1% and another 138 MiB of peak Torch
allocation. At inputs of 4K or more, measured peak allocation was 9369.9 MiB
for chunk 256, 9464.4 MiB for 1024, and 9602.4 MiB for 2048. These allocator
figures do not include all device residency. Non-server modes retain chunk 256.

The first 1K request at chunk 256 achieved only 75.96 tokens/second; its repeat
achieved 264.85. Small or first-use samples must not be used as the sustained
prefill rate. This study does not reproduce 750–2000 tokens/second or establish
equivalence with a different quantization/backend. Chunks above 2048 were not
tested. The plateau motivates subsequent GPU profiling; it does not identify
a particular native kernel as the bottleneck.

## Prefix-reuse checks

Prefix reuse has a separate correctness and latency protocol: compare generated
token IDs with reuse off and on for identical, appended, and changed inputs;
check cancellation recovery and seeded sampling; exercise MTP both enabled and
disabled. Cached tokens are reported separately from computed prefill tokens.
First-token latency measures the user-visible benefit, including checkpoint
restoration. CPU fixtures alone do not establish these GPU properties.

Both GPU configurations passed all 12 cases: cache-off references, identical
and appended hits, early-prefix misses, cancellation and recovery, and seeded
sampling with fresh request state. Output token IDs matched the corresponding
cache-off references. The repeated 4096-token prompt reused 3840 tokens and
computed 255; the last prompt token is the first decode input.

| Configuration | 4K cache miss, first token | 4K cache hit, first token |
| --- | ---: | ---: |
| MTP depth 6, confidence 0.6 | 14.29 s | 1.18 s |
| MTP disabled | 14.00 s | 1.05 s |

These are cache-reuse latency gains, not an increase to raw prefill throughput.
Each run completed 11 requests plus one deliberate cancellation, with zero
engine failures. The live recurrent stash stayed below its 1 GiB bound. The
native runtime printed a `SharedSignalPool` signal-leak warning at process
teardown; addressing native resource lifetime is outside this change.

## HTTP code workload at 120064 context

The final server used Q8 K/V, long attention, MTP depth 6/confidence 0.6,
chunk 1024 and prefix reuse. Nine production-API requests used actual repository
Python code, the saved chat template, thinking disabled, greedy sampling, and
32 output tokens. Each input size began with a distinct system prefix, ensuring
zero cache credit on its first request. All four repeated outputs matched.

| Prompt tokens | Fresh compute, tok/s | Fresh first token | Repeated first token | Reused tokens |
| ---: | ---: | ---: | ---: | ---: |
| 1037 | 173.69 | 6.06 s | 0.50 s | 1024 |
| 3785 | 277.16 | 13.89 s | 1.03 s | 3584 |
| 7048 | 281.47 | 25.29 s | 0.89 s | 6912 |
| 14992 | 274.56 | 54.86 s | 0.95 s | 14848 |

The 1037-token request was the first large-prompt request in this process; its
shape and first-use costs differ from the warmed exact-length chunk sweep.
A follow-up chat turn with 7102 prompt tokens reused 6912 tokens and reached
its first token in 0.98 seconds. The server stayed healthy, with zero engine
failures and 886.5 MiB of live recurrent checkpoints under the 1 GiB bound.
This validates the configured context capacity with occupied prompts up to
14992 tokens, not a fully occupied 120K prompt or broad answer quality.

The public controls and timing definitions are in [SERVING.md](SERVING.md).

The subsequent native projection optimization and its separate measurements
are recorded in [GPU-PERFORMANCE.md](GPU-PERFORMANCE.md).
