# Release validation

## mul1 and RDNA4 update: 2026-09-16

Implementation proceeded in two stages with bounded OMP assistance: mul1
small-M support first, then the shared RDNA4 WMMA adapter. Native builds used
the existing ROCm 7.2.4 environment and fresh output directories. CPU test
results and cross-compilation are not RDNA4 hardware validation.

- Linux: 452 tests, 442 passed and 10 Windows-only skips. Windows: all 67
  launcher tests and four new validator contract tests passed. CLI help and
  whitespace checks passed. The Linux suite includes model storage/codebook
  audit checks, legacy-binary decode-fusion fallback regressions and bounded
  conversion-error metrics against the original full-tensor formulas.
- The mul1-stage gfx1101 build passed 408 packed projection/graph cases and
  56 prefill GEMM checks. Coverage includes cb0/mul1, bits 2/3/4, rows 1–9,
  FP16/FP32 output, all three small-M variants, split-K, non-default streams,
  canaries, external capture and native MLP graph pointer patching. Packed
  decode matched the independent CPU reader exactly; worst projection/MLP
  relative L2 error was 0.000927 (gates 0.005/0.01).
- Standalone WMMA probes compiled for gfx1101, gfx1200 and gfx1201. Extracted
  RDNA4 code objects contain native FP16, BF16 and INT8 WMMA instructions.
  On gfx1101, all scalar-reference primitive checks and 2,304 checks of the
  actual RDNA4 input/accumulator adapters passed. Each conversion direction
  is checked independently, including integers above FP32's exact range.
- A fresh full extension compiled and linked all 110 source units for gfx1101,
  gfx1200 and gfx1201. All three code objects were extracted and confirmed;
  all 327 recorded native source/build fingerprints matched the final build.
  The final multi-target binary passed the same 408 packed/graph and 56
  prefill checks plus the primitive suite through the public guarded validator
  on gfx1101. Both validator and monitor exited successfully.
- The local dense Qwen-family 27B model matched the previously registered
  binary's input hashes and all 160 output tokens across a 32-token warmup
  and 128-token generation case. Settings: context 4096, Q8 K/V, MTP depth 4,
  WMMA register-B small-M and WMMA prefill. This is a bounded cb0 model
  regression, not a mul1 model test or throughput study. The accepted binary
  was registered locally with the previous registration retained as a backup.

- A complete BF16-source 27B mul1 model was converted to 9.402 decimal GB,
  including MTP and metadata. All 409 mul1 markers were verified. Large-head
  error reporting completed with bounded memory, and its packed head matched
  the pre-fix weights byte for byte. All 14 comparison processes and their
  monitors exited successfully; repeated speed-run output tokens matched.
- With the same mul1 weights, fused versus older fallback decode measured
  31.90 versus 12.37 tokens/s on SQL and 58.15 versus 17.72 on binary search,
  with exact output-token agreement. Both models recovered all three values
  from 119,808 occupied input tokens. Fixed 128-token coding decode at that
  occupancy measured 25.51 / 31.56 tokens/s for baseline/candidate.
- The candidate failed the first-occurrence coding task on duplicate inputs;
  the baseline passed. Target-only and older-binary controls reproduce the
  candidate's incorrect answer. Its head uses 3 bits versus the baseline's 4,
  and calibration is only 4 x 512 tokens. This is not a quality-equivalent
  replacement or a BF16-fidelity claim. See [MUL1-VALIDATION.md](MUL1-VALIDATION.md)
  for the complete protocol, quality review, memory measurements and limits.

The existing ROCm `SharedSignalPool` teardown warning remains reproducible
with baseline and candidate tests. No RDNA4 device was available. RDNA4
numerical inference, performance, sustained serving, clean installation and
broader model/MoE validation remain outstanding.
See [GPU-COMPATIBILITY.md](GPU-COMPATIBILITY.md) for the exact envelope,
implementation references and tester commands.

## Full CLI follow-up: 2026-09-14–15

The decode-first follow-up retained a small MTP host-readback reduction and
revised the guarded WMMA prefill kernel. OMP's default model assisted bounded
implementation and investigation; no new security-agent review was performed.

- Linux inference environment: 426 tests, 416 passed, 10 Windows-only skips.
- Windows: all 67 launcher tests passed. The earlier broad Windows test attempt
  had four dependency import errors (NumPy/FastAPI unavailable) and 18 skips;
  it is not reported as a passing full suite.
- Revised native binary: all 56 GPU operator checks passed, including both
  sides of the new 512-row dispatch boundary, partial tiles, long reductions,
  subnormals, strided canaries, invalid inputs, streams and graph replay.
- Full CLI at capacity 120064, Q8, long attention, chunk 1024, MTP 6/0.6 and
  prefix reuse: all 11 baseline/revised input hashes and raw output sequences
  matched. Warm prefill was approximately 556 / 531 / 473 tok/s at occupied
  prompt lengths 3874 / 15391 / 30706. A same-session 15K control confirmed
  a modest 3.3% prefill gain; decode was effectively unchanged.
- Chunk 512/2048 each matched all five shared request outputs. A combined
  decode-flag test matched all 10 outputs but offered no aggregate speed gain.
- The revised binary passed 11 separate HTTP reasoning, streaming, tool and
  lifecycle checks. Health ended with 18 completions, one deliberate
  cancellation and zero engine failures, followed by clean shutdown.
- Exact fresh-versus-cache-repeat output **failed at 30K on both kernels**
  after 241 matching generated tokens. Verification-round traces differ near
  token 15; numerical causality remains unproven. The original client assertion
  failed despite a clean server exit. This limit is retained in the report and
  does not count as a passing exact-reuse check.
- Final source fingerprints matched the built native extension, the registered
  binary hash matched, and the public-file path/credential-pattern scan was
  clean. Benchmark servers stopped and released the GPU lease.

See [RUNTIME-PERFORMANCE.md](RUNTIME-PERFORMANCE.md) for the full CLI protocol,
flag comparisons, recommended command and remaining limits. This does not
validate a fully occupied 120K prompt, broad answer quality, clean installation
on another machine or the absence of every security issue. The pre-existing
native probe `SharedSignalPool` teardown warning remains unresolved.

## Native prefill update: 2026-09-13

The optional `--prefill-gemm wmma` path was built and exercised on gfx1101 in
the existing ROCm environment. A full native build was followed by incremental
candidate builds that checked source/header, compiler-setting and reused-object
fingerprints. The final binary was hash-verified before inference. This does
not establish a clean package installation or compatibility with other GPUs.

- Linux inference environment: 419 tests run, 409 passed, 10 Windows-only skips.
- Windows: 64 launcher tests and 11 native optimization contract tests passed.
- Final native binary: all 39 GPU operator checks passed, including numerical
  references, long reductions, exact subnormals, strided-output canaries,
  fallbacks, malformed arguments, aliases, non-default streams and graph replay.
- Controlled BLAS/WMMA model comparison: all 11 input hashes and output token
  sequences agreed. Warm prefill improved 1.86x at 4K and 1.79x at 16K without
  prefix reuse. The same final binary was used for both implementations.
- Runtime flags: 112 requests across 12 configurations at capacity 32768,
  followed by 33 requests across three configurations at capacity 120064.
  All 33 larger-capacity outputs agreed with the control. The selected command
  uses chunk 1024, WMMA prefill, MTP depth 6/confidence 0.6 and Q8 K/V.
- Final production HTTP: 11 reasoning/SSE/tool/lifecycle checks, nine code and
  prefix-reuse requests, three browser-boundary rejection checks and a seeded
  sampling/reuse check passed. Health ended with 18 completions, one deliberate
  cancellation and zero engine failures. Explicit shutdown was confirmed clean.
- Live socket snapshots showed a loopback listener and the incoming local test
  connection, then only the listener at idle. No outbound connection was observed.
  Public-file scans found no personal workstation paths or credential patterns;
  private models, binaries, installation records and artifacts remain ignored.
- The security-engineer review found no confirmed reportable vulnerability in
  the native dispatch, runtime propagation or HTTP/prefix boundaries. This is
  a bounded review, not proof of universal memory safety or absence of leaks.

See [GPU-PERFORMANCE.md](GPU-PERFORMANCE.md) for the profile, flag sweep,
serving acceptance and limitations. The existing ROCm `SharedSignalPool`
teardown warning remains reproducible with both BLAS and WMMA.

## Storage and prefill update: 2026-09-13

Validated weight placement, chunk sizing, and opt-in process-local prefix reuse
on the same dense 27B Qwen-family EXL3 model and gfx1101 environment. The native
extension stayed unchanged and hash-verified. No native build was performed.

- Linux inference environment: 415 tests run, 405 passed, 10 Windows-only skips.
- Windows: all 63 launcher and 32 resource-limit tests passed. CLI help exposes
  the mode-specific chunk defaults and prefix-reuse option.
- GPU chunk matrix: 32 requests across 1K/4K/8K/16K prompts and chunks
  256/512/1024/2048, with no engine failures. Identical input hashes and eight
  output token IDs agreed across configurations and repeats.
- GPU prefix checks: all 12 cases passed with MTP enabled and all 12 passed
  with MTP disabled. Coverage includes cache-off references, identical/appended
  reuse, early-prefix misses, cancellation recovery, and seeded output agreement.
- Production HTTP at 120064 context: all 11 reasoning/SSE/tool/lifecycle checks
  and all nine code-workload requests passed. Four repeated code prompts gave
  identical greedy output; a follow-up chat turn reused the shared prefix.
  Final health reported 16 completed requests, one deliberate cancellation,
  zero engine failures, and bounded live checkpoint memory.

Serve defaults to chunk 1024; other modes retain 256. Prefix reuse defaults to
off and was explicitly enabled for the HTTP acceptance. Measured fresh prefill
improved about 9% in the chunk sweep; reused prompts showed much larger latency
reductions without inflating compute throughput. See the full
[storage and prefill measurements](PREFILL-PERFORMANCE.md) for timings and limits.

The configured 120064 capacity was exercised with prompts up to 14992 tokens;
this is not validation of a fully occupied 120K context. Generated outputs were
short and do not establish broad quality or sampling equivalence. The native
runtime emitted a `SharedSignalPool` teardown warning on the completed study
processes; native resource-lifetime investigation remains separate work.

The final security-engineer review found no reportable vulnerabilities in this
bounded change. Live Host/Origin/content-type rejection probes passed. The exact
runtime process owned one loopback listener and no established or other network
sockets at observation. This is not a guarantee against all data leakage. Prefix
reuse is intended for trusted local clients: cache timing and token counts can
reveal whether a guessed prefix was seen earlier, and logical invalidation is
not secure memory erasure. Keep reuse off across mixed-trust local clients.

## Runtime update: 2026-09-13

Validated the updated launcher, HTTP adapter, reasoning channels, sampling, and
console telemetry using the existing compatible ROCm environment and verified
native extension. No package installation or native rebuild was performed.

### Automated checks

- Linux inference environment: 399 tests run, 389 passed, 10 Windows-only skips.
- Windows: all 61 launcher and 32 resource-limit tests passed, including the
  Windows-specific cases skipped on Linux. Six serving-engine fixture tests also
  passed without loading a GPU model.
- CLI help passed; the removed `--execute` flag is rejected. Installation execution
  permissions, extension verification, resource limits, and the shared GPU lease
  remain enforced.
- Eight targeted security regressions passed, included in the Linux total.
- The full suite could not run in the default Windows Python because its NumPy
  and FastAPI dependencies were unavailable; the complete suite ran in WSL instead.

### Real GPU acceptance

A local dense 27B Qwen-family EXL3 model ran on an AMD Radeon RX 7800 XT
(gfx1101), with Q8 K/V cache and integrated MTP depth 6, confidence 0.6.

| Configuration | Result |
| --- | --- |
| 4096 context, default attention, allocator fraction 0.85 | 11 acceptance checks passed; 9 completed requests, one deliberate cancellation, zero engine failures; explicit clean shutdown |
| Seeded sampling on the 4096 server | Two requests with temperature, top-k, top-p, min-p, and penalties produced identical output with the same per-job seed |
| 120064 context, long attention, allocator fraction 0.90 | Loaded successfully; three short requests passed nonstream reasoning, SSE reasoning, and thinking-off checks; healthy afterward |
| 120064-context load | 145.93 seconds; 12568 MiB Torch reserved memory |

The 4096-context checks also covered truncated reasoning, rejected template
options, raw completions, structured tool calls, reasoning history on a tool-result
turn, and cancellation recovery. All tool results were synthetic, fixed data;
no generated command was executed. A warmed short request on the larger-context
server reported 71 prefill tokens in 0.484 seconds and 135 decode tokens in
2.536 seconds. Decode accounting excludes the first emitting MTP iteration.
These observations are smoke-test measurements, not a throughput benchmark.

The local llama.cpp build also loaded the corresponding 27B IQ3_M GGUF and
completed normal and streamed chat requests. Both exposed separate
`reasoning_content`; its terminal reported prompt and decode timings. It was
stopped after comparison. Different quantizations and decode configurations
prevent an apples-to-apples performance comparison.

### Security and privacy

The final security-engineer review led to two fixes: reject foreign Host/Origin
headers and non-JSON browser POSTs before request admission, and keep engine
exception details out of public HTTP/SSE errors. Live probes after inference
confirmed health 200, foreign Host 400, foreign Origin 403, and text/plain POST
415. The runtime process had one loopback listener and no established sockets
at the final observation. Offline loading and data-only generated tool calls
remain in place; routine serving status contains no prompt or completion text.

This is a bounded review and point-in-time network observation, not proof of
absence of every possible data leak. Other local processes remain unauthenticated
by design. Private artifacts inherit the volume's existing multi-user ACLs;
ignored files are not an access-control boundary. Raw evidence stays in ignored
local artifacts.

### Limits

The 120064 setting validates allocation and short requests, not a fully occupied
120K prompt or long-context answer quality. CPU MoE flags are wired to the existing
backend and reject incompatible dense models, but actual MoE execution was not
validated: this EXL3 model is dense. Other models, templates, GPU architectures,
broad sampling quality, fresh builds, and prolonged serving require separate
validation. OpenCode's reasoning settings were updated and the compatible API
was verified; the OpenCode UI itself was not exercised.

## Earlier source-export validation: 2026-09-11

The following records the earlier existing-environment source export smoke test,
not a clean-install or broad compatibility certification.

## CPU checks

- Windows: 355 tests, successful, 14 skipped.
- Linux inference environment: 355 tests, successful, 10 skipped.
- CLI help works without model loading or installation metadata.

Coverage includes launcher boundaries, model-free registration, resource limits, telemetry selection, cache/tool protocols, timing, HTTP behavior and socket lifecycle. Platform/dependency skips remain explicit. New checks exercise different detected GPU/RAM totals, configurable stop thresholds, missing/ambiguous adapter handling, allocator forwarding, and persistent-server timeout behavior. OMP's default model assisted implementation; final review and acceptance used the actual exported files.

## GPU server smoke

The public source loaded a local 27B Qwen-family EXL3 model on gfx1101 through an existing compatible ROCm environment and a previously verified native extension. No weights or binary are distributed.

| Setting / observation | Result |
| --- | --- |
| Context / cache | 4096 / Q8 K and V |
| Integrated MTP | Maximum 6, adaptive confidence 0.6 |
| Attention profile | Default |
| Torch allocator fraction | 0.85 for this bounded smoke |
| Model loads | 1 |
| Generation requests completed | 14 of 14 |
| GPU engine failures | 0 |
| Readiness | 58.93 seconds |
| Entire acceptance including shutdown | 95.33 seconds |
| Shutdown | Explicit stop; clean shutdown confirmed |

Checks covered basic generation, four repeated raw completions, multi-turn chat, named/required function calls, SSE, tool-result turns, multiple/reordered results, deliberate truncation errors and subsequent recovery. The two deliberately invalid tool outputs produced the expected protocol errors while the engine stayed healthy. No model-generated commands were executed.

The worker recorded `timeout: null` for serving. Resource guards stayed active; the smoke's external supervisor imposed its own bounded acceptance deadline and stopped the server explicitly. This does not reintroduce a server lifetime cap.

## Export review

The public tree excludes models, compiled binaries, private installation/configuration and build records, raw logs, reference/calibration corpora, developer jobs and research handoffs. Public documentation links were checked. A file-content scan checked for local identifiers/paths and credential patterns. Source files and recorded local configuration in the original runtime were fingerprinted before and after export and matched.

Raw acceptance logs, responses, environment details and fingerprints remain private. They are not required to launch the source release.

## Remaining limits

No new native build, package installation, throughput optimization, full 98K/120K run or BF16 comparison was performed for this export. The smoke reused an existing compatible binary; it does not establish that a fresh full build reproduces it. Other GPU architectures, other models/templates, broad quality and prolonged serving need separate validation. Q4/MTP exact equivalence remains unresolved. No 60/80 tok/s or universal harness/model claim follows from this acceptance.
