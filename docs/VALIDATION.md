# Release validation

## Decode profiling and narrow dense GEMM: 2026-09-28

RX 7800 XT / gfx1101, WSL `EXL3-Runtime`. Details in
[DECODE-PROFILING.md](DECODE-PROFILING.md).

- rocprofv3 1.1.0: kernel/HSA tracing unavailable on this WSL stack (no KFD
  topology; the WSL HSA runtime does not register with rocprofiler-sdk). HIP API
  tracing works after forcing one rocprofiler-sdk instance. Per-kernel time
  comes from `scripts/profile_decode_round.py` (event-based, GPU pre-queued).
- New build `narrow-v2`, SHA-256
  `f3c131266a3f8461bdd0b7f9bec36c0c986225b9e072fb1eeeb19923d9370047`: native
  suites narrow-gemm 119, hgemm 56, smallm 408, highbit-smallm 60, mlp-pair 143,
  packed-mid 929, packed-prefill 232, head-tiled 185, gdn-recurrent 75 and
  attention-schedule 116, all passed.
- MiMo 9B K5/H6 mul1, BF16 MTP 2, Q6/Q6, 131072 context, greedy, bracketed
  baseline/candidate/baseline: decode 77.6–78.1 → 90.7 tokens/s short and
  64.9–65.3 → 76.1 at 16K occupied (+16%). Unroll-only +0.8%.
- Greedy outputs: identical to baseline at short, 4K and 32K occupied for 256
  tokens; at 16K they diverge at a near-tie (token 114 without MTP, 40 with MTP)
  into equivalent text. The narrow GEMM changes accumulation order; unroll is
  bit-identical.
- Qwenseek 27B (GDN decode fusions, quantized MTP 2, Q6): within control drift,
  identical outputs.
- Serving smoke with the candidate (sampled 0.6/0.95/20 with speculative
  sampling, and greedy): 4 requests completed, repeated greedy outputs identical,
  clean shutdown (`RUN-20260928T205830Z-0726dd0b`).
- Unit suite: Windows 863 tests OK (66 dependency/platform skips); Linux
  runtime 863 OK (12 skips).
- Not validated: other GPUs, sampled-decode throughput, other models with
  narrow dense projections.

## Ideas 4–6 combined on the promoted binary: 2026-09-28

One MiMo 9B serving process with every default (`narrow-v2`, SHA-256
`f3c131266a3f8461bdd0b7f9bec36c0c986225b9e072fb1eeeb19923d9370047`; BF16 MTP
2, Q6/Q6, 131072 context, prefix cache on; `RUN-20260928T210253Z-0ac321a6`):

- **Startup:** warm-up 5.3 s, with 111 attention variants and 221 Triton
  specializations cached.
- **Decode:** 84.3 tokens/s with the serving sampler (8 × 512-token chat
  requests) and 87.4 greedy.
- **Prefill:** staged, 1757 tokens/s at 8.8K and 1088 tokens/s over 69K →
  128K occupied.
- **Correctness:** speculative sampling ran 1800 rounds with no fallbacks;
  `jit_after_ready = 0`; no failed requests.
- **Unit suites:**
  - Windows: 863 tests OK, 66 skips.
  - Linux runtime interpreter: 863 OK, 12 skips, with the server-deps path
    on `PYTHONPATH`. Without it, the 4 HTTP test modules cannot import
    `fastapi` and the other 761 pass.

## Long-context prefill: staged quantized KV and gfx1101 tiles: 2026-09-28

RX 7800 XT / gfx1101, MiMo 9B K5/H6 mul1, BF16 MTP 2, Q6/Q6, 131072 context,
chunk 1024, prefix cache on; extension SHA-256
`0ad3db65fcda9a59629a81d3a93b85f2af7dd40013207ced7088b19a6b0c132a`.
Details in [PREFILL-ATTENTION.md](PREFILL-ATTENTION.md).

- **Operator** (`benchmark_exl3_prefill_attention.py`, head_dim 256, 16/4
  heads):
  - The new tiles are 2.0–2.5× faster than the inherited tile at 17–2048
    query rows and 8K–128K context.
  - Relative L2 against FP32 is 2.8–3.0 × 10⁻⁴ for both, the FP16 output
    floor.
- **Full model** (prefix-hit spans of about 17K tokens):
  - 1008 against 463 tokens/s at 122K occupied context.
  - 1614 against 1126 at 35K.
  - 1742 against 1578 at 8.8K.
  - Chunk 2048: +1–2% for +248 MiB, so it was not adopted.
- **Greedy output:**
  - The first token after nine prefixes (8.8K–128K) was identical.
  - A 64-token continuation was identical at 77K. At 26K it diverged at a
    wording near-tie after about 45 tokens.
- **Memory:** reserved 9170 against 8658 MiB (the scratch).
  `jit_after_ready = 0` (111 warmed attention variants).
- **Not validated:** other GPUs, head sizes other than 256, Qwenseek 27B,
  quantized-cache queries under 256 rows (unchanged direct path), and cache
  formats other than Q6 end to end.

## Speculative sampling for MTP drafts: 2026-09-28

Same GPU, model and configuration. Details in
[SPECULATIVE-SAMPLING.md](SPECULATIVE-SAMPLING.md).

- **Unit coverage:** `tests/test_spec_sampling.py`, 20 tests. The WSL
  runtime interpreter runs them all, including the GPU Triton-against-reference
  test.
- **Distribution:** seeded 1500-request comparisons against MTP-off sampling
  (permutation χ²):
  - T=1.0 / top-k 50, fixed version: p = 0.22–0.93 at positions 1–6.
  - Spec off matches MTP off token for token.
  - A first version with whole-round tie fallback showed p = 0.002; it was
    fixed by the per-position handoff.
- **Speed (same session):**
  - Serving sampling: 73.1 against 72.8 t/s (neutral).
  - T=1.0 / top-k 50: 68.7 against 62.4 t/s (+10%).
  - Greedy output: token-identical with spec on and off.
- **Draft depth:** re-sweep under sampling keeps depth 2. Depth 3 runs at
  74.6 ms/round without native 4-row kernels; depth 4 at 39.1 ms/round for
  2.67 tokens.
- **Not validated:** batched sampled MTP, `--draft-confidence`, n-gram,
  DFlash or separate draft models, and models other than MiMo 9B.

## Cold start: autotune cache, warm-up and NVMe runtime: 2026-09-28

Validated on RX 7800 XT / gfx1101 with MiMo 9B K5/H6 mul1 (BF16 MTP 2, Q6/Q6,
131072 context, chunk 1024, prefix cache on); extension SHA-256
`0ad3db65fcda9a59629a81d3a93b85f2af7dd40013207ced7088b19a6b0c132a` unchanged.
Details in [COLD-START.md](COLD-START.md).

- Windows unit suite: 830 tests successful, 46 dependency/platform skips.
  New coverage: `tests/test_warmup.py`, `--warmup` forwarding/rejection and
  `TRITON_CACHE_AUTOTUNING` in `tests/test_launcher.py`.
- Warm-up on: `jit_after_ready = 0` in every run, including prefix-hit
  growth to 72K occupied context. Warm-up off: 33 late specializations
  (two runs).
- Greedy outputs: identical across warm-up-on processes and across
  warm-up-off processes. On vs off matched except on an off process's first
  request, which matched once preceded by a short request (per-process prefill
  `num_stages` timing pick).
- Cold-VM startup to ready: 143.9/146.5 s on the HDD-hosted distro and
  39.8/37.6 s on the NVMe `EXL3-Runtime` distro, including warm-up.
- Not validated: vision and native-attention/BC graph warm-up coverage,
  Qwenseek 27B, other cache formats and contexts. The first launch after a
  new extension or configuration still compiles its variants during warm-up.

## Native kernel integration into main: 2026-09-27

The three qualified native-kernel stages and their public validators, reports
and launch guidance were checked together before publication:

- Windows: 807 tests successful, with 46 dependency/platform skips.
- Linux inference environment with the registered HTTP dependency overlay:
  807 tests successful, with 10 Windows-only skips. An initial run without
  that overlay could not import four HTTP test modules because FastAPI was
  unavailable; using the existing configured overlay resolved those errors.
  No packages were installed.
- CLI help, native-checker help, syntax parsing of all six changed Python
  files, diff whitespace and 125 local Markdown links passed.
- The public-file audit found no model weights, compiled binaries, private
  installation records, workstation paths or credential-pattern matches.
- All 332 recorded native-build source fingerprints match both the working
  files and the staged Git contents. The registered binary's SHA-256 remains
  `0ad3db65fcda9a59629a81d3a93b85f2af7dd40013207ced7088b19a6b0c132a`.

The integration also fixes Chat Completions clients that explicitly send
`store: false`. JSON and SSE regressions pass; unsupported persistence values
are rejected before the engine runs, and model/unknown-field validation stays
strict. Installed OpenCode 2.0.18 received a streamed response using its actual
`store: false` payload against an isolated CPU fake engine. Its auxiliary
`reasoning_effort` request remains unsupported; the client's default fallback
succeeded. This checks request/stream interoperability, not live model output.
The running inference server was not restarted.

The GPU correctness and model-performance evidence below applies to this
unchanged native source. Publication did not rebuild the extension or repeat
those GPU benchmarks. Other installations must build and register a compatible
extension to use the new kernels; pulling source does not replace a registered
binary. Raw integration evidence stays in ignored
`artifacts/main-native-integration-20260927/`.

## Native kernel follow-up: 2026-09-27

Stage one is qualified and primary on the local RX 7800 XT / gfx1101 runtime:
specialized K5/K6 unpackers and conservative format-qualified narrow-row loop
unrolling. Both the initial and refined binaries passed 2,013 native checks.
The Windows unit suite completed 804 tests with 46 dependency/platform skips.
The highbit validator adds an independent packed-bit boundary regression.

The refined build preserved all 13,824 measured output tokens and exact MTP
draft windows across bracketed 9B and 27B comparisons. MiMo decode improved
4.8–8.8% against the faster control; the 27B result stayed within control
variation and is not a claimed speedup. The lower-bit loop regression from
the initial candidate was removed before promotion. The stage-one binary
had SHA-256 `8e9120b06a67d095b59851f066b3094a7409eea40c0c07678aec34ab9283cab9`.

Stage two is also qualified and primary: next-tile register staging in packed
prefill, selected automatically by format and geometry. The slower dense-tile
experiment was reverted. The refined and final builds each passed 288 native
GPU checks; all 216 real-weight operator cases per model were byte-identical
across the candidate and both controls. All eight 9B and seven original 27B
complete-model cases matched inputs, tokens and MTP windows exactly: 135
measured continuations / 17,280 tokens.

The original 27B 256-token case diverged between two unchanged-baseline
processes; the candidate matched the first control. It is excluded from speed
claims. A separate baseline/candidate/baseline diagnostic matched all nine
continuations / 1,152 tokens and draft windows. The original discrepancy's
cause is unresolved and remains a reproducibility limitation.

Short-prompt prefill improved 9.4–11.5% on MiMo and 10.0–12.1% on Qwenseek
against the faster control. Long-prompt prefill was largely unchanged; this
is not a new decode-speed claim. Small decode and total-time regressions in
individual cases are retained in the detailed report. Final line-ending cleanup
preserved GPU instructions/constants and host instructions/relocations exactly;
compiler-generated internal identifiers changed. The stage-two source-matched
primary included both stages and had SHA-256
`b286d3fbe4752425868a7c363893e19777e90d40d7560236fe913a065444e6c2`.
CLI help and diff whitespace checks passed.

Stage three is also qualified and primary: compact 128-thread GDN recurrence
for ROCm, batch one, 128-by-128 heads, at most 64 value heads and more than one
token. Its 75 recurrent and 60 convolution checks passed against independent
CPU references, including state/history, rewind, graph replay and rounding.
The final 9B/27B comparisons preserved all 108 measured continuations / 13,824
tokens and exact MTP draft windows. This total includes a reused preceding 27B
control, followed by a new candidate and a fresh control.

Qualifying MTP decode cases improved 0.86–1.32% on MiMo and 2.68–2.89% on
Qwenseek against the faster controls. Other cases were effectively unchanged
or within control variation; 27B 16K does not establish a gain. Small total-time
regressions and all case results are retained in the report. This stage does
not claim a general prefill speedup. Broader single-token replacement,
cross-token state retention and width-four convolution were rejected after
model regressions despite favorable isolated timings.

The current source-matched primary includes all three stages and has SHA-256
`0ad3db65fcda9a59629a81d3a93b85f2af7dd40013207ced7088b19a6b0c132a`.
All 14 preserved GDN kernel instruction bodies match the previous primary;
none of the 1,574 inspected selected specializations spills to private memory.
A fresh check through the promoted default registration passed all 75 recurrent
tests. Four focused validator CPU tests, CLI/checker help and diff whitespace
checks also passed. The full 804-test suite was not repeated for stage three.
See the [native kernel report](NATIVE-KERNEL-PERFORMANCE.md) for exact shapes,
model settings, measurements and limits. These results do not establish
other-GPU compatibility or hardware bandwidth saturation.

## Residual fusion and larger MTP graphs: 2026-09-26

Fresh profiles on the optimized MiMo 9B BF16-MTP/F16-cache and Qwenseek 27B
quantized-MTP/Q6-cache configurations preceded the next fusion and dispatch
experiments. Both ran on RX 7800 XT / gfx1101 using the existing verified
head/attention ABI-3 binary. No native rebuild was needed.

- The native residual/RMSNorm qualification passed 324 bit-exact comparisons.
  FP32 normalized output and FP16-residual/FP32-update combinations remain
  outside the new fusion's qualified envelope.
- Six completed comparison suites preserved all 28,672 output tokens across
  224 measured continuations, including exact MTP draft windows. Warmups and
  early trials that failed to activate the candidates are excluded.
- Residual fusion and batched readback did not establish useful repeatable
  gains. The wider graph improved warmed 9B decode by 0.41–0.79%, but fresh
  captures added first-token latency without a consistent total-request win.
  The 27B graph results were neutral or slower. No candidate was promoted.
- The final focused Linux suite passed 36 tests. After archiving and removing
  the experimental hooks, the primary runtime passed 804 tests on Windows
  (46 dependency/platform skips) and 804 on Linux (10 Windows-only skips).
  CLI help and diff whitespace checks passed.

The runtime source and registered binary remain unchanged by this round;
the retained repository changes are documentation. GPU profiling returned
inclusive event intervals but no usable kernel self-time or bandwidth
counters. These results do not establish memory-bandwidth saturation, broad
model compatibility, or a benefit from a persistent scheduler. Full tables,
capture costs, numerical limits and private evidence locations are in the
[investigation report](FUSION-DECODE-PERFORMANCE.md).

## Consolidated runtime integration: 2026-09-26

The pending packed-kernel/cache changes and the complete MiMo branch chain
were combined in one checkout, including conversion and packaging tools,
BF16 MTP, rowwise verification, optional vision, and their tests and reports.
Merge resolution preserves the qualified native extension sources exactly;
the registered head/attention ABI-3 binary was reused without rebuilding.
Validated packed kernels and attention schedules remain automatic, with
unsupported cases retaining their guarded fallbacks.

- Windows: 804 tests successful, with 46 dependency/platform skips.
- Linux inference environment: 804 tests successful, with 10 Windows-only skips.
- CLI help, syntax parsing of all 91 changed Python files, diff whitespace,
  and public-file credential/workstation-path checks passed.
- Ordinary MiMo CLI generation with BF16 MTP depth 2, rowwise verification,
  F16 cache and context 4096 preserved all 64 output IDs against diagnostic
  kernel controls. The primary recorded 128 packed-head calls; the control
  recorded zero. Both exercised 336 rowwise attention windows.
- The MiMo BF16-MTP vision server passed image JSON/SSE output agreement,
  text requests before/after the image, and remote-image URL rejection.
  Health recorded four completed requests, zero failures/cancellations,
  and 52 packed-head calls.
- The 27B Q6/MTP-depth-2 server processed the same 14333-token prompt through
  JSON and SSE, producing matching 32-token responses. Health recorded two
  completed requests, zero failures/cancellations, 108 packed-head calls,
  and 438 automatic attention calls.
- Both servers shut down cleanly through their stop files; every inference
  launcher exited zero. The temporary WSL keepalive was stopped afterward.

These are integration checks, not new throughput or broad-quality benchmarks.
The earlier 2117 native checks and 716 attention-oracle comparisons remain
documented in [the head/attention report](HEAD-ATTENTION-PERFORMANCE.md).
The image check used one synthetic color fixture. Existing model, cache-policy,
vision, precision and GPU compatibility limits still apply. Portable source
and reports are committed; raw evidence stays in ignored
`artifacts/main-integration-20260926/`.

## Vocabulary head and occupied-context attention: 2026-09-26

The next two roadmap items have native and model coverage on RX 7800 XT /
gfx1101. Repacked-head ABI 3 preserves FP16/FP32 arithmetic, shares its bounded
compressed view with MTP and selects supported rows automatically. Attention
uses cache format, query/KV geometry and occupied page-table bounds to choose
its schedule. Unsupported cases retain inherited dispatch.

The fresh 110-unit build passed 2117 GPU checks: 185 head, 116 attention graph/
stream checks and 1816 existing projection/prefill/MLP checks. All 332 native/
build fingerprints matched. Attention sweeps passed another 716 independent
FP32-oracle comparisons. Windows and Linux each ran 567 unit tests without
failures, with 25 Windows dependency/platform skips and 10 Linux Windows-only
skips. Four supervised OMP implementation packages completed successfully.

Corrected FP16 head trials preserved all 6912 measured output tokens and MTP
acceptance windows across the 9B and 27B models. The candidates recorded 1932
and 1776 packed-head calls, respectively. Decode gains were 0.44–1.94% for
MiMo BF16-MTP and 0.96–1.97% for the 27B quantized MTP model. Earlier FP32-only
full-model head trials never dispatched and are explicitly invalidated in the
report; their apparent model gains are withdrawn.

At 32K occupied tokens, Q8 attention improved target-only decode by 11.20%
on 9B and 10.80% on 27B. The 9B continuation changed after 84 common tokens;
both inspected binary-search functions passed 10801 examples, and timing the
shared prefix retained an 11.24% gain. This is a narrow functional check,
not proof of unchanged broad model quality. A warmed 27B Q6/MTP comparison
at 32K improved from 28.93 to 37.56 tok/s (29.82%), preserving all 768 measured
tokens and draft acceptance windows. Other attention comparisons and
the final public-interface checks are recorded in the
[complete report](HEAD-ATTENTION-PERFORMANCE.md). Extra future-model geometries
have operator coverage only; no additional GPU family is validated.

The qualified binary is registered locally as primary. Normal MiMo target-only
CLI generation matched 64 tokens against diagnostic controls and reported
96 packed-head calls plus 512 automatic attention calls. The normal 27B MTP2
server returned matching 64-token streaming/non-streaming responses and a
sampled response. Health reported three completed requests, zero failures,
175 packed-head calls and 748 automatic attention calls. Monitor-controlled
shutdown was clean and the launcher exited zero. The 9B BF16-MTP numbers use
the frozen Step4 evaluator; main-CLI MiMo coverage is target-only.

## MiMo tiled prefill and paired MLP: 2026-09-26

The next two roadmap items are implemented and the measured shape policy is
primary. Generation and serving automatically select verified packed prefill,
paired gate/up plus SwiGLU, and dense WMMA fallback. The new gfx1101 extension
passed 1,816 GPU checks; all 331 native/build source fingerprints matched.
Both CPU suites passed 542 tests (25 skips on Windows, 10 on Linux).

The final policy preserved all 8,640 output tokens across 108 measured
BF16-MTP requests with 256- and 1024-token prefill chunks. With 256-token
chunks, 1244/2924-token prompts improved from 865.74/882.83 to
1386.28/1401.85 prefill tok/s versus the original BLAS baseline. Improvements
over the faster WMMA controls are smaller, and decode gains remain modest.
The broader packed policy was rejected for both performance and changed
long-prompt token sequences; its losing shapes retain dense fallback.

The normal public CLI, using the newly registered binary with no config or
optimization flags, selected all three native capabilities and matched 125
target-only output tokens against disabled-control paths. Live loopback HTTP
streaming and non-streaming matched 112 tokens; monitor-controlled shutdown
completed with launcher exit zero. See the
[complete report](PREFILL-MLP-FUSION.md) for exact dispatch bounds, accepted and
rejected measurements, public-interface checks, binary identity and limits.
This does not establish new 27B, RDNA4 or broad model-quality coverage.

## MiMo packed projections: 2026-09-26

The requested MiMo 9B baseline, packed MLP scheduling sweep and packed
10–64-row implementation are complete on gfx1101. A fresh 110-unit native
build passed 1,441 GPU checks; all 329 source/build fingerprints matched.
Windows ran 520 unit tests (25 skips), and the Linux runtime ran the same 520
(10 Windows-only skips), with no failures in the correctly configured runs.

Repeated BF16-MTP model comparisons show about 1.98 times faster prefill on a
41-token prompt with exact output-token agreement. Longer prompts and decode
show no consistent gain. MLP warp overrides were not promoted: four warps
regressed decode, and a separate MLP-only 16-warp diagnostic changed one SQL
completion. Default MLP scheduling remains unchanged. The verified ABI-1 build
was subsequently promoted to the local primary registration; generation and
serving automatically select its packed 10–64-row path. Older binaries retain
fallback, and `--no-packed-mid` is available for diagnosis.

After promotion, both unit suites ran 531 tests successfully (25 Windows
dependency/platform skips; 10 Linux Windows-only skips). A normal MiMo launch
with no config or kernel flag selected packed-mid ABI 1 and matched all 64
output tokens from the earlier explicit packed run. Process and monitor exited
zero. The tested native binary did not change.

The actual main public CLI also passed a held-out target-only comparison with
64 identical generated tokens. See [PACKED-MLP-PERFORMANCE.md](PACKED-MLP-PERFORMANCE.md)
for baseline rates, both successful and rejected experiments, commands, binary
identity, OMP assistance, raw-evidence locations and limits. This is not a new
27B, RDNA4, broad-quality or saturated-memory-bandwidth validation.

## Checkpoint rename retry: 2026-09-24

Two MiMo conversion launches on WSL's Windows-mounted model drive completed
their module writes but failed with `PermissionError` at `ckpt_new -> ckpt`.
Checkpoint saving now retries only permission failures, rechecking sibling
paths and an absent destination before each attempt. Each rename is bounded
to 11 attempts and 7.5 seconds of requested sleep; existing process deadlines
remain in force. Persistent errors still stop conversion. The helper does not
copy/delete payloads or change quantization calculations, calibration or seeds.
The shared ownership contract remains necessary: path checks are not an
OS-level atomic no-replace primitive against arbitrary external writers.

- 16 targeted WSL unit tests passed; Windows passed 13 and skipped three
  unavailable symlink cases. Tests cover retry exhaustion, unrelated errors,
  changed paths during backoff, preserved bytes and preexisting destinations.
- Eight CPU filesystem cases with 64-MiB synthetic payloads did not reproduce
  the error. A 4,294,978,644-byte synthetic payload did: two permission failures
  were followed by success after 0.1/0.2-second sleeps. Independent readback
  matched SHA-256 `8818012e4c68c8ad2ea9bac4bdaea91c7fd6fb10a9afc7431b46aaf1b5fb9cc8`.
- This demonstrates a recoverable permission failure on the tested filesystem;
  it does not establish the underlying OS/handle cause or a universal size
  threshold. Full-model conversion with this helper is still pending.

Source: `vendor/rocm-exl3/exllamav3/conversion/checkpoint_io.py`;
tests: `tests/test_checkpoint_io.py`. Detailed MiMo provenance and retained
failed attempts are recorded in the research repository's step-3 report.

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
