# Packed MLP and 10–64-row projections

Completed on 2026-09-26 on the RX 7800 XT. The new packed path roughly
doubled prefill throughput for the measured 41-token MiMo prompt. It did not
improve longer prefill or establish a decode gain. MLP warp-count tuning did
not qualify a replacement for the existing default.

This work is limited to the requested MiMo 9B baseline, packed MLP scheduling,
and packed projections for 10–64 rows. Persistent/model-wide kernels, attention,
KV changes, larger-prefill GEMM changes and other roadmap items are excluded.

## Implementation

The new noncooperative kernel consumes EXL3 packed weights directly. Each block
computes 16 output columns and up to 16 input rows; it decodes a weight tile,
uses the existing RDNA WMMA primitives, and reduces split-K partials inside the
block. Rows are masked on the final tile. Input/output Hadamard transforms and
scales retain the existing semantics. No full FP16 weight matrix is allocated.
At 64 rows, four row tiles read the packed matrix. This is bounded reuse, not a
persistent whole-model kernel or a claim of saturated DRAM bandwidth.

The runtime automatically enables the path when
`quantlab_exl3_packed_mid_abi() == 1` in the hash-verified binary. Its envelope
is rows 10–64, positive input/output
dimensions divisible by 128, cb0 K2/K3/K4 or mul1 K2–K6, and FP16/FP32 output.
Unsupported shapes/formats and rows above 64 keep reconstruction fallback.
Explicit reconstruction and replacement modules retain precedence. Native MLP
graph eligibility remains at its existing limit; ordinary external stream
capture is a separate validation case.

`--smallm-mlp-warps 4|8|16` experiments with in-block split-K scheduling for
multi-row matrices with K >= 2048 and 2048 <= N <= 32768. This shape envelope
includes MLP projections and can include other projections with those dimensions.
Existing explicit global split-K/head controls take precedence. Default
scheduling is unchanged. This flag also requires packed-mid ABI 1, independently
of whether the 10–64-row path is enabled.

MiMo's existing high-bit dot extension is carried forward from the MiMo runtime:
mul1 K5/K6, rows 2/3/5, dot mode, high-bit ABI 1. This is needed because the
selected MiMo has K5 body projections and a K6 head. Older binaries retain their
old eligibility. The 3/5/9-row installation setting is not increased to 64.

```bash
python run.py -m MODEL_DIRECTORY \
  --decode-fusions off --cache-type f16 -c 4096 -b 256 -n 64 \
  -p "Give three tips for writing readable Python."
python scripts/check_exl3_kernels.py \
  --extension-dir CANDIDATE_LIB --expected-extension-sha256 CANDIDATE_SHA256 \
  --checks smallm highbit-smallm packed-mid hgemm --output artifacts/packed-checks
```

Rebuild in a fresh output directory following [BUILD.md](BUILD.md). A flag cannot
enable these kernels in an older binary. Default commands remain compatible
with old registered binaries. Native source lives under the vendored
`exllamav3_ext/rocm/quant` directory; kernel mirrors are kept synchronized.
Leave `--smallm-mlp-warps` unset for the measured candidate. It is a diagnostic
control, not a recommended speed setting.

The verified gfx1101 build is now the local registered primary binary, so normal
commands need neither a candidate config nor a kernel flag. The original
candidate configuration remains in `artifacts/packed-mlp-20260926/candidate.toml`
for reproducibility. Other installations with older binaries retain automatic
reconstruction fallback. Explicit `--packed-mid` still rejects a missing or
unknown ABI; `--no-packed-mid` disables the packed path for diagnosis. The
public CLI example is target-only; the existing BF16 MTP workflow described
below resides in the MiMo checkout.

## Baseline and protocol

Hardware: RX 7800 XT (gfx1101), 16 GiB, WSL Ubuntu, Torch 2.13.0+ROCm 7.2 and
HIP 7.2.53211, SDK 7.2.4, Triton 3.7.1. Model: retained MiMo 9B 5.0/H6 mul1
package, with original BF16 MTP donor. Text only, vision unloaded, CPU
embedding, F16 KV, context capacity
4096, prefill chunk 256, dot small-M and BLAS prefill, GPU allocator fraction
0.65. No prompt-cache reuse between requests. The retained package contains
25 files totaling 8,566,767,336 bytes, all individually SHA256 fingerprinted.
The target has 32 blocks, hidden width 4096, MLP width 12288, 24 GDN and eight
full-attention blocks, with K5 mul1 body projections and a K6 head.

The BF16-MTP baseline uses the existing MiMo checkout at `b502d37`, depth 2,
its rowwise target verification attention and off decode fusions. Those existing
MiMo features are held fixed, not introduced or retuned in this work. The native
baseline SHA256 is
`74ebb61648ac34c68352dafa0b2fe27f4d99fbaad6d88640f4730de3794fd4d3`.

For the model A/B, a frozen copy of that evaluator uses the same MiMo backend,
BF16 donor and rowwise attention, loading only this worktree's `compat.py` and
`optimizations.py`. Native small-M maximum remains five in this comparison.
This isolates the packed-kernel treatments without importing unrelated MiMo
features into the main runtime. A separate check uses the actual main public
CLI, current backend and target-only generation.

Each measured prefix has a discarded warmup and three fixed 128-token runs.
Decode timing excludes the entire first emitting GPU iteration (timing v2).
Prefill sums synchronized `Job.prefill` calls. Short-prompt rates must not be
extrapolated to long prefill. The final input token is held for the first decode
iteration; thus a 41-token prompt measures 40 prefill tokens. Cold model load,
extension build and per-shape warmups are excluded. All GPU jobs retain the
existing lease, binary verification, offline loading and resource guards.

Initial fresh baseline:

| BF16 MTP depth 2 workload | Input tokens | Median prefill tok/s | Median decode tok/s |
| --- | ---: | ---: | ---: |
| SQL parameters | 78 | 492.84 | 70.22 |
| First-index binary search | 93 | 579.67 | 77.35 |

| Target-only workload | Input tokens | Median prefill tok/s | Median decode tok/s |
| --- | ---: | ---: | ---: |
| SQL parameters | 78 | 512.13 | 47.71 |
| First-index binary search | 93 | 610.29 | 48.07 |

This initial baseline overlapped CPU-only native compilation. A subsequent
quiet control bracket, below, is the basis for candidate comparisons. MTP and
target-only decode rates are different workloads and are not compared as a
kernel speedup.

## Real MLP operator measurements

The layer-zero gate, up and down packed weights were read from the retained
model. Each sweep first checks real 128-by-256 weight slices against the
independent CPU EXL3 reader, then checks complete projections against runtime
reconstruction. Shapes are 4096 to 12288 for gate/up and 12288 to 4096 for down.
Sweeps cover rows 2, 3, 5, 10, 16, 17, 32, 33, 48 and 64, default/4/8/16 warps,
and reconstruction controls for rows at least 10. Each backend has 141 timing
cases and three independent real-weight slice checks.

Reported times include runtime allocation, Hadamard transforms, scaling and,
for the control, full weight reconstruction plus GEMM. Each case has three
warmups and 15 HIP-event samples. Both hot and cache-evicted measurements are
retained. The latter touch a 256-MiB buffer before the measured interval; this
is an eviction experiment, not a measurement of DRAM bandwidth utilization.
BLAS and WMMA reconstruction controls run in separate processes because the
existing GEMM backend selection is cached on first use.

Representative evicted BLAS comparisons, median microseconds (lower is better):

| Projection | Rows | Packed, default warps | Packed, four warps | Reconstruct + BLAS |
| --- | ---: | ---: | ---: | ---: |
| Down | 10 | 292.59 | 226.38 | 997.15 |
| Down | 16 | 293.02 | 227.54 | 1000.22 |
| Down | 32 | 390.97 | 330.66 | 988.42 |
| Down | 64 | 608.57 | 544.09 | 1434.53 |
| Gate | 10 | 276.08 | 260.22 | 858.94 |
| Gate | 16 | 280.33 | 263.04 | 847.30 |
| Gate | 32 | 389.06 | 357.05 | 858.18 |
| Gate | 64 | 627.78 | 590.53 | 1338.53 |
| Up | 10 | 273.94 | 256.86 | 864.09 |
| Up | 16 | 284.58 | 265.31 | 853.28 |
| Up | 32 | 391.33 | 355.22 | 861.53 |
| Up | 64 | 630.22 | 587.49 | 1342.70 |

At 64 rows in the separate WMMA-control process, default packed down/gate/up
took 613.43/629.98/629.19 microseconds versus 1279.91/1021.44/1030.93 for
reconstruction plus WMMA: about 2.09/1.62/1.64 times faster. These are individual
projection measurements, not whole-model prefill rates.

For three-row MLPs, four warps improved evicted down/gate/up times from
270.61/265.61/265.47 to 243.83/249.34/252.13 microseconds. Hot results did not
show the same preference: down was 133.7 microseconds at default, 145.9 with
four warps and 120.4 with 16. The model tests below determine acceptance;
selecting a decode setting from the evicted operator results alone would have
caused a regression.

## Whole-model comparison and tuning decisions

One loaded model ran six treatments in order: control-before, four warps,
16 warps, packed-mid only, packed-mid plus four warps, control-after. Each
treatment used a discarded warmup and three measured runs per prefix. SQL,
binary search and the short greeting generated 128 fixed tokens; the
1,244-token log prompt generated 32. The before/after controls expose drift
within the run rather than assuming identical clocks and cache state.

Packed-mid only, with the existing scheduling heuristic:

| Prompt | Input tokens | Control prefill range, tok/s | Packed prefill, tok/s | Control decode range, tok/s | Packed decode, tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| SQL parameters | 78 | 487.00–495.78 | 489.35 | 69.69–70.94 | 69.82 |
| First-index binary search | 93 | 582.99–590.94 | 583.31 | 78.51–79.27 | 78.43 |
| Brief greeting | 41 | 286.27–289.23 | 570.84 | 75.26–76.62 | 75.77 |
| Repeated log | 1244 | 864.56–867.38 | 867.82 | 70.48–70.97 | 70.09 |

The short-prompt prefill improves about 1.98 times, from approximately 139 ms
to 70 ms. End-to-end time for its 128-token completion falls from the mean of
the two control medians, 1.845 seconds, to 1.780 seconds (about 3.5%). Decode
does not improve. SQL and binary search exceed 64 prefill rows; the long case
uses 256-row chunks and a 219-row tail, so none exercises the new path.
These prompts show unchanged fallback behavior, not long-prefill acceleration.
The short case recorded 800 packed calls over its warmup and three measured
runs, across 200 target projections.

All 7,488 measured output tokens across the primary comparison's 72 requests
match their respective control exactly, including input hashes and stop
reasons. New-binary controls also match the old-binary SQL and binary-search
baseline outputs. The whole comparison completed in 284.97 seconds with no
resource stop; peak allocated/reserved GPU memory was 6,075,308,032 /
6,505,365,504 bytes.

MLP scheduling experiments, median decode tok/s:

| Treatment | SQL | Binary search | Short | Long | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Control-before | 70.94 | 79.27 | 76.62 | 70.48 | Reference |
| Four warps | 68.43 | 76.16 | 73.98 | 63.32 | Slower; reject |
| 16 warps | 70.62 | 79.18 | 76.58 | 71.49 | No consistent gain; reject |
| Packed-mid + four warps | 67.93 | 76.20 | 73.64 | 63.18 | Slower decode; reject |
| Control-after | 69.69 | 78.51 | 75.26 | 70.97 | Reference |

A second private diagnostic applied the override only during `GatedMLP.forward`,
excluding other similarly sized projections. Its 60 measured requests also
failed to qualify a default change: four warps stayed slower, and the
MLP-only 16-warp treatments (with and without packed-mid) changed the SQL
completion beginning at zero-based generated token index 21 (15094 to 7892).
The other three prefixes matched. The wrapper adds some Python overhead and
is only a diagnostic; it is not part of the production runtime. This numerical
divergence is a failed exact-output criterion, not evidence that the new
packed-mid-only setting changed that completion.

The measured candidate initially exposed the qualified 10–64-row path as opt-in.
It was subsequently promoted to the default at the user's request, retaining
the existing MLP scheduling heuristic. No additional optimization was started.

## Public CLI integration

Two separate main-runtime launches used the verified candidate config, F16 KV,
no MTP, decode fusions off, and the held-out prompt in the example above.
The second launch enabled `--packed-mid` and passed `--expected-results`
pointing to the first result. Both processes and monitors exited zero, all 32
warmup raw-logit checks per launch were finite, and all 64 generated tokens
matched exactly for the 43-token input.

Control versus packed measured prefill was 216.82 versus 70.31 ms, decode
48.53 versus 48.57 tok/s, and end-to-end 1.539 versus 1.391 seconds. This is a
single-pair integration smoke test with different process/cache warmup; it is
not a second statistically repeated speed benchmark. The candidate result
explicitly records `packed_mid: true`, ABI 1, and no MLP warp override.

## Default promotion: 2026-09-26

The user requested primary use after reviewing the measurements. The launcher,
evaluator and server now select packed-mid automatically from the verified
binary's capability. Explicit on/off settings remain available for diagnosis;
an older binary without the capability keeps the previous fallback. This
changes dispatch defaults only: the tested native binary and its 329 source
fingerprints remain identical, and MLP warp scheduling remains unset.

The local installation now points to the verified gfx1101 packed build and
its SHA256 recorded below. Only `extension_dir` and `extension_sha256` changed;
the previous installation is retained as
`.runtime/installation.before-packed-primary-20260926.toml`.

A fresh normal MiMo CLI launch used no `--config` or packed-kernel flag. Its
result recorded `packed_mid: true`, ABI 1 and `mlp_warps: null`, and all 64
output tokens for the 43-token prompt matched the previously validated packed
run exactly. The process and monitor exited zero with no resource stop. This
checks automatic selection, not a new throughput benchmark.

OMP assisted the default-selection regression coverage and exited zero within
its ten-minute deadline. The final Windows suite ran 531 tests: 506 passed and
25 were skipped. Linux ran the same 531: 521 passed and 10 Windows-only skips.
Coverage includes automatic new/legacy-binary selection, explicit disable,
unknown ABI rejection, environment reset, and launcher/worker parser behavior
for generation, benchmark and serving. The existing GPU-kernel acceptance
still applies to the unchanged binary. No new live HTTP test was performed
for this default-only change.

Promotion evidence is in `artifacts/packed-primary-20260926/`: before snapshots,
registration changes, full unit logs, `default-cli/`, and `promotion-evidence.json`.
OMP records are in `.codex/.omp-jobs/packed-primary-20260926/`.

## Build and correctness validation

OMP 18.1.16 with the configured `@default` model completed the bounded native
implementation with exit code 0 in about three minutes, within its 25-minute
deadline. The usage query did not expose remaining quota; a bounded probe and
the actual job succeeded. OMP used the configured Muse Code model. The
coordinator reviewed the diff and performed builds, integration, correctness
checks and performance acceptance.

The fresh gfx1101 build compiled all 110 native source units, without reusing
old objects, in 731.92 seconds. All 329 recorded native/build fingerprints
match the tested source. Binary SHA256:
`006ac53097629cdbdd4237050a5fec7feeef9a74af7a8c6a83cb0325361135f1`.

| Check | Result |
| --- | --- |
| Windows unit suite | 520 run: 495 passed, 25 dependency/platform skips |
| Linux runtime unit suite | 520 run: 510 passed, 10 Windows-only skips |
| Existing small-M GPU projections/graphs | 408 passed |
| MiMo high-bit GPU projections | 48 passed |
| Packed 10–64-row GPU projections | 929 passed |
| Existing prefill GEMM GPU checks | 56 passed |
| Real MLP operator checks | Both 141-case sweeps and all six slice checks passed |
| CLI help and whitespace | Passed |

The 1,441 public-validator GPU cases include every row count from 10 through
64, all supported codebook/bit combinations, FP16/FP32 outputs, guarded output
and scratch buffers, non-default streams, changed-input external graph replay,
and an uneven split-K shape. Worst packed-mid relative L2 error against the
independent CPU reference was 0.00037606 (0.0376%, below the 0.5% gate).
Native MLP graph support was not extended to these row counts. Unit coverage
checks old/unknown ABI rejection, state reset, routing boundaries, unsupported
formats, explicit reconstruction precedence and forwarding through both workers.

The first Linux unit invocation lacked the existing server-dependency directory
on its import path and failed with three import errors. The corrected command
used that existing directory, installed nothing, and passed the full suite.
Both logs are retained. The pre-existing ROCm `SharedSignalPool` teardown
warning remains reproducible with baseline and candidate; it was not fixed
as part of this kernel work.

## Evidence and limits

Ignored local evidence is retained under `artifacts/packed-mlp-20260926/`:

- `model-manifest.json`: all model file hashes and sizes.
- `baseline-target/`, `baseline-mtp/`: fresh baselines, exact requests, tokens,
  timing, resource telemetry and monitor outcomes.
- `operators.py`, `operators-mimo-blas/`, `operators-mimo-wmma/`: real-weight
  references and raw timing samples, including hot and evicted results.
- `mimo-evaluator-frozen.py`, `mimo-harness-provenance.json`, `mimo_ab.py`,
  `launch_mimo_ab.py`: fixed evaluator provenance and guarded comparison entry.
- `mimo-ab-v1/`, `analysis.json`: primary model treatments and recomputed medians.
- `mimo-ab-mlp-only/`, `analysis-mlp-only.json`: narrower diagnostic and failures.
- `public-control/`, `public-packed/`: actual main-CLI equivalence check.
- `gpu-checks-v1/`, unit logs, `build-source-verification.json`,
  `final-evidence.json`: correctness, source identity and acceptance records.
- `candidate.toml`, `mimo-candidate.toml`, `mimo-baseline.toml`: explicit local
  configurations used before the default promotion.

The fresh binary, build log and source manifest are in
`.runtime/gpu-tuning/packed-mlp-v1/`. OMP's work package, execution log,
implementation notes and completion record are in
`.codex/.omp-jobs/packed-mlp-20260926/`. `analyze.py` recomputes the primary
tables; `--ab mimo-ab-mlp-only --out analysis-mlp-only.json` selects the narrower
experiment. Raw model paths and outputs remain in local ignored artifacts.

This validates one gfx1101 device and the specified MiMo package. No new RDNA4
build or device test, 27B whole-model benchmark, sustained server run, broad
model-quality evaluation or BF16 fidelity claim is included. Numerical
operator tolerances and a handful of exact completions do not establish
universal token identity. No hardware DRAM counters were collected; 90–95%
memory-bandwidth utilization has not been demonstrated. Longer prefills,
attention, KV research, persistent kernels and other roadmap items remain
outside this completed task.
