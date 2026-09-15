# Full CLI performance follow-up

Measured September 14–15, 2026, on the same dense 27B Qwen-family EXL3 model,
RX 7800 XT (gfx1101) and existing WSL environment as
[the earlier native study](GPU-PERFORMANCE.md). These results do not establish
performance or answer quality on other models, GPUs or a fully occupied 120K
prompt. Weights, installation records and raw responses remain private.

## Method

Primary measurements launch the ordinary `run.py serve` CLI with capacity
120064, Q8 K/V, long attention, WMMA prefill, chunk 1024, MTP depth 6 with
confidence 0.6, and prefix reuse enabled. Only the specified setting changes.
Every server retains the GPU lease, extension hash verification and resource
guards. Different system prefixes force fresh requests; identical repeats
measure cache reuse. Requests are serialized and allow 256 output tokens.

Rates use the second fresh request after warmup. Decode divides counted tokens
by elapsed decode time, excluding the entire first emitting MTP iteration.
Aggregates divide summed tokens by summed time. Cached input tokens never count
as computed prefill throughput. Small differences of a few percent in the flag
sweep are not established speed improvements.

## Decode first

Python, reasoning and code-review requests supplied the aggregate below. The
baseline and readback trials also included tool calls; agreement counts include
warmup and repeats and compare identical input hashes and raw output token IDs.

| Setting | Combined decode, tok/s | Exact output agreement |
| --- | ---: | ---: |
| Initial control | 41.65 | 13/13 |
| Reduced MTP host readbacks | 42.06 | 13/13 |
| Fixed MTP depth 4 | 40.92 | 9/10 |
| Head warps 4 / 8 / 16 | 41.90–42.12 | 10/10 each |
| Split-K warps 4 | 40.68 | 9/10 |
| Split-K warps 16 | 42.91 | 7/10 |
| Cached draft FC | 41.62 | 10/10 |
| Cached draft attention | 40.97 | 10/10 |
| Cached draft MLP | 40.42 | 10/10 |
| Draft shortlist, 16 groups, packed / dense | 30.32–30.40 | 7/10 each |
| Draft shortlist, 64 groups, packed | 38.84 | 7/10 |
| Batched greedy | 42.45 | 10/10 |
| GDN + MLP fusion | 42.68 | 10/10 |
| Experimental draft-depth controller | 40.18 | 12/13 |

The retained host change copies each draft token to CPU once instead of twice.
During confidence-calibrator burn-in, when early stopping cannot occur, it
transfers confidence for the whole window once. Calibrated stopping and label
updates remain unchanged. Four CPU-Torch regression tests cover token
propagation, readback grouping, missing confidence and early stopping. The
measured decode change is small; no large decode gain is claimed.

The experimental controller was removed after regression. Shortlists retained
full-vocabulary target verification but reduced acceptance and throughput.
Output differences in other settings do not establish worse answer quality;
they do prevent claiming exact equivalence.

After installing the revised native kernel, a same-session comparison combined
`--decode-fusions gdn-mlp --head-warps 8 --batch-greedy`. It produced 42.78
combined decode tok/s versus 43.00 with normal settings; all 10 input hashes
and raw outputs matched. The combination is not recommended: it gave no
aggregate gain and batched greedy restricts request sampling and penalties.
Normal settings retain the default GDN fusion and the broader sampling API.

## Native prefill

Instrumented native-entry intervals measured approximately 5.53 seconds in
GEMM and 0.78 seconds in weight reconstruction during a separate prefill
request. These GPU-event intervals include submission gaps and are not exclusive
kernel self-time. Native work therefore targeted GEMM scheduling; a fused
reconstruction/GEMM implementation is not included in this follow-up.

The revised WMMA implementation uses the original serial 128×128×64 tile below
512 rows and a 256×64×32 tile with next-tile register prefetch at 512 rows and
above. Both use 512 threads, FP32 accumulation/output and the existing 256-K
partial-sum fold. Existing device, shape, stride, alignment, capacity and alias
guards remain. Compiler metadata reports 118 / 169 vector registers and no
scratch spills for the serial / prefetch branches. The build still exports
prefill GEMM ABI 1; the CLI flag and BLAS fallback are unchanged.

An earlier 128×128 prefetch variant did not improve HTTP prefill; a transposed
LDS layout regressed projection probes. Neither is selected. The final load-loop
revision restored the original small-input register use. It passed 56 native
GPU checks, extending the previous suite across the 512-row dispatch boundary,
partial tiles, strided canaries, subnormals, streams and graph replay.

The following HTTP runs use actual prompt lengths from the saved chat template.
Both kernels generated identical raw output IDs on all 11 matched requests.

| Prompt tokens | Original prefill | Revised prefill | Original / revised first token | Original / revised decode |
| ---: | ---: | ---: | ---: | ---: |
| 3,874 | 529.62 tok/s | 555.60 tok/s | 7.56 / 7.22 s | 42.63 / 43.74 tok/s |
| 15,391 | 510.27 tok/s | 530.91 tok/s | 30.39 / 29.23 s | 36.74 / 37.30 tok/s |
| 30,706 | 454.28 tok/s | 472.69 tok/s | 67.83 / 65.20 s | 33.73 / 34.12 tok/s |

The combined fresh prefill rate rose from 475.6 to 495.1 tok/s. This is a modest
improvement over the earlier WMMA implementation, not a new BLAS comparison.
See the earlier study for the larger BLAS-to-WMMA gain.

A same-session repeat of the original kernel measured 514.17 tok/s at 15,391
tokens, versus 530.91 for the revised kernel (3.3%). All five shared requests
matched input hashes and output token IDs. Decode was effectively tied at
37.53 versus 37.30 tok/s. This repeat supports a modest prefill gain rather
than attributing the earlier comparison entirely to conditions across sessions.

## Chunk comparison with the revised kernel

All three settings use capacity 120064 and the same 15,391-token HTTP prompt,
MTP configuration and 256-token output budget. The five shared requests
(warmup, two fresh prompts, repeat and appended turn) match input hashes and
output token IDs for both alternate chunk sizes.

| Chunk | Warm prefill, tok/s | Warm decode, tok/s | Engine first token | Client total |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 505.6 | 36.4 | 30.55 s | 39.45 s |
| 1024 | 530.9 | 37.3 | 29.23 s | 37.65 s |
| 2048 | 550.9 | 35.9 | 28.36 s | 37.28 s |

Chunk 1024 remains the balanced selection and has the broader 4K/15K/30K
validation above. Chunk 2048 offers about 3.8% more prefill in this 15K test,
with lower observed decode and only about 1% lower total client time. It is
an option to measure for prefill-heavy workloads, not a universal improvement.

## Prefix reuse and validation limits

With the revised kernel, repeats reused 3,840 / 15,360 / 30,464 input tokens
and reached their first token in 0.60 / 0.52 / 0.94 seconds. The appended
15K turn reused 15,360 tokens and reached its first token in 1.53 seconds.

Exact fresh-versus-repeat output checks passed at 4K and 15K but **failed at
30K on both the original and revised kernels**: the first 241 generated tokens
matched, followed by a wording difference. The original benchmark's client
assertion failed even though all 11 server requests completed and shutdown
returned zero. Subsequent diagnostics retain this failed exact-reproduction
result separately from server health and throughput.

Recorded MTP rounds confirm different verification grouping near output token
15, where prefix-relative recurrent checkpoint boundaries differ between fresh
and cached execution. This supports a floating-point/verification-shape
explanation; it does not prove the numerical cause or establish general cache
equivalence. No checkpoint policy or precision was changed to hide the failure.
Private request artifacts now retain the already collected draft-round tuples
to make this behavior inspectable; HTTP responses are unchanged.

The revised binary passed all 11 separate HTTP acceptance checks: reasoning,
SSE/non-stream agreement, thinking disabled, truncation, invalid options, raw
completion, tool-call parsing, synthetic tool-result turns and cancellation
recovery. No generated tool was executed. Final health recorded 18 completed
requests, one deliberate cancellation and zero engine failures, followed by
clean shutdown. These functional checks do not override the 30K exact-output
failure above. Peak reserved GPU memory was approximately 12.7 GiB.

Native probe processes still print the pre-existing `SharedSignalPool`
teardown warning; it was absent from this HTTP-server log. CPU coverage and
earlier validation limits are recorded in [VALIDATION.md](VALIDATION.md).

Final checks ran 426 tests in the Linux inference environment: 416 passed and
10 Windows-only tests skipped. All 67 Windows launcher tests passed separately.
CLI help and diff whitespace checks passed. A scan of 806 public files found
no personal workstation paths or matching credential patterns. This scan is
bounded; no new security-agent review was performed in this follow-up.

## Recommended command

Use a compatible extension built from the revised source and registered with
its actual hash, as described in [BUILD.md](BUILD.md). Replace the model
placeholder with a local EXL3 directory; no launch script is required.

```powershell
python run.py serve `
  -m "MODEL_DIRECTORY" `
  -c 120064 `
  --cache-type q8 `
  --attention-profile long `
  --prefill-gemm wmma `
  -b 1024 `
  --spec-type draft-mtp `
  --spec-draft-n-max 6 `
  --draft-confidence 0.6 `
  --prefix-cache on `
  --alias exl3 `
  --port 8092 `
  --request-timeout 900
```

Jinja templating is already enabled; adding `--jinja` does not improve loading
or inference speed. CPU MoE offload does not apply to this dense model. The
command retains Q8 precision and the normal resource guards. Prefix reuse is
optional and retains the exact-output limitation described above.
