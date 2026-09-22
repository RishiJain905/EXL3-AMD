# MiMo 9B: step 1 compatibility validation

Run `QEXP-002-20260922T031657Z-agent-mimo-step1-6c95bd2d`, 2026-09-22 UTC.
Completed bounded source/codec/operator validation on RX 7800 XT / gfx1101.
No full MiMo EXL3 model was converted or served. Stop after this stage.

| Gate | Measured result |
|---|---|
| Source mapping | 427 text tensors consumed; 333 vision tensors inventoried; no missing/mismatched/unconsumed text tensors |
| MTP | Config declares one layer but contains zero MTP tensors; suppress component in memory and reject explicit MTP requests |
| Token protocol | Full source/runtime vocabulary mapping and four chat-template fixtures match exactly; stop IDs 248046 and 248044 |
| Encoding | 11 saved mul1 artifacts: nine real 256×256 body slices at K4/K5/K6, one 128×4096 K6 head slice, one repeated K4 slice |
| Independent reload | Zero raw decoder mismatches; 110 projection checks passed at relative L2 ≤0.005; worst 0.000498002 |
| Head chunking | 12 sampled checks on a synthetic K6 4096×32896 packed matrix, 16/1024 input rows, FP16/FP32 output; worst relative L2 0.000913791 |
| Repeatability | Repeated same-seed K4 encoding has identical packed tensors and file SHA-256 |
| Vision preservation | Separate lab audit preserved all 333 BF16 payloads (912,020,960 bytes), independently re-read and hashed |
| Regression suite | Windows: 466 tests, 441 pass/25 skips; registered Linux environment: 466 tests, 456 pass/10 Windows-only skips |

Native extension SHA-256:
`df35ba80b0f6a0a61bb0fe31f4cac6ebfa90dca0fea2f62c38a584f041cb455e`.
Torch `2.13.0+rocm7.2`, HIP `7.2.53211`. No native build or installation.

## Changes and use

`mapped_text_config` checks actual MTP names after source-name mapping. Complete
absence disables only the optional in-memory component. Partial MTP sets still
undergo completeness checks; original config/weights are never rewritten.
The smoke harness now allows an explicitly MTP-free candidate when drafting is
off. It rejects drafting without available weights.

The independent NumPy oracle now validates K5/K6, with scalar circular-bitstream
fixtures and explicit Hadamard checks. Native optimized small-M support remains
K2/K3/K4. K5/K6 use single-row native GEMV, rowwise fallback for rows 2–9, and
reconstruction plus matrix multiplication for larger batches in the tested
compatibility route. No K5/K6 small-M optimization is claimed.

The conversion wrapper writes `conversion_identity.json` into each new work
directory. Resume compares source payload/metadata hashes, calibration content,
resolved quantization settings, recipe content, Python converter code, adapter,
compatibility helper and native extension identity. It rejects changed content
even when filenames are unchanged. Existing jobs without this identity fail
closed; do not backfill an invented identity. Interrupted full-model resume
was not executed in this stage; these guards have CPU regression coverage.

The bounded pilot has an explicit help interface:

```sh
python3 scripts/run_mimo_step1.py --help
python3 scripts/run_mimo_step1.py \
  --config INSTALLATION_TOML --source MIMO_BF16 --output NEW_PRIVATE_RUN \
  --mode inspect --execute
```

Linux/WSL only. The registered installation must allow probes and inference.
`encode`, `reload --input ENCODE_RUN`, and `head` are GPU modes. The parent
verifies the native binary, acquires the shared lease and monitors the child:
900-second timeout, minimum 4GiB WSL available RAM, 120GiB artifact disk,
70% of WSL MemTotal process-group RSS. GPU modes cap Torch's allocator at 8GiB;
this is not a cap on every native/driver allocation. Outputs must be new and
separate from the source. Pilot Gaussian data is not language calibration.

## Limits and retained warnings

- Attention, MLP and GDN **projections** were tested; this is not end-to-end
  attention/recurrent-state or model inference validation.
- The full 248320-output head, large streamed quantizer, 128×2048 language
  calibration and full-model memory peaks remain unmeasured.
- Quantization starts from original BF16 slices, converted through FP16 as the
  production loader does. Same-packed-weight correctness is distinct from
  BF16/model fidelity; no Q8-equivalence, task accuracy or speed claim follows.
- `SharedSignalPool` teardown warnings (122 signals for encode, 2 for each
  reload/head process) and a WSL topology warning were retained. All three GPU
  jobs exited 0; this is not long-duration leak or serving validation.
- The first Linux suite command omitted the registered server dependency
  directory and failed three imports. Restoring that existing search path
  passed the suite; no packages were installed.
- Vision file preservation does not add image inference or an mmproj switch.
  Donor MTP and vision runtime controls remain later stages.

Detailed source hashes, recipes, tables and private-evidence inventory are in
the [lab research record](https://github.com/RishiJain905/QuantizationResearch/blob/codex/mimo-step1-20260922/docs/mimo-exl3-step1.md).
