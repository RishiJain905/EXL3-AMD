# MiMo step 2: EXL3 candidate packaging and audit

For a separately acquired donor MTP component, use the independent
[derived-package builder](MIMO-MTP-PACKAGING.md). The original MTP rejection
and source-preservation contract below remain in force.

`scripts/package_mimo_exl3.py` is a stdlib-only CLI that joins a completed
MiMo text EXL3 conversion with the step 1 vision preservation pilot into one
final candidate directory, with an external JSON audit report. It never
modifies its inputs and fails closed: missing or incomplete expected inputs
abort before the output directory is created.

No image runtime support is claimed, and no MTP weights are added. Any
`mtp.*` tensor anywhere in the text candidate fails packaging.

## Invocation

```sh
python scripts/package_mimo_exl3.py \
  --source <ORIGINAL_BF16_DIR> \
  --text-dir <COMPILED_TEXT_CANDIDATE> \
  --vision-dir <VISION_PRESERVATION_PILOT> \
  --source-audit <SOURCE_AUDIT_JSON> \
  --mapping <MAPPING_JSON> \
  --output <NEW_FINAL_CANDIDATE> \
  --report <NEW_EXTERNAL_REPORT_JSON>
```

`--mapping` (step 1 `mapping.json`) is optional but preferred: it supplies
the exact 201 `qmap != None` projection keys and dimensions. Without it, the
tool falls back to the known MiMo projection suffix set and still requires
exactly 201 quantized projections with identical geometry checks. `--help`
needs no Torch or GPU.

`--output` and `--report` must be new paths. Source, text, vision, and output
directories must be pairwise separate and non-nested; the report must sit
outside all of them. The report embeds resolved local paths, so treat it as
private.

## Checks

- Safetensors headers across all text shards: duplicate names, offsets, file
  bounds, dtype sizes, contiguous payload coverage.
- Exactly the expected text tensor set: one `.trellis`/`.suh`/`.svh`/`.mul1`
  group per quantized projection, plus carried-over small tensors under
  unchanged names. Missing or unexpected tensors fail.
- Body projections K4, `lm_head` K6; trellis `I16` with shape
  `[in/16, out/16, K*16]`; `suh`/`svh` `F16` of length `in`/`out`; scalar
  `I32` mul1 marker exactly `0x83DCD12D`.
- Embedding stays `BF16` with identical shape and exact source payload hash
  (streamed, not loaded whole). Other small tensors must keep shape with
  `F16`/`BF16` dtype; `BF16 -> F16` conversions are listed in the report
  rather than treated as byte-identical.
- Vision file: container hash matches the source audit, all 333 canonical
  name/dtype/shape/payload hashes verify, no name collisions with text
  tensors, no reencoding (byte-exact copy, possibly renamed to avoid a shard
  filename collision).
- Source config must declare `Qwen3_5ForConditionalGeneration` text with
  4096 hidden size, 32 layers, and a vision config. The text candidate
  config must identify `quant_method=exl3`.

## Outputs

The output directory contains the copied text shards unchanged, the vision
file, source tokenizer/processor/template metadata (`.json`/`.jinja`/`.txt`,
plus `README`/`LICENSE` when present; never the stale source weight index,
dotfiles, subdirectories, or logs), compiled quantization metadata
(`quantization_config.json`, `conversion_identity.json` when present), a
patched `config.json`, and a regenerated `model.safetensors.index.json`.

The output config starts from the original source config (vision config
retained), sets `text_config.mtp_num_hidden_layers` to 0, and retains the
text candidate's `quantization_config`. The patch is recorded in the report.
The index covers every text plus vision shard tensor, with
`metadata.total_size` equal to actual payload bytes. Destination headers are
re-opened and re-audited against the index before the report is written.

The report records per-file hash/bytes, component byte totals, the body
parameter-weighted trellis-only stored rate separately from complete package
bytes/BPW, the exact realized per-projection bit map, embedding
dtype/hash preservation, vision hashes, source index/config hashes, the
generated config hash, copied/skipped metadata, and `null` for genuinely
unknown values.

## Failures

Validation failures before copying leave no output directory but still write
an `incomplete` report. Failures after the output directory is created also
leave a `PACKAGING-INCOMPLETE.json` marker inside it. Partial outputs are
preserved, never silently deleted, and nothing is ever written into the
input directories. A marker is only written to an output directory this run
created.

## Assumptions and limits

- The text candidate stores EXL3 projections as `LinearEXL3` storage:
  `<base>.trellis/.suh/.svh` with a `<base>.mul1` marker. Legacy
  `.su`/`.sv` or `.mcg` tensors fail as unexpected.
- Unquantized text tensors keep source names; only the embedding is
  required to stay `BF16`.
- Final full-shape validation and runtime reload belong to the parent
  stage, which also integrates this tool against real weights.
