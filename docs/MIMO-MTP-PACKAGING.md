# MiMo donor MTP packaging

`scripts/package_mimo_mtp.py` creates a fresh derived package from an existing
MiMo EXL3 target and a separately acquired official Qwen3.5-9B MTP component.
This is a CPU packaging utility. It does not acquire weights, run inference or
establish useful MTP acceptance/speed. No source or previous candidate is modified.

```text
python scripts/package_mimo_mtp.py \
  --target TARGET_PACKAGE \
  --target-manifest TARGET_FILE_MANIFEST.json \
  --donor-directory VERIFIED_DONOR_DIRECTORY \
  --output NEW_DERIVED_PACKAGE \
  --report NEW_EXTERNAL_REPORT.json
```

The output/report must not already exist and must be outside both inputs. Their
parents must exist. Only use output when the external report has `complete:
true`. A failed copy or verification preserves partial output with an incomplete
report. Invalid path requests write nothing. CLI symlinks and file-name
collisions are rejected.

## Required inputs

The target manifest is an array of `{file, bytes, sha256}` records for every
target file, as produced by the existing fresh-load smoke. The tool verifies
the complete inventory and hashes, dense Qwen3.5 configuration, MTP depth zero,
absence of MTP tensors, and exact shard/index coverage and payload byte counts.

The donor directory contains `manifest.json` plus these four files:

| File | Role |
| --- | --- |
| `mtp.safetensors` | Exactly the 15 canonical BF16 tensors for one MTP layer |
| `config.json` | Original donor geometry; MTP depth one, shared embedding/head |
| `tokenizer.json` | Must parse to the same JSON value as the target tokenizer |
| `LICENSE` | Donor license copied into the derived package |

The manifest has `schema_version: 1`, `complete: true`,
`repository: "Qwen/Qwen3.5-9B"`, a 40-hex commit `revision`, a `files` map for
exactly those four files with `{bytes, sha256}`, and a `tensors` map of
`{dtype, shape, bytes, sha256}` records. Tensor digests cover raw payload bytes.
Acquisition must separately establish the source/revision and preserve its
evidence; matching a caller-supplied manifest does not authenticate a publisher.

Architecture comparison includes attention/linear-attention geometry, layer
types, RoPE, norm settings and embedding-sharing configuration. Tokenizer
whitespace and JSON object ordering may differ; token-ID or semantic differences
fail closed. The target chat template and generation configuration are retained.

## Output and preservation

| Change | Contract |
| --- | --- |
| Target files | Independent copies; identical hashes except the two declared metadata changes |
| `config.json` | Only `text_config.mtp_num_hidden_layers` changes from 0 to 1 |
| `model.safetensors.index.json` | Add 15 donor entries and update actual tensor-payload `total_size`; preserve other fields |
| `mtp-donor-bf16.safetensors` | Byte-identical donor payload container |
| `MTP-DONOR-LICENSE` | Byte-identical donor license |
| `mtp-donor.json` | Whitelisted source/hashes/precision/sharing provenance without local paths |
| External report | Output inventory, hashes, size, outcome and precision limits; contains private local paths |

Donor tensors must have expected shapes, contiguous non-overlapping extents,
complete hashes and finite BF16 values. The builder does not quantize, cast or
add 1 to stored norms. Qwen's norm bias is applied at runtime; ordinary donor
projections load as FP16. The target keeps its embedding and EXL3 output head.
Vision storage is preserved; this utility does not implement image inference
or an mmproj control.

## Validation and next gate

Tiny stdlib fixtures exercise preservation, token/geometry mismatch, file/tensor
tampering, missing/extra/malformed/nonfinite weights, index mismatch, collisions,
protected-input/report boundaries and partial-copy failures.

```text
python -m unittest discover -s tests -p test_mimo_mtp_package.py -v
```

After packaging, the experiment still needs an independent fresh runtime load,
finite target/draft logits, MTP-off/on output parity, recurrent rollback/stopping
coverage and matched speed/memory measurements. MiMo step-4 preparation does
not yet provide that GPU evidence.
