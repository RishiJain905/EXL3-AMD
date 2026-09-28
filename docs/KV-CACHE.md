# KV cache precision

Attention caches: FP16 (`f16`, default), uniform integer `q8`, `q6`, `q5`, `q4`,
and the experimental nonuniform AsterKV-5 prototype (`aster5`).

`--attention-profile auto` is the default scheduling policy. It selects
measured launch geometry by cache format, query/KV heads, query rows and the
occupied page-table bound on gfx1101. It leaves cache precision unchanged.
Short contexts and unsupported cases retain inherited scheduling.
[Supported shapes and 9B/27B results](HEAD-ATTENTION-PERFORMANCE.md).
Explicit `--attention-profile default` retains the inherited control path.

With a quantized cache, prefill chunks of 256 or more rows are staged by
default (`--prefill-staging on`). The referenced window is dequantized once
into a shared FP16 scratch and attended with the FP16 kernel.

- **Speed.** 2.2× faster at 122K occupied context on MiMo 9B Q6.
- **Memory.** The scratch holds one layer's K/V for the whole pool:
  2 × context × KV heads × head dim × 2 bytes. That is 512 MiB for MiMo 9B at
  131072 tokens.
- **Diagnostic override.** `--prefill-staging off` restores in-kernel
  dequantization.

[Measurements](PREFILL-ATTENTION.md).

Experimental `--cache-policy POLICY.json` assigns precision per attention layer
with explicit target/draft coverage. See the [Stage 2 contract and research
gate](KVCache-Research/STAGE2.md). Profiles require a quantized base type and
matching model configuration/geometry; they do not change recurrent state.

```powershell
python run.py serve -m "MODEL_DIRECTORY" --cache-type f16 -c 4096 --alias exl3 --port 8000
python run.py serve -m "MODEL_DIRECTORY" --cache-type q8 -c 4096 --alias exl3 --port 8000
python run.py serve -m "MODEL_DIRECTORY" --cache-type q6 -c 4096 --alias exl3 --port 8000
python run.py serve -m "MODEL_DIRECTORY" --cache-type q5 -c 4096 --alias exl3 --port 8000
python run.py serve -m "MODEL_DIRECTORY" --cache-type q4 -c 4096 --alias exl3 --port 8000
```

Run one server at a time. Add MTP flags only for compatible weights. Separate `-ctk q8 -ctv q4` settings work, including `q6`/`q5` mixes; both sides must be quantized or both `f16`, and mixing FP16 with a quantized side is rejected.

Q8/Q6/Q5/Q4 use integer codes with group scales, not FP8 or GGUF `q8_0`/`q4_0`. Attention reads packed storage directly. Scale overhead means cache bytes are not exact fixed fractions of FP16. Recurrent state and workspaces are separate.

Q8 is a useful reduced-memory starting point. Q4 is more aggressive and can alter output; an earlier Q4/MTP comparison did not exactly match target-only output. Neither establishes full-precision KV or BF16 fidelity. Validate your tasks and drafting behavior.

Q6/Q5 are inherited uniform rotated integer formats, not AsterKV or TurboQuant.
Packed GPU checks and a limited model pilot are recorded in the research
[experiments](KVCache-Research/EXPERIMENTS.md); these do not establish broad
quality equivalence to Q8.

`--cache-type aster5` uses an experimental cubic approximation to a calibrated
32-entry grid, with five-bit packed indices and FP16 scales per 32 rotated
values. Both K and V must use `aster5`; mixing it
with uniform formats is rejected. This prototype requires Triton, supports
single-GPU execution, and uses ordinary attention dispatch rather than native
BC attention. The calibration is model-specific and quality remains experimental.

The opt-in `long` profile retains its Q8 geometry guard and adds a separate
Aster cubic schedule for gfx1101, batch one, 24 query heads, 4 KV heads and
head dimension 256. The Aster schedule uses a context bound of 4096–120064
tokens, capped by physical cache pages, and query lengths 1–8 for decode and
256–1024 for prefill; other
cases use ordinary dispatch. This is an experimental schedule, not a broader
hardware compatibility claim. Smaller KV storage can permit larger context
but does not remove history-reading costs. [Long context](LONG-CONTEXT.md).

[AsterKV research plan](KVCache-Research/README.md) lists work toward a
middle ground between Q4 and Q8, with ordered experiments and stopping criteria.
Later candidates remain research ideas rather than runtime options.
