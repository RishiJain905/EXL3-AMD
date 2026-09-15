# Long context

`-c` sets total capacity. Input, output and draft reserve must fit inside it and within model metadata and available memory. Cache allocation alone does not establish retrieval quality or speed.

Increase capacity incrementally. Reduced KV precision saves memory, but attention still reads more history as occupied context grows.

```powershell
python run.py context -m "MODEL_DIRECTORY" --cache-type q8 -c 32768
python run.py context-speed -m "MODEL_DIRECTORY" --cache-type q8 --spec-type draft-mtp --spec-draft-n-max 6 --draft-confidence 0.6 -c 32768
```

`context-speed` fills `context - 256` input tokens. Default task `docstring` generates 192 tokens; `--context-speed-task canonical` uses the fixed coding-suite question and 128 tokens. Keep tasks fixed across comparisons.

## Attention profile

`--attention-profile long` increases KV splits, reduction parallelism and grouped-query reuse. It applies only to Q8/Q8, gfx1101, batch one, 24 query heads / 4 KV heads / dimension 256, query lengths 1–8, global causal attention and effective page context 32768–120064.

Other configurations fall back. Inspect `applied_calls` to confirm the optimized path ran. Its 120064 bound is not a universal model limit. The CLI has no fixed 120K ceiling; metadata and memory still constrain capacity.

## Measurements

Use timing version 2, which excludes the complete first emitting GPU iteration. Track prefill, decode, actual token counts, peak memory, retrieval correctness and MTP/target output agreement separately. Older first-event MTP metrics are not directly comparable.

Q8 can remove an allocation blocker without sustaining short-prompt speed. This release makes no 60 or 80 tok/s long-context guarantee. [Cache precision](KV-CACHE.md).
