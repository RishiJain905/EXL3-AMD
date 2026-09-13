# KV cache precision

Supported attention caches: FP16 (`f16`, default), integer Q8 (`q8`) and integer Q4 (`q4`).

```powershell
python run.py serve -m "MODEL_DIRECTORY" --cache-type f16 -c 4096 --alias exl3 --port 8000 --execute
python run.py serve -m "MODEL_DIRECTORY" --cache-type q8 -c 4096 --alias exl3 --port 8000 --execute
python run.py serve -m "MODEL_DIRECTORY" --cache-type q4 -c 4096 --alias exl3 --port 8000 --execute
```

Run one server at a time. Add MTP flags only for compatible weights. Separate `-ctk q8 -ctv q4` settings work; mixing FP16 and quantized sides is rejected.

Q8/Q4 use integer codes with group scales, not FP8 or GGUF `q8_0`/`q4_0`. Attention reads packed storage directly. Scale overhead means cache bytes are not exactly half/quarter of FP16. Recurrent state and workspaces are separate.

Q8 is a useful reduced-memory starting point. Q4 is more aggressive and can alter output; an earlier Q4/MTP comparison did not exactly match target-only output. Neither establishes full-precision KV or BF16 fidelity. Validate your tasks and drafting behavior.

The opt-in `long` profile has a specific Q8 geometry guard and falls back elsewhere. Smaller KV storage can permit larger context but does not remove history-reading costs. [Long context](LONG-CONTEXT.md).
