# Familiar llama.cpp controls

EXL3-AMD uses ExLlamaV3 and local EXL3 weights. These controls expose existing
backend behavior; a llama.cpp command line is not interchangeable with this runtime.

| llama.cpp control | EXL3-AMD equivalent | Scope |
| --- | --- | --- |
| `-m`, `-c`, `-n`, `--alias`, `--port` | Same flags | `-n` is for CLI generation; HTTP requests set `max_tokens` |
| `--jinja` | Accepted; Jinja rendering is always enabled | Serve uses the model's saved Transformers chat template |
| `--chat-template-file` | Same flag | Local startup file only; no remote templates or HTTP template overrides |
| `--reasoning on/off/auto` | Same flag | Serve default; request `chat_template_kwargs.enable_thinking` overrides it |
| `--reasoning-format` | `auto`, `deepseek`, `none` | Qwen/DeepSeek-style `<think>` extraction; `none` preserves raw content |
| `--temp`, `--top-k`, `--top-p`, `--min-p` | Same flags; `--temperature` alias | Serve defaults and corresponding request fields |
| `--repeat-penalty` | Also `--repetition-penalty` | Request field `repetition_penalty`; applied before greedy or random sampling |
| Presence/frequency penalties, `--seed` | Same flags | Per-job seed; cross-device/version determinism is not promised |
| `-b`, `--batch-size` | Also `--prefill-chunk` | EXL3 prefill chunk, 256–8192 in multiples of 256; one active request, no continuous batching |
| Prompt-prefix reuse | `--prefix-cache on` | Process-local KV/recurrent checkpoint reuse; disabled by default, with fresh per-request sampling |
| `-ctk`, `-ctv` | `f16`, `q8`, `q4` | EXL3 KV formats; GGUF names such as `q8_0` are not aliases |
| `--spec-type draft-mtp` | Same flag | Integrated MTP weights required; depth and confidence controls supported |
| `-ncmoe`, `--n-cpu-moe N` | Same flags | Experimental CPU placement of experts in the first N eligible MoE layers |
| `-cmoe`, `--cpu-moe` | Also `--cpu-moe all` | Experimental CPU placement for all eligible MoE layers |
| CPU expert threads | `--moe-cpu-threads` | CPU MoE worker threads; requires supported native symbols and a compatible MoE model |

CPU MoE placement keeps expert weights in system RAM. It does **not** dynamically
evict only currently inactive experts. Dense models reject these flags. Eligibility
also depends on the inherited block-sparse expert/codebook implementation; native
symbol checks alone do not establish model compatibility. The recorded GPU tests
use a dense model, so CPU MoE execution remains unvalidated.

Separate draft models, draft CPU offload, `--gpu-layers`, `--mmap`, `--ubatch-size`,
RoPE overrides, arbitrary sampler chains, reasoning budgets, and grammar-constrained
decoding have no configured equivalent here. Unsupported flags fail rather than
silently changing an unrelated EXL3 setting. Generation/benchmark modes keep their
existing greedy protocol; the new sampling and reasoning defaults apply to `serve`.

The local llama.cpp comparison verified separate `reasoning_content` in normal and
streamed Chat Completions and terminal prompt/decode timing. EXL3 reports these
channels and measures prefill separately from first-token latency. Its decode rate
excludes the entire first emitting MTP iteration; compare measurements using the
documented accounting, not just the displayed number.

[Serving and timings](SERVING.md) · [Validation scope](VALIDATION.md) ·
[llama.cpp server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
