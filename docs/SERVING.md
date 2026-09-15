# Local HTTP server

After [building and registering](BUILD.md), run:

```powershell
python run.py serve -m "MODEL_DIRECTORY" --cache-type q8 -c 4096 --alias exl3 --port 8000
```

Add compatible MTP flags if the model includes drafting weights. The model remains loaded; every serialized request has its own generation job and sampler. There is one active GPU request and four queue slots, without continuous batching. Normal launches need no `--config`.

`--execute` is no longer a runtime CLI flag. Launching a command runs it once
both installation execution permissions are enabled; native hash checks and
resource guards still apply.

## API

Wait for readiness, then submit text:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/models
$body = @{
  model = 'exl3'
  messages = @(@{role='user'; content='Write Python binary search.'})
  max_tokens = 1024
  temperature = 0
} | ConvertTo-Json -Depth 5
$response = Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
$response.choices[0].message.content
```

`/v1/chat/completions` applies the saved chat template. `/v1/completions` accepts a raw string prompt. Streaming uses SSE ending with `[DONE]`; `stream_options: {"include_usage": true}` adds a usage chunk. A compatible client uses base URL `http://127.0.0.1:8000/v1` and the chosen alias. Client API-key placeholders are unused: the server has no authentication and binds only to loopback.

HTTP requests must use `127.0.0.1` or `localhost` in `Host`. POST requests require
`Content-Type: application/json`. A supplied browser `Origin` must match the
request's scheme and Host, including its port; cross-site requests are rejected
before GPU admission. Native API clients normally send no Origin. Unexpected
engine exceptions return a fixed public error; diagnostic details remain in the
private artifacts. Loopback restrictions do not authenticate other local users.

Supported roles: system, developer, user, assistant and tool. Content is text only. Output limits are 1–8192 tokens subject to remaining context, reasoning, and draft reserve. Chat also accepts `max_completion_tokens`. Stop strings, image/audio content, strict schemas and extra parameters are rejected. Bodies are capped at 1 MiB. Send the full conversation with each request.

## Reasoning

Thinking models emit `<think>...</think>` before their answer. The server splits that block into `reasoning_content`, separate from `content`, for both SSE deltas and non-streaming messages; tool-enabled turns buffer both channels the same way. Tool-looking text inside reasoning is never parsed as a call. No reasoning is invented: without markers the response is plain content.

`--reasoning on|off|auto` (default `auto`) forces or respects the template's thinking switch; `--reasoning-format none` keeps raw text instead of splitting. Per-request `chat_template_kwargs: {"enable_thinking": true|false}` overrides the default; only `enable_thinking` is accepted. Assistant `reasoning_content` history is passed to the native template on subsequent turns. `/v1/completions` is always raw. Generation budgets must leave room for reasoning: `max_tokens` counts reasoning plus visible tokens.

## Sampling

Greedy decoding is the default (`temperature: 0`, `top_p: 1`, `n: 1`). Serve flags set defaults and requests may override `temperature` [0,2], `top_p` (0,1], `top_k` [0,1000], `min_p` [0,1), `repetition_penalty` (0,2], `presence_penalty`/`frequency_penalty` [-2,2], and `seed` [0,2⁶³). Non-greedy requests use the vendored ComboSampler; `temperature: 0` or `top_k: 1` stays on the greedy path. `--batch-greedy` rejects stochastic sampling and penalties, including request overrides. Seeded sampling with MTP and penalties passed a dense Qwen-family GPU smoke test; broader sampling quality and determinism across devices or versions remain unvalidated.

## Backend controls

`-b / --prefill-chunk` (256–8192 in multiples of 256, serve default 1024) sets prompt tokens per prefill step, the analogue of llama.cpp `-b`. Larger chunks need more temporary VRAM; `-b 256` retains the previous setting. Non-server modes retain their 256-token default. `-ncmoe / --n-cpu-moe N` keeps routed experts of the first N MoE layers on CPU and `--cpu-moe all` offloads every MoE layer, backed by `cfg.infer_params.moe_cpu_offload`; this pins whole layers, not dynamic inactive-expert caching. Offload is rejected on dense models and when the extension lacks the CPU-MoE symbols, and is not GPU-validated here. Draft-model offload, `ngl`, `mmap`, and RoPE overrides are not offered.

`--prefill-gemm wmma` selects the native FP16-input, FP32-output GEMM on
gfx1101 for at least 64 rows and K/N dimensions divisible by 128. Accumulation
and output remain FP32. Other shapes and FP16 outputs keep the BLAS path.
The default is `blas`; opting in requires a hash-verified extension exporting
prefill GEMM ABI 1. This is separate from `--smallm-kernel`, which controls
packed decode projections.

## Prefix reuse

`--prefix-cache on` retains the backend's matching KV pages and recurrent
checkpoints between requests. It defaults to `off`. Reuse requires an identical
token prefix and, for hybrid models, a compatible recurrent checkpoint. Only
unchanged leading pages are reused; the changed portion is recomputed. MTP still
performs a real target pass before drafting.
The cache serves trusted local clients within one process; use `off` when
cross-request reuse is unwanted. It does not persist to disk or survive restart.
Timing and cached-token counts can reveal whether a guessed prefix was seen
earlier. Logical invalidation does not securely erase CPU/GPU memory.

The recurrent checkpoint cache is bounded to 1 GiB of host memory and the CPU KV
tier remains disabled. Requests have fresh sampling/RNG state. Cancellation,
generation failure, and uncertain cleanup discard reusable state. `/health`
reports reuse hits/misses, token totals, and live checkpoint memory.

See [storage and prefill measurements](PREFILL-PERFORMANCE.md) for the tested
chunk sizes, reuse correctness checks, and separate cold/reused latency results.

## Console and timings

The terminal shows model loading stages, device/context/KV/MTP/VRAM summary, readiness, and per-request prefill/first-token/decode timing plus cancellation, errors, and shutdown. The full child log stays in `process.log`; the console uses a bounded relay; routine serving status excludes prompts and completions. Usage carries a `timings` object with `prefill_seconds`, `first_token_seconds` (prefill plus first decode iteration, not pure prefill), `decode_seconds`, `total_seconds`, and matching `*_tokens_per_second` rates.

Prefill timing measures synchronized prefill calls and reports their actual token
count (normally prompt length minus the first decode input). First-token latency
also includes generator setup and the first decode iteration. Decode timing
excludes the entire first emitting iteration, including its MTP tokens; a single
emitting iteration has no measurable decode rate and prints `n/a`. MTP requests
also show accepted/rejected draft tokens. These additive fields are runtime
extensions to the standard usage counters.

With prefix reuse, `prefill_tokens` includes cached and computed input tokens;
`prefill_computed_tokens` and `prefill_cached_tokens` separate them. The prefill
rate divides only computed tokens by the time spent in prefill calls that did
computation. `prefill_wall_seconds` includes generator setup, cache lookup,
recurrent restoration, and gaps between chunks, ending before the first decode
iteration. Cached tokens never inflate compute throughput. First-token latency
is the useful comparison when evaluating prefix-reuse gains.

[llama.cpp flag mapping and limits](LLAMA-CPP-COMPATIBILITY.md).

[Function tools](TOOL-CALLING.md) use the supported Chat Completions contract without a client allowlist or OpenCode dependency. Responses API and Anthropic Messages are not implemented.

## Lifecycle

There is **no overall server lifetime timeout**. Stop with Ctrl+C or create the `stop` file in the printed run artifact directory. Resource and error safeguards remain active. `--request-timeout` defaults to 120 seconds (range 1–900), including body upload, queueing and generation. This is separate from server lifetime. Non-server modes retain a bounded capture duration.

A confirmed clean shutdown after Ctrl+C or an explicit stop exits successfully.
The private `monitor.json` records `stop_reason: "user_stop"` and retains the
child's original exit code, which may indicate SIGTERM after Uvicorn cleanup.
Missing or failed cleanup, resource-limit stops, and monitoring errors still
report failure.

Cancellation is observed between GPU iterations. Disconnects release slots; GPU generation failures mark the engine unavailable until restart. Invalid model tool syntax produces a protocol error without poisoning the engine. `/health` reports readiness, counters, detected tool protocol and allocator memory. Allocator bytes are not total adapter residency.

Run artifacts include prompts, output/token IDs, provenance, logs and monitoring. Request-finished records also retain the already collected MTP draft rounds (generated-token position, draft width, accepted count) for diagnostics. They are private and ignored. Public networking, authentication, broad model support and extended soak behavior require separate work.

Exact fresh-versus-reused output equivalence is not established for long MTP
continuations: a 30K prompt produced a late wording difference on both tested
WMMA kernels. See [RUNTIME-PERFORMANCE.md](RUNTIME-PERFORMANCE.md) for the
measured cache benefits, verification-round differences and validation limits.

## Optional HTTP dependencies

The launcher can add an ignored `.runtime/server-deps` overlay for serving. `runtime.server_deps` overrides its location. Alternatively install compatible HTTP dependencies in your dedicated inference environment.

Inspect the dependency plan first:

```bash
"$EXL3_PYTHON" -m pip install --dry-run -r requirements-server.txt
```

If shared dependencies already satisfy that plan, an unused overlay can hold the HTTP packages:

```bash
"$EXL3_PYTHON" -m pip install --target .runtime/server-deps --no-deps -r requirements-server.txt
PYTHONPATH=.runtime/server-deps:src "$EXL3_PYTHON" -m unittest discover -s tests -p test_server.py
```

`--no-deps` is appropriate only after checking shared dependencies; the pinned file is not a full inference environment. See [DEPENDENCIES.md](DEPENDENCIES.md).
