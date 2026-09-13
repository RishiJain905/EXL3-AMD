# Local HTTP server

After [building and registering](BUILD.md), run:

```powershell
python run.py serve -m "MODEL_DIRECTORY" --cache-type q8 -c 4096 --alias exl3 --port 8000 --execute
```

Add compatible MTP flags if the model includes drafting weights. The model remains loaded; every serialized request has fresh generation state. There is one active GPU request and four queue slots, without continuous batching. Normal launches need no `--config`.

## API

Wait for readiness, then submit text:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/models
$body = @{
  model = 'exl3'
  messages = @(@{role='user'; content='Write Python binary search.'})
  max_tokens = 256
  temperature = 0
} | ConvertTo-Json -Depth 5
$response = Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
$response.choices[0].message.content
```

`/v1/chat/completions` applies the saved chat template. `/v1/completions` accepts a raw string prompt. Streaming uses SSE ending with `[DONE]`; `stream_options: {"include_usage": true}` adds a usage chunk. A compatible client uses base URL `http://127.0.0.1:8000/v1` and the chosen alias. Client API-key placeholders are unused: the server has no authentication and binds only to loopback.

Supported roles: system, developer, user, assistant and tool. Content is text only. Sampling is greedy (`temperature=0`, `top_p=1`, `n=1`), thinking disabled. Output limits are 1–8192 tokens subject to remaining context/draft reserve. Chat also accepts `max_completion_tokens`. Unsupported sampling, stop strings, image/audio content, strict schemas and extra parameters are rejected. Bodies are capped at 1 MiB. Send the full conversation with each request.

[Function tools](TOOL-CALLING.md) use the supported Chat Completions contract without a client allowlist or OpenCode dependency. Responses API and Anthropic Messages are not implemented.

## Lifecycle

There is **no overall server lifetime timeout**. Stop with Ctrl+C or create the `stop` file in the printed run artifact directory. Resource and error safeguards remain active. `--request-timeout` defaults to 120 seconds (range 1–900), including body upload, queueing and generation. This is separate from server lifetime. Non-server modes retain a bounded capture duration.

Cancellation is observed between GPU iterations. Disconnects release slots; GPU generation failures mark the engine unavailable until restart. Invalid model tool syntax produces a protocol error without poisoning the engine. `/health` reports readiness, counters, detected tool protocol and allocator memory. Allocator bytes are not total adapter residency.

Run artifacts include prompts, output/token IDs, provenance, logs and monitoring. They are private and ignored. Public networking, authentication, broad model support and extended soak behavior require separate work.

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
