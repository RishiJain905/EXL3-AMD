# Function tools for compatible harnesses

Tool calling is part of the runtime's **OpenAI-compatible Chat Completions API**, available to any harness using that supported contract. OpenCode is one client, not a runtime dependency or special case. There is no client allowlist, User-Agent check, OpenCode process, or OpenCode configuration needed by the server.

The harness executes tools. The runtime accepts their definitions, renders them with the model's saved chat template, interprets generated calls, and returns the structured API response that lets a harness execute them. The runtime never runs a generated shell command or Python expression itself.

Tool support is automatic for a recognized chat template; no extra launch flag is needed. `/health` reports the detected `tool_protocol`; `qwen_xml` is the model protocol covered by the GPU smoke test. The parser uses that template's `<tool_call><function=...><parameter=...>` format. A Hermes-style JSON format has CPU fixtures, but no other model has been validated on the GPU. Unknown templates reject requests containing tools instead of guessing a format. The saved model template and weights are unchanged.

## Launch and client settings

Select a local model directory; the installed backend is selected automatically:

```powershell
python run.py serve -m "MODEL_DIRECTORY" --cache-type q8 -c 4096 --alias exl3 --port 8000 --request-timeout 900
```

Normal launches need no `--config`. First build and register a compatible backend using [BUILD.md](BUILD.md). Wait for `/health` before connecting a harness. The server has no overall lifetime timeout; `--request-timeout 900` applies to each request separately.

Cache and MTP flags also apply to tools when supported by the model. Keep client context and output budgets within the server allocation, including tool definitions and draft reserve. [Cache options and limitations](KV-CACHE.md).

Configure a compatible client with these values:

| Setting | Value |
| --- | --- |
| API protocol | OpenAI Chat Completions |
| Base URL | `http://127.0.0.1:8000/v1` |
| Model | The server's `--alias`, here `exl3` |
| API key if required by client | Any local placeholder; loopback server has no authentication |
| Sampling | Greedy default; `temperature`, `top_p`, `top_k`, `min_p`, penalties, `seed` overridable, `n: 1` |
| Reasoning | `reasoning_content` split from `<think>` blocks; interleaved field `reasoning_content` |
| Tool capability | Enable in the client if it requires a capability declaration |

Clients speaking only Anthropic Messages, OpenAI Responses, legacy `functions`, or requiring strict constrained tool schemas need a protocol adapter or additional runtime support. Those APIs are not implemented here. Compatibility means the documented Chat Completions contract, not every harness protocol or every EXL3 model.

### Optional OpenCode example

OpenCode's provider model entry must advertise `tool_call: true`. A minimal example, merged into an existing configuration rather than replacing it:

```json
{
  "provider": {
    "exl3-amd": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "EXL3 AMD (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "local",
        "timeout": 900000
      },
      "models": {
        "exl3": {
          "name": "exl3",
          "limit": {"context": 4096, "output": 2048},
          "tool_call": true,
          "temperature": true,
          "reasoning": true,
          "interleaved": {"field": "reasoning_content"},
          "attachment": false,
          "modalities": {"input": ["text"], "output": ["text"]}
        }
      }
    }
  }
}
```

Select `exl3-amd/exl3` in OpenCode. Keep the harness's normal permissions: enabling model tool support does not grant tools permission to run. The output budget leaves room for reasoning: `max_tokens` counts reasoning plus visible tokens. OpenCode documents [provider/model selection](https://opencode.ai/docs/models/) and [tool permissions](https://opencode.ai/docs/permissions/).

## API contract

Send standard Chat Completions function definitions:

```json
{
  "model": "exl3",
  "messages": [{"role": "user", "content": "Look up asset ALPHA7."}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "lookup_asset",
      "description": "Read the asset inventory.",
      "parameters": {
        "type": "object",
        "properties": {"asset": {"type": "string"}},
        "required": ["asset"],
        "additionalProperties": false
      }
    }
  }],
  "tool_choice": "auto",
  "parallel_tool_calls": false,
  "max_tokens": 1024,
  "temperature": 0
}
```

A call returns assistant `tool_calls` with unique IDs, function names and JSON **string** arguments, and `finish_reason: "tool_calls"`. Thinking models also return `reasoning_content`; resend it with the assistant message on tool round trips. Execute calls in the harness, then append the returned assistant message and one `role: "tool"` message per call, with the matching `tool_call_id` and text result. Resend that history for the next turn. Every pending call needs a result, including a text error result when execution fails. Multiple results may arrive in any order; the adapter associates them by ID and normalizes their order for the native template.

Supported choices are `auto` (default with definitions), `none`, `required`, and a named function object. `parallel_tool_calls: false` limits a response to one call; it does not control harness scheduling. Choice constraints are enforced after generation. Arguments receive basic type/required-property/enum validation; the harness remains responsible for its full schema, authorization, and execution. `strict: true`, legacy `functions`/`function_call`, built-in hosted tools and the Responses API are not supported.

With tools enabled, the runtime buffers the generated response until it has validated every complete call. SSE then emits indexed `delta.tool_calls`, followed by `finish_reason: "tool_calls"`, optional usage, and `[DONE]`. It does not stream partial arguments. This adds delivery latency for tool-enabled turns; ordinary requests without tools retain token streaming. Usage still counts all generated protocol tokens.

Malformed, unknown, truncated, or constraint-violating calls return HTTP 502 `model_output_error` / `invalid_tool_call`; an already-open SSE stream receives the equivalent error event and `[DONE]`. No partial call is emitted. Increase `max_tokens` if the response reached its output limit. A protocol failure leaves the loaded GPU engine healthy; it is separate from a GPU execution failure.

## Plain PowerShell tool round trip

This example needs only PowerShell and the running server. Its tool implementation is a fixed in-memory lookup. It does not execute generated commands.

```powershell
$ErrorActionPreference = 'Stop'
$uri = 'http://127.0.0.1:8000/v1/chat/completions'
$model = 'exl3' # Match --alias
$tool = @{type='function'; function=@{
  name='lookup_asset'; description='Read the owner of an asset from inventory.'
  parameters=@{type='object'; properties=@{asset=@{type='string'}}; required=@('asset'); additionalProperties=$false}
}}
$messages = @(@{role='user'; content='Look up ALPHA7 and tell me its owner.'})
$body = @{model=$model; messages=$messages; tools=@($tool); temperature=0; max_tokens=1024
  tool_choice=@{type='function'; function=@{name='lookup_asset'}}; parallel_tool_calls=$false}
$first = Invoke-RestMethod $uri -Method Post -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 20)
$assistant = $first.choices[0].message
if ($first.choices[0].finish_reason -ne 'tool_calls' -or @($assistant.tool_calls).Count -ne 1) { throw 'Expected one tool call' }
$call = $assistant.tool_calls[0]
$arguments = $call.function.arguments | ConvertFrom-Json
if ($call.function.name -ne 'lookup_asset' -or $arguments.asset -ne 'ALPHA7') { throw 'Unexpected lookup' }
$result = @{owner='Mira-927'} | ConvertTo-Json -Compress
$messages += $assistant
$messages += @{role='tool'; tool_call_id=$call.id; content=$result}
$body = @{model=$model; messages=$messages; tools=@($tool); tool_choice='none'; temperature=0; max_tokens=512}
$final = Invoke-RestMethod $uri -Method Post -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 20)
if ($final.choices[0].message.content -notmatch 'Mira-927') { throw 'Tool result was not recovered' }
$final.choices[0].message.content
```

## Verification

`scripts/validate_server_tools.py --output artifacts/<unique-check>` runs real nonstream/SSE calls and in-memory tool-result turns against a monitored server, including multiple/reordered results, output truncation and recovery. It executes no generated command. `tests/test_tool_calls.py` covers the parser, while `tests/test_server.py` covers HTTP validation, streaming and lifecycle behavior without loading a GPU model. See [VALIDATION.md](VALIDATION.md) for release acceptance and remaining limits.
