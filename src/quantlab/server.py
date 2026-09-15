"""Bounded OpenAI-compatible HTTP API over an injected inference engine.

Main agent supplies the GPU engine; this module owns only HTTP, validation,
admission control, and streaming. No torch import, no model loading here.

Engine contract:
  - ``model_name: str``, ``context: int``
  - ``status() -> dict`` with a ``ready`` boolean plus counters.
  - ``prepare(*, messages=None, prompt=None, max_tokens=256, tools=None,
    tool_choice=None, parallel_tool_calls=True, sampling=None,
    template_kwargs=None)`` -> opaque prepared object; raises ``ValueError``
    for context/template problems. The tool kwargs are only passed for
    tool-related chat requests; ordinary requests keep the original three
    kwargs. ``sampling`` is the validated sampling dict (always passed);
    ``template_kwargs`` carries ``enable_thinking`` when the client sent it.
  - ``generate(prepared)`` -> async generator yielding dicts with ``text``
    (incremental string), optional ``reasoning`` (incremental reasoning
    string, chat only), and ``done`` (bool); the final item carries
    ``finish_reason`` (``'stop'``/``'length'``/``'tool_calls'``) and ``usage``
    (``prompt_tokens``/``completion_tokens``/``total_tokens``, plus an
    optional ``timings`` object with ``*_seconds`` and
    ``*_tokens_per_second`` fields). A tool-call turn carries a complete
    ``tool_calls`` list (OpenAI assistant shape, with JSON-string arguments)
    in ONE event; the engine never emits partial executable calls.
    Malformed/truncated/constraint-violating model output raises
    ``quantlab.tool_calls.ToolCallError`` (a ``ValueError``).

All prepare/generate work is serialized through one ``asyncio.Lock`` held for
the whole request, streaming included. A bounded waiting count caps queueing;
overflow is 503 while ``/health`` stays available.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse

from quantlab.sampling import check_sampling as _sampling_params
from quantlab.tool_calls import ToolCallError, loads_json

MAX_BODY_BYTES = 1024 * 1024  # 1 MiB request cap, enforced while reading.
DEFAULT_MAX_TOKENS = 256
MAX_MAX_TOKENS = 8192
_ACQUIRE_POLL = 0.005  # lock wait poll interval; keeps queued-disconnect checks simple.

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class _Invalid(ValueError):
    """Client validation failure -> 400."""


class _TooLarge(Exception):
    """Request body exceeded MAX_BODY_BYTES -> 413."""


class _Gone(Exception):
    """Client disconnected -> 499 (nobody reads it, but keep the shape)."""


def _error(status: int, message: str, type: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type, "code": code}},
    )


def _invalid(message: str) -> JSONResponse:
    return _error(400, message, "invalid_request_error", "invalid_request_error")


def _sse(data: str) -> str:
    return f"data: {data}\n\n"


def _sse_json(obj: Any) -> str:
    return _sse(json.dumps(obj, ensure_ascii=False))


def _sse_error(status: int, message: str, type: str, code: str) -> str:
    return _sse_json({"error": {"message": message, "type": type, "code": code}})


# ---------------------------------------------------------------------------
# Strict validation: text-only OpenAI subset + function tools; rejects instead
# of ignoring.
# ---------------------------------------------------------------------------

_CHAT_KEYS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "n",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "chat_template_kwargs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }
)
_COMPLETION_KEYS = frozenset(
    {"model", "prompt", "max_tokens", "stream", "stream_options", "n", "temperature",
     "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
     "frequency_penalty", "seed"}
)
_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_TOOL_CHOICE_STRINGS = frozenset({"auto", "none", "required"})
_MAX_TOOLS = 128


class _LocalRequestsOnly:
    """Reject browser cross-site requests before body parsing or GPU admission."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = scope.get('headers', [])
        def values(name):
            return [value.decode('latin-1') for key, value in headers if key.lower() == name]
        hosts = values(b'host')
        match = re.fullmatch(r'(127\.0\.0\.1|localhost)(?::([0-9]{1,5}))?',
                             hosts[0], re.IGNORECASE) if len(hosts) == 1 else None
        if not match or (match[2] is not None and not 1 <= int(match[2]) <= 65535):
            return await _error(400, 'invalid local Host header', 'invalid_request_error',
                                'invalid_host')(scope, receive, send)
        origins = values(b'origin')
        expected_origin = scope.get('scheme', 'http') + '://' + hosts[0].lower()
        if len(origins) > 1 or (origins and origins[0].lower() != expected_origin):
            return await _error(403, 'cross-origin requests are not allowed', 'invalid_request_error',
                                'invalid_origin')(scope, receive, send)
        if scope.get('method') == 'POST':
            types = values(b'content-type')
            if len(types) != 1 or types[0].split(';', 1)[0].strip().lower() != 'application/json':
                return await _error(415, 'Content-Type must be application/json', 'invalid_request_error',
                                    'unsupported_media_type')(scope, receive, send)
        return await self.app(scope, receive, send)


def _is_int(value: Any) -> bool:
    return type(value) is int  # bool excluded: type(True) is bool


def _is_bool(value: Any) -> bool:
    return type(value) is bool


def _is_number(value: Any) -> bool:
    return type(value) in (int, float)


def _check_model(body: dict, model_name: str) -> str:
    if "model" not in body:
        return model_name
    model = body["model"]
    if not isinstance(model, str):
        raise _Invalid("model must be a string")
    if model != model_name:
        raise _Invalid(f"model '{model}' not found; this server serves '{model_name}'")
    return model


def _check_max_tokens(body: dict, allow_alias: bool) -> int:
    values: dict[str, int] = {}
    for key in ("max_tokens", "max_completion_tokens") if allow_alias else ("max_tokens",):
        if key in body:
            value = body[key]
            if not _is_int(value) or not 1 <= value <= MAX_MAX_TOKENS:
                raise _Invalid(
                    f"{key} must be an integer between 1 and {MAX_MAX_TOKENS}"
                )
            values[key] = value
    if len(values) == 2 and values["max_tokens"] != values["max_completion_tokens"]:
        raise _Invalid("max_tokens and max_completion_tokens disagree")
    if "max_tokens" in values:
        return values["max_tokens"]
    if "max_completion_tokens" in values:
        return values["max_completion_tokens"]
    return DEFAULT_MAX_TOKENS


def _check_stream(body: dict) -> tuple[bool, bool]:
    stream = body.get("stream", False)
    if not _is_bool(stream):
        raise _Invalid("stream must be a boolean")
    include_usage = False
    if "stream_options" in body:
        options = body["stream_options"]
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise _Invalid("stream_options must be an object")
        for key in options:
            if key != "include_usage":
                raise _Invalid(f"unsupported parameter: 'stream_options.{key}'")
        include_usage = options.get("include_usage", False)
        if not _is_bool(include_usage):
            raise _Invalid("stream_options.include_usage must be a boolean")
    return stream, include_usage


def _check_sampling(body: dict) -> dict:
    try:
        params = _sampling_params(body)
        return {key: value for key, value in params.items() if key in body}
    except ValueError as exc:
        raise _Invalid(str(exc))


_TEMPLATE_KWARGS_KEYS = frozenset({"enable_thinking"})


def _check_template_kwargs(body: dict) -> dict | None:
    if "chat_template_kwargs" not in body:
        return None
    kwargs = body["chat_template_kwargs"]
    if not isinstance(kwargs, dict):
        raise _Invalid("chat_template_kwargs must be an object")
    for key in kwargs:
        if key not in _TEMPLATE_KWARGS_KEYS:
            raise _Invalid(f"unsupported parameter: 'chat_template_kwargs.{key}'")
    if "enable_thinking" in kwargs and not _is_bool(kwargs["enable_thinking"]):
        raise _Invalid("chat_template_kwargs.enable_thinking must be a boolean")
    return dict(kwargs)


def _normalize_content(value: Any, ctx: str, *, allow_null: bool) -> str | None:
    """Plain string or text-only content array -> string. Rejects media parts."""
    if isinstance(value, str):
        return value
    if value is None and allow_null:
        return None
    if isinstance(value, list):
        if not value:
            raise _Invalid(f"{ctx} must be a string or a non-empty content array")
        texts: list[str] = []
        for j, part in enumerate(value):
            if not isinstance(part, dict):
                raise _Invalid(f"{ctx}[{j}] must be an object")
            for key in part:
                if key not in ("type", "text"):
                    raise _Invalid(f"unsupported parameter: '{ctx}[{j}].{key}'")
            ptype = part.get("type")
            if ptype != "text":
                raise _Invalid(
                    f"{ctx}[{j}].type '{ptype}' is not supported; "
                    "only text content parts are allowed"
                )
            text = part.get("text")
            if not isinstance(text, str):
                raise _Invalid(f"{ctx}[{j}].text must be a string")
            texts.append(text)
        return "".join(texts)
    if value is None:
        raise _Invalid(f"{ctx} must be a string")
    raise _Invalid(
        f"{ctx} must be a string or a text-only content array; "
        "multimodal content parts are not supported"
    )


def _parse_tool_arguments(raw: Any, ctx: str) -> dict:
    """Assistant tool-call arguments: JSON string -> mapping for the template."""
    if not isinstance(raw, str):
        raise _Invalid(f"{ctx} must be a JSON string decoding to an object")

    try:
        decoded = loads_json(raw)
    except (ValueError, RecursionError):
        raise _Invalid(f"{ctx} must be a JSON string decoding to an object")
    if not isinstance(decoded, dict):
        raise _Invalid(f"{ctx} must be a JSON string decoding to an object")
    return decoded


def _check_tools(body: dict) -> list[dict] | None:
    if "tools" not in body:
        return None
    tools = body["tools"]
    if not isinstance(tools, list):
        raise _Invalid("tools must be a list of function definitions")
    if len(tools) > _MAX_TOOLS:
        raise _Invalid(f"tools must contain at most {_MAX_TOOLS} definitions")
    normalized: list[dict] = []
    seen: set[str] = set()
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise _Invalid(f"tools[{i}] must be an object")
        for key in tool:
            if key not in ("type", "function"):
                raise _Invalid(f"unsupported parameter: 'tools[{i}].{key}'")
        if tool.get("type") != "function":
            raise _Invalid(f"tools[{i}].type must be 'function'")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise _Invalid(f"tools[{i}].function must be an object")
        for key in function:
            if key not in ("name", "description", "parameters", "strict"):
                raise _Invalid(f"unsupported parameter: 'tools[{i}].function.{key}'")
        name = function.get("name")
        if not isinstance(name, str) or not _TOOL_NAME_RE.match(name):
            raise _Invalid(
                f"tools[{i}].function.name must match [A-Za-z0-9_-]{{1,64}}"
            )
        if name in seen:
            raise _Invalid(f"duplicate tool name '{name}'")
        seen.add(name)
        spec: dict[str, Any] = {"name": name}
        if "description" in function:
            if not isinstance(function["description"], str):
                raise _Invalid(f"tools[{i}].function.description must be a string")
            spec["description"] = function["description"]
        if "parameters" in function:
            params = function["parameters"]
            if not isinstance(params, dict) or params.get("type") != "object":
                raise _Invalid(
                    f"tools[{i}].function.parameters must be "
                    "an object with root type 'object'"
                )
            spec["parameters"] = params
        else:
            spec["parameters"] = {"type": "object", "properties": {}}
        if "strict" in function:
            strict = function["strict"]
            if strict is None:
                pass
            elif strict is False:
                spec["strict"] = False
            elif strict is True:
                raise _Invalid(
                    f"tools[{i}].function.strict true is not supported; "
                    "no grammar/strict schema enforcement is available"
                )
            else:
                raise _Invalid(f"tools[{i}].function.strict must be false or null")
        normalized.append({"type": "function", "function": spec})
    return normalized


def _check_tool_choice(body: dict, tools: list[dict] | None) -> str | dict:
    names = {t["function"]["name"] for t in tools} if tools else set()
    if "tool_choice" not in body:
        return "auto" if names else "none"
    choice = body["tool_choice"]
    if isinstance(choice, str):
        if choice not in _TOOL_CHOICE_STRINGS:
            raise _Invalid(
                "tool_choice must be one of auto, none, required "
                "or a named function object"
            )
        if choice == "required" and not names:
            raise _Invalid("tool_choice 'required' requires tools")
        return choice
    if isinstance(choice, dict):
        for key in choice:
            if key not in ("type", "function"):
                raise _Invalid(f"unsupported parameter: 'tool_choice.{key}'")
        if choice.get("type") != "function":
            raise _Invalid("tool_choice.type must be 'function'")
        function = choice.get("function")
        if not isinstance(function, dict):
            raise _Invalid("tool_choice.function must be an object")
        for key in function:
            if key not in ("name",):
                raise _Invalid(f"unsupported parameter: 'tool_choice.function.{key}'")
        name = function.get("name")
        if not isinstance(name, str) or not _TOOL_NAME_RE.match(name):
            raise _Invalid("tool_choice.function.name must match [A-Za-z0-9_-]{1,64}")
        if not names:
            raise _Invalid("tool_choice requires tools")
        if name not in names:
            raise _Invalid(f"unknown tool '{name}'")
        return {"type": "function", "function": {"name": name}}
    raise _Invalid(
        "tool_choice must be one of auto, none, required or a named function object"
    )


def _check_parallel_tool_calls(body: dict) -> bool:
    if "parallel_tool_calls" not in body:
        return True
    parallel = body["parallel_tool_calls"]
    if not _is_bool(parallel):
        raise _Invalid("parallel_tool_calls must be a boolean")
    return parallel


_ASSISTANT_KEYS = frozenset({"role", "content", "tool_calls", "reasoning_content"})
_TOOL_KEYS = frozenset({"role", "content", "tool_call_id", "name"})
_TEXT_KEYS = frozenset({"role", "content"})


def _check_messages(body: dict) -> list[dict]:
    if "messages" not in body:
        raise _Invalid("messages is required")
    messages = body["messages"]
    if not isinstance(messages, list) or not messages:
        raise _Invalid("messages must be a non-empty list")
    checked: list[dict] = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise _Invalid(f"messages[{i}] must be an object")
        role = message.get("role")
        if not isinstance(role, str) or role not in _ROLES:
            raise _Invalid(
                f"messages[{i}].role must be one of system, developer, user, "
                "assistant, tool"
            )
        allowed = (
            _ASSISTANT_KEYS if role == "assistant"
            else _TOOL_KEYS if role == "tool"
            else _TEXT_KEYS
        )
        for key in message:
            if key not in allowed:
                raise _Invalid(f"unsupported parameter: 'messages[{i}].{key}'")
        ctx = f"messages[{i}].content"
        if role == "assistant":
            if "content" in message:
                content = _normalize_content(message["content"], ctx, allow_null=True)
            elif "tool_calls" in message:
                content = None
            else:
                raise _Invalid(f"messages[{i}].content is required")
            calls: list[dict] | None = None
            if "tool_calls" in message:
                raw_calls = message["tool_calls"]
                if not isinstance(raw_calls, list) or not raw_calls:
                    raise _Invalid(f"messages[{i}].tool_calls must be a non-empty list")
                calls = []
                seen_call: set[str] = set()
                for k, call in enumerate(raw_calls):
                    cctx = f"messages[{i}].tool_calls[{k}]"
                    if not isinstance(call, dict):
                        raise _Invalid(f"{cctx} must be an object")
                    for key in call:
                        if key not in ("id", "type", "function"):
                            raise _Invalid(f"unsupported parameter: '{cctx}.{key}'")
                    call_id = call.get("id")
                    if not isinstance(call_id, str) or not call_id:
                        raise _Invalid(f"{cctx}.id must be a non-empty string")
                    if call_id in seen_call:
                        raise _Invalid(f"duplicate tool_call id '{call_id}'")
                    seen_call.add(call_id)
                    if call.get("type") != "function":
                        raise _Invalid(f"{cctx}.type must be 'function'")
                    function = call.get("function")
                    if not isinstance(function, dict):
                        raise _Invalid(f"{cctx}.function must be an object")
                    for key in function:
                        if key not in ("name", "arguments"):
                            raise _Invalid(
                                f"unsupported parameter: '{cctx}.function.{key}'"
                            )
                    name = function.get("name")
                    if not isinstance(name, str) or not _TOOL_NAME_RE.match(name):
                        raise _Invalid(
                            f"{cctx}.function.name must match [A-Za-z0-9_-]{{1,64}}"
                        )
                    args = _parse_tool_arguments(
                        function.get("arguments"), f"{cctx}.function.arguments"
                    )
                    calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    )
            if content is None and not calls:
                raise _Invalid(
                    f"messages[{i}].content may be null "
                    "only when tool_calls are present"
                )
            if "reasoning_content" in message:
                history = message["reasoning_content"]
                if history is not None and not isinstance(history, str):
                    raise _Invalid(f"messages[{i}].reasoning_content must be a string")
            entry: dict[str, Any] = {"role": role, "content": content}
            if message.get("reasoning_content") is not None:
                entry["reasoning_content"] = message["reasoning_content"]
            if calls is not None:
                entry["tool_calls"] = calls
            checked.append(entry)
        elif role == "tool":
            if "content" not in message:
                raise _Invalid(f"messages[{i}].content is required")
            content = _normalize_content(message["content"], ctx, allow_null=False)
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise _Invalid(
                    f"messages[{i}].tool_call_id must be a non-empty string"
                )
            entry = {"role": role, "content": content, "tool_call_id": call_id}
            if "name" in message:
                if not isinstance(message["name"], str):
                    raise _Invalid(f"messages[{i}].name must be a string")
                entry["name"] = message["name"]
            checked.append(entry)
        else:
            if "content" not in message:
                raise _Invalid(f"messages[{i}].content is required")
            checked.append(
                {
                    "role": role,
                    "content": _normalize_content(
                        message["content"], ctx, allow_null=False
                    ),
                }
            )
    # Cross-turn tool protocol: IDs unique, every result matches one pending
    # call, and no other turn interleaves before pending calls resolve.
    seen_ids: set[str] = set()
    pending: dict[str, str] = {}
    resolved: set[str] = set()
    for i, message in enumerate(checked):
        if message["role"] == "assistant" and "tool_calls" in message:
            if pending:
                raise _Invalid(
                    f"messages[{i}] arrives before all pending tool calls have results"
                )
            for call in message["tool_calls"]:
                if call["id"] in seen_ids:
                    raise _Invalid(f"duplicate tool_call id '{call['id']}'")
                seen_ids.add(call["id"])
                pending[call["id"]] = call["function"]["name"]
        elif message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id in resolved or (call_id in seen_ids and call_id not in pending):
                raise _Invalid(f"duplicate tool result for '{call_id}'")
            if call_id not in pending:
                raise _Invalid(
                    f"messages[{i}] has no matching assistant tool_call id '{call_id}'"
                )
            if "name" in message and message["name"] != pending[call_id]:
                raise _Invalid(
                    f"messages[{i}].name '{message['name']}' does not match "
                    f"tool call '{call_id}' for '{pending[call_id]}'"
                )
            del pending[call_id]
            resolved.add(call_id)
        elif pending:
            raise _Invalid(
                f"messages[{i}] arrives before all pending tool calls have results"
            )
    if pending:
        raise _Invalid("all tool calls must have results before generation")
    # The native template does not encode IDs: normalize each contiguous tool
    # group to the preceding assistant tool_calls order.
    ordered: list[dict] = []
    i = 0
    n = len(checked)
    while i < n:
        message = checked[i]
        if message["role"] == "assistant" and "tool_calls" in message:
            ordered.append(message)
            order = {c["id"]: k for k, c in enumerate(message["tool_calls"])}
            j = i + 1
            group: list[dict] = []
            while j < n and checked[j]["role"] == "tool":
                group.append(checked[j])
                j += 1
            group.sort(key=lambda m: order.get(m["tool_call_id"], len(order)))
            ordered.extend(group)
            i = j
        else:
            ordered.append(message)
            i += 1
    return ordered


def _check_prompt(body: dict) -> str:
    if "prompt" not in body:
        raise _Invalid("prompt is required")
    prompt = body["prompt"]
    if not isinstance(prompt, str):
        raise _Invalid("prompt must be a string")
    return prompt


def _reject_unknown(body: dict, allowed: frozenset) -> None:
    for key in body:
        if key not in allowed:
            raise _Invalid(f"unsupported parameter: '{key}'")


# ---------------------------------------------------------------------------
# Admission gate: bounded waiters + one lock held for the whole request.
# ---------------------------------------------------------------------------


class _Gate:
    def __init__(self, max_pending: int) -> None:
        self.lock = asyncio.Lock()
        self.waiting = 0
        self.max_pending = max_pending


async def _watch_disconnect(request: Request) -> bool:
    """Block on the receive channel until the client goes away.

    Starlette's ``is_disconnected()`` is a non-blocking poll, which misses
    disconnects that the server hasn't observed yet; a blocking receive both
    notices promptly and (on uvicorn) resumes socket reads so a POST
    disconnect is actually noticed. Only http.disconnect can arrive here: the
    body was fully consumed before the watcher starts.
    """
    try:
        while True:
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                return True
    except asyncio.CancelledError:
        raise
    except Exception:
        return True  # receive error: treat the client as gone (safe direction)


async def _acquire(gate: _Gate, disc: asyncio.Task, deadline: float) -> str:
    """Take the request lock or report why not: 'ok', 'full', 'gone', 'timeout'.

    Polling keeps this obviously correct: no waiter task to cancel/reap, no
    lock-leak edge, and the disconnect check runs immediately before and
    after the atomic locked()/acquire() pair, so a disconnected queued caller
    never reaches the engine.
    """
    if time.monotonic() >= deadline:
        return "timeout"
    await asyncio.sleep(0)  # Let an already-disconnected receive channel report it.
    if disc.done():
        return "gone"
    if not gate.lock.locked():
        await gate.lock.acquire()  # completes synchronously when free: atomic
        if disc.done():
            gate.lock.release()
            return "gone"
        return "ok"
    if gate.waiting >= gate.max_pending:
        return "full"
    gate.waiting += 1
    try:
        while True:
            if disc.done():
                return "gone"
            if time.monotonic() >= deadline:
                return "timeout"
            if not gate.lock.locked():
                await gate.lock.acquire()
                if disc.done():
                    gate.lock.release()
                    return "gone"
                return "ok"
            await asyncio.sleep(_ACQUIRE_POLL)
    finally:
        gate.waiting -= 1


async def _settle(task: asyncio.Task) -> None:
    """Cancel a task we own and reap it so nothing leaks or warns."""
    if task.done():
        if not task.cancelled():
            task.exception()  # retrieve; caller is aborting anyway
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, StopAsyncIteration, Exception):
        pass


async def _pull(gen: Any, disc: asyncio.Task, deadline: float):
    """Next engine item, racing client disconnect and the request deadline.

    Returns ``(item, None)`` or ``(None, reason)`` with reason 'gone',
    'timeout', or 'exhausted' (generator ended without a done flag).
    Engine exceptions propagate to the caller.
    """
    if disc.done():
        return None, "gone"
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None, "timeout"
    task = asyncio.create_task(gen.__anext__())
    try:
        done, _ = await asyncio.wait({task, disc}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        # Our task is being cancelled/closed: stop driving the engine. The
        # pending cancellation unwinds the generator (running its finally
        # blocks); the caller still acloses it, tolerating the in-flight race.
        await _settle(task)
        raise
    if disc.done():
        await _settle(task)
        return None, "gone"
    if not done:
        await _settle(task)
        return None, "timeout"
    try:
        return task.result(), None
    except StopAsyncIteration:
        return None, "exhausted"


async def _close(gen: Any) -> None:
    """Explicitly close the engine iterator; always called in a finally."""
    if gen is None:
        return
    await gen.aclose()


class _ClosingStreamResponse(StreamingResponse):
    """StreamingResponse that always closes the body iterator.

    Under ASGI >= 2.4 a disconnect surfaces as an OSError from send() in this
    frame, abandoning a suspended generator un-closed (upstream fixed the same
    with CancellingStreamResponse over sse-starlette); aclose() forces our
    generator's finally, which closes the engine iterator and frees the lock.
    """

    def __init__(self, content, *, gen, gate, disc, deadline):
        super().__init__(content, media_type="text/event-stream",
                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        self.gen, self.gate, self.disc, self.deadline = gen, gate, disc, deadline

    async def __call__(self, scope, receive, send) -> None:
        # Our watcher owns receive. Avoid a second competing Starlette watcher.
        # Response-level ownership also covers failure before body iteration starts.
        try:
            async with asyncio.timeout(max(0, self.deadline-time.monotonic())):
                await self.stream_response(send)
        except (OSError, TimeoutError):
            pass  # Failed/blocked sends cannot retain a generation slot.
        finally:
            try:
                try:
                    await self.body_iterator.aclose()
                finally:
                    await _close(self.gen)
            finally:
                await _settle(self.disc)
                self.gate.lock.release()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(engine: Any, request_timeout: float = 120, max_pending: int = 4) -> FastAPI:
    """Build the API app around an injected engine (no torch, no loader)."""
    if isinstance(request_timeout, bool) or not isinstance(
        request_timeout, (int, float)
    ):
        raise TypeError("request_timeout must be a number of seconds")
    if not request_timeout > 0:
        raise ValueError("request_timeout must be positive")
    if type(max_pending) is not int or max_pending < 0:
        raise ValueError("max_pending must be a non-negative integer")

    gate = _Gate(max_pending)
    app = FastAPI(title="quantlab-server")
    app.add_middleware(_LocalRequestsOnly)
    app.state.gate = gate  # introspection/testing; not part of the API

    async def _read_body(request: Request) -> bytes:
        length = request.headers.get("content-length")
        if length is not None:
            try:
                if int(length) > MAX_BODY_BYTES:
                    raise _TooLarge()
            except ValueError:
                pass  # fall through to streaming enforcement
        total = 0
        chunks: list[bytes] = []
        try:
            async for chunk in request.stream():
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_BODY_BYTES:
                    raise _TooLarge()  # abandon the iterator; ASGI >= 2.4 allows it
                chunks.append(chunk)
        except ClientDisconnect:
            raise _Gone()
        return b"".join(chunks)

    def _parse(body: bytes) -> dict:
        try:
            data = json.loads(body) if body else None
        except (ValueError, UnicodeDecodeError):
            raise _Invalid("request body must be valid JSON")
        if not isinstance(data, dict):
            raise _Invalid("request body must be a JSON object")
        return data

    async def _engine_ready() -> tuple[bool, dict]:
        try:
            info = engine.status()
            if inspect.isawaitable(info):
                info = await info
        except Exception:
            return False, {}
        if not isinstance(info, dict):
            return False, {}
        return bool(info.get("ready", False)), info

    @app.get("/health")
    async def health():
        ready, info = await _engine_ready()
        payload = dict(info)
        payload["status"] = "ok" if ready else "unavailable"
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "quantlab",
                }
            ],
        }

    async def _run(
        request: Request,
        allowed: frozenset,
        kind: str,  # 'chat' or 'completion'
    ):
        deadline = time.monotonic() + request_timeout
        try:
            raw = await asyncio.wait_for(_read_body(request), timeout=request_timeout)
        except TimeoutError:
            return _error(504, "request body timed out", "timeout_error", "request_timeout")
        except _TooLarge:
            return _error(
                413,
                "request body exceeds the 1 MiB limit",
                "invalid_request_error",
                "payload_too_large",
            )
        except _Gone:
            return _error(
                499, "client disconnected", "client_closed_request", "client_closed_request"
            )
        try:
            body = _parse(raw)
            _reject_unknown(body, allowed)
            sampling = _check_sampling(body)
            model = _check_model(body, engine.model_name)
            max_tokens = _check_max_tokens(body, allow_alias=(kind == "chat"))
            stream, include_usage = _check_stream(body)
            template_kwargs = _check_template_kwargs(body) if kind == "chat" else None
            if kind == "chat":
                messages = _check_messages(body)
                prompt = None
                tools = _check_tools(body)
                tool_choice = _check_tool_choice(body, tools)
                parallel = _check_parallel_tool_calls(body)
            else:
                messages = None
                prompt = _check_prompt(body)
                tools = None
                tool_choice = "none"
                parallel = True
        except _Invalid as exc:
            return _invalid(str(exc))

        ready, _ = await _engine_ready()
        if not ready:
            return _error(
                503, "engine is not ready", "service_unavailable", "engine_not_ready"
            )

        disc = asyncio.create_task(_watch_disconnect(request))
        locked = False
        streaming = False
        try:
            outcome = await _acquire(gate, disc, deadline)
            if outcome == "full":
                return _error(
                    503,
                    "server busy: too many pending requests",
                    "service_unavailable",
                    "queue_full",
                )
            if outcome == "gone":
                return _error(
                    499,
                    "client disconnected",
                    "client_closed_request",
                    "client_closed_request",
                )
            if outcome == "timeout":
                return _error(
                    504,
                    f"request timed out after {request_timeout:g}s",
                    "timeout_error",
                    "request_timeout",
                )
            locked = True
            ready, _ = await _engine_ready()
            if not ready:
                return _error(503, "engine is not ready", "service_unavailable", "engine_not_ready")
            try:
                if kind == "chat":
                    history_uses_tools = any(
                        m.get("role") == "tool" or "tool_calls" in m
                        for m in messages
                    )
                    explicit_tools = (
                        "tools" in body
                        or "tool_choice" in body
                        or "parallel_tool_calls" in body
                    )
                    if explicit_tools or history_uses_tools:
                        prepared = engine.prepare(
                            messages=messages,
                            max_tokens=max_tokens,
                            tools=tools if tools is not None else [],
                            tool_choice=tool_choice,
                            parallel_tool_calls=parallel,
                            sampling=sampling,
                            template_kwargs=template_kwargs,
                        )
                    else:
                        prepared = engine.prepare(
                            messages=messages, max_tokens=max_tokens,
                            sampling=sampling, template_kwargs=template_kwargs,
                        )
                else:
                    prepared = engine.prepare(prompt=prompt, max_tokens=max_tokens,
                                              sampling=sampling)
            except ToolCallError as exc:
                return _error(
                    502,
                    str(exc) or "invalid model tool output",
                    "model_output_error",
                    "invalid_tool_call",
                )
            except ValueError as exc:
                message = str(exc) or "prompt rejected by engine"
                lowered = message.lower()
                code = (
                    "context_length_exceeded"
                    if any(
                        word in lowered
                        for word in ("context", "too long", "exceed", "length")
                    )
                    else "invalid_request_error"
                )
                public_message = ('Prompt plus output exceeds configured context' if code == 'context_length_exceeded'
                                  else 'Prompt, template, or generation options rejected by engine')
                return _error(400, public_message, "invalid_request_error", code)
            except Exception as exc:
                return _error(
                    500, "engine error; consult private server logs", "internal_error", "internal_error"
                )
            try:
                gen = engine.generate(prepared)
            except ToolCallError as exc:
                return _error(
                    502,
                    str(exc) or "invalid model tool output",
                    "model_output_error",
                    "invalid_tool_call",
                )
            except Exception as exc:
                return _error(
                    500, "engine error; consult private server logs", "internal_error", "internal_error"
                )
            if not hasattr(gen, "__anext__") or not hasattr(gen, "aclose"):
                return _error(
                    500,
                    "engine returned an invalid generator",
                    "internal_error",
                    "internal_error",
                )

            created = int(time.time())
            if kind == "chat":
                completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            else:
                completion_id = f"cmpl-{uuid.uuid4().hex}"

            if stream:
                streaming = True  # generator below owns lock + watcher + iterator
                return _ClosingStreamResponse(
                    _stream_body(
                        gen, gate, disc, deadline, kind, model, completion_id,
                        created, include_usage, request_timeout,
                    ),
                    gen=gen, gate=gate, disc=disc, deadline=deadline,
                )

            # Non-streaming: race every pull against disconnect/deadline, not
            # just between tokens, and close the generator on every exit.
            texts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[dict] = []
            finish_reason = "stop"
            usage: dict | None = None
            try:
                while True:
                    item, reason = await _pull(gen, disc, deadline)
                    if reason == "gone":
                        return _error(
                            499,
                            "client disconnected",
                            "client_closed_request",
                            "client_closed_request",
                        )
                    if reason == "timeout":
                        return _error(
                            504,
                            f"request timed out after {request_timeout:g}s",
                            "timeout_error",
                            "request_timeout",
                        )
                    if reason == "exhausted":
                        raise RuntimeError("engine ended without a final event")
                    texts.append(item.get("text", "") or "")
                    if kind == "chat":
                        reasoning_parts.append(item.get("reasoning", "") or "")
                    emitted = item.get("tool_calls")
                    if isinstance(emitted, list) and emitted:
                        tool_calls.extend(emitted)
                    if item.get("done"):
                        finish_reason = item.get("finish_reason") or "stop"
                        usage = item.get("usage") or None
                        if usage is None:
                            raise RuntimeError("engine final event lacks usage")
                        break
            except ToolCallError as exc:
                return _error(
                    502,
                    str(exc) or "invalid model tool output",
                    "model_output_error",
                    "invalid_tool_call",
                )
            except Exception as exc:
                return _error(
                    500, "engine error; consult private server logs", "internal_error", "internal_error"
                )
            finally:
                locked = False
                try:
                    await _close(gen)
                finally:
                    gate.lock.release()
            text = "".join(texts)
            reasoning_text = "".join(reasoning_parts)
            if kind == "chat":
                if tool_calls:
                    chat_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": text if text else None,
                        "tool_calls": tool_calls,
                    }
                else:
                    chat_message = {"role": "assistant", "content": text}
                if reasoning_text:
                    chat_message["reasoning_content"] = reasoning_text
                return JSONResponse(
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": chat_message,
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": usage,
                    }
                )
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "text": text, "finish_reason": finish_reason}
                    ],
                    "usage": usage,
                }
            )
        finally:
            if not streaming:
                await _settle(disc)
                if locked:
                    gate.lock.release()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _run(request, _CHAT_KEYS, "chat")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _run(request, _COMPLETION_KEYS, "completion")

    return app


async def _stream_body(
    gen: Any,
    gate: _Gate,
    disc: asyncio.Task,
    deadline: float,
    kind: str,
    model: str,
    completion_id: str,
    created: int,
    include_usage: bool,
    request_timeout: float,
):
    """SSE body generator. Owns the lock, watcher, and engine iterator;

    ``_ClosingStreamResponse`` guarantees this is aclosed (running the
    finally) on disconnect, send failure, or cancellation.
    """
    try:
        if disc.done():
            return
        if time.monotonic() >= deadline:
            yield _sse_error(
                504,
                f"request timed out after {request_timeout:g}s",
                "timeout_error",
                "request_timeout",
            )
            return

        def chat_chunk(delta: dict, finished: str | None = None, usage=None) -> str:
            payload: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finished}],
            }
            if usage is not None:
                payload["usage"] = usage
            return _sse_json(payload)

        def text_chunk(text: str, finished: str | None = None, usage=None) -> str:
            payload = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": finished}],
            }
            if usage is not None:
                payload["usage"] = usage
            return _sse_json(payload)

        try:
            if kind == "chat":
                yield chat_chunk({"role": "assistant"})
            finish_reason = "stop"
            usage: dict | None = None
            tool_index = 0
            while True:
                item, reason = await _pull(gen, disc, deadline)
                if reason == "gone":
                    return  # client left: truncated stream, no [DONE]
                if reason == "timeout":
                    yield _sse_error(
                        504,
                        f"request timed out after {request_timeout:g}s",
                        "timeout_error",
                        "request_timeout",
                    )
                    return
                if reason == "exhausted":
                    raise RuntimeError("engine ended without a final event")
                text = item.get("text", "") or ""
                thinking = item.get("reasoning", "") or "" if kind == "chat" else ""
                if thinking:
                    yield chat_chunk({"reasoning_content": thinking})
                if text:
                    yield chat_chunk({"content": text}) if kind == "chat" else text_chunk(
                        text
                    )
                emitted = item.get("tool_calls")
                if kind == "chat" and isinstance(emitted, list) and emitted:
                    deltas = []
                    for call in emitted:
                        fn = call.get("function", {}) or {}
                        deltas.append(
                            {
                                "index": tool_index,
                                "id": call.get("id"),
                                "type": "function",
                                "function": {
                                    "name": fn.get("name"),
                                    "arguments": fn.get("arguments"),
                                },
                            }
                        )
                        tool_index += 1
                    yield chat_chunk({"tool_calls": deltas})
                if item.get("done"):
                    finish_reason = item.get("finish_reason") or "stop"
                    usage = item.get("usage") or None
                    if usage is None:
                        raise RuntimeError("engine final event lacks usage")
                    break
            if kind == "chat":
                yield chat_chunk({}, finish_reason)
            else:
                yield text_chunk("", finish_reason)
            if include_usage:
                yield _sse_json(dict(id=completion_id,
                                    object="chat.completion.chunk" if kind == "chat" else "text_completion",
                                    created=created, model=model, choices=[], usage=usage))
            yield _sse("[DONE]")
        except (GeneratorExit, asyncio.CancelledError):
            raise
        except ToolCallError as exc:
            try:
                yield _sse_error(
                    502,
                    str(exc) or "invalid model tool output",
                    "model_output_error",
                    "invalid_tool_call",
                )
                yield _sse("[DONE]")
            except (GeneratorExit, asyncio.CancelledError):
                raise
            except Exception:
                pass
        except Exception as exc:
            try:
                yield _sse_error(
                    500, "engine error; consult private server logs", "internal_error", "internal_error"
                )
            except (GeneratorExit, asyncio.CancelledError):
                raise
            except Exception:
                pass
    finally:
        # Response owns cleanup even if this body was never started.
        pass
