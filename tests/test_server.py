"""CPU-only bounded server API tests; stdlib ASGI harness + fake engine."""
import asyncio
import json
import time
import unittest

from quantlab.server import MAX_BODY_BYTES, ToolCallError, create_app


# ---------------------------------------------------------------------------
# Fake engine (contract in src/quantlab/server.py docstring)
# ---------------------------------------------------------------------------

DEFAULT_USAGE = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
DEFAULT_ITEMS = [
    {"text": "Hello ", "done": False},
    {"text": "world", "done": True, "finish_reason": "stop", "usage": dict(DEFAULT_USAGE)},
]


class FakeEngine:
    def __init__(self, script=None, *, model_name="test-model", ready=True):
        self.model_name = model_name
        self.context = 4096
        self.ready = ready
        self.script = [dict(i) for i in (DEFAULT_ITEMS if script is None else script)]
        self.prepare_calls = []
        self.generate_calls = []
        self.prepare_error = None
        self.generate_error = None
        self.generate_error_after = -1  # yield index after which to raise
        self.item_delay = 0.0
        self.hang_event = None
        self.hang_at = -1  # hang before yielding script[hang_at]; len => after all
        self.invalid_generator = False
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.close_count = 0
        self.concurrent = 0
        self.max_concurrent = 0

    def status(self):
        return {"ready": self.ready, "context": self.context}

    def prepare(self, *, messages=None, prompt=None, max_tokens=256, **tool_kw):
        self.prepare_calls.append(
            {"messages": messages, "prompt": prompt, "max_tokens": max_tokens, **tool_kw}
        )
        if self.prepare_error is not None:
            raise self.prepare_error
        return {"messages": messages, "prompt": prompt, "max_tokens": max_tokens, **tool_kw}

    def generate(self, prepared):
        self.generate_calls.append(prepared)
        # Reached engine (lock held); set before first pull so queued tests
        # can observe promptly.
        try:
            self.started.set()
        except RuntimeError:
            pass
        if self.invalid_generator:
            return ["not-a-generator"]
        return self._gen()

    async def _gen(self):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            for i, item in enumerate(self.script):
                if self.hang_event is not None and i == self.hang_at:
                    await self.hang_event.wait()
                if self.item_delay:
                    await asyncio.sleep(self.item_delay)
                yield dict(item)
                if (
                    self.generate_error is not None
                    and i == self.generate_error_after
                ):
                    raise self.generate_error
            if self.hang_event is not None and self.hang_at >= len(self.script) and self.hang_at >= 0:
                await self.hang_event.wait()
        finally:
            self.concurrent -= 1
            self.close_count += 1
            try:
                self.closed.set()
            except RuntimeError:
                pass


# ---------------------------------------------------------------------------
# Stdlib ASGI harness: receive holds body once then blocks for disconnect.
# ---------------------------------------------------------------------------

def make_scope(path, method="POST", body=b"", headers=None):
    hdrs = [(b"host", b"127.0.0.1"), (b"content-type", b"application/json")]
    if body is not None and method in ("POST", "PUT", "PATCH"):
        hdrs.append((b"content-length", str(len(body)).encode()))
    if headers:
        hdrs.extend(headers)
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": hdrs,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }


class Harness:
    def __init__(self, body: bytes, *, body_delay=0.0, fail_from=None):
        self._body = body
        self._body_delay = body_delay
        self._sent_body = False
        self._disconnect = asyncio.Event()
        self.sent = []
        self._send_count = 0
        self._fail_from = fail_from  # send index from which to raise OSError

    async def receive(self):
        if not self._sent_body:
            if self._body_delay:
                await asyncio.sleep(self._body_delay)
            self._sent_body = True
            return {"type": "http.request", "body": self._body, "more_body": False}
        await self._disconnect.wait()
        return {"type": "http.disconnect"}

    def disconnect(self):
        self._disconnect.set()

    async def send(self, message):
        if self._fail_from is not None and self._send_count >= self._fail_from:
            self._send_count += 1
            raise OSError("fake send failure")
        self._send_count += 1
        self.sent.append(message)


def status_of(h):
    for m in h.sent:
        if m["type"] == "http.response.start":
            return m["status"]
    raise AssertionError(f"no response start in {h.sent!r}")


def body_bytes_of(h):
    return b"".join(
        m.get("body", b"") for m in h.sent if m["type"] == "http.response.body"
    )


def json_of(h):
    return json.loads(body_bytes_of(h).decode())


def sse_of(h):
    """Parse concatenated SSE body into [dict | '[DONE]'] payloads."""
    text = body_bytes_of(h).decode()
    parts = [p for p in text.split("\n\n") if p.strip()]
    out = []
    for p in parts:
        assert p.startswith("data: "), f"bad SSE part: {p!r}"
        data = p[len("data: "):]
        if data.strip() == "[DONE]":
            out.append("[DONE]")
        else:
            out.append(json.loads(data))
    return out


async def run_app(app, scope, harness, timeout=5.0):
    await asyncio.wait_for(app(scope, harness.receive, harness.send), timeout)


async def wait_until(pred, timeout=2.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        await asyncio.sleep(interval)
    return bool(pred())


def chat_body(**over):
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    body.update(over)
    return json.dumps(body).encode()


def completion_body(**over):
    body = {"prompt": "hello", "max_tokens": 16}
    body.update(over)
    return json.dumps(body).encode()


class ServerTestBase(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        # Let cancelled watcher/pull tasks settle; silence warnings.
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Health / models / success paths
# ---------------------------------------------------------------------------

class HealthModelsTests(ServerTestBase):
    async def test_health_ok(self):
        app = create_app(FakeEngine(), request_timeout=5, max_pending=4)
        h = Harness(b"", body_delay=0)
        await run_app(app, make_scope("/health", method="GET", body=b""), h)
        self.assertEqual(status_of(h), 200)
        payload = json_of(h)
        self.assertEqual(payload["status"], "ok")

    async def test_health_unavailable_when_not_ready(self):
        app = create_app(FakeEngine(ready=False), request_timeout=5, max_pending=4)
        h = Harness(b"")
        await run_app(app, make_scope("/health", method="GET", body=b""), h)
        self.assertEqual(status_of(h), 503)
        self.assertEqual(json_of(h)["status"], "unavailable")

    async def test_models(self):
        app = create_app(FakeEngine(), request_timeout=5, max_pending=4)
        h = Harness(b"")
        await run_app(app, make_scope("/v1/models", method="GET", body=b""), h)
        self.assertEqual(status_of(h), 200)
        payload = json_of(h)
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], "test-model")


class SuccessTests(ServerTestBase):
    async def test_chat_nonstream_success(self):
        eng = FakeEngine()
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h)
        self.assertEqual(status_of(h), 200)
        payload = json_of(h)
        self.assertTrue(payload["id"].startswith("chatcmpl-"))
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(
            payload["choices"][0]["message"],
            {"role": "assistant", "content": "Hello world"},
        )
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["usage"], DEFAULT_USAGE)
        self.assertFalse(app.state.gate.lock.locked())
        self.assertTrue(eng.closed.is_set())

    async def test_completion_nonstream_success(self):
        eng = FakeEngine()
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(completion_body())
        await run_app(app, make_scope("/v1/completions"), h)
        self.assertEqual(status_of(h), 200)
        payload = json_of(h)
        self.assertTrue(payload["id"].startswith("cmpl-"))
        self.assertEqual(payload["object"], "text_completion")
        self.assertEqual(payload["choices"][0]["text"], "Hello world")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["usage"], DEFAULT_USAGE)

    async def test_chat_stream_success(self):
        eng = FakeEngine()
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body(stream=True))
        await run_app(app, make_scope("/v1/chat/completions"), h)
        self.assertEqual(status_of(h), 200)
        payloads = sse_of(h)
        self.assertEqual(payloads[-1], "[DONE]")
        chunks = [p for p in payloads if p != "[DONE]"]
        self.assertGreaterEqual(len(chunks), 3)
        # Role chunk first.
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        # Content chunks then final finish chunk.
        texts = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
            if "choices" in c
        )
        self.assertEqual(texts, "Hello world")
        finish = [c for c in chunks if c["choices"][0]["finish_reason"]]
        self.assertEqual(len(finish), 1)
        self.assertEqual(finish[0]["choices"][0]["finish_reason"], "stop")
        # No usage without opt-in.
        self.assertFalse(any("usage" in c for c in chunks))
        self.assertFalse(app.state.gate.lock.locked())
        self.assertTrue(eng.closed.is_set())

    async def test_completion_stream_success(self):
        app = create_app(FakeEngine(), request_timeout=5, max_pending=4)
        h = Harness(completion_body(stream=True))
        await run_app(app, make_scope("/v1/completions"), h)
        self.assertEqual(status_of(h), 200)
        payloads = sse_of(h)
        self.assertEqual(payloads[-1], "[DONE]")
        chunks = [p for p in payloads if p != "[DONE]"]
        self.assertTrue(all(c["object"] == "text_completion" for c in chunks))
        self.assertEqual("".join(c["choices"][0]["text"] for c in chunks[:-1]), "Hello world")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    async def test_stream_include_usage_separate_chunk_chat(self):
        # Fixed behavior: usage arrives as its own SSE chunk, separate from
        # the finish_reason chunk (not merged into it).
        app = create_app(FakeEngine(), request_timeout=5, max_pending=4)
        h = Harness(chat_body(stream=True, stream_options={"include_usage": True}))
        await run_app(app, make_scope("/v1/chat/completions"), h)
        payloads = sse_of(h)
        self.assertEqual(payloads[-1], "[DONE]")
        chunks = [p for p in payloads if p != "[DONE]"]
        usage_chunks = [c for c in chunks if "usage" in c]
        self.assertEqual(len(usage_chunks), 1, f"chunks: {chunks!r}")
        self.assertEqual(usage_chunks[0]["usage"], DEFAULT_USAGE)
        choices = usage_chunks[0].get("choices", None)
        if choices == []:
            pass  # OpenAI-style empty-choices usage chunk
        else:
            self.assertTrue(
                all(c.get("finish_reason") is None for c in choices),
                f"usage chunk must not carry finish_reason: {usage_chunks[0]!r}",
            )
        finish = [c for c in chunks if c.get("choices") and any(x.get("finish_reason") for x in c["choices"])]
        self.assertEqual(len(finish), 1)
        self.assertNotIn("usage", finish[0], "usage must be separate from finish chunk")

    async def test_stream_include_usage_separate_chunk_completion(self):
        app = create_app(FakeEngine(), request_timeout=5, max_pending=4)
        h = Harness(completion_body(stream=True, stream_options={"include_usage": True}))
        await run_app(app, make_scope("/v1/completions"), h)
        payloads = sse_of(h)
        self.assertEqual(payloads[-1], "[DONE]")
        chunks = [p for p in payloads if p != "[DONE]"]
        usage_chunks = [c for c in chunks if "usage" in c]
        self.assertEqual(len(usage_chunks), 1, f"chunks: {chunks!r}")
        self.assertEqual(usage_chunks[0]["usage"], DEFAULT_USAGE)

    async def test_first_completed_fast_response(self):
        # Regression for asyncio.wait default ALL_COMPLETED: an immediately
        # ready engine must not wait out the full request timeout per token.
        eng = FakeEngine()
        app = create_app(eng, request_timeout=2.0, max_pending=4)
        h = Harness(chat_body())
        start = time.monotonic()
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        elapsed = time.monotonic() - start
        self.assertEqual(status_of(h), 200, json_of(h))
        self.assertLess(elapsed, 1.0, f"engine-ready pull waited too long: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Strict validation
# ---------------------------------------------------------------------------

class ValidationTests(ServerTestBase):
    async def _post(self, path, raw: bytes):
        app = create_app(FakeEngine(), request_timeout=5, max_pending=4)
        h = Harness(raw)
        await run_app(app, make_scope(path), h)
        return status_of(h), json_of(h)

    async def test_list_valued_role_rejected_400(self):
        # Unhashable role must be a 400, not an unhandled TypeError/500.
        status, payload = await self._post(
            "/v1/chat/completions",
            chat_body(messages=[{"role": ["user"], "content": "hi"}]),
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")

    async def test_bool_max_tokens_rejected_400(self):
        for path, raw in [
            ("/v1/chat/completions", chat_body(max_tokens=True)),
            ("/v1/completions", completion_body(max_tokens=True)),
            ("/v1/chat/completions", chat_body(max_tokens=False)),
        ]:
            with self.subTest(path=path, raw=raw):
                status, payload = await self._post(path, raw)
                self.assertEqual(status, 400, payload)

    async def test_strict_matrix(self):
        cases = [
            ("extra-key-chat", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stop": ["x"]}),
            ("extra-key-tools-completion", "/v1/completions", {"prompt": "hi", "tools": []}),
            ("extra-key-logit", "/v1/completions", {"prompt": "hi", "logit_bias": {}}),
            ("model-mismatch", "/v1/chat/completions", {"model": "other", "messages": [{"role": "user", "content": "hi"}]}),
            ("model-nonstr", "/v1/chat/completions", {"model": 7, "messages": [{"role": "user", "content": "hi"}]}),
            ("missing-messages", "/v1/chat/completions", {"max_tokens": 5}),
            ("empty-messages", "/v1/chat/completions", {"messages": []}),
            ("messages-image-content", "/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}]}),
            ("message-extra-key", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi", "name": "x"}]}),
            ("message-nondict", "/v1/chat/completions", {"messages": ["hi"]}),
            ("role-missing", "/v1/chat/completions", {"messages": [{"content": "hi"}]}),
            ("role-bad", "/v1/chat/completions", {"messages": [{"role": "admin", "content": "hi"}]}),
            ("role-none", "/v1/chat/completions", {"messages": [{"role": None, "content": "hi"}]}),
            ("role-dict", "/v1/chat/completions", {"messages": [{"role": {"r": "user"}, "content": "hi"}]}),
            ("missing-prompt", "/v1/completions", {"max_tokens": 5}),
            ("prompt-nonstring", "/v1/completions", {"prompt": ["hi"]}),
            ("max-tokens-zero", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}),
            ("max-tokens-huge", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8193}),
            ("max-tokens-float", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1.5}),
            ("max-tokens-str", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": "16"}),
            ("alias-disagree", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "max_completion_tokens": 6}),
            ("alias-in-completion", "/v1/completions", {"prompt": "hi", "max_completion_tokens": 5}),
            ("stream-nonstrict", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": "true"}),
            ("stream-int", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": 1}),
            ("stream-options-nondict", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream_options": True}),
            ("stream-options-unknown", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream_options": {"other": True}}),
            ("include-usage-nonstrict", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream_options": {"include_usage": "yes"}}),
            ("n-two", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "n": 2}),
            ("n-bool", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "n": True}),
            ("temperature-high", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "temperature": 2.5}),
            ("temperature-str", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "temperature": "0"}),
            ("top-p-zero", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "top_p": 0}),
            ("top-k-negative", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "top_k": -1}),
            ("min-p-one", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "min_p": 1.0}),
            ("repetition-zero", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "repetition_penalty": 0}),
            ("presence-high", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "presence_penalty": 3}),
            ("seed-negative", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "seed": -1}),
            ("template-kwargs-nondict", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": True}),
            ("template-kwargs-unknown", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"thinking": True}}),
            ("template-kwargs-nonstrict", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"enable_thinking": "yes"}}),
            ("template-kwargs-in-completion", "/v1/completions", {"prompt": "hi", "chat_template_kwargs": {"enable_thinking": True}}),
        ]
        for name, path, body in cases:
            with self.subTest(name=name):
                status, payload = await self._post(path, json.dumps(body).encode())
                self.assertEqual(status, 400, payload)
                self.assertIn("error", payload)

    async def test_allowed_sampling_passes(self):
        ok = {"messages": [{"role": "user", "content": "hi"}], "n": 1, "temperature": 0, "top_p": 1}
        status, payload = await self._post("/v1/chat/completions", json.dumps(ok).encode())
        self.assertEqual(status, 200, payload)
        eng = FakeEngine()
        app = create_app(eng, request_timeout=5, max_pending=4)
        ok2 = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7, "top_p": 0.9,
               "top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1, "presence_penalty": 0.1,
               "frequency_penalty": 0.2, "seed": 7,
               "chat_template_kwargs": {"enable_thinking": True}}
        h = Harness(json.dumps(ok2).encode())
        await run_app(app, make_scope("/v1/chat/completions"), h)
        self.assertEqual(status_of(h), 200, json_of(h))
        call = eng.prepare_calls[-1]
        self.assertEqual(call["sampling"]["temperature"], 0.7)
        self.assertEqual(call["sampling"]["seed"], 7)
        self.assertEqual(call["template_kwargs"], {"enable_thinking": True})

    async def test_invalid_json_400(self):
        status, payload = await self._post("/v1/chat/completions", b"{not json")
        self.assertEqual(status, 400, payload)

    async def test_non_object_json_400(self):
        for raw in [b"[]", b"3", b"null", b'""', b""]:
            with self.subTest(raw=raw):
                status, payload = await self._post("/v1/chat/completions", raw)
                self.assertEqual(status, 400, payload)


# ---------------------------------------------------------------------------
# Body cap / body deadline
# ---------------------------------------------------------------------------

class BodyCapTests(ServerTestBase):
    async def test_body_cap_413(self):
        big = "x" * (MAX_BODY_BYTES + 1024)
        raw = chat_body(messages=[{"role": "user", "content": big}])
        self.assertGreater(len(raw), MAX_BODY_BYTES)
        app = create_app(FakeEngine(), request_timeout=5, max_pending=4)
        h = Harness(raw)
        await run_app(app, make_scope("/v1/chat/completions"), h)
        self.assertEqual(status_of(h), 413)
        payload = json_of(h)
        self.assertEqual(payload["error"]["code"], "payload_too_large")

    async def test_body_cap_content_length_413(self):
        raw = chat_body()
        app = create_app(FakeEngine(), request_timeout=5, max_pending=4)
        scope = make_scope("/v1/chat/completions", headers=[(b"x-pad", b"1")])
        # Forge a lying large content-length; server must reject pre-read.
        scope["headers"] = [
            (k, v) for (k, v) in scope["headers"] if k != b"content-length"
        ] + [(b"content-length", str(MAX_BODY_BYTES + 1).encode())]
        h = Harness(raw)
        await run_app(app, scope, h)
        self.assertEqual(status_of(h), 413)

    async def test_body_deadline_enforced(self):
        # Fixed behavior: a body that arrives after the request deadline
        # times out instead of stalling the handler.
        app = create_app(FakeEngine(), request_timeout=0.05, max_pending=4)
        h = Harness(chat_body(), body_delay=0.5)
        start = time.monotonic()
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        elapsed = time.monotonic() - start
        self.assertEqual(status_of(h), 504, json_of(h))
        self.assertLess(elapsed, 2.0)


# ---------------------------------------------------------------------------
# Admission: overflow / serialization / timeout
# ---------------------------------------------------------------------------

class AdmissionTests(ServerTestBase):
    async def test_queue_overflow_503_and_health_available(self):
        gate_event = asyncio.Event()
        eng = FakeEngine(
            script=[{"text": "done", "done": True, "finish_reason": "stop",
                     "usage": dict(DEFAULT_USAGE)}],
        )
        eng.hang_event = gate_event
        eng.hang_at = 0
        app = create_app(eng, request_timeout=5, max_pending=0)
        h1 = Harness(chat_body())
        t1 = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h1, timeout=5))
        self.assertTrue(await wait_until(lambda: eng.started.is_set(), timeout=2))
        # Lock held, no waiters allowed: overflow must fail fast.
        h2 = Harness(chat_body())
        start = time.monotonic()
        await run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertEqual(status_of(h2), 503)
        self.assertEqual(json_of(h2)["error"]["code"], "queue_full")
        # Health stays available under saturation.
        hh = Harness(b"")
        await run_app(app, make_scope("/health", method="GET", body=b""), hh, timeout=5)
        self.assertEqual(status_of(hh), 200)
        gate_event.set()
        await asyncio.wait_for(t1, timeout=5)
        self.assertEqual(status_of(h1), 200)
        self.assertFalse(app.state.gate.lock.locked())

    async def test_queue_overflow_with_one_waiter(self):
        gate_event = asyncio.Event()
        eng = FakeEngine(
            script=[{"text": "d", "done": True, "finish_reason": "stop",
                     "usage": dict(DEFAULT_USAGE)}],
        )
        eng.hang_event = gate_event
        eng.hang_at = 0
        app = create_app(eng, request_timeout=5, max_pending=1)
        h1 = Harness(chat_body())
        t1 = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h1, timeout=5))
        self.assertTrue(await wait_until(lambda: eng.started.is_set(), timeout=2))
        h2 = Harness(chat_body())
        t2 = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5))
        self.assertTrue(await wait_until(lambda: app.state.gate.waiting >= 1, timeout=2))
        h3 = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h3, timeout=5)
        self.assertEqual(status_of(h3), 503)
        gate_event.set()
        await asyncio.wait_for(t1, timeout=5)
        await asyncio.wait_for(t2, timeout=5)
        self.assertEqual(status_of(h1), 200)
        self.assertEqual(status_of(h2), 200)
        self.assertEqual(eng.max_concurrent, 1)

    async def test_serialization_one_at_a_time(self):
        eng = FakeEngine()
        eng.item_delay = 0.02
        app = create_app(eng, request_timeout=5, max_pending=4)
        h1, h2 = Harness(chat_body()), Harness(completion_body())
        t1 = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h1, timeout=5))
        t2 = asyncio.create_task(run_app(app, make_scope("/v1/completions"), h2, timeout=5))
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=5)
        self.assertEqual(status_of(h1), 200)
        self.assertEqual(status_of(h2), 200)
        self.assertEqual(eng.max_concurrent, 1)
        self.assertFalse(app.state.gate.lock.locked())

    async def test_timeout_and_recovery(self):
        gate_event = asyncio.Event()
        eng = FakeEngine(
            script=[{"text": "late", "done": True, "finish_reason": "stop",
                     "usage": dict(DEFAULT_USAGE)}],
        )
        eng.hang_event = gate_event
        eng.hang_at = 0
        app = create_app(eng, request_timeout=0.05, max_pending=4)
        h1 = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h1, timeout=5)
        self.assertEqual(status_of(h1), 504)
        self.assertEqual(json_of(h1)["error"]["code"], "request_timeout")
        self.assertTrue(await wait_until(lambda: eng.closed.is_set(), timeout=2))
        self.assertFalse(app.state.gate.lock.locked())
        # Lock recovered: next request succeeds once the gate opens.
        gate_event.set()
        eng.closed.clear()
        h2 = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5)
        self.assertEqual(status_of(h2), 200)

    async def test_timeout_includes_queue_wait(self):
        # Queued time counts toward the deadline: with identical timeouts the
        # holder succeeds while the waiter, delayed by the queue, times out
        # during its own generation (it would succeed unqueued).
        eng = FakeEngine()
        eng.item_delay = 0.3  # ~0.6s per request; leave scheduling margin on WSL.
        app = create_app(eng, request_timeout=0.9, max_pending=4)
        h1 = Harness(chat_body())
        t1 = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h1, timeout=5))
        self.assertTrue(await wait_until(lambda: eng.started.is_set(), timeout=2))
        h2 = Harness(chat_body())
        t2 = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5))
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=5)
        self.assertEqual(status_of(h1), 200, json_of(h1))
        self.assertEqual(status_of(h2), 504, json_of(h2))
        self.assertFalse(app.state.gate.lock.locked())

    async def test_stream_timeout_truncates_and_recovers(self):
        gate_event = asyncio.Event()
        eng = FakeEngine(
            script=[{"text": "late", "done": True, "finish_reason": "stop",
                     "usage": dict(DEFAULT_USAGE)}],
        )
        eng.hang_event = gate_event
        eng.hang_at = 0
        app = create_app(eng, request_timeout=0.05, max_pending=4)
        h = Harness(chat_body(stream=True))
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertEqual(status_of(h), 200)  # headers already sent
        payloads = sse_of(h)
        # Truncated: never a successful [DONE]; an in-band timeout error may
        # or may not win the race with the response-level deadline.
        self.assertNotIn("[DONE]", payloads)
        errors = [p for p in payloads if isinstance(p, dict) and "error" in p]
        self.assertLessEqual(len(errors), 1, payloads)
        self.assertTrue(await wait_until(lambda: eng.closed.is_set(), timeout=2))
        self.assertFalse(app.state.gate.lock.locked())
        gate_event.set()
        h2 = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5)
        self.assertEqual(status_of(h2), 200)


# ---------------------------------------------------------------------------
# Disconnects
# ---------------------------------------------------------------------------

class DisconnectTests(ServerTestBase):
    async def test_queued_disconnect_never_starts(self):
        gate_event = asyncio.Event()
        eng = FakeEngine(
            script=[{"text": "d", "done": True, "finish_reason": "stop",
                     "usage": dict(DEFAULT_USAGE)}],
        )
        eng.hang_event = gate_event
        eng.hang_at = 0
        app = create_app(eng, request_timeout=5, max_pending=4)
        h1 = Harness(chat_body())
        t1 = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h1, timeout=5))
        self.assertTrue(await wait_until(lambda: eng.started.is_set(), timeout=2))
        h2 = Harness(chat_body())
        t2 = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5))
        self.assertTrue(await wait_until(lambda: app.state.gate.waiting >= 1, timeout=2))
        n_prepare = len(eng.prepare_calls)
        self.assertEqual(n_prepare, 1)
        h2.disconnect()
        await asyncio.wait_for(t2, timeout=5)
        self.assertEqual(status_of(h2), 499)
        self.assertEqual(len(eng.prepare_calls), n_prepare, "queued disconnect reached engine")
        self.assertEqual(len(eng.generate_calls), 1)
        gate_event.set()
        await asyncio.wait_for(t1, timeout=5)
        self.assertEqual(status_of(h1), 200)
        self.assertFalse(app.state.gate.lock.locked())

    async def test_active_nonstream_disconnect_closes_engine(self):
        gate_event = asyncio.Event()
        eng = FakeEngine(script=[{"text": "part", "done": False}])
        eng.hang_event = gate_event
        eng.hang_at = 1  # yield one token, then stall mid-generation
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body())
        t = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h, timeout=5))
        self.assertTrue(await wait_until(lambda: eng.started.is_set(), timeout=2))
        await asyncio.sleep(0.05)  # let the first pull complete; second pends
        h.disconnect()
        await asyncio.wait_for(t, timeout=5)
        self.assertEqual(status_of(h), 499)
        self.assertTrue(await wait_until(lambda: eng.closed.is_set(), timeout=2))
        self.assertFalse(app.state.gate.lock.locked())
        # Recovery still works.
        gate_event.set()

    async def test_active_stream_disconnect_closes_engine(self):
        gate_event = asyncio.Event()
        eng = FakeEngine(script=[{"text": "part", "done": False}])
        eng.hang_event = gate_event
        eng.hang_at = 1
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body(stream=True))
        t = asyncio.create_task(run_app(app, make_scope("/v1/chat/completions"), h, timeout=5))
        # Wait for headers + first SSE chunk, then drop the client.
        self.assertTrue(
            await wait_until(
                lambda: any(m["type"] == "http.response.body" for m in h.sent),
                timeout=2,
            )
        )
        h.disconnect()
        await asyncio.wait_for(t, timeout=5)
        self.assertTrue(await wait_until(lambda: eng.closed.is_set(), timeout=2))
        self.assertFalse(app.state.gate.lock.locked())
        body = body_bytes_of(h).decode()
        self.assertNotIn("[DONE]", body)
        gate_event.set()


# ---------------------------------------------------------------------------
# Send failures
# ---------------------------------------------------------------------------

class SendFailureTests(ServerTestBase):
    async def test_header_send_failure_releases_lock(self):
        # Fixed behavior: failed sends are swallowed at the response layer;
        # the request lock is released even when headers fail before the
        # engine iterator is ever iterated (unstarted generators have no
        # finally to observe, so only the lock + recovery are asserted).
        eng = FakeEngine()
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body(stream=True), fail_from=0)
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertTrue(eng.started.is_set())
        self.assertFalse(app.state.gate.lock.locked())
        # Server still serves after the failure.
        h2 = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5)
        self.assertEqual(status_of(h2), 200)
    async def test_midstream_send_failure_releases_lock(self):
        eng = FakeEngine()
        app = create_app(eng, request_timeout=5, max_pending=4)
        # 0: headers, 1: first body chunk; fail on the next send.
        h = Harness(chat_body(stream=True), fail_from=2)
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertTrue(await wait_until(lambda: eng.closed.is_set(), timeout=2))
        self.assertFalse(app.state.gate.lock.locked())
        body = body_bytes_of(h).decode()
        self.assertNotIn("[DONE]", body)
        h2 = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5)
        self.assertEqual(status_of(h2), 200)

# ---------------------------------------------------------------------------
# Engine errors / exhaustion / readiness
# ---------------------------------------------------------------------------

class EngineErrorTests(ServerTestBase):
    async def test_prepare_value_error_context_400(self):
        eng = FakeEngine()
        eng.prepare_error = ValueError("prompt exceeds context length")
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertEqual(status_of(h), 400)
        payload = json_of(h)
        self.assertEqual(payload["error"]["code"], "context_length_exceeded")
        self.assertFalse(app.state.gate.lock.locked())

    async def test_prepare_value_error_generic_400(self):
        eng = FakeEngine()
        eng.prepare_error = ValueError("bad template")
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertEqual(status_of(h), 400)
        self.assertEqual(json_of(h)["error"]["code"], "invalid_request_error")

    async def test_prepare_generic_500_and_recovery(self):
        eng = FakeEngine()
        eng.prepare_error = RuntimeError("boom")
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertEqual(status_of(h), 500)
        self.assertFalse(app.state.gate.lock.locked())
        eng.prepare_error = None
        h2 = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5)
        self.assertEqual(status_of(h2), 200)

    async def test_generate_raises_nonstream_500(self):
        eng = FakeEngine(script=[{"text": "a", "done": False}])
        eng.generate_error = RuntimeError("kernel fault")
        eng.generate_error_after = 0
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertEqual(status_of(h), 500)
        self.assertTrue(eng.closed.is_set())
        self.assertFalse(app.state.gate.lock.locked())

    async def test_generate_raises_stream_sse_error(self):
        eng = FakeEngine(script=[{"text": "a", "done": False}])
        eng.generate_error = RuntimeError("kernel fault")
        eng.generate_error_after = 0
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body(stream=True))
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        payloads = sse_of(h)
        errors = [p for p in payloads if isinstance(p, dict) and "error" in p]
        self.assertEqual(len(errors), 1, payloads)
        self.assertNotIn("[DONE]", payloads)
        self.assertTrue(eng.closed.is_set())
        self.assertFalse(app.state.gate.lock.locked())

    async def test_exhausted_iterator_nonstream_500(self):
        # Fixed behavior: engine ending without a done event is an error,
        # not a silent truncated 200.
        eng = FakeEngine(script=[{"text": "partial", "done": False}])
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertEqual(status_of(h), 500, body_bytes_of(h).decode())
        self.assertTrue(eng.closed.is_set())
        self.assertFalse(app.state.gate.lock.locked())

    async def test_exhausted_iterator_stream_error(self):
        eng = FakeEngine(script=[{"text": "partial", "done": False}])
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body(stream=True))
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        payloads = sse_of(h)
        errors = [p for p in payloads if isinstance(p, dict) and "error" in p]
        self.assertGreaterEqual(len(errors), 1, payloads)
        self.assertNotIn("[DONE]", payloads)
        self.assertTrue(eng.closed.is_set())
        self.assertFalse(app.state.gate.lock.locked())

    async def test_engine_not_ready_503(self):
        app = create_app(FakeEngine(ready=False), request_timeout=5, max_pending=4)
        h = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertEqual(status_of(h), 503)
        self.assertEqual(json_of(h)["error"]["code"], "engine_not_ready")

    async def test_invalid_generator_500(self):
        eng = FakeEngine()
        eng.invalid_generator = True
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertEqual(status_of(h), 500)
        self.assertFalse(app.state.gate.lock.locked())


# ---------------------------------------------------------------------------
# Tools: Chat Completions function-tool subset (fake engine; no XML parsing)
# ---------------------------------------------------------------------------

def _tool(name="get_weather", **fn_over):
    fn = {"name": name, "parameters": {"type": "object", "properties": {}}}
    fn.update(fn_over)
    return {"type": "function", "function": fn}


def _call_msg(call_id, name, args, content=None, msg_id=None):
    if not isinstance(args, str):
        args = json.dumps(args)
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": args}}
        ],
    }


def _tool_msg(call_id, content="ok", **over):
    body = {"role": "tool", "content": content, "tool_call_id": call_id}
    body.update(over)
    return body


def _tool_script(calls, usage=None, prefix_text=""):
    items = []
    if prefix_text:
        items.append({"text": prefix_text, "done": False})
    items.append({
        "text": "", "done": True, "finish_reason": "tool_calls",
        "usage": dict(usage or DEFAULT_USAGE), "tool_calls": calls,
    })
    return items


def _emit_call(call_id, name, args):
    if not isinstance(args, str):
        args = json.dumps(args)
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args}}


class ToolCallSuccessTests(ServerTestBase):
    async def _post(self, eng, raw):
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(raw)
        await run_app(app, make_scope("/v1/chat/completions"), h)
        return app, h

    async def test_single_tool_call_nonstream(self):
        calls = [_emit_call("call_1", "get_weather", {"city": "Paris"})]
        eng = FakeEngine(script=_tool_script(calls))
        app, h = await self._post(eng, chat_body(tools=[_tool()]))
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        payload = json_of(h)
        msg = payload["choices"][0]["message"]
        self.assertIsNone(msg["content"])
        self.assertEqual(msg["tool_calls"], calls)
        self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(payload["usage"], DEFAULT_USAGE)
        prep = eng.prepare_calls[0]
        self.assertEqual(len(prep["tools"]), 1)
        self.assertEqual(prep["tool_choice"], "auto")
        self.assertTrue(prep["parallel_tool_calls"])
        self.assertFalse(app.state.gate.lock.locked())

    async def test_multiple_tool_calls_nonstream(self):
        calls = [_emit_call("call_1", "get_weather", {"city": "Paris"}),
                 _emit_call("call_2", "get_time", {"zone": "CET"})]
        eng = FakeEngine(script=_tool_script(calls))
        _, h = await self._post(
            eng, chat_body(tools=[_tool(), _tool("get_time")]))
        self.assertEqual(status_of(h), 200)
        msg = json_of(h)["choices"][0]["message"]
        self.assertIsNone(msg["content"])
        self.assertEqual([c["id"] for c in msg["tool_calls"]],
                         ["call_1", "call_2"])

    async def test_text_prefix_with_tool_calls_keeps_content(self):
        calls = [_emit_call("call_1", "get_weather", {})]
        eng = FakeEngine(script=_tool_script(calls, prefix_text="checking "))
        _, h = await self._post(eng, chat_body(tools=[_tool()]))
        msg = json_of(h)["choices"][0]["message"]
        self.assertEqual(msg["content"], "checking ")
        self.assertEqual(msg["tool_calls"], calls)

    async def test_ordinary_request_keeps_tool_kwargs_absent(self):
        eng = FakeEngine()
        app, h = await self._post(eng, chat_body())
        self.assertEqual(status_of(h), 200)
        self.assertEqual(set(eng.prepare_calls[0]),
                         {"messages", "prompt", "max_tokens", "sampling", "template_kwargs"})
        self.assertIsNone(eng.prepare_calls[0]["template_kwargs"])
        self.assertFalse(app.state.gate.lock.locked())


class ToolHistoryTests(ServerTestBase):
    async def _post(self, eng, raw):
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(raw)
        await run_app(app, make_scope("/v1/chat/completions"), h)
        return app, h

    async def test_nullable_round_trip_reordered_results(self):
        history = [
            {"role": "user", "content": "weather and time?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "get_weather", "arguments": '{"city":"Paris"}'}},
                {"id": "call_2", "type": "function", "function": {
                    "name": "get_time", "arguments": '{"zone":"CET"}'}},
            ]},
            _tool_msg("call_2", "12:00"),  # reversed arrival order
            _tool_msg("call_1", "sunny"),
            {"role": "user", "content": "thanks"},
        ]
        eng = FakeEngine()
        app, h = await self._post(
            eng, chat_body(messages=history, tools=[_tool(), _tool("get_time")]))
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        sent = eng.prepare_calls[0]["messages"]
        asst = sent[1]
        self.assertIsNone(asst["content"])
        # Arguments parsed to mappings for the template; IDs preserved.
        self.assertEqual(asst["tool_calls"][0]["function"]["arguments"],
                         {"city": "Paris"})
        # Contiguous result group normalized to call order for the template.
        self.assertEqual([m["tool_call_id"] for m in sent[2:4]],
                         ["call_1", "call_2"])
        self.assertFalse(app.state.gate.lock.locked())

    async def test_single_round_trip_with_followup(self):
        history = [
            {"role": "user", "content": "weather?"},
            _call_msg("call_1", "get_weather", {"city": "Paris"}),
            _tool_msg("call_1", "sunny", name="get_weather"),
            {"role": "user", "content": "and tomorrow?"},
        ]
        eng = FakeEngine()
        _, h = await self._post(eng, chat_body(messages=history, tools=[_tool()]))
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        sent = eng.prepare_calls[0]["messages"]
        self.assertEqual(sent[2]["tool_call_id"], "call_1")

    async def test_past_tools_need_not_match_current(self):
        history = [
            {"role": "user", "content": "old"},
            _call_msg("call_9", "retired_tool", {"x": 1}),
            _tool_msg("call_9", "done"),
            {"role": "user", "content": "now?"},
        ]
        eng = FakeEngine()
        _, h = await self._post(eng, chat_body(messages=history, tools=[_tool()]))
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())

    async def test_developer_and_text_arrays_normalized(self):
        history = [
            {"role": "developer", "content": "be brief"},
            {"role": "user",
             "content": [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]},
        ]
        eng = FakeEngine()
        _, h = await self._post(eng, chat_body(messages=history))
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        sent = eng.prepare_calls[0]["messages"]
        self.assertEqual(sent[0], {"role": "developer", "content": "be brief"})
        self.assertEqual(sent[1], {"role": "user", "content": "hello"})

    async def test_history_only_request_passes_none_choice(self):
        history = [
            {"role": "user", "content": "hi"},
            _call_msg("call_1", "get_weather", {}),
            _tool_msg("call_1", "sunny"),
            {"role": "user", "content": "again"},
        ]
        eng = FakeEngine()
        _, h = await self._post(eng, chat_body(messages=history))
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        prep = eng.prepare_calls[0]
        self.assertEqual(prep["tools"], [])
        self.assertEqual(prep["tool_choice"], "none")
        self.assertTrue(prep["parallel_tool_calls"])


class ToolChoiceTests(ServerTestBase):
    async def _choice(self, **over):
        eng = FakeEngine()
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body(**over))
        await run_app(app, make_scope("/v1/chat/completions"), h)
        return eng, h

    async def test_named_choice(self):
        eng, h = await self._choice(
            tools=[_tool(), _tool("get_time")],
            tool_choice={"type": "function", "function": {"name": "get_time"}})
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        self.assertEqual(eng.prepare_calls[0]["tool_choice"],
                         {"type": "function", "function": {"name": "get_time"}})

    async def test_required_choice(self):
        eng, h = await self._choice(tools=[_tool()], tool_choice="required")
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        self.assertEqual(eng.prepare_calls[0]["tool_choice"], "required")

    async def test_none_choice_with_tools(self):
        eng, h = await self._choice(tools=[_tool()], tool_choice="none")
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        self.assertEqual(eng.prepare_calls[0]["tool_choice"], "none")
        self.assertEqual(json_of(h)["choices"][0]["message"]["content"], "Hello world")

    async def test_parallel_false_propagates(self):
        eng, h = await self._choice(tools=[_tool()], parallel_tool_calls=False)
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        self.assertFalse(eng.prepare_calls[0]["parallel_tool_calls"])

    async def test_default_parameters_schema(self):
        eng, h = await self._choice(
            tools=[{"type": "function", "function": {"name": "bare"}}])
        self.assertEqual(status_of(h), 200, body_bytes_of(h).decode())
        fn = eng.prepare_calls[0]["tools"][0]["function"]
        self.assertEqual(fn["parameters"], {"type": "object", "properties": {}})


class ToolValidationTests(ServerTestBase):
    async def _post(self, raw: bytes):
        app = create_app(FakeEngine(), request_timeout=5, max_pending=4)
        h = Harness(raw)
        await run_app(app, make_scope("/v1/chat/completions"), h)
        return status_of(h), json_of(h)

    async def test_tool_matrix_400(self):
        many = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(129)]
        cases = [
            ("tools-nonstr", {"tools": {}}),
            ("tools-too-many", {"tools": many}),
            ("tools-dup-names", {"tools": [_tool("a"), _tool("a")]}),
            ("tools-nondict", {"tools": ["x"]}),
            ("tools-bad-type", {"tools": [{"type": "retrieval", "function": {"name": "a"}}]}),
            ("tools-extra-key", {"tools": [{"type": "function", "function": {"name": "a"}, "extra": 1}]}),
            ("tools-fn-nondict", {"tools": [{"type": "function", "function": "a"}]}),
            ("tools-fn-extra", {"tools": [{"type": "function", "function": {"name": "a", "bogus": 1}}]}),
            ("tools-bad-name", {"tools": [{"type": "function", "function": {"name": "bad name!"}}]}),
            ("tools-empty-name", {"tools": [{"type": "function", "function": {"name": ""}}]}),
            ("tools-long-name", {"tools": [{"type": "function", "function": {"name": "x" * 65}}]}),
            ("tools-missing-name", {"tools": [{"type": "function", "function": {}}]}),
            ("tools-desc-nonstr", {"tools": [{"type": "function", "function": {"name": "a", "description": 5}}]}),
            ("tools-params-nondict", {"tools": [{"type": "function", "function": {"name": "a", "parameters": []}}]}),
            ("tools-params-no-type", {"tools": [{"type": "function", "function": {"name": "a", "parameters": {"properties": {}}}}]}),
            ("tools-params-array", {"tools": [{"type": "function", "function": {"name": "a", "parameters": {"type": "array"}}}]}),
            ("tools-strict-true", {"tools": [{"type": "function", "function": {"name": "a", "strict": True}}]}),
            ("tools-strict-str", {"tools": [{"type": "function", "function": {"name": "a", "strict": "yes"}}]}),
            ("choice-bad-string", {"tools": [_tool()], "tool_choice": "sometimes"}),
            ("choice-nonobj", {"tools": [_tool()], "tool_choice": 5}),
            ("choice-required-no-tools", {"tool_choice": "required"}),
            ("choice-named-no-tools", {"tool_choice": {"type": "function", "function": {"name": "a"}}}),
            ("choice-unknown-tool", {"tools": [_tool("a")], "tool_choice": {"type": "function", "function": {"name": "b"}}}),
            ("choice-bad-type", {"tools": [_tool()], "tool_choice": {"type": "other", "function": {"name": "get_weather"}}}),
            ("choice-extra-key", {"tools": [_tool()], "tool_choice": {"type": "function", "function": {"name": "get_weather"}, "extra": 1}}),
            ("choice-fn-extra", {"tools": [_tool()], "tool_choice": {"type": "function", "function": {"name": "get_weather", "x": 1}}}),
            ("parallel-nonstrict", {"tools": [_tool()], "parallel_tool_calls": "yes"}),
            ("parallel-int", {"parallel_tool_calls": 1}),
        ]
        for name, over in cases:
            with self.subTest(name=name):
                body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
                body.update(over)
                status, payload = await self._post(json.dumps(body).encode())
                self.assertEqual(status, 400, payload)

    async def test_history_matrix_400(self):
        u = {"role": "user", "content": "hi"}
        cases = [
            ("orphan-result", [u, _tool_msg("call_zzz", "x")]),
            ("duplicate-result", [u, _call_msg("call_1", "get_weather", {}),
                                  _tool_msg("call_1", "a"), _tool_msg("call_1", "b")]),
            ("duplicate-call-ids", [u, _call_msg("call_1", "get_weather", {}),
                                    _tool_msg("call_1", "a"),
                                    _call_msg("call_1", "get_time", {}),
                                    _tool_msg("call_1", "b")]),
            ("user-before-pending", [u, _call_msg("call_1", "get_weather", {}),
                                     {"role": "user", "content": "wait"}]),
            ("assistant-before-pending", [u, _call_msg("call_1", "get_weather", {}),
                                          {"role": "assistant", "content": "hm"}]),
            ("unresolved-at-end", [u, _call_msg("call_1", "get_weather", {})]),
            ("partial-group", [u, {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
                _tool_msg("call_1", "a"), {"role": "user", "content": "early"}]),
            ("args-nonstr", [u, {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {}}}]}]),
            ("args-bad-json", [u, _call_msg("call_1", "get_weather", "{oops")]),
            ("args-non-object", [u, _call_msg("call_1", "get_weather", "[1,2]")]),
            ("args-nan", [u, _call_msg("call_1", "get_weather", '{"x": NaN}')]),
            ("args-infinity", [u, _call_msg("call_1", "get_weather", '{"x": Infinity}')]),
            ("tool-name-mismatch", [u, _call_msg("call_1", "get_weather", {}),
                                    _tool_msg("call_1", "a", name="get_time")]),
            ("tool-missing-id", [u, {"role": "tool", "content": "x"}]),
            ("tool-empty-id", [u, _tool_msg("", "x")]),
            ("assistant-null-no-calls", [u, {"role": "assistant", "content": None}]),
            ("assistant-bad-call-type", [u, {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "other", "function": {"name": "get_weather", "arguments": "{}"}}]}]),
            ("assistant-empty-id", [u, {"role": "assistant", "content": None, "tool_calls": [
                {"id": "", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]}]),
            ("assistant-bad-name", [u, {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "bad name", "arguments": "{}"}}]}]),
            ("assistant-extra-key", [u, {"role": "assistant", "content": "x", "name": "bot"}]),
            ("audio-part", [u, {"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]),
            ("empty-parts", [u, {"role": "user", "content": []}]),
            ("part-extra-key", [u, {"role": "user", "content": [{"type": "text", "text": "hi", "cache": 1}]}]),
        ]
        for name, messages in cases:
            with self.subTest(name=name):
                status, payload = await self._post(chat_body(messages=messages))
                self.assertEqual(status, 400, payload)

    async def test_invalid_json_arguments_with_otherwise_resolved_history(self):
        for raw in ('{"x":1e999}', '{"x":1,"x":2}', '{"x":NaN}', '[1]', '{oops'):
            with self.subTest(raw=raw):
                history = [{"role": "user", "content": "Check"},
                           _call_msg("call_1", "get_weather", raw),
                           _tool_msg("call_1", "result")]
                status, payload = await self._post(chat_body(messages=history))
                self.assertEqual(status, 400, payload)
                self.assertIn('arguments', payload['error']['message'])

    async def test_strict_false_and_null_accepted(self):
        for strict in (False, None):
            with self.subTest(strict=strict):
                status, payload = await self._post(chat_body(
                    tools=[{"type": "function",
                            "function": {"name": "a", "strict": strict}}]))
                self.assertEqual(status, 200, payload)


class ToolStreamTests(ServerTestBase):
    async def test_sse_tool_calls_indexes_and_args(self):
        calls = [_emit_call("call_10", "get_weather", {"city": "Paris"}),
                 _emit_call("call_11", "get_time", {"zone": "CET"})]
        eng = FakeEngine(script=_tool_script(calls))
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body(stream=True, stream_options={"include_usage": True},
                              tools=[_tool(), _tool("get_time")]))
        await run_app(app, make_scope("/v1/chat/completions"), h)
        self.assertEqual(status_of(h), 200)
        payloads = sse_of(h)
        self.assertEqual(payloads[-1], "[DONE]")
        chunks = [p for p in payloads if p != "[DONE]"]
        tool_deltas = [c["choices"][0]["delta"]["tool_calls"] for c in chunks
                       if c.get("choices") and "tool_calls" in c["choices"][0]["delta"]]
        self.assertEqual(len(tool_deltas), 1)
        self.assertEqual([d["index"] for d in tool_deltas[0]], [0, 1])
        self.assertEqual([d["id"] for d in tool_deltas[0]], ["call_10", "call_11"])
        self.assertEqual(tool_deltas[0][0]["function"]["name"], "get_weather")
        self.assertEqual(tool_deltas[0][0]["function"]["arguments"], '{"city": "Paris"}')
        self.assertEqual(tool_deltas[0][1]["function"]["arguments"], '{"zone": "CET"}')
        finish = [c for c in chunks if c.get("choices") and c["choices"][0]["finish_reason"]]
        self.assertEqual(len(finish), 1)
        self.assertEqual(finish[0]["choices"][0]["finish_reason"], "tool_calls")
        usage_chunks = [c for c in chunks if "usage" in c]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["usage"], DEFAULT_USAGE)
        self.assertFalse(app.state.gate.lock.locked())
        self.assertTrue(eng.closed.is_set())


class ToolCallErrorTests(ServerTestBase):
    async def test_nonstream_502_and_recovery(self):
        eng = FakeEngine(script=[{"text": "part", "done": False}])
        eng.generate_error = ToolCallError("truncated tool markup")
        eng.generate_error_after = 0
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body(tools=[_tool()]))
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        self.assertEqual(status_of(h), 502, body_bytes_of(h).decode())
        payload = json_of(h)
        self.assertEqual(payload["error"]["type"], "model_output_error")
        self.assertEqual(payload["error"]["code"], "invalid_tool_call")
        self.assertTrue(eng.closed.is_set())
        self.assertFalse(app.state.gate.lock.locked())
        eng.generate_error = None
        eng.script = [dict(i) for i in DEFAULT_ITEMS]
        h2 = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5)
        self.assertEqual(status_of(h2), 200)
    async def test_stream_error_plus_done_and_recovery(self):
        eng = FakeEngine(script=[{"text": "part", "done": False}])
        eng.generate_error = ToolCallError("bad tool args")
        eng.generate_error_after = 0
        app = create_app(eng, request_timeout=5, max_pending=4)
        h = Harness(chat_body(stream=True, tools=[_tool()]))
        await run_app(app, make_scope("/v1/chat/completions"), h, timeout=5)
        payloads = sse_of(h)
        self.assertEqual(payloads[-1], "[DONE]")
        errors = [p for p in payloads if isinstance(p, dict) and "error" in p]
        self.assertEqual(len(errors), 1, payloads)
        self.assertEqual(errors[0]["error"]["type"], "model_output_error")
        self.assertEqual(errors[0]["error"]["code"], "invalid_tool_call")
        self.assertTrue(await wait_until(lambda: eng.closed.is_set(), timeout=2))
        self.assertFalse(app.state.gate.lock.locked())
        eng.generate_error = None
        eng.closed.clear()
        eng.script = [dict(i) for i in DEFAULT_ITEMS]
        h2 = Harness(chat_body())
        await run_app(app, make_scope("/v1/chat/completions"), h2, timeout=5)
        self.assertEqual(status_of(h2), 200)

if __name__ == "__main__":
    unittest.main()
