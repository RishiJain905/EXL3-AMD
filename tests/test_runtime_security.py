"""Focused HTTP boundary and privacy regressions for the local runtime."""

import asyncio
import json
import unittest

from quantlab.server import create_app


_USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class _Engine:
    model_name = "test-model"
    context = 4096

    def __init__(self):
        self.prepare_calls = 0
        self.prepare_error = None
        self.generate_error = None

    def status(self):
        return {"ready": True, "context": self.context}

    def prepare(self, **kwargs):
        self.prepare_calls += 1
        if self.prepare_error is not None:
            raise self.prepare_error
        return kwargs

    def generate(self, prepared):
        async def events():
            yield {"text": "partial", "done": False}
            if self.generate_error is not None:
                raise self.generate_error
            yield {"text": "", "done": True, "finish_reason": "stop", "usage": _USAGE}

        return events()


class _Exchange:
    def __init__(self, body=b""):
        self.body = body
        self.sent_body = False
        self.disconnect = asyncio.Event()
        self.messages = []

    async def receive(self):
        if not self.sent_body:
            self.sent_body = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self.disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(self, message):
        self.messages.append(message)


def _scope(path, *, method="POST", headers=()):
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
        "headers": list(headers),
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8092),
    }


async def _request(app, path, body=b"", *, method="POST", headers=()):
    exchange = _Exchange(body)
    await asyncio.wait_for(
        app(_scope(path, method=method, headers=headers), exchange.receive, exchange.send),
        timeout=2,
    )
    status = next(m["status"] for m in exchange.messages if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in exchange.messages if m["type"] == "http.response.body")
    return status, raw


def _chat_body(**extra):
    body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
    body.update(extra)
    return json.dumps(body).encode()


class BrowserBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_foreign_host_is_rejected_before_health_disclosure(self):
        app = create_app(_Engine(), request_timeout=1)
        status, raw = await _request(
            app,
            "/health",
            method="GET",
            headers=((b"host", b"attacker.example"),),
        )
        self.assertGreaterEqual(status, 400, raw)

    async def test_malformed_and_repeated_authority_headers_are_rejected(self):
        cases = (
            ((b"host", b"127.0.0.1.attacker.example"),),
            ((b"host", b"127.0.0.1:8092"), (b"host", b"attacker.example")),
            (
                (b"host", b"127.0.0.1:8092"),
                (b"origin", b"http://127.0.0.1:8092"),
                (b"origin", b"https://attacker.example"),
            ),
        )
        for headers in cases:
            with self.subTest(headers=headers):
                app = create_app(_Engine(), request_timeout=1)
                status, raw = await _request(app, "/health", method="GET", headers=headers)
                self.assertGreaterEqual(status, 400, raw)

    async def test_foreign_origin_cannot_trigger_generation(self):
        engine = _Engine()
        app = create_app(engine, request_timeout=1)
        status, raw = await _request(
            app,
            "/v1/chat/completions",
            _chat_body(),
            headers=(
                (b"host", b"127.0.0.1:8092"),
                (b"origin", b"https://attacker.example"),
                (b"content-type", b"application/json"),
            ),
        )
        self.assertGreaterEqual(status, 400, raw)
        self.assertEqual(engine.prepare_calls, 0)

    async def test_local_json_client_remains_available(self):
        engine = _Engine()
        app = create_app(engine, request_timeout=1)
        status, raw = await _request(
            app,
            "/v1/chat/completions",
            _chat_body(),
            headers=(
                (b"host", b"127.0.0.1:8092"),
                (b"content-type", b"application/json; charset=utf-8"),
            ),
        )
        self.assertEqual(status, 200, raw)
        self.assertEqual(engine.prepare_calls, 1)

    async def test_simple_text_plain_request_cannot_trigger_generation(self):
        engine = _Engine()
        app = create_app(engine, request_timeout=1)
        status, raw = await _request(
            app,
            "/v1/chat/completions",
            _chat_body(),
            headers=(
                (b"host", b"127.0.0.1:8092"),
                (b"content-type", b"text/plain"),
            ),
        )
        self.assertGreaterEqual(status, 400, raw)
        self.assertEqual(engine.prepare_calls, 0)


class ErrorRedactionTests(unittest.IsolatedAsyncioTestCase):
    async def _post(self, engine, *, stream=False):
        app = create_app(engine, request_timeout=1)
        return await _request(
            app,
            "/v1/chat/completions",
            _chat_body(stream=stream),
            headers=(
                (b"host", b"127.0.0.1:8092"),
                (b"content-type", b"application/json"),
            ),
        )

    async def test_prepare_exception_does_not_disclose_local_path(self):
        marker = "C:/private-user/model/weights.safetensors"
        engine = _Engine()
        engine.prepare_error = RuntimeError("failed to open " + marker)
        status, raw = await self._post(engine)
        self.assertEqual(status, 500, raw)
        self.assertNotIn(marker.encode(), raw)

    async def test_generation_exception_does_not_disclose_local_path(self):
        marker = "C:/private-user/runtime/process.log"
        engine = _Engine()
        engine.generate_error = RuntimeError("kernel failure near " + marker)
        status, raw = await self._post(engine)
        self.assertEqual(status, 500, raw)
        self.assertNotIn(marker.encode(), raw)

    async def test_stream_exception_does_not_disclose_local_path(self):
        marker = "C:/private-user/runtime/events.jsonl"
        engine = _Engine()
        engine.generate_error = RuntimeError("kernel failure near " + marker)
        status, raw = await self._post(engine, stream=True)
        self.assertEqual(status, 200, raw)
        self.assertNotIn(marker.encode(), raw)


if __name__ == "__main__":
    unittest.main()
