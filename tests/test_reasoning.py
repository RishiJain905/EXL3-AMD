"""CPU-only reasoning splitter, sampling, and reasoning_content API tests."""
import asyncio
import json
import unittest

from quantlab.reasoning import ReasoningSplitter, has_unclosed_think, split_complete
from quantlab.sampling import DEFAULTS, is_plain_greedy, build_sampler, check_sampling, is_greedy
from quantlab.server import create_app
from quantlab.tool_calls import parse_response


class SplitterTests(unittest.TestCase):
    def test_no_markers_pass_through_raw(self):
        splitter = ReasoningSplitter()
        thinking, visible = splitter.feed("plain answer")
        t, v = splitter.flush()
        self.assertEqual((thinking + t, visible + v), ("", "plain answer"))

    def test_complete_block_splits(self):
        splitter = ReasoningSplitter()
        thinking, visible = splitter.feed(" <think>plan</think> done")
        tail_thinking, tail_visible = splitter.flush()
        self.assertEqual(thinking + tail_thinking, "plan")
        self.assertEqual(visible + tail_visible, "  done")

    def test_markers_split_across_fragments(self):
        text = "a<think>deep thought</think>b"
        for width in (1, 2, 3, 5, 7):
            splitter = ReasoningSplitter()
            thinking, visible = "", ""
            for i in range(0, len(text), width):
                t, v = splitter.feed(text[i:i + width])
                thinking += t
                visible += v
            t, v = splitter.flush()
            self.assertEqual((thinking + t, visible + v), ("deep thought", "ab"),
                             msg=f"width={width}")

    def test_prefix_open_continues_reasoning(self):
        self.assertTrue(has_unclosed_think("prompt <think>\n"))
        self.assertFalse(has_unclosed_think("user mentioned <think> earlier\nassistant:"))
        self.assertFalse(has_unclosed_think("prompt <think>x</think> tail"))
        self.assertFalse(has_unclosed_think("no markers"))
        splitter = ReasoningSplitter(prefix_open=True)
        thinking, visible = splitter.feed("more</think>answer")
        t, v = splitter.flush()
        self.assertEqual(thinking + t, "more")
        self.assertEqual(visible + v, "answer")

    def test_truncated_thinking_stays_reasoning(self):
        self.assertEqual(split_complete("head<think>cut off"),
                         ("cut off", "head"))
        splitter = ReasoningSplitter()
        thinking, visible = splitter.feed("head<think>cut")
        t, v = splitter.flush()
        self.assertEqual(thinking + t, "cut")
        self.assertEqual(visible + v, "head")

    def test_only_first_block_splits(self):
        thinking, visible = split_complete("<think>a</think>mid<think>b</think>")
        self.assertEqual(thinking, "a")
        self.assertEqual(visible, "mid<think>b</think>")

    def test_tool_text_inside_reasoning_never_parses(self):
        policy = {"definitions": {"get_time": {"name": "get_time", "parameters": {}}},
                  "choice": "auto", "parallel": True, "protocol": "hermes_json"}
        raw = '<think>{"name": "get_time", "arguments": {}}</think>done'
        thinking, visible = split_complete(raw)
        content, calls = parse_response(visible, policy, "stop")
        self.assertEqual(thinking, '{"name": "get_time", "arguments": {}}')
        self.assertEqual((content, calls), ("done", []))


class SamplingTests(unittest.TestCase):
    def test_defaults_are_greedy(self):
        params = check_sampling({})
        self.assertEqual(params, dict(DEFAULTS, seed=None))
        self.assertTrue(is_greedy(params))

    def test_ranges_reject(self):
        for body in ({"temperature": -0.1}, {"temperature": 2.1},
                     {"top_p": 0}, {"top_p": 1.5}, {"top_k": -1},
                     {"min_p": 1.0}, {"repetition_penalty": 0},
                     {"presence_penalty": 2.5}, {"frequency_penalty": -3},
                     {"seed": -1}, {"seed": 2 ** 63}, {"n": 2}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                check_sampling(body)

    def test_overrides_merge(self):
        params = check_sampling({"temperature": 0.7, "seed": 3})
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["seed"], 3)
        self.assertEqual(params["top_p"], 1.0)
        self.assertFalse(is_greedy(params))
        self.assertTrue(is_greedy({"temperature": 0.7, "top_k": 1}))

    def test_build_sampler_selects_vendor_class(self):
        made = []

        class Argmax:
            def __init__(self):
                made.append("argmax")

        class Combo:
            def __init__(self, **kwargs):
                made.append(("combo", kwargs))

        build_sampler({"temperature": 0, "top_k": 0}, ArgmaxSampler=Argmax,
                      ComboSampler=Combo)
        sampler = build_sampler(
            {"temperature": 0.8, "top_k": 0, "top_p": 0.9, "min_p": 0.1,
             "repetition_penalty": 1.2, "presence_penalty": 0.1,
             "frequency_penalty": 0.2},
            ArgmaxSampler=Argmax, ComboSampler=Combo)
        self.assertEqual(made[0], "argmax")
        self.assertEqual(made[1][0], "combo")
        self.assertEqual(made[1][1]["temperature"], 0.8)
        self.assertIsInstance(sampler, Combo)

    def test_greedy_penalties_are_not_ignored(self):
        self.assertFalse(is_plain_greedy(dict(DEFAULTS, repetition_penalty=1.2)))
        self.assertFalse(is_plain_greedy(dict(DEFAULTS, presence_penalty=0.5)))
        self.assertFalse(is_plain_greedy(dict(DEFAULTS, frequency_penalty=0.5)))
        made = build_sampler(dict(DEFAULTS, repetition_penalty=1.2),
                             ArgmaxSampler=lambda: 'argmax', ComboSampler=lambda **kw: kw)
        self.assertEqual(made['rep_p'], 1.2)
        self.assertEqual(made['temperature'], 0)


class _Engine:
    model_name = "test-model"
    context = 4096

    def __init__(self, script):
        self.script = script
        self.prepare_calls = []

    def status(self):
        return {"ready": True}

    def prepare(self, *, messages=None, prompt=None, max_tokens=256, **kwargs):
        self.prepare_calls.append({"messages": messages, "prompt": prompt,
                                   "max_tokens": max_tokens, **kwargs})
        return {}

    def generate(self, prepared):
        return self._gen()

    async def _gen(self):
        for item in self.script:
            yield dict(item)


def _run(app, path, body):
    sent = []

    async def receive():
        if sent:
            await asyncio.sleep(3600)
            return {"type": "http.disconnect"}
        sent.append(True)
        return {"type": "http.request", "body": body, "more_body": False}

    messages = []

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": path,
             "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"host", b"127.0.0.1"), (b"content-type", b"application/json"),
                         (b"content-length", str(len(body)).encode())],
             "client": ("test", 1), "server": ("test", 2)}
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, raw


class ReasoningResponseTests(unittest.TestCase):
    def test_omitted_sampling_does_not_reset_cli_defaults(self):
        from quantlab.server import _check_sampling
        self.assertEqual(_check_sampling({}), {})
        self.assertEqual(_check_sampling({'temperature': 0.8}), {'temperature': 0.8})

    def test_nonstream_carries_reasoning_content(self):
        script = [{"text": "Hel", "reasoning": "pla", "done": False},
                  {"text": "lo", "reasoning": "n", "done": False},
                  {"text": "", "done": True, "finish_reason": "stop",
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}]
        app = create_app(_Engine(script), request_timeout=5)
        status, raw = _run(app, "/v1/chat/completions",
                            json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
        self.assertEqual(status, 200, raw)
        message = json.loads(raw)["choices"][0]["message"]
        self.assertEqual(message["content"], "Hello")
        self.assertEqual(message["reasoning_content"], "plan")

    def test_empty_reasoning_omitted(self):
        script = [{"text": "hi", "done": True, "finish_reason": "stop",
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}]
        app = create_app(_Engine(script), request_timeout=5)
        status, raw = _run(app, "/v1/chat/completions",
                            json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
        self.assertEqual(status, 200, raw)
        self.assertNotIn("reasoning_content", json.loads(raw)["choices"][0]["message"])

    def test_stream_emits_reasoning_deltas(self):
        script = [{"text": "a", "reasoning": "r", "done": False},
                  {"text": "", "done": True, "finish_reason": "stop",
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}]
        app = create_app(_Engine(script), request_timeout=5)
        status, raw = _run(app, "/v1/chat/completions",
                            json.dumps({"messages": [{"role": "user", "content": "hi"}],
                                        "stream": True}).encode())
        self.assertEqual(status, 200, raw)
        deltas = []
        for line in raw.decode().splitlines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", []):
                    deltas.append(choice.get("delta", {}))
        self.assertIn({"reasoning_content": "r"}, deltas)
        self.assertIn({"content": "a"}, deltas)

    def test_reasoning_ignored_on_raw_completions(self):
        script = [{"text": "a", "reasoning": "r", "done": False},
                  {"text": "", "done": True, "finish_reason": "stop",
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}]
        app = create_app(_Engine(script), request_timeout=5)
        status, raw = _run(app, "/v1/completions",
                            json.dumps({"prompt": "hi"}).encode())
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["choices"][0]["text"], "a")

    def test_reasoning_history_preserved_for_native_template(self):
        script = [{"text": "ok", "done": True, "finish_reason": "stop",
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}]
        engine = _Engine(script)
        app = create_app(engine, request_timeout=5)
        body = {"messages": [{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "thinking done",
                              "reasoning_content": "prior plan"},
                             {"role": "user", "content": "go"}]}
        status, raw = _run(app, "/v1/chat/completions", json.dumps(body).encode())
        self.assertEqual(status, 200, raw)
        sent = engine.prepare_calls[0]["messages"]
        self.assertEqual(sent[1]["reasoning_content"], "prior plan")
        self.assertEqual(sent[1]["content"], "thinking done")


if __name__ == "__main__":
    unittest.main()
