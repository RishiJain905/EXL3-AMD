"""Exercise the real serving engine with a deterministic CPU backend fixture."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import serve_exl3 as serving
from quantlab.sampling import DEFAULTS


class Tokens:
    def __init__(self, values):
        self.values = values

    def numel(self):
        return len(self.values)

    def flatten(self):
        return self

    def tolist(self):
        return self.values


class RuntimeControlsTests(unittest.TestCase):
    def make_engine(self):
        engine = serving.Engine.__new__(serving.Engine)
        engine.args = SimpleNamespace(batch_greedy=False)
        engine.ready = True
        engine.active = False
        engine.started = engine.completed = engine.cancelled = engine.failed = 0
        engine.invalid_tool_outputs = 0
        engine.reasoning = 'auto'
        engine.reasoning_format = 'auto'
        engine.sampling_defaults = dict(DEFAULTS, temperature=0.7, seed=123)
        engine.depth = 0
        engine.context = 4096
        engine.chunk_size = 256
        engine.draft_confidence = None
        engine.cfg = SimpleNamespace(eos_token_id_list=[99])
        engine.model = SimpleNamespace(caps={})
        engine.cache = engine.draft = engine.draft_cache = None
        engine.Sampler = lambda: 'argmax'
        engine.ComboSampler = lambda **kwargs: kwargs
        engine.memory = lambda: {}
        engine.record = lambda *args, **kwargs: None
        engine.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        engine.tokenizer = SimpleNamespace(actual_vocab_size=100,
            hf_tokenizer=SimpleNamespace(chat_template='<think></think>'),
            encode=lambda *args, **kwargs: Tokens([1, 2, 3, 4]),
            hf_render_chat_template=lambda messages, **kwargs:
                '<im_start>assistant\n<think>\n' if kwargs.get('enable_thinking', True)
                else '<im_start>assistant\n<think>\n</think>\n')
        return engine

    def test_prepare_respects_defaults_and_explicit_reasoning_override(self):
        engine = self.make_engine()
        prepared = engine.prepare(messages=[{'role': 'user', 'content': 'hi'}], sampling={})
        self.assertEqual(prepared['sampling']['temperature'], 0.7)
        self.assertEqual(prepared['sampling']['seed'], 123)
        self.assertTrue(prepared['prefix_open'])
        engine.reasoning = 'off'
        prepared = engine.prepare(messages=[{'role': 'user', 'content': 'hi'}],
                                  template_kwargs={'enable_thinking': True})
        self.assertTrue(prepared['split_reasoning'])
        self.assertTrue(prepared['prefix_open'])
        prepared = engine.prepare(messages=[{'role': 'user', 'content': 'hi'}],
                                  template_kwargs={'enable_thinking': False})
        self.assertFalse(prepared['split_reasoning'])

    def test_batch_argmax_rejects_sampling_and_penalties_per_request(self):
        engine = self.make_engine()
        engine.args.batch_greedy = True
        for params in ({'temperature': 0.8}, {'temperature': 0, 'repetition_penalty': 1.2}):
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, 'batch-greedy'):
                engine.prepare(prompt='hello', sampling=params)

    def run_generation(self, fragments, *, policy=None, prefix_open=False, split=True):
        engine = self.make_engine()
        clock = [100.0]
        jobs = []
        messages = []

        class Job:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.sequences = [SimpleNamespace(kv_position=0)]
                jobs.append(self)

            def is_prefill_done(self):
                return self.sequences[0].kv_position == 3

            def prefill(self, results):
                if not self.is_prefill_done():
                    clock[0] += 0.25
                    self.sequences[0].kv_position = 3

        class Generator:
            def __init__(self, *args, **kwargs):
                self.index = 0

            def enqueue(self, job):
                self.job = job

            def num_remaining_jobs(self):
                return self.index < len(fragments)

            def iterate(self):
                self.job.prefill([])
                clock[0] += 10.0
                text = fragments[self.index]
                self.index += 1
                return [{'text': text, 'token_ids': Tokens([5]),
                         'eos': self.index == len(fragments), 'eos_reason': 'stop_token'}]

            def clear_queue(self):
                pass

        engine.Job, engine.Generator = Job, Generator
        cache = SimpleNamespace(attention_backends=lambda _: {}, attention_profile_status=lambda: {})
        async def consume():
            prepared = dict(ids=Tokens([1, 2, 3, 4]), max_tokens=len(fragments),
                            sampling=dict(DEFAULTS, seed=123), tool_policy=policy,
                            split_reasoning=split, prefix_open=prefix_open, recognize_reasoning=True)
            gen = engine.generate(prepared)
            items = []
            try:
                async for item in gen:
                    items.append(item)
                    if item.get('done'):
                        break  # HTTP adapter closes immediately after this event.
            finally:
                await gen.aclose()
            return items

        with patch.object(serving.time, 'monotonic', lambda: clock[0]), \
             patch.object(serving, '_say', messages.append), \
             patch.object(serving, '_cache_precision', lambda: cache):
            items = asyncio.run(consume())
        return engine, items, jobs, messages

    def test_prefill_excludes_first_decode_and_done_log_precedes_close(self):
        engine, items, jobs, messages = self.run_generation(['plan', '</think>answer'], prefix_open=True)
        timings = items[-1]['usage']['timings']
        self.assertEqual(timings['prefill_seconds'], 0.25)
        self.assertEqual(timings['prefill_tokens'], 3)
        self.assertEqual(timings['first_token_seconds'], 10.25)
        self.assertEqual(timings['decode_tokens'], 1)
        self.assertEqual(timings['decode_seconds'], 10.0)
        self.assertEqual(jobs[0].kwargs['seed'], 123)
        self.assertTrue(any('done tokens=2' in row for row in messages))
        self.assertFalse(any('cancelled' in row for row in messages))
        self.assertEqual(engine.failed, 0)

    def test_single_emission_logs_unavailable_decode_rate_without_error(self):
        engine, items, jobs, messages = self.run_generation(['x'], prefix_open=True)
        self.assertIsNone(items[-1]['usage']['timings']['decode_seconds'])
        self.assertTrue(any('rate=n/a' in row for row in messages))
        self.assertTrue(engine.ready)

    def test_buffered_tools_strip_reasoning_started_in_prompt(self):
        policy = {'definitions': {'lookup': {'name': 'lookup', 'parameters': {'type': 'object'}}},
                  'choice': 'required', 'parallel': False, 'protocol': 'hermes_json'}
        # A bogus call inside reasoning must never reach the executable-call parser.
        fragments = ['<tool_call>{"name":"bogus"}</tool_call>',
                     '</think><tool_call>{"name":"lookup","arguments":{}}</tool_call>']
        engine, items, _, _ = self.run_generation(fragments, policy=policy, prefix_open=True)
        calls = [call for item in items for call in item.get('tool_calls', [])]
        self.assertEqual([call['function']['name'] for call in calls], ['lookup'])
        self.assertTrue(any('bogus' in item.get('reasoning', '') for item in items))
        self.assertEqual(items[-1]['finish_reason'], 'tool_calls')
        self.assertTrue(engine.ready)

    def test_raw_reasoning_display_still_protects_tool_parser(self):
        policy = {'definitions': {'lookup': {'name': 'lookup', 'parameters': {'type': 'object'}}},
                  'choice': 'auto', 'parallel': False, 'protocol': 'hermes_json'}
        fragments = ['<tool_call>{"name":"bogus"}</tool_call>', '</think>just an answer']
        engine, items, _, _ = self.run_generation(fragments, policy=policy, prefix_open=True, split=False)
        self.assertFalse(any(item.get('tool_calls') for item in items))
        self.assertFalse(any(item.get('reasoning') for item in items))
        self.assertIn('bogus', ''.join(item.get('text', '') for item in items))
        self.assertTrue(engine.ready)


class PrefixCacheTests(unittest.TestCase):
    def make_engine(self, prefix_cache=False):
        engine = RuntimeControlsTests().make_engine()
        engine.model_name = 'exl3'
        engine.tool_protocol = None
        engine.prefix_cache = prefix_cache
        engine._prefix_gen = None
        engine.prefix_hits = engine.prefix_misses = 0
        engine.prefix_cached_tokens = engine.prefix_computed_tokens = 0
        return engine

    def backend(self, engine, job_plans, gen_plans):
        builds, jobs, clock = [], [], [100.0]

        class Job:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                plan = job_plans.pop(0)
                self.sequences = [SimpleNamespace(kv_position=plan.get('kv_start', 0))]
                self.cached_pages = 0
                self.cached_tokens = plan.get('cached_tokens', 0)
                jobs.append(self)

            def is_prefill_done(self):
                return self.sequences[0].kv_position == 3

            def prefill(self, results):
                if not self.is_prefill_done():
                    clock[0] += 0.25
                    self.sequences[0].kv_position = 3

        class Generator:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                builds.append(self)
                self.pool_closed = False
                self.filter_pool = SimpleNamespace(shutdown=self.close_pool)

            def close_pool(self, **kwargs):
                self.pool_closed = True

            def enqueue(self, job):
                plan = gen_plans.pop(0)
                self.job = job
                self.fragments = plan['fragments']
                self.fail = plan.get('fail', False)
                self.cleanup_fail = plan.get('cleanup_fail', False)
                self.index = 0

            def num_remaining_jobs(self):
                return self.index < len(self.fragments)

            def iterate(self):
                if self.fail:
                    raise RuntimeError('boom')
                self.job.prefill([])
                clock[0] += 10.0
                text = self.fragments[self.index]
                self.index += 1
                return [{'text': text, 'token_ids': Tokens([5]),
                         'eos': self.index == len(self.fragments), 'eos_reason': 'stop_token'}]

            def clear_queue(self):
                if self.cleanup_fail:
                    raise RuntimeError('cleanup failed')

        engine.Job, engine.Generator = Job, Generator
        return builds, jobs, clock

    def consume(self, engine, clock, seed, max_tokens=2):
        prepared = dict(ids=Tokens([1, 2, 3, 4]), max_tokens=max_tokens,
                        sampling=dict(DEFAULTS, seed=seed), tool_policy=None,
                        split_reasoning=False, prefix_open=False, recognize_reasoning=False)
        cache = SimpleNamespace(attention_backends=lambda _: {}, attention_profile_status=lambda: {})

        async def go():
            gen = engine.generate(prepared)
            items = []
            try:
                async for item in gen:
                    items.append(item)
                    if item.get('done'):
                        break
            finally:
                await gen.aclose()
            return items

        with patch.object(serving.time, 'monotonic', lambda: clock[0]), \
             patch.object(serving, '_say', lambda message: None), \
             patch.object(serving, '_cache_precision', lambda: cache):
            return asyncio.run(go())

    def status(self, engine):
        cache = SimpleNamespace(attention_backends=lambda _: {}, attention_profile_status=lambda: {})
        with patch.object(serving, '_cache_precision', lambda: cache):
            return engine.status()

    def test_off_builds_fresh_generator_per_request(self):
        engine = self.make_engine(prefix_cache=False)
        builds, _, clock = self.backend(engine, [{}, {}],
                                        [{'fragments': ['a', 'b']}, {'fragments': ['c', 'd']}])
        self.consume(engine, clock, seed=1)
        self.consume(engine, clock, seed=2)
        self.assertEqual(len(builds), 2)
        self.assertIsNone(engine._prefix_gen)
        self.assertFalse(self.status(engine)['prefix_cache']['enabled'])

    def test_on_reuses_generator_with_fresh_job_and_sampler(self):
        engine = self.make_engine(prefix_cache=True)
        builds, jobs, clock = self.backend(engine, [{}, {'kv_start': 2, 'cached_tokens': 2}],
                                           [{'fragments': ['a', 'b']}, {'fragments': ['c', 'd']}])
        self.consume(engine, clock, seed=1)
        self.consume(engine, clock, seed=2)
        self.assertEqual(len(builds), 1)
        self.assertIs(engine._prefix_gen, builds[0])
        self.assertEqual(len(jobs), 2)
        self.assertIsNot(jobs[0], jobs[1])
        self.assertEqual((jobs[0].kwargs['seed'], jobs[1].kwargs['seed']), (1, 2))

    def test_hit_telemetry_separates_computed_and_cached(self):
        engine = self.make_engine(prefix_cache=True)
        builds, _, clock = self.backend(engine, [{}, {'kv_start': 2, 'cached_tokens': 2}],
                                        [{'fragments': ['a', 'b']}, {'fragments': ['c', 'd']}])
        miss = self.consume(engine, clock, seed=1)[-1]['usage']['timings']
        self.assertEqual((miss['prefill_computed_tokens'], miss['prefill_cached_tokens']), (3, 0))
        self.assertEqual(miss['prefill_tokens_per_second'], 12.0)
        hit = self.consume(engine, clock, seed=2)[-1]['usage']
        timings = hit['timings']
        self.assertEqual(hit['prompt_tokens'], 4)
        self.assertEqual(timings['prefill_tokens'], 3)
        self.assertEqual(timings['prefill_computed_tokens'], 1)
        self.assertEqual(timings['prefill_cached_tokens'], 2)
        self.assertEqual(timings['prefill_seconds'], 0.25)
        self.assertEqual(timings['prefill_wall_seconds'], 0.25)
        # Compute-only rate: 1 computed token over 0.25s, never 3/0.25.
        self.assertEqual(timings['prefill_tokens_per_second'], 4.0)
        self.assertEqual(timings['decode_tokens'], 1)
        self.assertEqual((engine.prefix_hits, engine.prefix_misses), (1, 1))
        self.assertEqual((engine.prefix_cached_tokens, engine.prefix_computed_tokens), (2, 4))
        cached = self.status(engine)['prefix_cache']
        self.assertTrue(cached['enabled'])
        self.assertEqual((cached['hits'], cached['misses']), (1, 1))

    def test_bounded_cache_configuration(self):
        engine = self.make_engine(prefix_cache=True)
        builds, _, clock = self.backend(engine, [{}], [{'fragments': ['a', 'b']}])
        self.consume(engine, clock, seed=1)
        kwargs = builds[0].kwargs
        self.assertEqual(kwargs['recurrent_cache_size'], 1024 ** 3)
        self.assertEqual(kwargs['cpu_cache_size'], 0)
        self.assertEqual(kwargs['max_batch_size'], 1)
        self.assertNotIn('recurrent_checkpoint_interval', kwargs)
        self.assertNotIn('recurrent_checkpoint_interval_pp', kwargs)

    def test_cancellation_discards_reusable_state(self):
        engine = self.make_engine(prefix_cache=True)
        builds, _, clock = self.backend(engine, [{}, {}, {}],
                                        [{'fragments': ['a', 'b']}, {'fragments': ['c', 'd']},
                                         {'fragments': ['e', 'f']}])
        self.consume(engine, clock, seed=1)
        self.assertIsNotNone(engine._prefix_gen)
        prepared = dict(ids=Tokens([1, 2, 3, 4]), max_tokens=2,
                        sampling=dict(DEFAULTS, seed=2), tool_policy=None,
                        split_reasoning=False, prefix_open=False, recognize_reasoning=False)
        cache = SimpleNamespace(attention_backends=lambda _: {}, attention_profile_status=lambda: {})

        async def cancel_midstream():
            gen = engine.generate(prepared)
            try:
                await gen.__anext__()
            finally:
                await gen.aclose()

        with patch.object(serving.time, 'monotonic', lambda: clock[0]), \
             patch.object(serving, '_say', lambda message: None), \
             patch.object(serving, '_cache_precision', lambda: cache):
            asyncio.run(cancel_midstream())
        self.assertIsNone(engine._prefix_gen)
        self.assertEqual(engine.cancelled, 1)
        self.assertTrue(engine.ready)
        self.consume(engine, clock, seed=3)
        self.assertEqual(len(builds), 2)
        self.assertIsNotNone(engine._prefix_gen)

    def test_error_discards_reusable_state(self):
        engine = self.make_engine(prefix_cache=True)
        builds, _, clock = self.backend(engine, [{}, {}],
                                        [{'fragments': ['a', 'b']}, {'fragments': ['c'], 'fail': True}])
        self.consume(engine, clock, seed=1)
        self.assertIsNotNone(engine._prefix_gen)
        with self.assertRaisesRegex(RuntimeError, 'boom'):
            self.consume(engine, clock, seed=2)
        self.assertIsNone(engine._prefix_gen)
        self.assertFalse(engine.ready)
        self.assertEqual(engine.failed, 1)

    def test_cleanup_failure_closes_pool_and_disables_reuse(self):
        engine = self.make_engine(prefix_cache=True)
        builds, _, clock = self.backend(engine, [{}],
                                        [{'fragments': ['a', 'b'], 'cleanup_fail': True}])
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            self.consume(engine, clock, seed=1)
        self.assertIsNone(engine._prefix_gen)
        self.assertTrue(builds[0].pool_closed)
        self.assertFalse(engine.ready)

    def test_shutdown_discards_persistent_generator(self):
        import tempfile
        import time
        engine = self.make_engine(prefix_cache=True)
        builds, _, clock = self.backend(engine, [{}], [{'fragments': ['a', 'b']}])
        self.consume(engine, clock, seed=1)
        self.assertIsNotNone(engine._prefix_gen)
        with tempfile.TemporaryDirectory() as tmp:
            engine.output = Path(tmp) / 'run'
            engine.output.mkdir()
            engine.t0 = time.monotonic()
            engine.events = (engine.output / 'events.jsonl').open('w', encoding='utf-8')
            cache = SimpleNamespace(attention_backends=lambda _: {}, attention_profile_status=lambda: {})
            with patch.object(serving, '_cache_precision', lambda: cache), \
                 patch.object(serving, '_say', lambda message: None):
                engine.shutdown()
        self.assertIsNone(engine._prefix_gen)
        self.assertFalse(engine.ready)


if __name__ == '__main__':
    unittest.main()
