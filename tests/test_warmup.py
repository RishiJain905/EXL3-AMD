"""CPU tests for the serving startup warm-up plan, flag and state handling."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import serve_exl3 as serving
from quantlab.methods.exl3 import warmup
from quantlab.sampling import DEFAULTS


def _cdiv(a, b):
    return -(-a // b)


class PlanTests(unittest.TestCase):
    def test_table_widths_cross_every_16_page_boundary(self):
        widths = warmup.table_widths(512)
        self.assertEqual(widths, sorted(set(widths)))
        self.assertTrue(set(range(1, 17)) <= set(widths))
        for edge in range(16, 513, 16):
            self.assertIn(edge - 1, widths)
            self.assertIn(edge, widths)
            if edge < 512:
                self.assertIn(edge + 1, widths)
        self.assertEqual((widths[0], widths[-1]), (1, 512))
        self.assertLess(len(widths), 120)
        self.assertEqual(warmup.table_widths(4), [1, 2, 3, 4])
        self.assertEqual(warmup.table_widths(40)[-1], 40)

    def test_query_rows_cover_decode_and_every_prefill_block_in_both_residues(self):
        for chunk in (256, 1024, 2048):
            rows = warmup.query_rows(chunk)
            self.assertEqual(rows[:16], list(range(1, 17)))
            self.assertEqual(rows[-1], chunk)
            for block_m in (32, 64, 128):
                for block in range(1, _cdiv(chunk, block_m) + 1):
                    members = [n for n in rows if n > 16 and _cdiv(n, block_m) == block]
                    with self.subTest(chunk=chunk, block_m=block_m, block=block):
                        self.assertTrue(any(n % 16 == 0 for n in members))
                        self.assertTrue(any(n % 16 for n in members))

    def test_prefill_segments_cover_conv_buckets_in_both_residues(self):
        self.assertEqual(warmup.prefill_segments(256), [256, 47, 48, 95, 96, 191, 192])
        segments = warmup.prefill_segments(1024)
        self.assertEqual(segments[0], 1024)
        for low, high in ((33, 64), (65, 128), (129, 256), (257, 1024)):
            bucket = [n for n in segments if low <= n <= high]
            with self.subTest(bucket=(low, high)):
                self.assertTrue(any(n % 16 == 0 for n in bucket))
                self.assertTrue(any(n % 16 for n in bucket))


class FakeKernel:
    name = 'fake'

    def __init__(self):
        self.loaded = 0

    def _init_handles(self):
        self.loaded += 1


class FakeJit:
    def __init__(self):
        self.calls = []
        self.kernel = FakeKernel()

    def run(self, *args, grid, warmup, **kwargs):
        self.calls.append((args, grid, warmup, kwargs))
        return self.kernel


class CompileOnlyTests(unittest.TestCase):
    def test_launches_compile_without_running_and_restore(self):
        module = ModuleType('fake_paged')
        launched, helper = FakeJit(), FakeJit()
        module._paged_kernel, module._helper = launched, helper
        with warmup.compile_only(module, FakeJit) as kernels:
            self.assertIsNot(module._paged_kernel, launched)
            self.assertIs(module._helper, helper)
            module._paged_kernel[(2, 3)](1, 'x', num_warps=4)
            module._paged_kernel[(2, 3)](1, 'x', num_warps=4)
        self.assertIs(module._paged_kernel, launched)
        self.assertEqual(launched.calls[0], ((1, 'x'), (2, 3), True, dict(num_warps=4)))
        self.assertEqual(len(kernels), 1)
        self.assertEqual(launched.kernel.loaded, 2)

    def test_restores_kernels_after_errors(self):
        module = ModuleType('fake_paged')
        launched = FakeJit()
        module._paged_kernel = launched
        with self.assertRaises(RuntimeError):
            with warmup.compile_only(module, FakeJit):
                raise RuntimeError('boom')
        self.assertIs(module._paged_kernel, launched)


class JitMonitorTests(unittest.TestCase):
    def test_counts_specializations_and_chains_previous_hook(self):
        seen = []
        knobs = SimpleNamespace(runtime=SimpleNamespace(jit_post_compile_hook=lambda **kw: seen.append(kw['key'])))
        monitor = warmup.JitMonitor(knobs)
        heard = []
        knobs.runtime.jit_post_compile_hook(key=1, fn=SimpleNamespace(name='decode'))
        monitor.listener = heard.append
        knobs.runtime.jit_post_compile_hook(key=2, fn=SimpleNamespace(name='decode'))
        self.assertEqual((monitor.count, monitor.kernels['decode']), (2, 2))
        self.assertEqual(seen, [1, 2])
        self.assertEqual(heard, ['decode'])


class Ids:
    def __init__(self, values):
        self.values = values

    def flatten(self):
        return self

    def tolist(self):
        return list(self.values)


class EngineWarmupTests(unittest.TestCase):
    def make_engine(self, events):
        engine = serving.Engine.__new__(serving.Engine)
        engine.chunk_size, engine.context, engine.depth = 256, 4096, 2
        engine.jit = SimpleNamespace(count=5)
        engine.sampling_defaults = dict(DEFAULTS, temperature=0.6, seed=3)
        engine.Sampler = lambda: 'argmax'
        engine.ComboSampler = lambda **kwargs: ('combo', kwargs['temperature'])
        engine.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: Ids(range(100, 140)))
        engine.torch = SimpleNamespace(long='long', tensor=lambda data, dtype=None: data,
                                       cuda=SimpleNamespace(synchronize=lambda: None))
        engine.model = SimpleNamespace(quantlab_shortlist=SimpleNamespace(reset=lambda: events.append('reset')))
        engine.cache, engine.draft, engine.draft_cache = 'cache', 'draft', 'draft_cache'
        engine._prefix_gen = None
        engine.memory = lambda: {}
        engine.record = lambda stage, **fields: events.append((stage, fields))
        return engine

    def test_throwaway_generation_then_attention_leaves_no_state(self):
        events, jobs, builds = [], [], []
        paged = ModuleType('triton_paged')
        paged._auto_decode_calls = paged._qc_long_decode_calls = 7

        class Job:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                jobs.append(self)

        class Generator:
            def __init__(self):
                self.pending = 0
                self.closed = False
                self.filter_pool = SimpleNamespace(shutdown=lambda **kwargs: setattr(self, 'closed', True))
                builds.append(self)

            def enqueue(self, job):
                self.pending = 1

            def num_remaining_jobs(self):
                return self.pending

            def iterate(self):
                paged._auto_decode_calls += 1
                self.pending = 0

            def clear_queue(self):
                events.append('clear')

        engine = self.make_engine(events)
        engine.Job = Job
        engine._build_generator = Generator
        attention_calls = []

        def warm_attention(torch, pairs, *, chunk):
            attention_calls.append((pairs, chunk))
            return dict(calls=3, variants=2)

        modules = {'exllamav3': ModuleType('exllamav3'), 'exllamav3.modules': ModuleType('exllamav3.modules'),
                   'exllamav3.modules.attention_fn': ModuleType('exllamav3.modules.attention_fn'),
                   'exllamav3.modules.attention_fn.triton_paged': paged}
        modules['exllamav3.modules.attention_fn'].triton_paged = paged
        with patch.dict(sys.modules, modules), \
             patch.object(warmup, 'attention_layers', lambda model, cache: [(model, cache)]), \
             patch.object(warmup, 'warm_attention', warm_attention), \
             patch.object(serving, '_say', lambda message: None):
            engine._warmup()

        self.assertEqual([len(job.kwargs['input_ids'][0]) for job in jobs],
                         [n + 1 for n in warmup.prefill_segments(256)])
        self.assertEqual(len({job.kwargs['input_ids'][0][0] for job in jobs}), len(jobs))
        self.assertEqual([job.kwargs['sampler'] for job in jobs[:3]],
                         [('combo', 0.6), 'argmax', ('combo', 0.6)])
        self.assertEqual(jobs[0].kwargs['max_new_tokens'], 16 + 1 + 2)
        self.assertEqual(jobs[0].kwargs['stop_conditions'], [])
        self.assertEqual(len(builds), 1)
        self.assertTrue(builds[0].closed)
        self.assertIsNone(engine._prefix_gen)
        self.assertEqual(attention_calls, [([(engine.model, 'cache'), ('draft', 'draft_cache')], 256)])
        self.assertEqual((paged._auto_decode_calls, paged._qc_long_decode_calls), (7, 7))
        self.assertIn('reset', events)
        stage, fields = events[-1]
        self.assertEqual(stage, 'warmup')
        self.assertTrue(fields['enabled'])
        self.assertEqual(fields['attention'], dict(calls=3, variants=2))
        self.assertEqual(fields['jit_specializations'], 0)


class ServeParserTests(unittest.TestCase):
    def test_warmup_defaults_on_with_off_override(self):
        base = ['--config', 'c', '--candidate', 'm', '--source-dir', 's', '--extension-dir', 'e',
                '--output', 'o', '--expected-extension-sha256', '0' * 64]
        self.assertEqual(serving.parser().parse_args(base).warmup, 'on')
        self.assertEqual(serving.parser().parse_args(base + ['--warmup', 'off']).warmup, 'off')


if __name__ == '__main__':
    unittest.main()
