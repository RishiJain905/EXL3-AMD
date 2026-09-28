"""CPU tests for the decode-round profiler's aggregation and argument handling."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('profile_decode_round', ROOT / 'scripts' / 'profile_decode_round.py')
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)


class ProfilerTests(unittest.TestCase):
    def test_projection_role_drops_layer_indices(self):
        self.assertEqual(profiler.projection_role('model.layers.7.mlp.down_proj'), 'mlp.down_proj')
        self.assertEqual(profiler.projection_role('model.layers.30.linear_attn.in_proj_qkv'),
                         'linear_attn.in_proj_qkv')
        self.assertEqual(profiler.projection_role('lm_head'), 'lm_head')
        self.assertEqual(profiler.projection_role(''), '?')

    def test_summary_skips_prefill_and_first_decode_rounds(self):
        rounds = [dict(prefill=True, ops={'a': [9, 9.0]})]
        rounds += [dict(prefill=False, ops={'a': [9, 9.0]}) for _ in range(2)]
        rounds += [dict(prefill=False, ops={'a': [2, 1.0], 'b': [1, 3.0]}),
                   dict(prefill=False, ops={'a': [2, 3.0]})]
        summary = profiler.summarize_rounds(rounds)
        self.assertEqual(summary['rounds'], 2)
        self.assertEqual([row['label'] for row in summary['labels']], ['a', 'b'])
        self.assertEqual(summary['labels'][0], dict(label='a', calls_per_round=2.0, ms_per_round=2.0))
        self.assertEqual(summary['labels'][1]['ms_per_round'], 1.5)
        self.assertEqual(profiler.summarize_rounds(rounds[:3]), dict(rounds=0, labels=[]))

    def test_median_rounds_ignores_prefill_and_warm_rounds(self):
        rounds = [dict(prefill=True, ms=100.0)] + [dict(prefill=False, ms=v) for v in (50.0, 40.0, 30.0, 31.0, 35.0)]
        self.assertEqual(profiler.median_rounds(rounds, 'ms'), 31.0)
        self.assertIsNone(profiler.median_rounds(rounds[:3], 'ms'))

    def test_cli_requires_linux_and_rejects_non_exl3_env(self):
        if sys.platform != 'linux':
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                profiler.main(['--output', 'x', '-m', 'y'])
        self.assertIn('--attention-profile', profiler.DEFAULT_SERVE_FLAGS)
        self.assertEqual(profiler.DEFAULT_SERVE_FLAGS[profiler.DEFAULT_SERVE_FLAGS.index('--temperature') + 1], '0')


if __name__ == '__main__':
    unittest.main()
