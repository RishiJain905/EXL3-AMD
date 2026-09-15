"""CPU boundaries for the paged-attention plan and occupied-coding mode; no GPU work."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_exl3_attention import MAX_CASES, MAX_CONFIGS_PER_CASE, PAGE_SIZE, validate_plan
import evaluate_exl3_candidate as evaluator
import launch_runtime as launcher


def _case(name='attn-4k', **overrides):
    case = dict(name=name, cache_type='f16', length=4000, q_len=1,
                configs=[dict(name='wide')])
    case.update(overrides)
    return case


class ValidatePlanTests(unittest.TestCase):
    def test_normalizes_capacity_and_defaults(self):
        (case,) = validate_plan([_case(length=4000, q_len=1)])
        self.assertEqual(case['capacity'], 4096)
        self.assertEqual(case['seed'], 4197)
        self.assertEqual(case['configs'][0]['num_warps'], 4)
        self.assertEqual(case['configs'][0]['num_stages'], 2)
        self.assertIsNone(case['configs'][0]['block_n'])

    def test_partial_final_page_rounds_up_to_full_pages(self):
        (case,) = validate_plan([_case(length=4095, q_len=2)])
        self.assertEqual(case['capacity'] % PAGE_SIZE, 0)
        self.assertGreaterEqual(case['capacity'], 4095 + 2)

    def test_case_count_bound(self):
        ok = [_case(name='c%d' % i) for i in range(MAX_CASES)]
        self.assertEqual(len(validate_plan(ok)), MAX_CASES)
        with self.assertRaisesRegex(ValueError, 'at most'):
            validate_plan(ok + [_case(name='overflow')])

    def test_duplicate_case_names_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_plan([_case(name='dup'), _case(name='dup')])

    def test_config_count_bound(self):
        configs = [dict(name='k%d' % i) for i in range(MAX_CONFIGS_PER_CASE)]
        (case,) = validate_plan([_case(configs=configs)])
        self.assertEqual(len(case['configs']), MAX_CONFIGS_PER_CASE)
        with self.assertRaisesRegex(ValueError, 'at most'):
            validate_plan([_case(configs=configs + [dict(name='overflow')])])

    def test_duplicate_and_reserved_config_names_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_plan([_case(configs=[dict(name='dup'), dict(name='dup')])])
        with self.assertRaisesRegex(ValueError, 'reserved'):
            validate_plan([_case(configs=[dict(name='control')])])

    def test_invalid_shapes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-empty'):
            validate_plan([])
        with self.assertRaisesRegex(ValueError, 'cache_type'):
            validate_plan([_case(cache_type='fp8')])
        with self.assertRaisesRegex(ValueError, 'q_len'):
            validate_plan([_case(q_len=17)])
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            validate_plan([_case(length=131072, q_len=16)])

    def test_decode_kwargs_defaults(self):
        (case,) = validate_plan([_case(length=4000, q_len=1)])
        self.assertIsNone(case['configs'][0]['head_block'])
        self.assertIs(case['configs'][0]['parallel_combine'], False)

    def test_head_block_product_boundaries(self):
        for q_len, head_block in ((1, 16), (4, 4), (4, 16), (8, 2), (8, 8), (16, 1), (16, 4)):
            with self.subTest(q_len=q_len, head_block=head_block):
                (case,) = validate_plan([_case(length=4000, q_len=q_len,
                                               configs=[dict(name='wide', head_block=head_block,
                                                             parallel_combine=True)])])
                self.assertEqual(case['configs'][0]['head_block'], head_block)
                self.assertIs(case['configs'][0]['parallel_combine'], True)

    def test_invalid_parallel_combine_rejected(self):
        for bad in (0, 1, 'true', None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'parallel_combine'):
                    validate_plan([_case(configs=[dict(name='wide', parallel_combine=bad)])])

    def test_invalid_head_block_rejected(self):
        for bad in (True, 0, 3, 32, '8'):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'head_block'):
                    validate_plan([_case(length=4000, q_len=4,
                                         configs=[dict(name='wide', head_block=bad)])])
        for q_len, head_block in ((1, 8), (8, 1), (16, 8), (16, 16)):
            with self.subTest(q_len=q_len, head_block=head_block):
                with self.assertRaisesRegex(ValueError, 'head_block'):
                    validate_plan([_case(length=4000, q_len=q_len,
                                         configs=[dict(name='wide', head_block=head_block)])])

    def test_table_pad_pages_defaults_to_one(self):
        (case,) = validate_plan([_case(length=4000, q_len=1)])
        self.assertEqual(case['table_pad_pages'], 1)

    def test_table_pad_pages_valid_sixteen(self):
        (case,) = validate_plan([_case(length=4000, q_len=1, table_pad_pages=16)])
        self.assertEqual(case['table_pad_pages'], 16)

    def test_invalid_table_pad_pages_rejected(self):
        for bad in (True, False, 0, 3, '16'):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'table_pad_pages'):
                    validate_plan([_case(table_pad_pages=bad)])


class OccupiedCodingModeTests(unittest.TestCase):
    def _evaluator_mode(self, mode):
        args = evaluator.parser().parse_args([
            '--config', 'c', '--candidate', 'd', '--source-dir', 's',
            '--extension-dir', 'e', '--suite', 'u', '--output', 'o',
            '--expected-extension-sha256', 'x', '--mode', mode])
        return args.mode

    def _evaluator_task(self, mode, task=None):
        argv = ['--config', 'c', '--candidate', 'd', '--source-dir', 's',
                '--extension-dir', 'e', '--suite', 'u', '--output', 'o',
                '--expected-extension-sha256', 'x', '--mode', mode]
        if task is not None:
            argv += ['--context-speed-task', task]
        return evaluator.parser().parse_args(argv).context_speed_task

    def test_both_harnesses_accept_context_speed(self):
        self.assertEqual(self._evaluator_mode('context-speed'), 'context-speed')
        self.assertEqual(launcher.parser().parse_args(['context-speed']).mode, 'context-speed')

    def test_existing_context_mode_preserved(self):
        self.assertEqual(self._evaluator_mode('context'), 'context')
        self.assertEqual(launcher.parser().parse_args(['context']).mode, 'context')

    def test_context_speed_task_defaults_to_docstring(self):
        self.assertEqual(self._evaluator_task('context-speed'), 'docstring')
        self.assertEqual(launcher.parser().parse_args(['context-speed']).context_speed_task, 'docstring')
        self.assertEqual(launcher.parser().parse_args([]).context_speed_task, 'docstring')

    def test_context_speed_task_choices_match_across_harnesses(self):
        for task in ('docstring', 'canonical'):
            with self.subTest(task=task):
                self.assertEqual(self._evaluator_task('context-speed', task), task)
                forwarded = launcher.parser().parse_args(['context-speed', '--context-speed-task', task])
                self.assertEqual(forwarded.context_speed_task, task)

    def test_evaluator_rejects_canonical_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / 'never-created')
            argv = ['evaluate_exl3_candidate.py', '--config', 'missing.toml', '--candidate', 'd',
                   '--source-dir', 's', '--extension-dir', 'e', '--suite', 'u', '--output', output,
                   '--expected-extension-sha256', 'x', '--mode', 'speed',
                   '--context-speed-task', 'canonical']
            saved, sys.argv = sys.argv, argv
            try:
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        evaluator.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('--context-speed-task', stderr.getvalue())
            finally:
                sys.argv = saved
            self.assertFalse(Path(output).exists())

    def test_launcher_rejects_canonical_before_config(self):
        argv = ['launch_runtime.py', 'generate', '--context-speed-task', 'canonical',
                '--prompt', 'hi', '--config', 'definitely-missing.toml']
        saved, sys.argv = sys.argv, argv
        try:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    launcher.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('--context-speed-task', stderr.getvalue())
        finally:
            sys.argv = saved


if __name__ == '__main__':
    unittest.main()
