import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('mimo_mtp_analysis', Path(__file__).resolve().parents[1]/'scripts/mimo_mtp_evaluation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(mtp=False, seconds=10):
    rows = []
    for prefix in ('code', 'prose'):
        for repeat in range(3):
            rows.append(dict(name=f'{prefix}-{repeat}', warmup=False, input_sha256=prefix,
                             input_tokens=10, output_tokens=128, output_token_ids=[1]*128,
                             output_limit=128, fixed_length=True, eos_reason='max_new_tokens',
                             end_to_end_seconds=seconds+repeat-1, draft_windows=[]))
    rows.append(dict(rows[0], name='warmup', warmup=True, end_to_end_seconds=999))
    return dict(status='completed', mode='speed', mtp=mtp, cache_tokens=4096,
                cache_k='f16', cache_v='f16', thinking=False, results=rows)


class EvaluationTests(unittest.TestCase):
    def test_exact_parity(self):
        self.assertTrue(m.compare_records(fixture(), fixture(True))['passed'])

    def test_first_token_mismatch(self):
        a, b = fixture(), fixture(True)
        b['results'][0]['output_token_ids'] = [1]*7+[2]+[1]*120
        result = m.compare_records(a, b)
        self.assertFalse(result['passed'])
        self.assertEqual(result['cases'][0]['first_output_mismatch'], 7)

    def test_input_stop_and_missing_case(self):
        for key, value in (('input_sha256', 'other'), ('eos_reason', 'stop_token')):
            with self.subTest(key=key):
                b = fixture(True)
                b['results'][0][key] = value
                self.assertFalse(m.compare_records(fixture(), b)['passed'])
        b = fixture(True)
        b['results'].pop()
        self.assertFalse(m.compare_records(fixture(), b)['passed'])

    def test_malformed_and_incomplete_records(self):
        for mutation in ('duplicate', 'incomplete', 'count'):
            with self.subTest(mutation=mutation):
                b = fixture(True)
                if mutation == 'duplicate': b['results'].append(b['results'][0])
                if mutation == 'incomplete': b['status'] = 'failed'
                if mutation == 'count': b['results'][0]['output_tokens'] = 127
                with self.assertRaises(ValueError): m.compare_records(fixture(), b)

    def test_bracket_medians_and_geometric_mean(self):
        a, b, c = fixture(seconds=10), fixture(True, 5), fixture(seconds=10.2)
        for row in b['results']:
            if row['name'].startswith('prose-'): row['end_to_end_seconds'] += 5
        result = m.analyze_speed(a, b, c)
        self.assertTrue(result['qualified'])
        self.assertAlmostEqual(result['speed_ratio'], math.sqrt((10.1/5)*(10.1/10)))

    def test_warmup_time_is_excluded(self):
        b = fixture(True, 5)
        b['results'][-1]['end_to_end_seconds'] = 1e9
        self.assertAlmostEqual(m.analyze_speed(fixture(), b, fixture())['speed_ratio'], 2)

    def test_drift_invalidates_comparison(self):
        result = m.analyze_speed(fixture(), fixture(True, 5), fixture(seconds=15))
        self.assertFalse(result['valid'])
        self.assertFalse(result['qualified'])

    def test_no_gain_is_valid_negative_result(self):
        result = m.analyze_speed(fixture(), fixture(True, 11), fixture())
        self.assertTrue(result['valid'])
        self.assertFalse(result['qualified'])

    def test_missing_repetition_is_rejected(self):
        b = fixture(True)
        b['results'].pop(0)
        with self.assertRaises(ValueError): m.analyze_speed(fixture(), b, fixture())

    def test_nonfinite_time_is_rejected(self):
        b = fixture(True)
        b['results'][0]['end_to_end_seconds'] = float('nan')
        with self.assertRaises(ValueError): m.analyze_speed(fixture(), b, fixture())

    def test_draft_denominator_includes_untested_tail(self):
        stats = m.draft_statistics(dict(accepted_draft_tokens=3, rejected_draft_tokens=5,
            draft_windows=[[3,4,2],[5,4,1]]))
        self.assertEqual(stats['non_final_truncated_windows'], 1)
        self.assertEqual(stats['accepted_generated_fraction'], 3/8)
        self.assertEqual(stats['accepted_prefix_histogram'], {'2':1,'1':1})

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, out = [Path(tmp)/n for n in ('a.json','b.json','out.json')]
            a.write_text(json.dumps(fixture())); b.write_text(json.dumps(fixture(True))); out.write_text('keep')
            with self.assertRaises(SystemExit):
                m.main(['compare','--reference',str(a),'--candidate',str(b),'--output',str(out)])
            self.assertEqual(out.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
