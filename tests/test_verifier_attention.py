import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

spec = importlib.util.spec_from_file_location('verifier_attention', Path(__file__).resolve().parents[1]
        / 'src/quantlab/methods/exl3/verifier_attention.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@unittest.skipIf(torch is None, 'CPU PyTorch unavailable')
class VerifierAttentionTests(unittest.TestCase):
    def arguments(self, rows=3):
        q = torch.arange(rows*4, dtype=torch.float32).reshape(1, rows, 1, 4)
        return dict(q=q, k=q.clone(), v=q.clone(), cache=object(),
                    cache_seqlens=torch.tensor([74]), causal=True)

    def test_appends_each_row_at_its_own_causal_position(self):
        calls = []
        def dispatch(**kw):
            calls.append(kw)
            return kw['q'] + kw['cache_seqlens'].item()
        args = self.arguments()
        counter = dict(windows=0, rows=0)
        y = m.rowwise_attention(dispatch, counter, **args)
        self.assertEqual([x['cache_seqlens'].item() for x in calls], [74,75,76])
        self.assertTrue(all(x['q'].shape[1] == x['k'].shape[1] == x['v'].shape[1] == 1 for x in calls))
        self.assertTrue(torch.equal(y, args['q'] + torch.tensor([74,75,76]).reshape(1,3,1,1)))
        self.assertEqual(args['cache_seqlens'].item(), 74)
        self.assertEqual(counter, dict(windows=1, rows=3))

    def test_long_prefill_and_single_row_delegate_unchanged(self):
        for rows in (1,10):
            calls = []
            args = self.arguments(rows)
            def dispatch(**kw):
                calls.append(kw)
                return kw['q']
            result = m.rowwise_attention(dispatch, dict(windows=0,rows=0), **args)
            self.assertIs(result, args['q'])
            self.assertEqual(len(calls), 1)

    def test_noncausal_or_misaligned_inputs_fail_before_writing_cache(self):
        for change in ('causal','k','non_causal_spans'):
            args = self.arguments()
            args[change] = False if change == 'causal' else args['k'][:,:1] if change == 'k' else [(0,3)]
            calls = []
            with self.assertRaises(ValueError):
                m.rowwise_attention(lambda **kw: calls.append(kw), dict(windows=0,rows=0), **args)
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
