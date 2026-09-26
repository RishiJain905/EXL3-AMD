"""Focused checks for the offline cache-precision allocation helper."""
import copy
import importlib.util
import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('allocate_cache_precision', Path(__file__).resolve().parents[1] / 'kernels/exl3/allocate_cache_precision.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
select_precision = module.select_precision

PAIRS = [('aster5', 'aster5'), ('q4', 'q4'), ('q5', 'q5'), ('q6', 'q6'), ('q8', 'q8'), ('q8', 'q6'), ('q6', 'q8'), ('q4', 'q8')]


def _layer(idx, *options):
    return dict(layer_idx=idx, options=[dict(k=k, v=v, bytes=size, loss=loss) for k, v, size, loss in options])


def _brute(layers, budget):
    ordered = sorted(layers, key=lambda item: item['layer_idx'])
    best = None
    for combo in itertools.product(*[item['options'] for item in ordered]):
        total = sum(option['bytes'] for option in combo)
        if total > budget:
            continue
        loss = 0.0
        for option in combo:
            loss += float(option['loss'])
        key = (loss, total, tuple((option['k'], option['v']) for option in combo))
        if best is None or key < best[0]:
            best = (key, combo, total, loss)
    if best is None:
        raise ValueError('infeasible')
    _, combo, total, loss = best
    picked = [dict(layer_idx=item['layer_idx'], k=o['k'], v=o['v'], bytes=o['bytes'], loss=float(o['loss'])) for item, o in zip(ordered, combo)]
    return dict(total_bytes=total, total_loss=loss, layers=picked)


class OracleTests(unittest.TestCase):
    def test_brute_force_oracle_over_random_tiny_problems(self):
        rng = random.Random(20260919)
        checked = 0
        for _ in range(20):
            idxs = rng.sample(range(8), rng.randint(1, 4))
            layers = []
            for idx in idxs:
                pairs = rng.sample(PAIRS, rng.randint(1, 4))
                options = [dict(k=k, v=v, bytes=rng.randint(1, 6), loss=rng.choice((0.0, 0.5, 1.0, 1.5, 2.0))) for k, v in pairs]
                layers.append(dict(layer_idx=idx, options=options))
            cheapest = sum(min(o['bytes'] for o in item['options']) for item in layers)
            priciest = sum(max(o['bytes'] for o in item['options']) for item in layers)
            budget = rng.randint(cheapest - 1, priciest)
            if budget < cheapest:
                with self.assertRaises(ValueError):
                    select_precision(layers, budget)
            else:
                self.assertEqual(select_precision(layers, budget), _brute(layers, budget))
                checked += 1
        self.assertGreater(checked, 0)


class AllocationTests(unittest.TestCase):
    def test_unequal_byte_costs_force_constrained_tradeoff(self):
        layers = [
            _layer(0, ('q8', 'q8', 1, 100.0), ('q6', 'q6', 2, 0.0)),
            _layer(1, ('q8', 'q8', 1, 100.0), ('q6', 'q6', 2, 0.0)),
        ]
        expected = dict(total_bytes=3, total_loss=100.0, layers=[
            dict(layer_idx=0, k='q6', v='q6', bytes=2, loss=0.0),
            dict(layer_idx=1, k='q8', v='q8', bytes=1, loss=100.0),
        ])
        result = select_precision(layers, 3)
        self.assertEqual(result, expected)
        self.assertIs(type(result['total_bytes']), int)
        self.assertIs(type(result['total_loss']), float)

    def test_exact_boundary_and_infeasible_budgets(self):
        layers = [_layer(0, ('q8', 'q8', 3, 0.5), ('q6', 'q6', 5, 0.1))]
        self.assertEqual(select_precision(layers, 5)['layers'][0]['k'], 'q6')
        self.assertEqual(select_precision(layers, 4)['layers'][0]['k'], 'q8')
        self.assertEqual(select_precision(layers, 3)['total_bytes'], 3)
        for budget in (2, 1, 0):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    select_precision(layers, budget)

    def test_ties_prefer_fewer_bytes_then_lexicographic_codecs(self):
        layers = [_layer(0, ('q8', 'q8', 6, 1.0), ('q6', 'q6', 4, 1.0))]
        self.assertEqual(select_precision(layers, 6)['layers'], [dict(layer_idx=0, k='q6', v='q6', bytes=4, loss=1.0)])
        layers = [_layer(0, ('q8', 'q8', 4, 1.0), ('q6', 'q6', 4, 1.0))]
        result = select_precision(layers, 4)
        self.assertEqual(result['layers'], [dict(layer_idx=0, k='q6', v='q6', bytes=4, loss=1.0)])
        self.assertEqual(result['total_loss'], 1.0)

    def test_multilayer_ties_and_input_order_independence(self):
        base = [
            _layer(5, ('q8', 'q6', 4, 1.0), ('q6', 'q8', 4, 1.0)),
            _layer(2, ('q8', 'q8', 4, 1.0), ('q6', 'q6', 4, 1.0)),
        ]
        expected = dict(total_bytes=8, total_loss=2.0, layers=[
            dict(layer_idx=2, k='q6', v='q6', bytes=4, loss=1.0),
            dict(layer_idx=5, k='q6', v='q8', bytes=4, loss=1.0),
        ])
        self.assertEqual(select_precision(base, 8), expected)
        flipped = [dict(layer_idx=item['layer_idx'], options=list(reversed(item['options']))) for item in reversed(base)]
        self.assertEqual(select_precision(flipped, 8), expected)

    def test_input_is_not_mutated(self):
        layers = [_layer(2, ('q8', 'q6', 4, 1.5), ('aster5', 'aster5', 3, 2.0)), _layer(0, ('q4', 'q4', 2, 0.5))]
        snapshot = copy.deepcopy(layers)
        select_precision(layers, 7)
        self.assertEqual(layers, snapshot)

    def test_frontier_bound_fails_explicitly(self):
        layers = []
        for i in range(13):
            layers.append(dict(layer_idx=i, options=[
                dict(k='q8', v='q8', bytes=1, loss=10000.0),
                dict(k='q6', v='q6', bytes=1 + 2 ** i, loss=10000.0 - 2 ** i),
            ]))
        with self.assertRaises(ValueError):
            select_precision(layers, 8204)


class ValidationTests(unittest.TestCase):
    def test_malformed_inputs_raise_value_error(self):
        good = _layer(0, ('q8', 'q8', 4, 1.0))
        cases = [
            (None, 4), ('x', 4), ([], 4), ({}, 4), (['x'], 4),
            ([good], None), ([good], True), ([good], -1), ([good], 4.0), ([good], '4'),
            ([dict(layer_idx=0)], 4),
            ([dict(layer_idx=0, options=good['options'], extra=1)], 4),
            ([dict(layer_idx=True, options=good['options'])], 4),
            ([dict(layer_idx=-1, options=good['options'])], 4),
            ([dict(layer_idx=0.0, options=good['options'])], 4),
            ([good, dict(layer_idx=0, options=good['options'])], 4),
            ([dict(layer_idx=0, options=[])], 4),
            ([dict(layer_idx=0, options={})], 4),
            ([dict(layer_idx=i, options=[dict(k='q8', v='q8', bytes=1, loss=0.0)]) for i in range(257)], 10 ** 9),
        ]
        bad_options = [
            dict(k='q8', v='q8', bytes=4),
            dict(k='q8', v='q8', bytes=4, loss=1.0, extra=0),
            dict(k=8, v='q8', bytes=4, loss=1.0),
            dict(k='q8', v=None, bytes=4, loss=1.0),
            dict(k='q8', v='q8', bytes=True, loss=1.0),
            dict(k='q8', v='q8', bytes=0, loss=1.0),
            dict(k='q8', v='q8', bytes=-2, loss=1.0),
            dict(k='q8', v='q8', bytes=4.0, loss=1.0),
            dict(k='q8', v='q8', bytes=4, loss=True),
            dict(k='q8', v='q8', bytes=4, loss=-0.5),
            dict(k='q8', v='q8', bytes=4, loss=float('nan')),
            dict(k='q8', v='q8', bytes=4, loss=float('inf')),
            dict(k='q8', v='q8', bytes=4, loss='1.0'),
            dict(k='q8', v='q8', bytes=4, loss=None),
        ]
        for bad in bad_options:
            cases.append(([dict(layer_idx=0, options=[bad])], 8))
        opt = dict(k='q8', v='q8', bytes=4, loss=1.0)
        cases.append(([dict(layer_idx=0, options=[dict(opt), dict(opt)])], 8))
        for layers, budget in cases:
            with self.subTest(layers=layers, budget=budget):
                with self.assertRaises(ValueError):
                    select_precision(layers, budget)

    def test_unsupported_pairs_raise_value_error(self):
        for k, v in [('f16', 'f16'), ('aster5', 'q8'), ('q8', 'aster5'), ('q3', 'q3'), ('Q8', 'Q8'), ('q8', 'f16'), ('', 'q8')]:
            layers = [_layer(0, (k, v, 4, 1.0))]
            with self.subTest(k=k, v=v):
                with self.assertRaises(ValueError):
                    select_precision(layers, 8)

    def test_supported_pairs_and_count_bounds(self):
        quads = ('q4', 'q5', 'q6', 'q8')
        sixteen = [dict(k=k, v=v, bytes=1, loss=1.0) for k in quads for v in quads]
        result = select_precision([dict(layer_idx=0, options=sixteen)], 1)
        self.assertEqual(result['total_bytes'], 1)
        seventeen = sixteen + [dict(k='aster5', v='aster5', bytes=1, loss=1.0)]
        with self.assertRaises(ValueError):
            select_precision([dict(layer_idx=0, options=seventeen)], 1)
        layers = [dict(layer_idx=i, options=[dict(k='q8', v='q8', bytes=2, loss=0.5)]) for i in range(256)]
        result = select_precision(layers, 512)
        self.assertEqual(result['total_bytes'], 512)
        self.assertEqual(len(result['layers']), 256)


class CliTests(unittest.TestCase):
    def test_round_trip_and_refused_overwrite(self):
        layers = [_layer(3, ('aster5', 'aster5', 4, 0.5), ('q8', 'q8', 6, 0.1)), _layer(1, ('q6', 'q6', 5, 0.2))]
        expected = select_precision(layers, 10)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'request.json'
            dst = Path(tmp) / 'result.json'
            src.write_text(json.dumps(dict(layers=layers, budget_bytes=10)), encoding='utf-8')
            self.assertEqual(module.main([str(src), '--output', str(dst)]), 0)
            self.assertEqual(json.loads(dst.read_text(encoding='utf-8')), expected)
            before = dst.read_text(encoding='utf-8')
            with self.assertRaises(SystemExit) as ctx:
                module.main([str(src), '--output', str(dst)])
            self.assertNotEqual(ctx.exception.code, 0)
            self.assertEqual(dst.read_text(encoding='utf-8'), before)

    def test_bad_input_exits_nonzero(self):
        payloads = ['{"layers": [], "budget_bytes": 1}', '{"layers": [], "layers": [], "budget_bytes": 1}', '{"layers": [], "budget_bytes": NaN}']
        for text in payloads:
            with self.subTest(text=text):
                with tempfile.TemporaryDirectory() as tmp:
                    src = Path(tmp) / 'request.json'
                    dst = Path(tmp) / 'result.json'
                    src.write_text(text, encoding='utf-8')
                    with self.assertRaises(SystemExit) as ctx:
                        module.main([str(src), '--output', str(dst)])
                    self.assertNotEqual(ctx.exception.code, 0)
                    self.assertFalse(dst.exists())


if __name__ == '__main__':
    unittest.main()
