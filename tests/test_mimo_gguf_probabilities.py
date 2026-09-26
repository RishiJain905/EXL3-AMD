"""CPU-only checks for deterministic GGUF teacher-forced probability summary."""
import importlib.util
import math
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("mimo_gguf_control_under_test",
                                               SCRIPTS / "run_mimo_gguf_control.py")
control = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(control)


def make_top(logprobs_by_id, order=None):
    if order is None:
        order = sorted(logprobs_by_id)
    return [{"id": token_id, "logprob": logprobs_by_id[token_id]} for token_id in order]


class SummarizeTeacherForcedTests(unittest.TestCase):
    def test_valid_and_excluded_masses(self):
        # Valid probs sum to 0.9375, excluded to 0.0625, total 1.0.
        logprobs = {0: math.log(0.5), 1: math.log(0.25), 2: math.log(0.125),
                    3: math.log(0.0625), 4: math.log(0.03125), 5: math.log(0.03125)}
        result = control.summarize_teacher_forced(make_top(logprobs), 1,
                                                  valid_size=4, stored_size=6)
        self.assertEqual(result["target_id"], 1)
        self.assertEqual(result["top1_id"], 0)
        self.assertEqual(result["valid_vocabulary_size"], 4)
        self.assertEqual(result["returned_vocabulary_size"], 6)
        # NLL normalized over valid IDs only: log(0.9375 / 0.25) = log(3.75).
        self.assertAlmostEqual(result["target_nll"], math.log(3.75), places=12)
        # Total-normalized NLL would be log(4.0); valid-only must differ.
        self.assertGreater(abs(math.log(4.0) - result["target_nll"]), 0.05)
        self.assertAlmostEqual(result["excluded_probability_mass"], 0.0625, places=12)

    def test_top1_tie_breaks_to_smallest_id_despite_order(self):
        tie = math.log(0.4)
        logprobs = {0: math.log(0.1), 1: tie, 2: math.log(0.1), 3: tie,
                    4: math.log(0.9), 5: math.log(0.01)}
        # Larger tied ID first; excluded ID 4 exceeds the valid maximum but
        # must not win because top-1 is valid-only.
        top = make_top(logprobs, order=[5, 3, 2, 0, 4, 1])
        result = control.summarize_teacher_forced(top, 0, valid_size=4, stored_size=6)
        self.assertEqual(result["top1_id"], 1)

    def test_excluded_mass_uses_returned_invalid_ids(self):
        # Valid sum 0.6, excluded sum 0.3, total 0.9: subtraction would give
        # 1 - 0.6 = 0.4, while the returned-mass fraction is 1/3.
        logprobs = {0: math.log(0.3), 1: math.log(0.2), 2: math.log(0.1),
                    3: math.log(0.2), 4: math.log(0.1)}
        result = control.summarize_teacher_forced(make_top(logprobs), 0,
                                                  valid_size=3, stored_size=5)
        self.assertAlmostEqual(result["excluded_probability_mass"], 1 / 3, places=12)
        self.assertGreater(abs(result["excluded_probability_mass"] - 0.4), 0.05)
        # Valid-only NLL is log(0.6 / 0.3) = log(2); total-normalized log(3).
        self.assertAlmostEqual(result["target_nll"], math.log(2.0), places=12)
        self.assertGreater(abs(math.log(3.0) - result["target_nll"]), 0.3)

    def test_missing_id_rejected(self):
        logprobs = {0: math.log(0.3), 1: math.log(0.3), 2: math.log(0.2),
                    3: math.log(0.1), 4: math.log(0.1)}
        # Stored size claims 6 IDs but only 5 rows are present.
        with self.assertRaises(RuntimeError):
            control.summarize_teacher_forced(make_top(logprobs), 0,
                                             valid_size=4, stored_size=6)

    def test_duplicate_id_rejected(self):
        top = [{"id": 0, "logprob": math.log(0.3)},
               {"id": 0, "logprob": math.log(0.2)},
               {"id": 1, "logprob": math.log(0.2)},
               {"id": 2, "logprob": math.log(0.1)},
               {"id": 3, "logprob": math.log(0.1)},
               {"id": 4, "logprob": math.log(0.1)}]
        with self.assertRaises(RuntimeError):
            control.summarize_teacher_forced(top, 1, valid_size=4, stored_size=6)

    def test_nonfinite_rejected(self):
        base = {0: math.log(0.4), 1: math.log(0.3), 2: math.log(0.2),
                3: math.log(0.05), 4: math.log(0.05)}
        for bad in (float("nan"), float("inf"), float("-inf")):
            for victim in (0, 1, 4):
                with self.subTest(bad=bad, victim=victim):
                    logprobs = dict(base)
                    logprobs[victim] = bad
                    with self.assertRaises(RuntimeError):
                        control.summarize_teacher_forced(make_top(logprobs), 0,
                                                         valid_size=3, stored_size=5)

    def test_target_underflow_rejected(self):
        logprobs = {0: -2e30, 1: math.log(0.5), 2: math.log(0.4),
                    3: math.log(0.05), 4: math.log(0.05)}
        with self.assertRaisesRegex(RuntimeError, "underflow"):
            control.summarize_teacher_forced(make_top(logprobs), 0,
                                             valid_size=3, stored_size=5)

    def test_target_id_validation(self):
        logprobs = {0: math.log(0.4), 1: math.log(0.3), 2: math.log(0.2),
                    3: math.log(0.05), 4: math.log(0.05)}
        top = make_top(logprobs)
        for bad_target in (-1, 3, 4, 5):
            with self.subTest(target=bad_target):
                with self.assertRaises(RuntimeError):
                    control.summarize_teacher_forced(top, bad_target,
                                                     valid_size=3, stored_size=5)

    def test_malformed_rows_rejected(self):
        good = {0: math.log(0.4), 1: math.log(0.3), 2: math.log(0.2),
                3: math.log(0.05), 4: math.log(0.05)}
        cases = [
            None,
            [{"id": 0}],  # missing logprob, wrong length
            make_top(good)[:-1] + [{"noid": 4, "logprob": math.log(0.05)}],
            make_top(good)[:-1] + [{"id": "4", "logprob": math.log(0.05)}],
            make_top(good)[:-1] + [{"id": 4, "logprob": "x"}],
        ]
        for index, top in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(RuntimeError):
                    control.summarize_teacher_forced(top, 0,
                                                     valid_size=3, stored_size=5)


if __name__ == "__main__":
    unittest.main()
