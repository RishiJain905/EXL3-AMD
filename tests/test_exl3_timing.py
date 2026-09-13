"""CPU-only MTP first-iteration timing; no Torch, GPU, model, or network."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from exl3_timing import TIMING_VERSION, FirstIterationTiming


class FirstIterationTimingTests(unittest.TestCase):
    def test_version_marks_grouped_iteration_format(self):
        self.assertEqual(TIMING_VERSION, 2)

    def test_empty_prefill_then_multi_event_first_iteration(self):
        timing = FirstIterationTiming()
        self.assertEqual(timing.observe(10.0, []), 0)
        self.assertEqual(timing.observe(11.0, [0, 0]), 0)
        self.assertIsNone(timing.first_time)
        self.assertIsNone(timing.first_count)
        self.assertIsNone(timing.last_time)
        # One GPU iterate returns two token events sharing one timestamp.
        self.assertEqual(timing.observe(12.0, [2, 1]), 3)
        self.assertEqual(timing.first_time, 12.0)
        self.assertEqual(timing.first_count, 3)
        self.assertEqual(timing.last_time, 12.0)
        # Later iterations advance last_time without touching first_count.
        self.assertEqual(timing.observe(13.0, [4]), 4)
        self.assertEqual(timing.observe(14.0, [1, 1]), 2)
        self.assertEqual(timing.first_count, 3)
        self.assertEqual(timing.last_time, 14.0)
        self.assertEqual(timing.decode_seconds(), 2.0)
        self.assertEqual(timing.decode_tokens(9), 6)
        self.assertAlmostEqual(timing.tokens_per_second(9), 3.0)

    def test_single_emitting_iteration_has_null_rate(self):
        timing = FirstIterationTiming()
        timing.observe(10.0, [])
        timing.observe(11.0, [1, 2])
        self.assertEqual(timing.first_count, 3)
        self.assertIsNone(timing.decode_seconds())
        self.assertEqual(timing.decode_tokens(3), 0)
        self.assertIsNone(timing.tokens_per_second(3))

    def test_no_emission_preserves_nulls(self):
        timing = FirstIterationTiming()
        timing.observe(10.0, [])
        timing.observe(11.0, [0])
        self.assertIsNone(timing.first_time)
        self.assertIsNone(timing.first_count)
        self.assertIsNone(timing.last_time)
        self.assertIsNone(timing.decode_seconds())
        self.assertEqual(timing.decode_tokens(0), 0)
        self.assertIsNone(timing.tokens_per_second(0))


if __name__ == '__main__':
    unittest.main()
