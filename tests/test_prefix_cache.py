"""CPU-only checks for prefix-reuse prefill accounting."""
import unittest

from quantlab.methods.exl3.prefix_cache import (
    PAGE_SIZE, cached_credit, compute_rate, prefill_telemetry, split_prefill_advance,
)
from types import SimpleNamespace


class PrefixCacheHelperTests(unittest.TestCase):
    def test_cached_credit_combines_pages_and_tokens(self):
        job = SimpleNamespace(cached_pages=3, cached_tokens=7)
        self.assertEqual(cached_credit(job), 3 * PAGE_SIZE + 7)

    def test_cached_credit_missing_attributes_mean_no_reuse(self):
        self.assertEqual(cached_credit(SimpleNamespace()), 0)
        self.assertEqual(cached_credit(SimpleNamespace(cached_pages=None, cached_tokens=None)), 0)

    def test_split_counts_only_counter_delta_as_cached(self):
        computed, cached = split_prefill_advance(256, 0, 256)
        self.assertEqual((computed, cached), (0, 256))
        computed, cached = split_prefill_advance(10, 512, 512)
        self.assertEqual((computed, cached), (10, 0))

    def test_split_clamps_racing_counters(self):
        computed, cached = split_prefill_advance(3, 0, 9)
        self.assertEqual((computed, cached), (0, 3))
        computed, cached = split_prefill_advance(0, 4, 4)
        self.assertEqual((computed, cached), (0, 0))

    def test_compute_rate_never_divides_reused_tokens(self):
        self.assertEqual(compute_rate(1, 0.5), 2.0)
        self.assertIsNone(compute_rate(0, 1.0))
        self.assertIsNone(compute_rate(5, 0.0))

    def test_telemetry_rate_uses_computed_only(self):
        fields = prefill_telemetry(computed_tokens=1, cached_tokens=255,
                                   compute_seconds=0.5, wall_seconds=0.6)
        self.assertEqual(fields['prefill_computed_tokens'], 1)
        self.assertEqual(fields['prefill_cached_tokens'], 255)
        self.assertEqual(fields['prefill_wall_seconds'], 0.6)
        self.assertEqual(fields['prefill_tokens_per_second'], 2.0)


if __name__ == '__main__':
    unittest.main()
