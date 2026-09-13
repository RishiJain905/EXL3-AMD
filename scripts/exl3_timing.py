"""MTP first-iteration decode timing (Torch-free, CPU-only).

One ``gen.iterate()`` returns multiple token events sharing one synchronized
timestamp. Throughput must exclude the whole first *emitting* GPU iteration,
not just its first event.
"""

TIMING_VERSION = 2


class FirstIterationTiming:
    """Accumulate whole-first-emitting-iteration decode timing."""

    __slots__ = ("first_time", "first_count", "last_time")

    def __init__(self):
        self.first_time = None
        self.first_count = None
        self.last_time = None

    def observe(self, now, event_counts):
        """Account one ``gen.iterate()`` result.

        ``event_counts`` holds the per-event emitted-token count for this
        iteration. Empty prefill iterations never count. Returns the
        iteration total.
        """
        total = 0
        for count in event_counts:
            total += count
        if total:
            if self.first_time is None:
                self.first_time = now
                self.first_count = total
            self.last_time = now
        return total

    def decode_seconds(self):
        """Span between first and last emitting iterations, else None."""
        if (self.first_time is not None and self.last_time is not None
                and self.last_time > self.first_time):
            return self.last_time - self.first_time
        return None

    def decode_tokens(self, total_tokens):
        """Tokens after excluding the entire first emitting iteration."""
        return total_tokens - (self.first_count or 0)

    def tokens_per_second(self, total_tokens):
        """Decode rate, else None for a single emitting iteration."""
        seconds = self.decode_seconds()
        if not seconds:
            return None
        return self.decode_tokens(total_tokens) / seconds
