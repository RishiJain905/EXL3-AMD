"""Prefix-reuse prefill accounting; Torch-free by design.

Separates actually computed prefill tokens from prompt tokens revived from a
persistent vendored Generator (hash-matched KV pages plus matching recurrent
checkpoints). The compute rate must never divide reused tokens by compute
time, so all rate math lives here in one place.
"""
from __future__ import annotations

# Mirror of exllamav3.constants.PAGE_SIZE. The vendored package needs Torch
# and the native extension, so this helper pins the value instead of
# importing it; a mismatch would only skew the cached/computed split.
PAGE_SIZE = 256


def cached_credit(job) -> int:
    """Prompt tokens already covered by cache state for this job.

    Vendored Job.allocate_pages credits full revived pages in cached_pages
    and partial-page copies in cached_tokens; prefill adds in-call skips to
    the same counters. Missing attributes (CPU fixtures) mean no reuse.
    """
    pages = getattr(job, "cached_pages", 0) or 0
    tokens = getattr(job, "cached_tokens", 0) or 0
    return int(pages) * PAGE_SIZE + int(tokens)


def split_prefill_advance(advanced: int, credit_before: int, credit_after: int):
    """Split one prefill call's kv advance into (computed, cached) tokens.

    kv_position advances over both recomputed chunks and skipped cached
    pages; only the counter delta identifies the skipped part. Clamped so a
    racing counter can never report negative compute.
    """
    cached_step = max(0, int(credit_after) - int(credit_before))
    cached_step = min(cached_step, max(0, int(advanced)))
    return max(0, int(advanced)) - cached_step, cached_step


def compute_rate(computed_tokens: int, compute_seconds: float):
    """Prefill compute rate, else None when nothing was computed."""
    if compute_seconds and computed_tokens > 0:
        return computed_tokens / compute_seconds
    return None


def prefill_telemetry(*, computed_tokens: int, cached_tokens: int,
                      compute_seconds: float, wall_seconds):
    """New usage-timing fields for one request's prefill phase."""
    return {
        "prefill_computed_tokens": int(computed_tokens),
        "prefill_cached_tokens": int(cached_tokens),
        "prefill_wall_seconds": wall_seconds,
        "prefill_tokens_per_second": compute_rate(computed_tokens, compute_seconds),
    }
