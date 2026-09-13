"""Pure request limits shared by the vendored EXL3 HTTP server."""

from __future__ import annotations

import json


MAX_COMPLETIONS_PER_REQUEST = 16
MAX_DRY_BREAKERS = 64
MAX_DRY_BREAKER_BYTES = 256
MAX_DRY_BREAKERS_TOTAL_BYTES = 4096
MAX_DRY_BREAKER_CACHE_ENTRIES = 32


def normalize_dry_breakers(value: list[str] | tuple[str, ...] | str) -> tuple[str, ...]:
    """Validate and canonicalize request-provided DRY sequence breakers."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("dry_sequence_breakers must be a JSON array of strings") from None

    if not isinstance(value, (list, tuple)):
        raise ValueError("dry_sequence_breakers must be a JSON array of strings")
    if len(value) > MAX_DRY_BREAKERS:
        raise ValueError(
            f"dry_sequence_breakers supports at most {MAX_DRY_BREAKERS} entries"
        )

    total_bytes = 0
    unique = set()
    for breaker in value:
        if not isinstance(breaker, str):
            raise ValueError("dry_sequence_breakers must be a JSON array of strings")
        try:
            size = len(breaker.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("dry_sequence_breakers must contain valid UTF-8 text") from None
        if size > MAX_DRY_BREAKER_BYTES:
            raise ValueError(
                f"each dry sequence breaker must be at most {MAX_DRY_BREAKER_BYTES} UTF-8 bytes"
            )
        total_bytes += size
        if total_bytes > MAX_DRY_BREAKERS_TOTAL_BYTES:
            raise ValueError(
                "dry_sequence_breakers must total at most "
                f"{MAX_DRY_BREAKERS_TOTAL_BYTES} UTF-8 bytes"
            )
        if breaker:
            unique.add(breaker)

    # Breaker order and exact duplicates do not affect the resulting token-ID set.
    return tuple(sorted(unique))
