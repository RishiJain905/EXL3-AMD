"""Validated sampling controls over the vendored ComboSampler.

Greedy (temperature=0) stays the default and maps to ArgmaxSampler, matching
prior behavior. Any non-greedy request maps to ComboSampler with explicit,
range-checked parameters; unsupported keys are rejected by the caller, never
silently ignored.
"""

from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": None,
}

_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
         "presence_penalty", "frequency_penalty", "seed")


def _is_number(value: Any) -> bool:
    return type(value) in (int, float)


def check_sampling(body: dict) -> dict[str, Any]:
    """Validate per-request sampling overrides; returns the full parameter set."""
    if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
        raise ValueError("only n=1 is supported")
    params = dict(DEFAULTS)
    if "temperature" in body:
        value = body["temperature"]
        if not _is_number(value) or not 0 <= value <= 2:
            raise ValueError("temperature must be a number between 0 and 2")
        params["temperature"] = value
    if "top_p" in body:
        value = body["top_p"]
        if not _is_number(value) or not 0 < value <= 1:
            raise ValueError("top_p must be a number in (0, 1]")
        params["top_p"] = value
    if "top_k" in body:
        value = body["top_k"]
        if type(value) is not int or not 0 <= value <= 1000:
            raise ValueError("top_k must be an integer between 0 and 1000")
        params["top_k"] = value
    if "min_p" in body:
        value = body["min_p"]
        if not _is_number(value) or not 0 <= value < 1:
            raise ValueError("min_p must be a number in [0, 1)")
        params["min_p"] = value
    if "repetition_penalty" in body:
        value = body["repetition_penalty"]
        if not _is_number(value) or not 0 < value <= 2:
            raise ValueError("repetition_penalty must be a number in (0, 2]")
        params["repetition_penalty"] = value
    for key in ("presence_penalty", "frequency_penalty"):
        if key in body:
            value = body[key]
            if not _is_number(value) or not -2 <= value <= 2:
                raise ValueError(f"{key} must be a number between -2 and 2")
            params[key] = value
    if "seed" in body:
        value = body["seed"]
        if type(value) is not int or not 0 <= value < 2 ** 63:
            raise ValueError("seed must be an integer between 0 and 2**63 - 1")
        params["seed"] = value
    return params


def is_greedy(params: dict) -> bool:
    """Mirror ComboSampler's greedy rule: temperature 0 or top_k 1."""
    return params.get("temperature", 0) == 0 or params.get("top_k") == 1


def build_sampler(params: dict, *, ArgmaxSampler: Any, ComboSampler: Any) -> Any:
    """Instantiate the vendored sampler for validated parameters."""
    params = dict(DEFAULTS, **params)
    if is_plain_greedy(params):
        return ArgmaxSampler()
    return ComboSampler(
        rep_p=params["repetition_penalty"],
        freq_p=params["frequency_penalty"],
        pres_p=params["presence_penalty"],
        temperature=params["temperature"],
        min_p=params["min_p"],
        top_k=params["top_k"],
        top_p=params["top_p"],
    )


def is_plain_greedy(params: dict) -> bool:
    """True only when the optimized argmax path preserves every control."""
    return (is_greedy(params)
            and params.get("repetition_penalty", 1.0) == 1.0
            and params.get("presence_penalty", 0.0) == 0.0
            and params.get("frequency_penalty", 0.0) == 0.0)
