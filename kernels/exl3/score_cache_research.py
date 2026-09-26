"""Frozen stage2-score-v2 semantic/format grading for the synthetic research suite.

Accepts explicitly equivalent structured answers, never executes model output.
JSON-only compliance is separate from recovering the requested values.
"""
import json
import math

VERSION = 'stage2-score-v2'


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate answer key')
        result[key] = value
    return result


def _constant(unused):
    raise ValueError('Nonfinite answer value')


def _answers(text):
    decoder = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_constant)
    try:
        return [decoder.decode(text.strip())], True
    except ValueError:
        pass
    values = []
    pos = 0
    while pos < len(text):
        if text[pos] not in '[{':
            pos += 1
            continue
        try:
            value, consumed = decoder.raw_decode(text[pos:])
        except ValueError:
            # Do not award a nested fragment from a malformed or duplicate-key answer.
            return [], False
        values.append(value)
        pos += consumed
    return values, False


def _same(actual, expected):
    if type(expected) is int:
        if type(actual) is int:
            return actual == expected
        return type(actual) is float and math.isfinite(actual) and actual == expected
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def score(text, family, expected):
    values, strict = _answers(text)
    original = values[-1] if values else None
    actual = original
    if family == 'reasoning' and isinstance(actual, dict):
        keys = ('additional_tokens_per_club', 'tokens_left_over')
        if set(actual) == set(keys):
            actual = [actual[k] for k in keys]
    elif family == 'instructions' and isinstance(actual, dict):
        keys = ('alpha', 'beta', 'gamma')
        if set(actual) == set(keys):
            actual = [actual[k] for k in keys]
    elif family == 'code' and isinstance(actual, dict) and len(actual) == 1:
        for key in ('output', 'result'):
            if key in actual:
                actual = actual[key]
                break
    if isinstance(expected, dict):
        items = {key: isinstance(actual, dict) and _same(actual.get(key), value)
                 for key, value in expected.items()}
        correct = all(items.values())
        format_ok = strict and isinstance(original, dict) and set(original) == set(expected)
    else:
        items = None
        correct = _same(actual, expected)
        format_ok = strict and isinstance(original, list)
        if family in ('reasoning', 'instructions') and format_ok:
            format_ok = len(original) == len(expected)
    return dict(correct=correct, item_scores=items, format_compliant=format_ok, parsed=actual)
