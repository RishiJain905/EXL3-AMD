"""Exact offline allocation of per-layer KV-cache precision under a byte budget.

Calibration-only helper: each option's ``loss`` is a local
attention-reconstruction proxy measured on calibration corpora, and ``bytes``
is that option's measured per-layer tensor size for one layer. Minimizing the
summed proxy under a budget does not establish model quality, downstream
accuracy, or speed; validate any assignment on held-out tasks before use.

``select_precision`` picks one option per layer minimizing summed loss subject
to summed bytes <= ``budget_bytes`` (exact inclusive integer bytes, not an
approximate bit count). Ties break deterministically by (total loss, total
bytes, lexicographic codec pairs in sorted ``layer_idx`` order). Losses sum
left to right in sorted ``layer_idx`` order with binary floating-point
addition. Search is sparse dynamic programming over byte totals with Pareto
pruning (a higher-byte state survives only with strictly lower loss than every
smaller-byte state); the retained frontier is bounded to 4096 states and the
call fails explicitly instead of approximating beyond that bound.

Supported pairs are ``aster5``/``aster5`` or any uniform ``q4``/``q5``/``q6``/
``q8`` K/V combination. Anything else (``f16``, mixed Aster, unknown formats)
is rejected, as are duplicate layers, duplicate pairs, unknown fields,
booleans in numeric positions, and nonfinite or negative losses.

Standard-library only; no Torch, no model discovery, no machine paths, and no
network access. The minimal CLI reads one portable JSON document holding
exactly ``{layers, budget_bytes}`` and writes the result to ``--output``,
refusing to overwrite::

    python allocate_cache_precision.py request.json --output result.json
"""

import argparse
import json
import math

MAX_FRONTIER = 4096
MAX_INPUT_BYTES = 1024 * 1024
MAX_LAYERS = 256
MAX_OPTIONS = 16

_UNIFORM = frozenset(('q4', 'q5', 'q6', 'q8'))
_LAYER_KEYS = frozenset(('layer_idx', 'options'))
_OPTION_KEYS = frozenset(('k', 'v', 'bytes', 'loss'))
_REQUEST_KEYS = frozenset(('layers', 'budget_bytes'))


def _pair_codecs(choices):
    return tuple((choice[0], choice[1]) for choice in choices)


def select_precision(layers, budget_bytes):
    """Allocate one precision option per layer under an exact byte budget."""
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError('Invalid budget_bytes')
    if not isinstance(layers, list) or not 1 <= len(layers) <= MAX_LAYERS:
        raise ValueError('Invalid layers')
    normalized = []
    seen = set()
    for layer in layers:
        if not isinstance(layer, dict) or set(layer) != _LAYER_KEYS:
            raise ValueError('Invalid layer fields')
        idx = layer['layer_idx']
        if type(idx) is not int or idx < 0:
            raise ValueError('Invalid layer_idx')
        if idx in seen:
            raise ValueError('Duplicate layer_idx')
        seen.add(idx)
        options = layer['options']
        if not isinstance(options, list) or not 1 <= len(options) <= MAX_OPTIONS:
            raise ValueError('Invalid options')
        pairs = set()
        clean = []
        for option in options:
            if not isinstance(option, dict) or set(option) != _OPTION_KEYS:
                raise ValueError('Invalid option fields')
            k = option['k']
            v = option['v']
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError('Invalid precision format')
            if (k, v) != ('aster5', 'aster5') and (k not in _UNIFORM or v not in _UNIFORM):
                raise ValueError('Unsupported precision pair')
            if (k, v) in pairs:
                raise ValueError('Duplicate precision pair')
            pairs.add((k, v))
            size = option['bytes']
            if type(size) is not int or size <= 0:
                raise ValueError('Invalid option bytes')
            loss = option['loss']
            if type(loss) is int:
                if loss < 0:
                    raise ValueError('Invalid option loss')
                try:
                    loss = float(loss)
                except OverflowError:
                    raise ValueError('Invalid option loss') from None
            elif type(loss) is float:
                if not math.isfinite(loss) or loss < 0:
                    raise ValueError('Invalid option loss')
            else:
                raise ValueError('Invalid option loss')
            clean.append((k, v, size, loss))
        normalized.append((idx, clean))
    normalized.sort(key=lambda item: item[0])

    frontier = {0: (0.0, ())}
    for _, options in normalized:
        nxt = {}
        for used, (used_loss, prefix) in frontier.items():
            for choice in options:
                total = used + choice[2]
                if total > budget_bytes:
                    continue
                entry = (used_loss + choice[3], prefix + (choice,))
                prev = nxt.get(total)
                if prev is None or (entry[0], _pair_codecs(entry[1])) < (prev[0], _pair_codecs(prev[1])):
                    nxt[total] = entry
        pruned = {}
        best_loss = math.inf
        for total in sorted(nxt):
            loss_value, choices = nxt[total]
            if loss_value < best_loss:
                best_loss = loss_value
                pruned[total] = (loss_value, choices)
        if len(pruned) > MAX_FRONTIER:
            raise ValueError('Allocation frontier exceeds 4096 states; refine inputs')
        frontier = pruned
        if not frontier:
            break
    if not frontier:
        raise ValueError('No feasible allocation within budget')
    winner = min(frontier, key=lambda total: (frontier[total][0], total, _pair_codecs(frontier[total][1])))
    total_loss, choices = frontier[winner]
    if not math.isfinite(total_loss):
        raise ValueError('Nonfinite total loss')
    picked = [dict(layer_idx=idx, k=c[0], v=c[1], bytes=c[2], loss=c[3]) for idx, c in zip([i for i, _ in normalized], choices)]
    return dict(total_bytes=winner, total_loss=total_loss, layers=picked)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate input JSON key')
        result[key] = value
    return result


def _nonfinite(unused):
    raise ValueError('Nonfinite input JSON value')


def _load_request(path):
    try:
        with open(path, 'rb') as handle:
            raw = handle.read(MAX_INPUT_BYTES + 1)
    except OSError:
        raise ValueError('Cannot read input file') from None
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError('Input exceeds size limit')
    try:
        text = raw.decode('utf-8')
    except UnicodeError:
        raise ValueError('Cannot read input file') from None
    try:
        payload = json.loads(text, object_pairs_hook=_unique, parse_constant=_nonfinite)
    except (json.JSONDecodeError, RecursionError):
        raise ValueError('Invalid input JSON') from None
    if not isinstance(payload, dict) or set(payload) != _REQUEST_KEYS:
        raise ValueError('Input must hold exactly layers and budget_bytes')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description='Exact offline cache-precision allocation under a byte budget.')
    parser.add_argument('input', help='JSON file with exactly {layers, budget_bytes}')
    parser.add_argument('--output', required=True, help='Result JSON path; refuses to overwrite')
    args = parser.parse_args(argv)
    try:
        payload = _load_request(args.input)
        result = select_precision(payload['layers'], payload['budget_bytes'])
    except ValueError as exc:
        parser.exit(1, 'error: %s\n' % exc)
    try:
        with open(args.output, 'x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
    except FileExistsError:
        parser.exit(1, 'error: refusing to overwrite %s\n' % args.output)
    except OSError as exc:
        parser.exit(1, 'error: cannot write output: %s\n' % exc)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
