"""Analyze frozen MiMo MTP parity and bracketed speed runs, without inference."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def cases(record):
    if record.get('status') != 'completed' or not isinstance(record.get('results'), list):
        raise ValueError('Expected a completed evaluator result')
    result = {}
    for row in record['results']:
        name = row.get('name')
        tokens = row.get('output_token_ids')
        if not isinstance(name, str) or not name or name in result:
            raise ValueError('Missing or duplicate case name')
        if (not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens)
                or type(row.get('output_tokens')) is not int or row['output_tokens'] != len(tokens)
                or type(row.get('input_tokens')) is not int or row['input_tokens'] <= 0
                or not isinstance(row.get('input_sha256'), str)
                or type(row.get('warmup')) is not bool or type(row.get('fixed_length')) is not bool):
            raise ValueError('Malformed token/count/control fields: ' + name)
        result[name] = row
    if not result:
        raise ValueError('No cases')
    return result


def draft_statistics(row):
    accepted, rejected = row.get('accepted_draft_tokens'), row.get('rejected_draft_tokens')
    for value in (accepted, rejected):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('Invalid draft counter')
    windows = row.get('draft_windows', [])
    histogram = Counter()
    drafted = accepted_prefixes = truncated = 0
    for i, window in enumerate(windows):
        if (not isinstance(window, (list, tuple)) or len(window) != 3
                or any(type(v) is not int for v in window)):
            raise ValueError('Invalid draft-window record')
        position, length, prefix = window
        if position < 0 or length <= 0 or not 0 <= prefix <= length:
            raise ValueError('Invalid draft-window extents')
        histogram[str(prefix)] += 1
        drafted += length
        accepted_prefixes += prefix
        truncated += int(i < len(windows)-1 and prefix < length)
    return dict(accepted=accepted, rejected=rejected, windows=len(windows),
                drafted_positions=drafted if windows else None,
                accepted_prefix_histogram=dict(histogram),
                accepted_generated_fraction=accepted_prefixes/drafted if drafted else None,
                non_final_truncated_windows=truncated if windows else None,
                denominator='All generated draft positions, including untested tails; not conditional prediction accuracy')


def compare_records(reference, candidate):
    before, after = cases(reference), cases(candidate)
    controls = ('mode', 'cache_tokens', 'cache_k', 'cache_v', 'thinking', 'decode_fusions')
    control_match = all(reference.get(k) == candidate.get(k) for k in controls)
    rows = []
    for name in sorted(set(before) | set(after)):
        if name not in before or name not in after:
            rows.append(dict(name=name, passed=False, error='case inventory mismatch'))
            continue
        a, b = before[name], after[name]
        same_input = a['input_sha256'] == b['input_sha256'] and a['input_tokens'] == b['input_tokens']
        same_tokens = a['output_token_ids'] == b['output_token_ids']
        same_stop = a.get('eos_reason') == b.get('eos_reason')
        same_controls = all(a.get(k) == b.get(k) for k in ('warmup', 'fixed_length', 'output_limit'))
        mismatch = next((i for i, (x, y) in enumerate(zip(a['output_token_ids'], b['output_token_ids'])) if x != y), None)
        if not same_tokens and mismatch is None:
            mismatch = min(len(a['output_token_ids']), len(b['output_token_ids']))
        rows.append(dict(name=name, passed=same_input and same_tokens and same_stop and same_controls,
                         input_match=same_input, output_match=same_tokens, stopping_match=same_stop,
                         controls_match=same_controls, first_output_mismatch=mismatch,
                         candidate_draft=draft_statistics(b)))
    return dict(passed=control_match and all(row['passed'] for row in rows),
                controls_match=control_match, cases=rows)


def measured_medians(record):
    prefixes = {}
    for name, row in cases(record).items():
        if row['warmup']:
            continue
        prefix, sep, repeat = name.rpartition('-')
        seconds = row.get('end_to_end_seconds')
        if (not prefix or not sep or repeat not in ('0', '1', '2')
                or not row['fixed_length'] or row.get('output_limit') != 128
                or row['output_tokens'] != 128 or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError('Invalid fixed-length speed case: ' + name)
        prefixes.setdefault(prefix, {})[repeat] = seconds
    if len(prefixes) != 2 or any(set(v) != {'0', '1', '2'} for v in prefixes.values()):
        raise ValueError('Speed requires two prefixes with three repetitions each')
    return {name: statistics.median(values.values()) for name, values in prefixes.items()}


def analyze_speed(before, treatment, after):
    parity_before = compare_records(before, treatment)
    parity_after = compare_records(before, after)
    medians = [measured_medians(r) for r in (before, treatment, after)]
    if any(set(m) != set(medians[0]) for m in medians):
        raise ValueError('Speed prefix inventory mismatch')
    controls = before.get('mtp') is False and after.get('mtp') is False and treatment.get('mtp') is True
    rows = []
    for name in sorted(medians[0]):
        a, b, c = (m[name] for m in medians)
        control = (a+c)/2
        rows.append(dict(prefix=name, before_median_seconds=a, treatment_median_seconds=b,
                         after_median_seconds=c, reference_seconds=control,
                         speed_ratio=control/b, control_drift=abs(a-c)/control))
    parity = parity_before['passed'] and parity_after['passed']
    drift_ok = all(row['control_drift'] <= .10 for row in rows)
    ratio = math.exp(sum(math.log(row['speed_ratio']) for row in rows)/len(rows))
    gain = ratio >= 1.05
    prefix_gate = all(row['speed_ratio'] >= 1/1.05 for row in rows)
    valid = controls and parity and drift_ok
    return dict(valid=valid, parity_passed=parity, controls_match=controls,
                control_drift_passed=drift_ok, speed_ratio=ratio, gain_passed=gain,
                prefix_regression_passed=prefix_gate,
                qualified=valid and gain and prefix_gate, prefixes=rows,
                parity_before=parity_before, parity_after=parity_after,
                claim_limit='Warmed request-time comparison on two frozen prefixes; no general speed or BF16 fidelity claim')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode, fields in (('compare', ('reference', 'candidate')), ('speed', ('before', 'treatment', 'after'))):
        child = sub.add_parser(mode)
        for field in fields:
            child.add_argument('--'+field, required=True, type=Path)
        child.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    fields = ('reference', 'candidate') if args.mode == 'compare' else ('before', 'treatment', 'after')
    output = args.output.resolve()
    paths = [getattr(args, field).resolve() for field in fields]
    if args.output.is_symlink() or output.exists() or output in paths:
        parser.error('Output must be a new file distinct from the inputs')
    records = [json.loads(path.read_text(encoding='utf8')) for path in paths]
    result = compare_records(*records) if args.mode == 'compare' else analyze_speed(*records)
    result['source_sha256'] = {field: hashlib.sha256(path.read_bytes()).hexdigest() for field, path in zip(fields, paths)}
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if result.get('passed', result.get('valid')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
