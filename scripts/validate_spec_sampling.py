"""Speculative sampling checks against already running local servers.

collect  Seeded short completions of one prompt, for a token-frequency comparison.
speed    Longer sampled chat generations; reports decode rate and MTP acceptance.
compare  Two-sample test on the token IDs that two servers recorded in their
         result/events.jsonl (permutation chi-square homogeneity per position).

Stdlib only. Performs no model loading or server management; start each server
with run.py (for example spec sampling on, off, or MTP off) and stop it with
its run directory's stop file.
"""
import argparse
import json
from pathlib import Path
import random
import time
import urllib.request

# Every position keeps entropy (one digit token each), so each drafted position is tested
PROMPT = 'Forty random decimal digits, separated by spaces: 7 2 9 4 0 6'
SPEED_PROMPTS = (
    'Write a Python function that parses an ISO 8601 duration string and returns seconds. Include tests.',
    'Explain how a B-tree differs from a binary search tree, with an example of an insertion.',
    'Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.',
    'A train leaves at 9:40 and travels 212 km at 83 km/h. When does it arrive? Show your reasoning.',
)


def post(port, path, payload, timeout=600):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def sampling(args):
    return dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)


def collect(args):
    rows = []
    start = time.monotonic()
    for seed in range(args.requests):
        body = post(args.port, '/v1/completions', dict(prompt=PROMPT, max_tokens=args.max_tokens,
                                                       seed=seed, **sampling(args)))
        rows.append(dict(seed=seed, text=body['choices'][0]['text'], usage=body.get('usage')))
    write(args.output, dict(kind='collect', prompt=PROMPT, sampling=sampling(args),
                            max_tokens=args.max_tokens, seconds=time.monotonic() - start, rows=rows))
    print(f'collected={len(rows)} seconds={time.monotonic() - start:.1f}')


def speed(args):
    rows = []
    for repeat in range(args.repeats):
        for index, prompt in enumerate(SPEED_PROMPTS):
            body = post(args.port, '/v1/chat/completions', dict(
                messages=[dict(role='user', content=prompt)], max_tokens=args.max_tokens,
                seed=1000 * repeat + index, **sampling(args)))
            usage = body['usage']
            timings = usage.get('timings', {})
            drafts = usage.get('draft_tokens') or {}
            rows.append(dict(prompt=index, repeat=repeat, completion_tokens=usage['completion_tokens'],
                             decode_tokens_per_second=timings.get('decode_tokens_per_second'),
                             accepted=drafts.get('accepted'), rejected=drafts.get('rejected')))
            print(json.dumps(rows[-1]), flush=True)
    rates = [r['decode_tokens_per_second'] for r in rows if r['decode_tokens_per_second']]
    accepted = sum(r['accepted'] or 0 for r in rows)
    proposed = accepted + sum(r['rejected'] or 0 for r in rows)
    summary = dict(requests=len(rows), mean_decode_tokens_per_second=sum(rates) / len(rates) if rates else None,
                   acceptance=accepted / proposed if proposed else None)
    write(args.output, dict(kind='speed', sampling=sampling(args), max_tokens=args.max_tokens,
                            rows=rows, summary=summary))
    print(json.dumps(summary))


def finished(events, first=0, count=None):
    """Output token IDs and draft rounds of completed requests [first, first + count)."""
    out = []
    for line in Path(events).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('stage') == 'request_finished' and row.get('completed'):
            out.append(row)
    return out[first:None if count is None else first + count]


def chi_square(a, b):
    """Homogeneity statistic for two label lists; categories under 5 expected are pooled."""
    counts = {}
    for label in a:
        counts.setdefault(label, [0, 0])[0] += 1
    for label in b:
        counts.setdefault(label, [0, 0])[1] += 1
    n_a, n_b = len(a), len(b)
    total = n_a + n_b
    pooled, cells = [0, 0], []
    for pair in counts.values():
        if sum(pair) * min(n_a, n_b) / total < 5:
            pooled[0] += pair[0]
            pooled[1] += pair[1]
        else:
            cells.append(pair)
    if sum(pooled):
        cells.append(pooled)
    stat = 0.0
    for x, y in cells:
        n = x + y
        for observed, size in ((x, n_a), (y, n_b)):
            expected = n * size / total
            stat += (observed - expected) ** 2 / expected
    return stat, len(cells)


def permutation_p(a, b, rounds, rng):
    observed, cells = chi_square(a, b)
    labels = list(a) + list(b)
    hits = 0
    for _ in range(rounds):
        rng.shuffle(labels)
        if chi_square(labels[:len(a)], labels[len(a):])[0] >= observed:
            hits += 1
    return observed, cells, (hits + 1) / (rounds + 1)


def compare(args):
    runs = [finished(path, first, args.count) for path, first in zip(args.events, args.first)]
    rng = random.Random(0)
    report = dict(events=[str(p) for p in args.events], requests=[len(r) for r in runs], positions=[])
    for position in args.positions:
        for kind, pick in (('token', lambda ids: ids[position - 1]), ('prefix', lambda ids: tuple(ids[:position]))):
            labels = [[pick(row['output_token_ids']) for row in run
                       if len(row['output_token_ids']) >= position] for run in runs]
            stat, cells, p = permutation_p(labels[0], labels[1], args.permutations, rng)
            distinct = [len(set(x)) for x in labels]
            report['positions'].append(dict(position=position, kind=kind, samples=[len(x) for x in labels],
                                            distinct=distinct, chi_square=stat, cells=cells, p_value=p))
            print(f'{kind}@{position} samples={[len(x) for x in labels]} distinct={distinct} '
                  f'chi2={stat:.1f} cells={cells} p={p:.3f}')
    for run, path in zip(runs, args.events):
        rounds = [r for row in run for r in row.get('draft_rounds') or ()]
        if rounds:
            accepted = sum(r[2] for r in rounds)
            print(f'{path}: rounds={len(rounds)} tokens_per_round={1 + accepted / len(rounds):.3f}')
    if args.output:
        write(args.output, report)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('collect', 'speed'):
        s = sub.add_parser(name)
        s.add_argument('--port', type=int, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--temperature', type=float, default=0.6)
        s.add_argument('--top-p', type=float, default=0.95)
        s.add_argument('--top-k', type=int, default=20)
    sub.choices['collect'].add_argument('--requests', type=int, default=1500)
    sub.choices['collect'].add_argument('--max-tokens', type=int, default=4)
    sub.choices['speed'].add_argument('--repeats', type=int, default=2)
    sub.choices['speed'].add_argument('--max-tokens', type=int, default=512)
    c = sub.add_parser('compare')
    c.add_argument('events', type=Path, nargs=2)
    c.add_argument('--positions', type=int, nargs='+', default=[1, 2, 3, 4])
    c.add_argument('--permutations', type=int, default=1000)
    c.add_argument('--first', type=int, nargs=2, default=[0, 0], help='index of the first collected request per run')
    c.add_argument('--count', type=int, default=None)
    c.add_argument('--output', type=Path)
    args = p.parse_args()
    dict(collect=collect, speed=speed, compare=compare)[args.command](args)


if __name__ == '__main__':
    main()
