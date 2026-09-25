"""Bounded image integration checks; external supervisor owns resource limits."""
import asyncio
import hashlib
import json
from pathlib import Path
import time
import tomllib

from serve_exl3 import Engine, parser
from quantlab.images import ImageInputError, local_image_url


def main():
    p = parser()
    p.add_argument('--fixtures', type=Path, required=True)
    args = p.parse_args()
    config = tomllib.loads(args.config.read_text())
    if not all(config['execution'].get(k) is True for k in ('allow_local_inference', 'allow_backend_probes')):
        p.error('Both local execution permissions required')
    fixture_bytes = args.fixtures.read_bytes()
    fixtures = json.loads(fixture_bytes)
    engine = Engine(args)
    counters = dict(raw_target_rows=0, raw_target_finite=True)
    original_sampler = engine.Sampler
    def sampler_factory(*a, **kw):
        sampler = original_sampler(*a, **kw)
        original = sampler.forward
        def forward(logits, *args, **kwargs):
            rows = logits[..., :engine.tokenizer.actual_vocab_size]
            counters['raw_target_rows'] += rows.numel() // rows.shape[-1]
            if not engine.torch.isfinite(rows).all():
                counters['raw_target_finite'] = False
                raise RuntimeError('Nonfinite target logits')
            return original(logits, *args, **kwargs)
        sampler.forward = forward
        return sampler
    engine.Sampler = sampler_factory
    cases = []
    async def run():
        for fixture in fixtures['cases']:
            parts = [dict(type='image_url', image_url=dict(url=local_image_url(args.fixtures.parent / image)))
                     for image in fixture.get('images', [])]
            parts.append(dict(type='text', text=fixture['prompt']))
            messages = [dict(role='user', content=parts if fixture.get('images') else fixture['prompt'])]
            engine.torch.cuda.synchronize()
            engine.torch.cuda.reset_peak_memory_stats()
            before = engine.memory()
            start = time.monotonic()
            try:
                prepared = engine.prepare(messages=messages, max_tokens=fixture.get('max_tokens', 32))
            except ImageInputError as exc:
                if fixture.get('images') and not engine.vision_enabled:
                    cases.append(dict(name=fixture['name'], rejected=True, reason=str(exc)))
                    continue
                raise
            response = []
            async for event in engine.generate(prepared):
                response.append(event)
            engine.torch.cuda.synchronize()
            cases.append(dict(name=fixture['name'], seconds=time.monotonic()-start,
                input_token_ids=prepared['ids'].flatten().tolist(), response=response,
                memory_before=before, memory_after=engine.memory()))
            (args.output/'cases.partial.json').write_text(json.dumps(cases, indent=2))
    try:
        with engine.torch.inference_mode():
            asyncio.run(run())
        result = dict(status='completed', fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
            vision=engine.vision_status, mtp_depth=engine.depth, counters=counters, cases=cases)
        (args.output/'result.json').write_text(json.dumps(result, indent=2))
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
