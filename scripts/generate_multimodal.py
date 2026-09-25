"""One bounded image-aware chat request using the same engine as the server."""
import asyncio
import hashlib
import json
from pathlib import Path
import tomllib

from serve_exl3 import Engine, parser
from quantlab.images import local_image_url, validate_vision_options


def main():
    p = parser()
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--image', type=Path, action='append', default=[])
    p.set_defaults(mode='generate', reasoning='off')
    args = p.parse_args()
    try:
        validate_vision_options(args)
        if not 1 <= args.max_tokens <= 8192:
            raise ValueError('--max-tokens must be in [1,8192]')
        config = tomllib.loads(args.config.read_text())
        if not all(config['execution'].get(k) is True for k in ('allow_local_inference', 'allow_backend_probes')):
            raise ValueError('Both local execution permissions are required')
        candidate, output = args.candidate.resolve(), args.output.resolve()
        if candidate == output or candidate in output.parents or output in candidate.parents:
            raise ValueError('Output and candidate must be separate nonnested paths')
        parts = [dict(type='image_url', image_url=dict(url=local_image_url(path))) for path in args.image]
        parts.append(dict(type='text', text=args.prompt_file.read_text(encoding='utf-8')))
        settings_bytes = (Path(__file__).resolve().parents[1]/'configs/evaluation.json').read_bytes()
        settings = json.loads(settings_bytes)
        # Preserve ordinary generate-mode's existing system message. Enabling
        # residency alone must not silently change the text chat protocol.
        messages = [dict(role='system', content=settings['system']),
                    dict(role='user', content=parts)]
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    engine = Engine(args)
    engine.record('cli_prompt_settings', sha256=hashlib.sha256(settings_bytes).hexdigest())
    async def run():
        prepared = engine.prepare(messages=messages, max_tokens=args.max_tokens)
        response = []
        async for event in engine.generate(prepared):
            response.append(event)
        return response, int(prepared['ids'].numel())
    try:
        with engine.torch.inference_mode():
            response, input_tokens = asyncio.run(run())
        records = [json.loads(line) for line in (args.output/'events.jsonl').read_text().splitlines()]
        finished = next(row for row in reversed(records) if row['stage']=='request_finished')
        result = dict(name='generate', warmup=False, input_tokens=input_tokens,
            output_tokens=len(finished['output_token_ids']), output_token_ids=finished['output_token_ids'],
            output_text=finished['output_text'], eos_reason=finished['eos_reason'],
            decode_tokens_per_second=finished['decode_tokens_per_second'])
        (args.output/'result.json').write_text(json.dumps(dict(status='completed', events=response, results=[result]), indent=2))
    finally:
        engine.shutdown()


if __name__ == '__main__':
    main()
