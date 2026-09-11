"""Bounded HTTP acceptance checks against an already monitored local server.

Uses stdlib clients only. Saves responses and failures privately; performs no
model loading, installation, tuning, or server process management.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--lifecycle-only', action='store_true', help='Only restart/raw-state checks; do not repeat quality tests')
    p.add_argument('--startup-timeout', type=int, default=540, help='Bounded readiness wait; cold model loading can exceed 180s')
    args = p.parse_args()
    if not args.lifecycle_only and args.reference is None:
        p.error('--reference is required for the full acceptance checks')
    if not 1 <= args.startup_timeout <= 600:
        p.error('--startup-timeout must be in [1,600]')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    suite = json.loads((root/'configs/evaluation.json').read_text())
    refs = {r['name']: r for r in json.loads(args.reference.read_text())['results']} if args.reference else {}
    base = f'http://127.0.0.1:{args.port}'
    observations = []

    def record(name, **fields):
        row = dict(name=name, **fields)
        observations.append(row)
        (args.output/'observations.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')
        print(name+': '+str(fields.get('passed', 'observed')), flush=True)

    def request(path, payload=None, expected=200):
        raw = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base+path, data=raw, headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(req, timeout=125)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            result = json.load(response)
            assert response.status == expected, (response.status, result)
            return result

    def chat(prompt, limit=128, stream=False):
        return dict(messages=[dict(role='system', content=suite['system']), dict(role='user', content=prompt)],
                    max_tokens=limit, temperature=0, stream=stream)

    def stream(payload, cancel=False):
        payload = dict(payload, stream=True, stream_options={'include_usage': True})
        conn = http.client.HTTPConnection('127.0.0.1', args.port, timeout=125)
        conn.request('POST', '/v1/chat/completions', json.dumps(payload), {'Content-Type':'application/json'})
        response = conn.getresponse()
        assert response.status == 200, response.read()
        parts, chunks, usage = [], [], None
        done = False
        try:
            while line := response.readline():
                if not line.startswith(b'data: '):
                    continue
                text = line[6:].decode().strip()
                if text == '[DONE]':
                    done = True
                    break
                chunk = json.loads(text)
                assert 'error' not in chunk, chunk
                chunks.append(chunk)
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    part = choice.get('delta', {}).get('content', '')
                    parts.append(part)
                    if cancel and part:
                        return dict(cancelled=True, text=''.join(parts))
        finally:
            response.close()
            conn.close()
        assert done, 'SSE stream ended without [DONE]'
        return dict(text=''.join(parts), usage=usage, chunks=chunks)

    def health_idle(timeout=15):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            state = request('/health')
            if not state['active']:
                return state
            time.sleep(.1)
        raise AssertionError('Engine did not return to idle')

    try:
        deadline = time.monotonic()+args.startup_timeout
        while True:
            try:
                state = request('/health')
                break
            except (OSError, AssertionError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(1)
        record('loaded_once', passed=state['model_load_count']==1, health=state, models=request('/v1/models'))
        if args.lifecycle_only:
            first = None
            memories = []
            for repeat in range(4):
                result = request('/v1/completions', dict(prompt='The capital of France is', max_tokens=16))
                text = result['choices'][0]['text']
                first = text if first is None else first
                assert text == first and 'Paris' in text
                health = health_idle()
                memories.append(health['allocator']['allocated_bytes'])
                record('raw_repeat_'+str(repeat), passed=True, response=result, health=health)
            assert len(set(memories[1:])) == 1, memories
            result = request('/v1/chat/completions', dict(messages=[
                dict(role='user', content='Remember the planet Mars.'),
                dict(role='assistant', content='I will remember Mars.'),
                dict(role='user', content='Which planet? Reply with one word.')], max_tokens=16))
            assert 'Mars' in result['choices'][0]['message']['content']
            record('multi_turn', passed=True, response=result, health=health_idle())
            return
        warm = request('/v1/chat/completions', chat('Reply with only the word READY.', 16))
        record('warmup', passed=bool(warm['choices'][0]['message']['content']), response=warm)
        quality = {}
        for task in suite['tasks']:
            result = request('/v1/chat/completions', chat(task['prompt'], suite['quality_max_tokens']))
            text = result['choices'][0]['message']['content']
            matched = text == refs[task['id']]['output_text']
            record('retained_'+task['id'], passed=matched, response=result, health=health_idle())
            assert matched, 'Retained text mismatch: '+task['id']
            quality[task['id']] = result
        prompt = suite['tasks'][0]['prompt']
        expected = quality['sql_parameters']
        streamed = stream(chat(prompt, suite['quality_max_tokens']))
        assert streamed['text'] == expected['choices'][0]['message']['content']
        assert streamed['usage'] == expected['usage']
        record('stream_equivalence', passed=True, response=streamed, health=health_idle())
        for limit in (1, 2, 7):
            result = request('/v1/chat/completions', chat('List integers from 1 to 100.', limit))
            assert result['usage']['completion_tokens'] == limit, result
            assert result['choices'][0]['finish_reason'] == 'length'
            record('limit_'+str(limit), passed=True, response=result)
        before = health_idle()
        cancelled = stream(chat('Write a detailed tutorial on implementing a binary search tree in Python.', 512), cancel=True)
        after = health_idle()
        assert after['cancelled'] == before['cancelled']+1, (before, after)
        record('stream_cancel', passed=True, response=cancelled, health=after)
        # Close a nonstreaming TCP client only after its generation has started.
        before = health_idle()
        client = socket.create_connection(('127.0.0.1', args.port), timeout=10)
        body = json.dumps(chat('Write a detailed tutorial on binary search trees in Python.', 512)).encode()
        client.sendall(f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n'.encode()+body)
        try:
            deadline = time.monotonic()+10
            while not request('/health')['active']:
                assert time.monotonic() < deadline, 'Nonstream request never became active'
                time.sleep(.05)
        finally:
            client.close()
        after = health_idle()
        assert after['cancelled'] == before['cancelled']+1, (before, after)
        record('nonstream_cancel', passed=True, health=after)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request, '/v1/chat/completions', chat(prompt, suite['quality_max_tokens'])) for _ in range(2)]
            results = [f.result() for f in futures]
        assert all(r['choices'][0]['message']['content'] == expected['choices'][0]['message']['content'] for r in results)
        record('overlap_and_recovery', passed=True, responses=results, health=health_idle())
        before = health_idle()
        rejected = request('/v1/chat/completions', chat('word '*5000, 32), expected=400)
        after = health_idle()
        assert before['started'] == after['started']
        record('context_rejected_before_generation', passed=True, response=rejected)
        raw = request('/v1/completions', dict(prompt='The capital of France is', max_tokens=16))
        record('raw_completion', passed=bool(raw['choices'][0]['text']), response=raw)
        final = health_idle()
        assert final['ready'] and final['failed'] == 0 and final['model_load_count'] == 1
        assert final['allocator']['allocated_bytes'] <= before['allocator']['allocated_bytes']+128*2**20
        record('final_health', passed=True, health=final)
    except BaseException as exc:
        record('failure', passed=False, error=type(exc).__name__+': '+str(exc))
        raise


if __name__ == '__main__':
    main()
