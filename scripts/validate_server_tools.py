"""Exercise real tool generation and tool-result turns on a monitored server.

This client executes only a fixed in-memory lookup fixture. It never evaluates
model text as code or runs a shell. Responses are private run artifacts.
"""
import argparse
import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--startup-timeout', type=int, default=540)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = f'http://127.0.0.1:{args.port}'
    observations = []

    def record(name, **values):
        observations.append(dict(name=name, **values))
        (args.output/'observations.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')
        print(name, values.get('passed'), flush=True)

    def request(path='/v1/chat/completions', payload=None, expected=200):
        req = urllib.request.Request(base+path, data=json.dumps(payload).encode() if payload else None,
                                     headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(req, timeout=240)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            result = json.load(response)
            if response.status != expected:
                record('unexpected_http', passed=False, status=response.status, response=result, payload=payload)
            assert response.status == expected, (response.status, result)
            return result

    def stream(payload):
        conn = http.client.HTTPConnection('127.0.0.1', args.port, timeout=240)
        conn.request('POST', '/v1/chat/completions', json.dumps(dict(payload, stream=True,
            stream_options={'include_usage': True})), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        assert response.status == 200, response.read()
        chunks, done = [], False
        try:
            while line := response.readline():
                if not line.startswith(b'data: '):
                    continue
                text = line[6:].decode().strip()
                if text == '[DONE]':
                    done = True
                    break
                chunks.append(json.loads(text))
        finally:
            response.close()
            conn.close()
        assert done
        return chunks

    tool = {'type': 'function', 'function': {'name': 'lookup_asset',
        'description': 'Read the owner and patch status for an asset from the local inventory.',
        'parameters': {'type': 'object', 'properties': {'asset': {'type': 'string'}},
                       'required': ['asset'], 'additionalProperties': False}}}
    fixtures = {'ALPHA7': {'owner': 'Mira-927', 'patch_state': 'current'},
                'BETA2': {'owner': 'Noel-416', 'patch_state': 'overdue'}}

    def body(prompt, **options):
        return dict(messages=[{'role': 'system', 'content': 'Use the supplied tools when needed. Never invent tool results.'},
                              {'role': 'user', 'content': prompt}],
                    tools=[tool], max_tokens=256, temperature=0, **options)

    def calls(response, assets):
        choice = response['choices'][0]
        assert choice['finish_reason'] == 'tool_calls', choice
        result = choice['message']['tool_calls']
        assert len(result) == len(assets), result
        assert len({c['id'] for c in result}) == len(result)
        assert all(c['type'] == 'function' and c['function']['name'] == 'lookup_asset' for c in result)
        assert sorted(json.loads(c['function']['arguments'])['asset'] for c in result) == sorted(assets)
        assert '<tool_call>' not in (choice['message'].get('content') or '')
        return result

    try:
        deadline = time.monotonic()+args.startup_timeout
        while True:
            try:
                health = request('/health')
                if health['ready']:
                    break
            except (OSError, AssertionError):
                if time.monotonic() >= deadline:
                    raise
            time.sleep(1)
        record('ready', passed=health['tool_protocol'] == 'qwen_xml', health=health)
        payload = body('Look up ALPHA7 now.', tool_choice={'type': 'function', 'function': {'name': 'lookup_asset'}},
                       parallel_tool_calls=False)
        response = request(payload=payload)
        first_calls = calls(response, ['ALPHA7'])
        record('named_nonstream', passed=True, payload=payload, response=response)
        history = payload['messages']+[response['choices'][0]['message']]
        for call in first_calls:
            asset = json.loads(call['function']['arguments'])['asset']
            history.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(fixtures[asset])})
        response = request(payload=dict(messages=history, tools=[tool], max_tokens=192, temperature=0))
        assert response['choices'][0]['finish_reason'] == 'stop', response
        assert 'Mira-927' in response['choices'][0]['message']['content'], response
        record('tool_result_turn', passed=True, response=response)

        payload = body('Look up BETA2 now.', tool_choice='required', parallel_tool_calls=False)
        chunks = stream(payload)
        assert all('error' not in chunk for chunk in chunks), chunks
        deltas = [c for chunk in chunks for choice in chunk.get('choices', [])
                  for c in choice.get('delta', {}).get('tool_calls', [])]
        assert len(deltas) == 1 and deltas[0]['index'] == 0, chunks
        assert json.loads(deltas[0]['function']['arguments']) == {'asset': 'BETA2'}, chunks
        assert any(c.get('usage', {}).get('completion_tokens', 0) > 0 for c in chunks), chunks
        assert any(choice.get('finish_reason') == 'tool_calls' for c in chunks for choice in c.get('choices', []))
        record('required_sse', passed=True, payload=payload, chunks=chunks)

        payload = body('Look up BOTH ALPHA7 and BETA2 in this response. Emit two calls, one for each asset.',
                       tool_choice='required', parallel_tool_calls=True)
        response = request(payload=payload)
        multiple = calls(response, ['ALPHA7', 'BETA2'])
        record('multiple_calls', passed=True, payload=payload, response=response)
        history = payload['messages']+[response['choices'][0]['message']]
        for call in reversed(multiple):
            asset = json.loads(call['function']['arguments'])['asset']
            history.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(fixtures[asset])})
        history.append({'role': 'user', 'content': 'State each asset and its owner. Use the existing results.'})
        response = request(payload=dict(messages=history, tools=[tool], tool_choice='none', max_tokens=192, temperature=0))
        content = response['choices'][0]['message']['content']
        assert 'Mira-927' in content and 'Noel-416' in content, response
        assert 'tool_calls' not in response['choices'][0]['message']
        record('reordered_results_none', passed=True, response=response)

        payload = body('Look up ALPHA7 now.', tool_choice='required')
        payload['max_tokens'] = 8
        response = request(payload=payload, expected=502)
        assert response['error']['code'] == 'invalid_tool_call', response
        record('truncated_nonstream', passed=True, response=response)
        chunks = stream(payload)
        assert any(c.get('error', {}).get('code') == 'invalid_tool_call' for c in chunks)
        assert not any(choice.get('delta', {}).get('tool_calls') for c in chunks for choice in c.get('choices', []))
        record('truncated_sse', passed=True, chunks=chunks)
        response = request(payload={'messages': [{'role': 'user', 'content': 'What is 2 + 2? Answer briefly.'}],
                                    'max_tokens': 32, 'temperature': 0})
        assert '4' in response['choices'][0]['message']['content'], response
        health = request('/health')
        assert health['ready'] and not health['active'] and health['failed'] == 0, health
        assert health['invalid_tool_outputs'] >= 2, health
        record('recovery', passed=True, response=response, health=health)
    except Exception as exc:
        record('failure', passed=False, error=type(exc).__name__+': '+str(exc))
        raise


if __name__ == '__main__':
    main()
