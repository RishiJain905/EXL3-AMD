"""Native Qwen XML / Hermes JSON calls -> Chat Completions function calls.

This module never executes tools. Responses with tools enabled are buffered until
generation ends: incomplete or invalid calls must not reach an executing client.
Validation covers argument types, required properties and enums, not all JSON
Schema keywords or constrained decoding. The harness still validates its inputs.
"""
from __future__ import annotations

import copy
import json
import math
import re
import uuid


class ToolCallError(ValueError):
    """Invalid generated protocol; this is not a GPU/engine failure."""


def _bad_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON property: {key}")
        result[key] = value
    return result


def loads_json(text):
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Non-finite JSON number')
        return result
    return json.loads(text, parse_constant=_bad_constant, parse_float=finite_float,
                      object_pairs_hook=_unique_object)


def protocol_for_template(template):
    """Detect only documented markers; do not guess support from a model name."""
    if not isinstance(template, str) or '<tool_call>' not in template or 'tools' not in template:
        return None
    if '<function=' in template and '<parameter=' in template:
        return 'qwen_xml'
    if 'arguments' in template and 'name' in template:
        return 'hermes_json'
    return None


def prepare_tools(messages, tools, tool_choice, parallel_tool_calls, template):
    """Copy messages; preserve the saved template and normalize policy guidance."""
    tools = tools or []
    choice = tool_choice if tool_choice is not None else ('auto' if tools else 'none')
    protocol = protocol_for_template(template)
    if tools and protocol is None:
        raise ValueError('Tool calling requires a supported Qwen XML or Hermes JSON chat template')
    definitions = {t['function']['name']: t['function'] for t in tools}
    selected = choice['function']['name'] if isinstance(choice, dict) else None
    hints = []
    if choice == 'none':
        exposed = []
        hints.append('Do not call any tools in this response. Answer with ordinary text.')
    else:
        exposed = [t for t in tools if not selected or t['function']['name'] == selected]
        if selected:
            hints.append(f'You must call the function {selected} in this response.')
        elif choice == 'required':
            hints.append('You must call at least one of the provided functions in this response.')
        if not parallel_tool_calls:
            hints.append('Call at most one function in this response. Wait for its result before another call.')
    rendered_messages = copy.deepcopy(messages)
    if hints:
        rendered_messages.insert(0, {'role': 'system', 'content': '\n'.join(hints)})
    return rendered_messages, exposed, dict(definitions=definitions, choice=choice,
                                            parallel=parallel_tool_calls, protocol=protocol)


def _resolved(schema, root):
    if not isinstance(schema, dict):
        return {}
    # Resolve local references only, with a cycle guard; never fetch a schema.
    seen = set()
    while '$ref' in schema:
        ref = schema['$ref']
        if not isinstance(ref, str) or not ref.startswith('#/') or ref in seen:
            raise ToolCallError('Unsupported or cyclic argument schema reference')
        seen.add(ref)
        value = root
        try:
            for key in ref[2:].split('/'):
                value = value[key.replace('~1', '/').replace('~0', '~')]
        except (KeyError, TypeError):
            raise ToolCallError('Unresolved argument schema reference') from None
        schema = value
        if not isinstance(schema, dict):
            raise ToolCallError('Argument schema reference is not an object')
    return schema


def _validate(value, schema, root, path='arguments', depth=0):
    if depth > 64:
        raise ToolCallError('Arguments exceed supported nesting depth')
    schema = _resolved(schema, root)
    for key in ('anyOf', 'oneOf'):
        if key in schema:
            matches = 0
            for option in schema[key]:
                try:
                    _validate(value, option, root, path, depth+1)
                    matches += 1
                except ToolCallError:
                    pass
            if matches == 0 or (key == 'oneOf' and matches != 1):
                raise ToolCallError(f'{path} does not match {key}')
    expected = schema.get('type')
    types = expected if isinstance(expected, list) else [expected] if expected else []
    valid = {'string': isinstance(value, str), 'object': isinstance(value, dict),
             'array': isinstance(value, list), 'boolean': type(value) is bool,
             'integer': type(value) is int, 'number': type(value) in (int, float),
             'null': value is None}
    if types and not any(valid.get(t, False) for t in types):
        raise ToolCallError(f'{path} has the wrong argument type')
    if 'enum' in schema and value not in schema['enum']:
        raise ToolCallError(f'{path} is outside its enum')
    if 'const' in schema and value != schema['const']:
        raise ToolCallError(f'{path} differs from its const')
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        missing = set(schema.get('required', []))-value.keys()
        if missing:
            raise ToolCallError(f'{path} is missing required arguments: {", ".join(sorted(missing))}')
        for key, child in value.items():
            if key not in properties and schema.get('additionalProperties') is False:
                raise ToolCallError(f'{path} has an unknown argument: {key}')
            child_schema = properties.get(key, schema.get('additionalProperties', {}))
            _validate(child, child_schema, root, f'{path}.{key}', depth+1)
    if isinstance(value, list) and isinstance(schema.get('items'), dict):
        for child in value:
            _validate(child, schema['items'], root, path+'[]', depth+1)


def _parameter(raw, schema, root):
    schema = _resolved(schema, root)
    options = schema.get('anyOf', schema.get('oneOf', []))
    types = schema.get('type', [])
    types = [types] if isinstance(types, str) else types
    string_allowed = 'string' in types or any(_resolved(o, root).get('type') == 'string' for o in options)
    # A string-valued "123" or "null" must remain a string. Preserve multiline
    # contents; remove only the template's single framing newline on each side.
    if raw.startswith('\r\n'):
        raw = raw[2:]
    elif raw.startswith('\n'):
        raw = raw[1:]
    if raw.endswith('\r\n'):
        raw = raw[:-2]
    elif raw.endswith('\n'):
        raw = raw[:-1]
    if string_allowed:
        return raw
    try:
        return loads_json(raw)
    except ValueError:
        if not types and not options:
            return raw
        raise ToolCallError('A non-string argument is not valid JSON') from None


def _parse_call(body, definitions, protocol):
    body = body.strip()
    if protocol == 'hermes_json':
        try:
            call = loads_json(body)
        except ValueError:
            raise ToolCallError('Malformed JSON tool call') from None
        if not isinstance(call, dict) or set(call) != {'name', 'arguments'}:
            raise ToolCallError('JSON call requires exactly name and arguments')
        name, arguments = call['name'], call['arguments']
    else:
        match = re.fullmatch(r'<function=([A-Za-z0-9_-]{1,64})>(.*?)</function>', body, re.S)
        if not match:
            raise ToolCallError('Malformed Qwen function block')
        name, params = match.groups()
        if name not in definitions:
            raise ToolCallError('Model called an unknown function')
        root = definitions[name].get('parameters', {})
        properties = _resolved(root, root).get('properties', {})
        arguments = {}
        while params.strip():
            match = re.match(r'\s*<parameter=([^<>\s=]+)>(.*?)</parameter>', params, re.S)
            if not match:
                raise ToolCallError('Malformed Qwen parameter block')
            key, raw = match.groups()
            if key in arguments:
                raise ToolCallError('Model repeated a function argument')
            arguments[key] = _parameter(raw, properties.get(key, {}), root)
            params = params[match.end():]
    if not isinstance(name, str) or name not in definitions:
        raise ToolCallError('Model called an unknown function')
    if not isinstance(arguments, dict):
        raise ToolCallError('Function arguments must be an object')
    schema = definitions[name].get('parameters', {})
    _validate(arguments, schema, schema)
    return {'id': 'call_'+uuid.uuid4().hex, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False, allow_nan=False)}}


def _parse_response(text, policy, finish_reason):
    """Validate an entire response, then return (visible content, complete calls)."""
    calls, content = [], []
    pos = 0
    fence = None
    # Only interpret markers outside Markdown fences. This also keeps examples
    # in ordinary answers from becoming executable calls.
    tokens = re.finditer(r'^\s*(`{3,}|~{3,})[^\n]*|<tool_call>|</tool_call>', text, re.M)
    for token in tokens:
        if token.start() < pos:
            continue
        marker = token.group()
        if token.group(1):
            current = token.group(1)
            if fence is None:
                fence = current
            elif current[0] == fence[0] and len(current) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        if marker == '</tool_call>':
            raise ToolCallError('Unexpected tool call closing marker')
        end = text.find('</tool_call>', token.end())
        if end < 0 or '<tool_call>' in text[token.end():end]:
            raise ToolCallError('Incomplete or nested tool call')
        between = text[pos:token.start()]
        if calls and between.strip():
            raise ToolCallError('Unexpected text between tool calls')
        content.append(between)
        calls.append(_parse_call(text[token.end():end], policy['definitions'], policy['protocol']))
        pos = end+len('</tool_call>')
    tail = text[pos:]
    if calls and tail.strip():
        raise ToolCallError('Unexpected text after tool calls')
    # Token limits may stop even inside the opening marker. Do not return it to
    # clients as a successful call, including after a preceding complete call.
    if calls and finish_reason == 'length':
        raise ToolCallError('Tool response reached the output limit; retry with more max_tokens')
    choice = policy['choice']
    if calls and choice == 'none':
        raise ToolCallError('Model called a function despite tool_choice=none')
    if not calls and (choice == 'required' or isinstance(choice, dict)):
        raise ToolCallError('Model did not emit the required function call')
    if not policy['parallel'] and len(calls) > 1:
        raise ToolCallError('Model emitted multiple calls despite parallel_tool_calls=false')
    if isinstance(choice, dict) and any(c['function']['name'] != choice['function']['name'] for c in calls):
        raise ToolCallError('Model emitted a different function than tool_choice')
    if not calls:
        # An unfinished opening marker must not be mistaken for successful text.
        if fence is None and re.search(r'<tool[_a-z]*$', text):
            raise ToolCallError('Incomplete tool call marker')
        return text, []
    return ''.join(content).rstrip(), calls


def parse_response(text, policy, finish_reason='stop'):
    """Normalize model/schema format failures without poisoning the GPU engine."""
    try:
        return _parse_response(text, policy, finish_reason)
    except ToolCallError:
        raise
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise ToolCallError('Malformed function arguments or unsupported schema structure') from None
