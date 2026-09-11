import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from quantlab.tool_calls import ToolCallError, parse_response, prepare_tools, protocol_for_template

SCHEMA = {'type': 'object', 'properties': {
    'path': {'type': 'string'}, 'count': {'type': 'integer'},
    'enabled': {'type': 'boolean'}, 'items': {'type': 'array', 'items': {'type': 'string'}},
    'options': {'type': 'object', 'properties': {'safe': {'type': 'boolean'}},
                'required': ['safe'], 'additionalProperties': False},
}, 'required': ['path'], 'additionalProperties': False}
TOOLS = [{'type': 'function', 'function': {'name': 'inspect_file', 'parameters': SCHEMA}},
         {'type': 'function', 'function': {'name': 'ping', 'parameters': {'type': 'object'}}}]
TEMPLATE = 'tools <tool_call> <function=name> <parameter=arg>'


def policy(choice='auto', parallel=True, tools=TOOLS, template=TEMPLATE):
    return prepare_tools([{'role': 'user', 'content': 'Inspect'}], tools, choice, parallel, template)[2]


def xml(params='', name='inspect_file'):
    return f'<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>'


class ToolProtocolTests(unittest.TestCase):
    def test_detection_is_template_based(self):
        self.assertEqual(protocol_for_template(TEMPLATE), 'qwen_xml')
        self.assertEqual(protocol_for_template('tools <tool_call> name arguments'), 'hermes_json')
        for text in ('Qwen', 'tools', '<tool_call>', None, {}):
            self.assertIsNone(protocol_for_template(text))
        with self.assertRaises(ValueError):
            policy(template='unsupported')

    def test_preparation_copies_and_filters_forced_function(self):
        messages = [{'role': 'user', 'content': 'hello'}]
        original = copy.deepcopy(messages)
        rendered, tools, p = prepare_tools(messages, TOOLS,
            {'type': 'function', 'function': {'name': 'ping'}}, False, TEMPLATE)
        self.assertEqual(messages, original)
        self.assertEqual(tools, [TOOLS[1]])
        self.assertIn('must call the function ping', rendered[0]['content'])
        self.assertFalse(p['parallel'])

    def test_xml_types_unicode_escaping_and_multiline(self):
        text = xml('<parameter=path>\nC:\\work\\caf\u00e9 "x".py\n</parameter>\n'
                   '<parameter=count>12</parameter><parameter=enabled>false</parameter>'
                   '<parameter=items>["a", "b"]</parameter>'
                   '<parameter=options>{"safe":true}</parameter>')
        content, calls = parse_response('I will inspect it.\n'+text, policy())
        self.assertEqual(content, 'I will inspect it.')
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {
            'path': 'C:\\work\\caf\u00e9 "x".py', 'count': 12, 'enabled': False,
            'items': ['a', 'b'], 'options': {'safe': True}})
        self.assertTrue(calls[0]['id'].startswith('call_'))

    def test_string_looking_like_json_stays_string(self):
        for value in ('123', 'null', 'true', '{"x":1}', ' a\n b '):
            _, calls = parse_response(xml(f'<parameter=path>{value}</parameter>'), policy())
            self.assertEqual(json.loads(calls[0]['function']['arguments'])['path'], value)

    def test_zero_argument_and_multiple_calls(self):
        content, calls = parse_response(xml(name='ping')+'\n'+xml(name='ping'), policy())
        self.assertEqual(content, '')
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]['id'], calls[1]['id'])
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {})

    def test_json_protocol(self):
        p = policy(template='tools <tool_call> name arguments')
        _, calls = parse_response('<tool_call>{"name":"inspect_file","arguments":{"path":"a"}}</tool_call>', p)
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {'path': 'a'})

    def test_code_fences_are_plain_content(self):
        for fence in ('```', '~~~~'):
            text = f'Example:\n{fence}xml\n{xml(name="ping")}\n{fence}\nDone.'
            self.assertEqual(parse_response(text, policy()), (text, []))

    def test_call_after_example(self):
        example = '```xml\n'+xml(name='ping')+'\n```\n'
        text, calls = parse_response(example+xml(name='ping'), policy())
        self.assertEqual(text, example.rstrip())
        self.assertEqual(len(calls), 1)

    def test_framing_removes_only_one_newline(self):
        for value in ('\nline\n', '\r\nline\r\n', '\nline\r\n'):
            text = xml('<parameter=path>\r\n'+value+'\r\n</parameter>')
            _, calls = parse_response(text, policy())
            self.assertEqual(json.loads(calls[0]['function']['arguments'])['path'], value)

    def test_invalid_calls_and_arguments_fail(self):
        cases = [xml(), xml(name='unknown'), xml('<parameter=path>x'),
            xml('<parameter=path>x</parameter><parameter=path>y</parameter>'),
            xml('<parameter=path>x</parameter><parameter=unknown>x</parameter>'),
            xml('<parameter=path>x</parameter><parameter=count>true</parameter>'),
            xml('<parameter=path>x</parameter><parameter=count>NaN</parameter>'),
            xml('<parameter=path>x</parameter><parameter=count>1e999</parameter>'),
            xml('<parameter=path>x</parameter><parameter=options>{"safe":1}</parameter>'),
            xml('<parameter=path>x</parameter><parameter=items>[1]</parameter>'),
            '<tool_call>{}</tool_call>', '</tool_call>', '<tool_call>'+xml(name='ping'),
            '<tool_call><function=ping>', '<tool_c',
            xml(name='ping')+' unstructured suffix', xml(name='ping')+' text '+xml(name='ping')]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ToolCallError):
                parse_response(text, policy())

    def test_limit_never_exposes_preceding_complete_call(self):
        with self.assertRaises(ToolCallError):
            parse_response(xml(name='ping'), policy(), 'length')

    def test_choice_constraints(self):
        for text, p in [('hello', policy('required')),
                        (xml(name='ping'), policy('none')),
                        (xml(name='ping')+xml(name='ping'), policy(parallel=False)),
                        (xml(name='ping'), policy({'type': 'function', 'function': {'name': 'inspect_file'}}))]:
            with self.assertRaises(ToolCallError):
                parse_response(text, p)
        self.assertEqual(parse_response('Hello', policy('none')), ('Hello', []))

    def test_json_duplicate_keys_nonfinite_and_nonobject_rejected(self):
        p = policy(template='tools <tool_call> name arguments')
        for args in ('[]', '{"path":"x","path":"y"}', '{"path":NaN}', '{"path":1e999}'):
            with self.assertRaises(ToolCallError):
                parse_response('<tool_call>{"name":"inspect_file","arguments":'+args+'}</tool_call>', p)

    def test_schema_ref_and_union(self):
        schema = {'type': 'object', '$defs': {'count': {'type': 'integer'}},
                  'properties': {'n': {'$ref': '#/$defs/count'},
                                 'label': {'anyOf': [{'type': 'null'}, {'type': 'string'}]}}}
        tools = [{'type': 'function', 'function': {'name': 'ping', 'parameters': schema}}]
        _, calls = parse_response(xml('<parameter=n>7</parameter><parameter=label>null</parameter>', 'ping'), policy(tools=tools))
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {'n': 7, 'label': 'null'})


if __name__ == '__main__':
    unittest.main()
