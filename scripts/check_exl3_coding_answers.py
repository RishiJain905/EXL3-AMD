"""Check manually reviewed pure-function answers in short-lived Python processes.

The AST/builtin restrictions are defense in depth, not an OS security sandbox.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

CHILD = '''
import copy,json,sys
data=json.load(sys.stdin)
safe={k:getattr(__import__('builtins'),k) for k in
      ('len','list','dict','set','tuple','range','enumerate','sorted','min','max','zip','reversed','abs','sum','bool','int','str')}
namespace={'__builtins__':safe}
exec(compile(data['code'],'<reviewed-answer>','exec'),namespace)
fn=namespace[data['name']]
failures=[]
for i,case in enumerate(data['cases']):
    args=copy.deepcopy(case['args']); before=copy.deepcopy(args)
    try:
        actual=fn(*args)
        if actual != case['expected'] or args != before:
            failures.append({'case':i,'actual':actual,'expected':case['expected'],'input_mutated':args!=before})
    except Exception as error:
        failures.append({'case':i,'error':type(error).__name__+': '+str(error)})
print(json.dumps({'cases':len(data['cases']),'failures':failures,'passed':len(data['cases'])-len(failures)}))
'''


def extract_function(text, name):
    text = text.strip()
    match = re.fullmatch(r'```(?:python|py)?\s*\n(.*?)\n```', text, re.DOTALL)
    code = match.group(1) if match else text
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != name:
        raise ValueError('Answer must contain only the requested function')
    fn = tree.body[0]
    if fn.decorator_list or fn.args.defaults or fn.args.kw_defaults:
        raise ValueError('Decorators and default expressions not admitted')
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise ValueError('Unsupported code outside a pure function')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('Private/dunder attributes not admitted')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('Dunder names not admitted')
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute-reviewed', action='store_true')
    args = p.parse_args()
    if not args.execute_reviewed:
        p.error('Read each answer first, then pass --execute-reviewed')
    if args.output.exists():
        p.error('Output exists')
    cases = json.loads(args.cases.read_text())
    results = {r['name']:r for r in json.loads(args.results.read_text())['results']}
    records = []
    for name, tests in cases.items():
        row = dict(name=name)
        try:
            result = results[name]
            if result['eos_reason'] != 'stop_token':
                raise ValueError('Answer did not finish normally')
            code = extract_function(result['output_text'], name)
            row['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
            with tempfile.TemporaryDirectory(prefix='quantlab-code-check-') as work:
                child = subprocess.run([sys.executable, '-I', '-S', '-c', CHILD],
                    input=json.dumps(dict(code=code, name=name, cases=tests)), text=True,
                    capture_output=True, cwd=work, timeout=3, check=True)
            row.update(json.loads(child.stdout))
            row['status'] = 'passed' if not row['failures'] else 'failed'
        except Exception as error:
            row.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        records.append(row)
    args.output.write_text(json.dumps(dict(
        status='completed', cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        answers_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(), records=records), indent=2))
    print(json.dumps(records))


if __name__ == '__main__':
    main()
