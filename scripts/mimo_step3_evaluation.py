"""MiMo step-3 prospective scorer + final-suite generator.

Stdlib only. Implements the exact extraction priority from the step-3
preregistration (thinking pre-strip, then final-block / direct-JSON /
terminal-fenced-block / last-nonempty-line). Gold-blind: the expected
answer is never used during extraction.

Reuses strict-JSON, typed-equality, result-pair and thinking-strip
helpers from scripts/mimo_validation_suite.py and
scripts/mimo_thinking_supplement.py through normal scripts imports.
Model output is parsed as data only; never executed.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path


def _load_script_module(name, filename):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_script_module("mimo_validation_suite_step3_base", "mimo_validation_suite.py")
_think = _load_script_module("mimo_thinking_supplement_step3_base", "mimo_thinking_supplement.py")

# Reused helpers (no local copies of strict/equality/pair/strip semantics).
_parse_json = _base._parse_json
json_equal = _base.json_equal
_result_pairs = _base._result_pairs
wilson_ci = _base.wilson_ci
strip_thinking_block = _think.strip_thinking_block

SEED = 20260923
SCHEMA = "mimo-step3-final-v1"
SYSTEM = ("Solve the task accurately. You may explain your work. Regardless of any earlier "
          "request for JSON only, finish with exactly one <final>JSON_VALUE</final> block "
          "containing the requested JSON value. Put no explanation inside that block and no "
          "text after it.")
QUALITY_MAX_TOKENS = 2048

FINAL_OPEN = "<final>"
FINAL_CLOSE = "</final>"
FENCE = "```"
_ASCII_WS = " \t\n\r\f\v"

_PROMPT_CHAR_LIMIT = 3000
_EXPECTED_CHAR_LIMIT = 600

CLAIM_LIMITS = ("Authored step-3 final task families (8 new families x 4), not a public "
                "benchmark, not an equivalence test, not general BF16 fidelity. "
                "No model-generated code execution and no real tool-use success claim.")


# --------------------------------------------------------------------------
# step-3 extraction (gold-blind)
# --------------------------------------------------------------------------

def _strict_parse(text):
    """Strict JSON parse of already-trimmed text; returns (value|None, error|None)."""
    try:
        return _parse_json(text), None
    except ValueError as exc:
        return None, str(exc)


def extract_after_thinking(answer):
    """Apply prereg rules 1-4 to post-thinking-strip text.

    Returns (value|None, route|None, format_ok: bool, error|None).
    route in {"final_block","direct_json","fenced_json","last_line"} on success.
    format_ok is True only for a valid terminal <final> block with strict-JSON
    content. Malformed final markers never fall through. Fence presence is
    exclusive: any fence marker routes to the fenced rule only.
    """
    if not isinstance(answer, str):
        return None, None, False, "response_not_string"
    has_open = FINAL_OPEN in answer
    has_close = FINAL_CLOSE in answer
    if has_open or has_close:
        n_open = answer.count(FINAL_OPEN)
        n_close = answer.count(FINAL_CLOSE)
        if n_open != 1 or n_close != 1:
            if n_open == 0 or n_close == 0:
                return None, None, False, "final_block_unbalanced"
            return None, None, False, "final_block_multiple"
        start = answer.find(FINAL_OPEN)
        end = answer.find(FINAL_CLOSE)
        if end < start:
            return None, None, False, "final_block_order"
        content = answer[start + len(FINAL_OPEN):end]
        after = answer[end + len(FINAL_CLOSE):]
        if FINAL_OPEN in content or FINAL_CLOSE in content:
            return None, None, False, "final_block_nested"
        if after.strip(_ASCII_WS):
            return None, None, False, "final_block_trailing_text"
        value, parse_error = _strict_parse(content.strip())
        if parse_error is not None:
            return None, None, False, f"final_block_invalid_json: {parse_error}"
        return value, "final_block", True, None
    # Rule 2: whole answer as strict JSON.
    value, _ = _strict_parse(answer.strip())
    if _ is None:
        return value, "direct_json", False, None
    # Rule 3: fenced block (exclusive once any fence marker exists).
    if FENCE in answer:
        count = answer.count(FENCE)
        if count != 2:
            if count % 2 == 1:
                return None, None, False, "fence_unclosed"
            return None, None, False, "multiple_fenced_blocks"
        start = answer.find(FENCE)
        end = answer.find(FENCE, start + len(FENCE))
        after = answer[end + len(FENCE):]
        if after.strip(_ASCII_WS):
            return None, None, False, "fence_trailing_text"
        inner = answer[start + len(FENCE):end]
        stripped = inner.strip()
        lower = stripped[:4].lower()
        if lower == "json" and (len(stripped) == 4 or stripped[4:5] in (" ", "\t", "\r", "\n")):
            stripped = stripped[4:].strip()
        elif stripped and not stripped[0] in "{[\"0123456789-tfn":
            # Language tag present (or non-JSON start with a tag line).
            first_line, _, rest = inner.strip().partition("\n")
            tag = first_line.strip().lower()
            if tag not in ("", "json"):
                return None, None, False, "fence_language_rejected"
            stripped = rest.strip()
            if tag == "json":
                pass
        # Bare fence with JSON content falls through to strict parse.
        # Reject empty fences explicitly.
        if not stripped:
            return None, None, False, "fence_empty"
        value, parse_error = _strict_parse(stripped)
        if parse_error is not None:
            return None, None, False, f"fence_invalid_json: {parse_error}"
        return value, "fenced_json", False, None
    # Rule 4: last nonempty line only.
    last = None
    for line in answer.splitlines():
        if line.strip(_ASCII_WS):
            last = line
    if last is None:
        return None, None, False, "last_line_missing"
    value, parse_error = _strict_parse(last.strip())
    if parse_error is not None:
        return None, None, False, f"last_line_invalid_json: {parse_error}"
    return value, "last_line", False, None


def extract_step3_response(text):
    """Full pipeline: thinking pre-strip then rules 1-4.

    Returns (value|None, route|None, format_ok, reasoning_chars|None,
             strip_error|None, extraction_error|None). Gold-blind.
    """
    answer, reasoning_chars, strip_error = strip_thinking_block(text)
    if strip_error is not None:
        return None, None, False, None, strip_error, None
    value, route, format_ok, extraction_error = extract_after_thinking(answer)
    return value, route, format_ok, reasoning_chars, None, extraction_error


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def score_final_suite(suite, results_obj):
    """Score a final suite against supported result shapes.

    Per-case fields: correct (=passed), format_ok, route, extraction_error,
    strip_error, reasoning_chars, actual, expected, error (high-level).
    Totals and by_category/by_family include correctness, format, extracted
    counts, conditional correctness given extraction and
    Wilson intervals for correctness.
    """
    tasks = suite.get("tasks", [])
    pairs, format_error = _result_pairs(results_obj)
    counts = {}
    for task_id, _ in pairs:
        counts[task_id] = counts.get(task_id, 0) + 1
    duplicates = sorted(t for t, c in counts.items() if c > 1)
    first = {}
    for task_id, text in pairs:
        first.setdefault(task_id, text)
    by_id = {t.get("id"): t for t in tasks}
    unknown = sorted(t for t in first if t not in by_id)
    cases = []
    for task in tasks:
        task_id = task.get("id")
        skeleton = {"id": task_id, "category": task.get("category"),
                    "family": task.get("family"), "expected": task.get("expected")}
        if format_error is not None:
            cases.append({**skeleton, "correct": False, "passed": False, "format_ok": False,
                          "route": None, "extraction_error": None, "strip_error": None,
                          "reasoning_chars": None,
                          "error": format_error, "actual": None})
            continue
        if task_id in duplicates:
            cases.append({**skeleton, "correct": False, "passed": False, "format_ok": False,
                          "route": None, "extraction_error": None, "strip_error": None,
                          "reasoning_chars": None,
                          "error": "duplicate_result", "actual": None})
            continue
        if task_id not in first:
            cases.append({**skeleton, "correct": False, "passed": False, "format_ok": False,
                          "route": None, "extraction_error": None, "strip_error": None,
                          "reasoning_chars": None,
                          "error": "missing_result", "actual": None})
            continue
        raw = first[task_id]
        if not isinstance(raw, str):
            cases.append({**skeleton, "correct": False, "passed": False, "format_ok": False,
                          "route": None, "extraction_error": "response_not_string",
                          "strip_error": None, "reasoning_chars": None,
                          "error": "parse_error", "actual": None})
            continue
        value, route, format_ok, reasoning_chars, strip_error, extraction_error = (
            extract_step3_response(raw))
        if strip_error is not None:
            cases.append({**skeleton, "correct": False, "passed": False, "format_ok": False,
                          "route": None, "extraction_error": None, "strip_error": strip_error,
                          "reasoning_chars": None,
                          "error": "thinking_parse_error", "actual": None})
            continue
        if extraction_error is not None:
            cases.append({**skeleton, "correct": False, "passed": False, "format_ok": False,
                          "route": None, "extraction_error": extraction_error,
                          "strip_error": None, "reasoning_chars": reasoning_chars,
                          "error": "parse_error", "actual": None})
            continue
        correct = json_equal(task.get("expected"), value)
        cases.append({**skeleton, "correct": correct, "passed": correct, "format_ok": format_ok,
                      "route": route, "extraction_error": None, "strip_error": None,
                      "reasoning_chars": reasoning_chars,
                      "error": None if correct else "mismatch", "actual": value})

    def summarize(items):
        correct = sum(1 for c in items if c["correct"])
        format_ok = sum(1 for c in items if c["format_ok"])
        extracted = sum(1 for c in items if c["route"] is not None)
        total = len(items)
        low, high = wilson_ci(correct, total)
        return {"correct": correct, "passed": correct, "total": total,
                "rate": (correct / total) if total else None,
                "wilson_low": low, "wilson_high": high,
                "format_ok": format_ok,
                "format_rate": (format_ok / total) if total else None,
                "extracted": extracted,
                "conditional_rate": (correct / extracted) if extracted else None}

    by_category = {}
    by_family = {}
    for case in cases:
        by_category.setdefault(case["category"], []).append(case)
        by_family.setdefault(case["family"], []).append(case)
    digest = hashlib.sha256(
        json.dumps(suite, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    totals = summarize(cases)
    return {
        "schema": SCHEMA,
        "suite_sha256": digest,
        "totals": totals,
        "by_category": {k: summarize(v) for k, v in sorted(by_category.items())},
        "by_family": {k: summarize(v) for k, v in sorted(by_family.items())},
        "missing_ids": sorted(t.get("id") for t in tasks if t.get("id") not in first),
        "duplicate_ids": duplicates,
        "unknown_ids": unknown,
        "format_error": format_error,
        "cases": cases,
    }


score_suite = score_final_suite


# --------------------------------------------------------------------------
# final-suite generators (8 new families x 4)
# --------------------------------------------------------------------------

def _gen_code_stack_machine(rng, k):
    patterns = [
        ["push", "push", "add", "push", "push", "multiply"],
        ["push", "push", "push", "add", "multiply", "push"],
        ["push", "push", "pop", "push", "add", "push", "push", "multiply", "add"],
        ["push", "push", "multiply", "push", "add", "pop", "push", "push", "add"],
    ]
    ops_template = patterns[k]
    values = [rng.randint(1, 9) for _ in ops_template if _ == "push"]
    ops = []
    it = iter(values)
    for name in ops_template:
        if name == "push":
            ops.append({"op": "push", "value": next(it)})
        else:
            ops.append({"op": name})
    stack = []
    for op in ops:
        if op["op"] == "push":
            stack.append(op["value"])
        elif op["op"] == "pop":
            stack.pop()
        elif op["op"] == "add":
            a = stack.pop()
            b = stack.pop()
            stack.append(b + a)
        elif op["op"] == "multiply":
            a = stack.pop()
            b = stack.pop()
            stack.append(b * a)
    lines = []
    for i, op in enumerate(ops, 1):
        if op["op"] == "push":
            lines.append(f"{i}. push {op['value']}")
        else:
            lines.append(f"{i}. {op['op']}")
    numbered = "\n".join(lines)
    prompt = ("Simulate an integer stack machine. The stack starts empty; represent it "
              "bottom-to-top.\n"
              "Semantics: `push N` appends N to the top. `pop` removes the top element. "
              "`add` pops a=top then b=second-from-top and pushes b+a. `multiply` pops "
              "a=top then b=second-from-top and pushes b*a. All operations are valid.\n"
              f"Operations in order:\n{numbered}\n"
              "What is the final whole stack bottom-to-top? "
              "Return the requested JSON value (a JSON array of integers).")
    reference = {"initial_stack": [], "operations": ops,
                 "operand_order": "pop a=top, b=second; push b+a / b*a"}
    return prompt, list(stack), reference


def _nested_spec(rng, k):
    if k == 0:
        t = rng.randint(8, 12)
        s = rng.randint(1, 3)
        src = (f"def transform(x):\n    if x % 2 == 0:\n        if x >= {t}:\n"
               f"            return x // 2\n        else:\n            return x + 1\n"
               f"    else:\n        if x < {s}:\n            return x * 3\n"
               f"        else:\n            return x - 2")

        def fn(x):
            if x % 2 == 0:
                return x // 2 if x >= t else x + 1
            return x * 3 if x < s else x - 2

        params = {"threshold": t, "small": s}
    elif k == 1:
        t = rng.randint(6, 10)
        src = (f"def transform(x):\n    if x % 3 == 0:\n        if x > {t}:\n"
               f"            return x - 5\n        else:\n            return x * 2\n"
               f"    else:\n        if x % 2 == 0:\n            return x + 10\n"
               f"        else:\n            return x + 1")

        def fn(x):
            if x % 3 == 0:
                return x - 5 if x > t else x * 2
            return x + 10 if x % 2 == 0 else x + 1

        params = {"threshold": t}
    elif k == 2:
        t = rng.randint(5, 9)
        s = rng.randint(1, 4)
        src = (f"def transform(x):\n    if x < {t}:\n        if x % 2 == 0:\n"
               f"            return x * x\n        else:\n            return x + {s}\n"
               f"    else:\n        if x % 3 == 0:\n            return x // 3\n"
               f"        else:\n            return x - {s}")

        def fn(x):
            if x < t:
                return x * x if x % 2 == 0 else x + s
            return x // 3 if x % 3 == 0 else x - s

        params = {"threshold": t, "adjust": s}
    else:
        t = rng.randint(7, 11)
        s = rng.randint(2, 5)
        src = (f"def transform(x):\n    if x >= {t}:\n        if x % 2 == 0:\n"
               f"            return x - {s}\n        else:\n            return x // 2\n"
               f"    else:\n        if x % 5 == 0:\n            return x + {s}\n"
               f"        else:\n            return x * 2")

        def fn(x):
            if x >= t:
                return x - s if x % 2 == 0 else x // 2
            return x + s if x % 5 == 0 else x * 2

        params = {"threshold": t, "adjust": s}
    return src, fn, params


def _gen_code_nested_transform(rng, k):
    src, fn, params = _nested_spec(rng, k)
    values = [rng.randint(0, 15) for _ in range(6)]
    expected = [fn(x) for x in values]
    prompt = ("Apply this exact Python function to each array element in order.\n"
              f"{src}\nvalues = {values}\n"
              "result = [transform(x) for x in values]\n"
              "What is the final value of `result`? "
              "Return the requested JSON value (a JSON array of integers).")
    reference = {"values": values, "variant": k, "params": params, "function_source": src}
    return prompt, expected, reference


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _gen_reasoning_clock_calendar(rng, k):
    start_day = rng.choice(_WEEKDAYS)
    start_hour = rng.randint(0, 23)
    start_minute = rng.randint(0, 59)
    span = [(30, 300), (700, 1500), (1500, 5000), (5000, 10000)][k]
    duration = rng.randint(*span)
    total = start_hour * 60 + start_minute + duration
    days_added = total // 1440
    rem = total % 1440
    expected = {"weekday": _WEEKDAYS[(_WEEKDAYS.index(start_day) + days_added) % 7],
                "hour": rem // 60, "minute": rem % 60}
    prompt = ("A clock uses a 24-hour day and this exact weekday cycle in order: "
              "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, then Monday again.\n"
              f"Start: {start_day} at {start_hour:02d}:{start_minute:02d}. "
              f"Add exactly {duration} minutes.\n"
              "What is the resulting weekday and time? Return the requested JSON value as "
              '{"weekday": "<name>", "hour": <0-23>, "minute": <0-59>}.')
    reference = {"start_weekday": start_day, "start_hour": start_hour,
                 "start_minute": start_minute, "duration_minutes": duration,
                 "weekday_cycle": list(_WEEKDAYS)}
    return prompt, expected, reference


def _gen_reasoning_sets(rng, k):
    if k in (0, 1):
        only_a = rng.randint(5, 30)
        only_b = rng.randint(5, 30)
        both = rng.randint(5, 25)
        neither = rng.randint(5, 30)
        total = only_a + only_b + both + neither
        size_a = only_a + both
        size_b = only_b + both
        union = only_a + only_b + both
        expected = {"union": union, "neither": neither}
        if k == 0:
            prompt = (f"An invented harbor census covers {total} residents. "
                      f"{size_a} collect stamps (Group A). {size_b} collect coins (Group B). "
                      f"{both} collect both. Every resident is counted once.\n"
                      "How many collect at least one (union), and how many collect neither? "
                      'Return the requested JSON value as {"union": <count>, "neither": <count>}.')
        else:
            prompt = (f"An invented garden club has {total} members. "
                      f"{size_a} grow roses (Group A). {size_b} grow tulips (Group B). "
                      f"{both} grow both. Every member is counted once.\n"
                      "How many grow at least one (union), and how many grow neither? "
                      'Return the requested JSON value as {"union": <count>, "neither": <count>}.')
        reference = {"population": total, "group_a": size_a, "group_b": size_b,
                     "intersection_ab": both,
                     "regions": {"only_a": only_a, "only_b": only_b, "both": both,
                                 "neither": neither}}
        return prompt, expected, reference
    only_a = rng.randint(2, 12)
    only_b = rng.randint(2, 12)
    only_c = rng.randint(2, 12)
    ab = rng.randint(1, 10)
    ac = rng.randint(1, 10)
    bc = rng.randint(1, 10)
    abc = rng.randint(1, 8)
    neither = rng.randint(2, 15)
    total = only_a + only_b + only_c + ab + ac + bc + abc + neither
    size_a = only_a + ab + ac + abc
    size_b = only_b + ab + bc + abc
    size_c = only_c + ac + bc + abc
    inter_ab = ab + abc
    inter_ac = ac + abc
    inter_bc = bc + abc
    union = total - neither
    expected = {"union": union, "neither": neither}
    if k == 2:
        prompt = (f"An invented school survey covers {total} pupils. "
                  f"{size_a} like apples (A). {size_b} like bread (B). {size_c} like cheese (C). "
                  f"{inter_ab} like both A and B. {inter_ac} like both A and C. "
                  f"{inter_bc} like both B and C. {abc} like all three. "
                  "Pairwise counts INCLUDE pupils in all three groups. "
                  "Every pupil is counted once.\n"
                  "How many like at least one (union), and how many like none (neither)? "
                  'Return the requested JSON value as {"union": <count>, "neither": <count>}.')
    else:
        prompt = (f"An invented workshop roster covers {total} workers. "
                  f"{size_a} can weld (A). {size_b} can paint (B). {size_c} can wire (C). "
                  f"{inter_ab} can both weld and paint. {inter_ac} can both weld and wire. "
                  f"{inter_bc} can both paint and wire. {abc} can do all three. "
                  "Pairwise counts INCLUDE workers in all three groups. "
                  "Every worker is counted once.\n"
                  "How many can do at least one (union), and how many can do none (neither)? "
                  'Return the requested JSON value as {"union": <count>, "neither": <count>}.')
    reference = {"population": total, "group_a": size_a, "group_b": size_b, "group_c": size_c,
                 "intersection_ab": inter_ab, "intersection_ac": inter_ac,
                 "intersection_bc": inter_bc, "intersection_abc": abc,
                 "regions": {"only_a": only_a, "only_b": only_b, "only_c": only_c,
                             "ab_only": ab, "ac_only": ac, "bc_only": bc,
                             "abc": abc, "neither": neither}}
    return prompt, expected, reference


def _gen_tools_bind_arguments(rng, k):
    if k == 0:
        item = rng.choice(["bolts", "screws", "nails"])
        dozen_n = rng.randint(1, 3)
        urg = rng.choice(["urgently", "with rush handling"])
        request = f"Restock {dozen_n} dozen {item} {urg}."
        provided = {}
        expected_args = {"item": item, "quantity": dozen_n * 12, "express": True}
        tool_spec = {"name": "pantry_restock", "schema": {
            "item": "string, required",
            "quantity": "integer pieces, default 1",
            "express": "boolean, default false"}}
        prompt = ("Bind arguments for exactly one fictional tool call. Do not execute anything.\n"
                  f"Tool: {json.dumps(tool_spec)}\n"
                  "Rules: convert '<N> dozen' to N*12 pieces; 'pieces' stay as-is. "
                  "Words 'urgent'/'urgently'/'rush'/'express' set express=true, else default. "
                  "Precedence: `provided_arguments` override request text; defaults fill missing.\n"
                  f"Request: {request}\nprovided_arguments = {json.dumps(provided)}\n"
                  'Return the requested JSON value as {"name": "<tool>", "arguments": {...}} '
                  "with normalized integer quantity and boolean express.")
        reference = {"tool": "pantry_restock", "request_text": request,
                     "provided_arguments": provided,
                     "defaults": {"quantity": 1, "express": False},
                     "conversions": {"dozen": 12, "express_words": ["urgent", "urgently", "rush", "express"]},
                     "expected_arguments": expected_args}
        return prompt, {"name": "pantry_restock", "arguments": expected_args}, reference
    if k == 1:
        label = rng.choice(["bake", "soak", "cooldown"])
        minutes = rng.randint(2, 10)
        override = rng.randint(30, 600)
        request = f"Start a {label} timer for {minutes} minutes."
        provided = {"seconds": override}
        expected_args = {"label": label, "seconds": override, "repeat": False}
        tool_spec = {"name": "timer_start", "schema": {
            "label": "string, required",
            "seconds": "integer, required",
            "repeat": "boolean, default false"}}
        prompt = ("Bind arguments for exactly one fictional tool call. Do not execute anything.\n"
                  f"Tool: {json.dumps(tool_spec)}\n"
                  "Rules: convert minutes to seconds (1 minute = 60 seconds); plain integers are "
                  "seconds. Precedence: `provided_arguments` override request text; defaults fill "
                  "missing. The word 'repeat' would set repeat=true; otherwise default.\n"
                  f"Request: {request}\nprovided_arguments = {json.dumps(provided)}\n"
                  'Return the requested JSON value as {"name": "<tool>", "arguments": {...}} '
                  "with integer seconds and boolean repeat.")
        reference = {"tool": "timer_start", "request_text": request,
                     "provided_arguments": provided, "defaults": {"repeat": False},
                     "conversions": {"minutes_to_seconds": 60},
                     "expected_arguments": expected_args}
        return prompt, {"name": "timer_start", "arguments": expected_args}, reference
    if k == 2:
        room = rng.choice(["study", "patio", "loft"])
        pct = rng.randint(10, 90)
        request = f"Set the {room} lamp to {pct} percent."
        provided = {}
        expected_args = {"room": room, "brightness": pct, "color": "white"}
        tool_spec = {"name": "lamp_set", "schema": {
            "room": "string, required",
            "brightness": "integer 0-100, required",
            "color": 'string, default "white"'}}
        prompt = ("Bind arguments for exactly one fictional tool call. Do not execute anything.\n"
                  f"Tool: {json.dumps(tool_spec)}\n"
                  "Rules: '<N> percent' converts to integer N. If no color is stated in the "
                  "request or provided_arguments, use the default. Precedence: "
                  "`provided_arguments` override request text; defaults fill missing.\n"
                  f"Request: {request}\nprovided_arguments = {json.dumps(provided)}\n"
                  'Return the requested JSON value as {"name": "<tool>", "arguments": {...}}.')
        reference = {"tool": "lamp_set", "request_text": request,
                     "provided_arguments": provided, "defaults": {"color": "white"},
                     "conversions": {"percent_to_int": 1},
                     "expected_arguments": expected_args}
        return prompt, {"name": "lamp_set", "arguments": expected_args}, reference
    title = rng.choice(["Harbor inventory", "Night checklist", "Shed repairs"])
    provided = {"priority": "medium"}
    request = f"Log note titled '{title}' marked important with #alpha #beta."
    expected_args = {"title": title, "priority": "medium", "tags": ["alpha", "beta"]}
    tool_spec = {"name": "log_note", "schema": {
        "title": "string, required",
        "priority": "one of low/medium/high, default low",
        "tags": "array of strings, default []"}}
    prompt = ("Bind arguments for exactly one fictional tool call. Do not execute anything.\n"
              f"Tool: {json.dumps(tool_spec)}\n"
              "Rules: the word 'important' maps to high and 'follow-up' maps to medium; "
              "words '#tag' collect in order as tags. Precedence: `provided_arguments` "
              "override request text; defaults fill missing.\n"
              f"Request: {request}\nprovided_arguments = {json.dumps(provided)}\n"
              'Return the requested JSON value as {"name": "<tool>", "arguments": {...}}.')
    reference = {"tool": "log_note", "request_text": request,
                 "provided_arguments": provided,
                 "defaults": {"priority": "low", "tags": []},
                 "conversions": {"important": "high", "follow-up": "medium", "hashtag": "tag"},
                 "expected_arguments": expected_args}
    return prompt, {"name": "log_note", "arguments": expected_args}, reference


def _topo_lexicographic(tools, deps):
    remaining = {t: set(deps.get(t, ())) for t in tools}
    order = []
    while remaining:
        ready = sorted(t for t, ds in remaining.items() if ds <= set(order))
        if not ready:
            raise ValueError("cyclic dependencies")
        order.append(ready[0])
        del remaining[ready[0]]
    return order


def _gen_tools_dependency_order(rng, k):
    graphs = [
        (["archive_export", "cache_warm", "index_build", "notify_done"],
         {"archive_export": [], "cache_warm": [], "index_build": ["archive_export", "cache_warm"],
          "notify_done": ["index_build"]}),
        (["alpha_sync", "beta_sync", "merge_sync", "report_sync"],
         {"alpha_sync": [], "beta_sync": [], "merge_sync": ["alpha_sync", "beta_sync"],
          "report_sync": ["merge_sync"]}),
        (["compile_shard", "link_shard", "test_shard", "pack_bundle", "ship_bundle"],
         {"compile_shard": [], "link_shard": ["compile_shard"], "test_shard": ["compile_shard"],
          "pack_bundle": ["link_shard", "test_shard"], "ship_bundle": ["pack_bundle"]}),
        (["draft_memo", "edit_memo", "approve_memo", "file_memo", "audit_log"],
         {"draft_memo": [], "edit_memo": ["draft_memo"], "approve_memo": ["edit_memo"],
          "file_memo": ["approve_memo"], "audit_log": ["draft_memo"]}),
    ]
    tools, deps = graphs[k]
    order = _topo_lexicographic(tools, deps)
    presented = list(tools)
    rng.shuffle(presented)
    lines = []
    for tool in presented:
        ds = deps[tool]
        if not ds:
            lines.append(f"- {tool}: no dependencies")
        else:
            ds_shown = list(ds)
            rng.shuffle(ds_shown)
            lines.append(f"- {tool}: depends on {', '.join(ds_shown)}")
    body = "\n".join(lines)
    prompt = ("Order fictional tool calls. Do not execute anything. The graph is acyclic.\n"
              f"Tools and declared dependencies:\n{body}\n"
              "Rule: a tool may run only after all its dependencies. Tie-break: repeatedly pick "
              "the lexicographically smallest ready tool (standard string comparison).\n"
              "Return the requested JSON value (a JSON array of tool names in legal order).")
    reference = {"tools": list(tools), "dependencies": {t: list(deps[t]) for t in tools},
                 "tie_break": "lexicographic_smallest_ready_first"}
    return prompt, list(order), reference


def _shortest_route(nodes, edges, start, goal):
    adj = {n: [] for n in nodes}
    for e in edges:
        adj[e["from"]].append((e["to"], e["weight"]))
    best = None
    best_cost = None
    stack = [(start, [start], 0)]
    while stack:
        node, path, cost = stack.pop()
        if best_cost is not None and cost > best_cost:
            continue
        if node == goal:
            if best_cost is None or cost < best_cost or (
                    cost == best_cost and tuple(path) < tuple(best)):
                best, best_cost = list(path), cost
            continue
        for nxt, w in adj.get(node, []):
            if nxt in path:
                continue
            stack.append((nxt, path + [nxt], cost + w))
    if best is None:
        raise ValueError("no route")
    return best, best_cost


def _gen_general_graph_route(rng, k):
    specs = [
        (["A", "B", "C", "D"],
         [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D")], "A", "D"),
        (["S", "T", "U", "V", "W"],
         [("S", "T"), ("S", "U"), ("T", "V"), ("U", "V"), ("V", "W"), ("T", "W"), ("U", "W")],
         "S", "W"),
        (["N1", "N2", "N3", "N4", "N5"],
         [("N1", "N2"), ("N1", "N3"), ("N2", "N4"), ("N3", "N4"), ("N4", "N5"),
          ("N2", "N5"), ("N3", "N5")], "N1", "N5"),
        (["P", "Q", "R", "S", "T"],
         [("P", "Q"), ("P", "R"), ("Q", "S"), ("R", "S"), ("S", "T"), ("Q", "T"), ("R", "T")],
         "P", "T"),
    ]
    nodes, pairs, start, goal = specs[k]
    edges = [{"from": a, "to": b, "weight": rng.randint(1, 9)} for a, b in pairs]
    route, cost = _shortest_route(nodes, edges, start, goal)
    lines = "\n".join(f"- {e['from']} -> {e['to']} weight {e['weight']}" for e in edges)
    prompt = ("Find the shortest directed route. All weights are positive integers.\n"
              f"Nodes: {', '.join(nodes)}. Start: {start}. Goal: {goal}.\n"
              f"Edges:\n{lines}\n"
              "Tie-break: if several routes share the minimal total cost, choose the "
              "lexicographically smallest route (compare the node-name sequences).\n"
              'Return the requested JSON value as {"route": [<nodes>], "cost": <total>}.')
    reference = {"nodes": list(nodes), "edges": edges, "start": start, "goal": goal,
                 "tie_break": "lexicographic_route"}
    return prompt, {"route": route, "cost": cost}, reference


def _gen_general_fact_updates(rng, k):
    if k == 0:
        alda = rng.randint(5, 20)
        bram = rng.randint(5, 20)
        cur = {"Alda": alda, "Bram": bram}
        updates = []
        give = rng.randint(1, min(5, cur["Alda"]))
        updates.append({"kind": "transfer", "from": "Alda", "to": "Bram", "amount": give,
                        "text": f"Alda gives {give} apples to Bram."})
        cur["Alda"] -= give
        cur["Bram"] += give
        set_b = rng.randint(1, 20)
        updates.append({"kind": "set", "entity": "Bram", "value": set_b,
                        "text": f"Bram's apple count is set to {set_b}."})
        cur["Bram"] = set_b
        give2 = rng.randint(1, min(5, cur["Bram"]))
        updates.append({"kind": "transfer", "from": "Bram", "to": "Alda", "amount": give2,
                        "text": f"Bram gives {give2} apples to Alda."})
        cur["Bram"] -= give2
        cur["Alda"] += give2
        expected = dict(cur)
        lines = "\n".join(f"{i + 1}. {u['text']}" for i, u in enumerate(updates))
        prompt = ("Track invented apple counts. Apply the updates in the listed order; transfers "
                  "subtract from the source and add to the destination; assignments overwrite.\n"
                  f"Initial: Alda has {alda} apples; Bram has {bram} apples.\n"
                  f"Updates:\n{lines}\n"
                  "What are the final counts? Return the requested JSON value as "
                  '{"Alda": <count>, "Bram": <count>}.')
        reference = {"initial": {"Alda": alda, "Bram": bram}, "updates": updates,
                     "query": ["Alda", "Bram"], "semantics": "ordered transfers/set"}
        return prompt, expected, reference
    if k == 1:
        places = ["depot", "dock", "shed"]
        crate = rng.choice(places)
        barrel = rng.choice(places)
        cur = {"crate": crate, "barrel": barrel}
        updates = []
        for entity in ("crate", "barrel", "crate", "barrel"):
            dest = rng.choice([p for p in places if p != cur[entity]])
            updates.append({"kind": "move", "entity": entity, "to": dest,
                            "text": f"The {entity} is moved to the {dest}."})
            cur[entity] = dest
        expected = dict(cur)
        lines = "\n".join(f"{i + 1}. {u['text']}" for i, u in enumerate(updates))
        prompt = ("Track invented item locations. Apply the moves in the listed order; each move "
                  "overwrites that item's previous location.\n"
                  f"Initial: crate at {crate}; barrel at {barrel}.\n"
                  f"Updates:\n{lines}\n"
                  "What are the final locations? Return the requested JSON value as "
                  '{"crate": "<place>", "barrel": "<place>"}.')
        reference = {"initial": {"crate": crate, "barrel": barrel}, "updates": updates,
                     "query": ["crate", "barrel"], "semantics": "ordered overwrite"}
        return prompt, expected, reference
    if k == 2:
        names = ["Cid", "Dee", "Eli"]
        initial = {n: rng.randint(3, 15) for n in names}
        cur = dict(initial)
        updates = []
        a, b = rng.sample(names, 2)
        amt = rng.randint(1, min(4, cur[a]))
        updates.append({"kind": "transfer", "from": a, "to": b, "amount": amt,
                        "text": f"{a} gives {amt} tokens to {b}."})
        cur[a] -= amt
        cur[b] += amt
        c = rng.choice(names)
        val = rng.randint(0, 15)
        updates.append({"kind": "set", "entity": c, "value": val,
                        "text": f"{c}'s token count is set to {val}."})
        cur[c] = val
        a2, b2 = rng.sample(names, 2)
        amt2 = rng.randint(1, min(4, cur[a2])) if cur[a2] > 0 else 0
        if amt2 == 0:
            val2 = rng.randint(1, 10)
            updates.append({"kind": "set", "entity": a2, "value": val2,
                            "text": f"{a2}'s token count is set to {val2}."})
            cur[a2] = val2
        else:
            updates.append({"kind": "transfer", "from": a2, "to": b2, "amount": amt2,
                            "text": f"{a2} gives {amt2} tokens to {b2}."})
            cur[a2] -= amt2
            cur[b2] += amt2
        expected = dict(cur)
        lines = "\n".join(f"{i + 1}. {u['text']}" for i, u in enumerate(updates))
        prompt = ("Track invented token counts. Apply the updates in the listed order; transfers "
                  "subtract from the source and add to the destination; assignments overwrite.\n"
                  f"Initial: {', '.join(f'{n} has {initial[n]}' for n in names)}.\n"
                  f"Updates:\n{lines}\n"
                  "What are the final counts? Return the requested JSON value as "
                  '{"Cid": <n>, "Dee": <n>, "Eli": <n>}.')
        reference = {"initial": initial, "updates": updates, "query": list(names),
                     "semantics": "ordered transfers/set"}
        return prompt, expected, reference
    north = rng.randint(10, 40)
    south = rng.randint(10, 40)
    cur = {"north": north, "south": south}
    updates = []
    ship = rng.randint(1, min(8, cur["north"]))
    updates.append({"kind": "transfer", "from": "north", "to": "south", "amount": ship,
                    "text": f"Ship {ship} units from north store to south store."})
    cur["north"] -= ship
    cur["south"] += ship
    restock = rng.randint(5, 20)
    updates.append({"kind": "add", "entity": "north", "amount": restock,
                    "text": f"Restock north store with {restock} units."})
    cur["north"] += restock
    audit_val = cur["south"]
    updates.append({"kind": "set", "entity": "south", "value": audit_val,
                    "text": f"Audit confirms south store at {audit_val} units (set, no change)."})
    expected = dict(cur)
    lines = "\n".join(f"{i + 1}. {u['text']}" for i, u in enumerate(updates))
    prompt = ("Track invented store stock. Apply the updates in the listed order; shipments move "
              "units between stores; restocks add; audits set the stated value.\n"
              f"Initial: north has {north}; south has {south}.\n"
              f"Updates:\n{lines}\n"
              "What are the final stock levels? Return the requested JSON value as "
              '{"north": <count>, "south": <count>}.')
    reference = {"initial": {"north": north, "south": south}, "updates": updates,
                 "query": ["north", "south"], "semantics": "ordered shipment/restock/set"}
    return prompt, expected, reference


_FAMILIES = (
    ("code", "code_stack_machine", _gen_code_stack_machine),
    ("code", "code_nested_transform", _gen_code_nested_transform),
    ("reasoning", "reasoning_clock_calendar", _gen_reasoning_clock_calendar),
    ("reasoning", "reasoning_sets", _gen_reasoning_sets),
    ("tools", "tools_bind_arguments", _gen_tools_bind_arguments),
    ("tools", "tools_dependency_order", _gen_tools_dependency_order),
    ("general", "general_graph_route", _gen_general_graph_route),
    ("general", "general_fact_updates", _gen_general_fact_updates),
)


def build_final_suite(seed=SEED):
    """Build the deterministic 32-task final suite (8 families x 4)."""
    rng = random.Random(seed)
    tasks = []
    for category, family, gen in _FAMILIES:
        for k in range(4):
            prompt, expected, reference = gen(rng, k)
            if len(prompt) >= _PROMPT_CHAR_LIMIT:
                raise ValueError(f"prompt too long: {family}-{k:02d}")
            blob = json.dumps(expected, ensure_ascii=False)
            if len(blob) >= _EXPECTED_CHAR_LIMIT:
                raise ValueError(f"gold too long: {family}-{k:02d}")
            # Strict-JSON round-trip guard for the gold value.
            _parse_json(blob)
            tasks.append({"id": f"{family}-{k:02d}", "category": category,
                          "family": family, "prompt": prompt, "expected": expected,
                          "reference_inputs": reference})
    return {
        "schema": SCHEMA,
        "seed": seed,
        "system": SYSTEM,
        "quality_max_tokens": QUALITY_MAX_TOKENS,
        "claim_limits": CLAIM_LIMITS,
        "tasks": tasks,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _refuse_overwrite(path):
    if Path(path).exists():
        print(f"refusing to overwrite existing file: {path}", file=sys.stderr)
        raise SystemExit(2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-final", help="Generate the deterministic 32-task final suite.")
    create.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score", help="Score results against a suite.")
    score.add_argument("--suite", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create-final":
        _refuse_overwrite(args.output)
        suite = build_final_suite()
        args.output.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"wrote {len(suite['tasks'])} tasks to {args.output}")
    elif args.command == "score":
        _refuse_overwrite(args.output)
        suite = json.loads(args.suite.read_text(encoding="utf-8"))
        results_obj = json.loads(args.results.read_text(encoding="utf-8"))
        report = score_final_suite(suite, results_obj)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        totals = report["totals"]
        rate = totals["rate"]
        rate_str = f"{rate:.3f}" if rate is not None else "n/a"
        print(f"passed {totals['passed']}/{totals['total']} "
              f"rate={rate_str} "
              f"format={totals['format_ok']}/{totals['total']} "
              f"extracted={totals['extracted']}")


if __name__ == "__main__":
    main()
