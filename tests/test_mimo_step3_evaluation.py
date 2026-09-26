"""CPU-only checks for the MiMo step-3 scorer + final-suite generator."""
import copy
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load("mimo_step3_evaluation_under_test", "mimo_step3_evaluation.py")

EXPECTED_SYSTEM = ("Solve the task accurately. You may explain your work. Regardless of any earlier "
                   "request for JSON only, finish with exactly one <final>JSON_VALUE</final> block "
                   "containing the requested JSON value. Put no explanation inside that block and no "
                   "text after it.")


def _mini_suite():
    return {
        "schema": mod.SCHEMA,
        "seed": mod.SEED,
        "system": mod.SYSTEM,
        "quality_max_tokens": 2048,
        "claim_limits": "test",
        "tasks": [
            {"id": "t-num", "category": "c", "family": "f", "prompt": "p", "expected": 42},
            {"id": "t-obj", "category": "c", "family": "f", "prompt": "p",
             "expected": {"a": 1}},
        ],
    }


class ExtractionRouteTests(unittest.TestCase):
    def test_final_block_with_leading_prose(self):
        value, route, fmt, err = mod.extract_after_thinking(
            'My reasoning says 7.\n<final>42</final>')
        self.assertEqual((value, route, fmt, err), (42, "final_block", True, None))

    def test_final_block_ignores_earlier_misleading_answer(self):
        # Earlier JSON / fences must not win once a final block exists.
        text = '{"a": 999}\n```json\n[1,2]\n```\nlast line decoy\n<final>{"a": 1}</final>'
        value, route, fmt, err = mod.extract_after_thinking(text)
        self.assertEqual((value, route, fmt), ({"a": 1}, "final_block", True))
        self.assertIsNone(err)

    def test_final_malformed_never_falls_through(self):
        bad = [
            "<final>42</final> trailing prose",
            "<final>42</final><final>42</final>",
            "<final>42",
            "42</final>",
            "</final><final>42",
            "<final>not json</final>",
            "<final>{\"a\": 1, \"a\": 2}</final>",
            "<final>NaN</final>",
        ]
        for text in bad:
            value, route, fmt, err = mod.extract_after_thinking(text)
            self.assertIsNone(value, msg=text)
            self.assertIsNone(route, msg=text)
            self.assertFalse(fmt, msg=text)
            self.assertIsNotNone(err, msg=text)

    def test_direct_json_route(self):
        value, route, fmt, err = mod.extract_after_thinking('  {"a": 1}  ')
        self.assertEqual((value, route, fmt, err), ({"a": 1}, "direct_json", False, None))

    def test_direct_json_strictness(self):
        for text in ('{"a": 1, "a": 2}', "NaN", "Infinity", '{"a": 1} trailing'):
            value, route, fmt, err = mod.extract_after_thinking(text)
            # Must not extract via direct route; either fence/last-line fails or parses
            # last line only. Here single-line inputs fail last-line strict parse too.
            self.assertIsNone(value, msg=text)
            self.assertIsNotNone(err, msg=text)

    def test_fenced_json_terminal_with_leading_prose(self):
        value, route, fmt, err = mod.extract_after_thinking(
            "Here is the answer:\n```json\n{\"a\": 1}\n```")
        self.assertEqual((value, route, fmt, err), ({"a": 1}, "fenced_json", False, None))
        value, route, fmt, err = mod.extract_after_thinking("```\n42\n```")
        self.assertEqual((value, route, fmt), (42, "fenced_json", False))

    def test_fence_rejections_exclusive(self):
        bad = [
            "```json\n1\n```\n```\n2\n```",  # multiple
            "```json\n1\n``` trailing",  # trailing prose
            "```python\n1\n```",  # language rejected
            "```json\nnot json\n```",  # invalid inner
            "```json\n1\n",  # unclosed
        ]
        for text in bad:
            value, route, fmt, err = mod.extract_after_thinking(text)
            self.assertIsNone(value, msg=text)
            self.assertIsNone(route, msg=text)
            self.assertIsNotNone(err, msg=text)

    def test_last_line_only(self):
        value, route, fmt, err = mod.extract_after_thinking(
            "First guess: 7\nI think {\"a\": 999}\n{\"a\": 1}")
        self.assertEqual((value, route, fmt), ({"a": 1}, "last_line", False))
        self.assertIsNone(err)
        # Earlier lines never searched when last line is bad.
        value, route, fmt, err = mod.extract_after_thinking('{"a": 1}\nnot json')
        self.assertIsNone(value)
        self.assertIsNotNone(err)

    def test_thinking_prefix_then_final(self):
        value, route, fmt, rc, serr, eerr = mod.extract_step3_response(
            "<think>scratch 999</think>work\n<final>42</final>")
        self.assertEqual((value, route, fmt), (42, "final_block", True))
        self.assertIsNone(serr)
        self.assertIsNone(eerr)
        self.assertEqual(rc, len("scratch 999"))

    def test_thinking_decoys_fail(self):
        bad = [
            "Answer: <think>x</think>42",
            "<think>unclosed 42",
            "42</think>",
            "<think>a</think>1<think>b</think>",
            "<think>t</think>{\"k\": \"<think>\"}",
        ]
        for text in bad:
            value, route, fmt, rc, serr, eerr = mod.extract_step3_response(text)
            self.assertIsNone(value, msg=text)
            self.assertIsNotNone(serr, msg=text)

    def test_thinking_trace_numbers_never_used(self):
        # Gold-looking number only in thinking trace; answer has no JSON.
        value, route, fmt, rc, serr, eerr = mod.extract_step3_response(
            "<think>the answer is 42</think>no json here")
        self.assertIsNone(value)
        self.assertIsNone(serr)
        self.assertIsNotNone(eerr)


class ScorerTests(unittest.TestCase):
    def test_correctness_format_separation(self):
        suite = _mini_suite()
        # t-num: wrong value but well formatted; t-obj: correct but no final block.
        results = {"t-num": "explanation\n<final>7</final>",
                   "t-obj": '{"a": 1}'}
        report = mod.score_final_suite(suite, results)
        by_id = {c["id"]: c for c in report["cases"]}
        self.assertFalse(by_id["t-num"]["correct"])
        self.assertTrue(by_id["t-num"]["format_ok"])
        self.assertEqual(by_id["t-num"]["route"], "final_block")
        self.assertEqual(by_id["t-num"]["error"], "mismatch")
        self.assertTrue(by_id["t-obj"]["correct"])
        self.assertFalse(by_id["t-obj"]["format_ok"])
        self.assertEqual(by_id["t-obj"]["route"], "direct_json")
        self.assertEqual(report["totals"]["correct"], 1)
        self.assertEqual(report["totals"]["format_ok"], 1)
        self.assertEqual(report["totals"]["extracted"], 2)
        self.assertAlmostEqual(report["totals"]["conditional_rate"], 0.5)

    def test_bool_is_not_number(self):
        suite = {"schema": mod.SCHEMA, "tasks": [
            {"id": "b", "category": "c", "family": "f", "prompt": "p", "expected": 1}]}
        report = mod.score_final_suite(suite, {"b": "<final>true</final>"})
        case = report["cases"][0]
        self.assertFalse(case["correct"])
        self.assertTrue(case["format_ok"])
        self.assertEqual(case["error"], "mismatch")

    def test_missing_duplicate_unknown_and_shapes(self):
        suite = _mini_suite()
        before = copy.deepcopy(suite)
        results = [
            {"task_id": "t-num", "response": "<final>42</final>"},
            {"task_id": "t-num", "response": "<final>42</final>"},
            {"task_id": "t-ghost", "response": "<final>0</final>"},
        ]
        snap = copy.deepcopy(results)
        report = mod.score_final_suite(suite, results)
        self.assertEqual(suite, before)
        self.assertEqual(results, snap)
        self.assertEqual(report["duplicate_ids"], ["t-num"])
        self.assertEqual(report["unknown_ids"], ["t-ghost"])
        self.assertEqual(report["missing_ids"], ["t-obj"])
        by_id = {c["id"]: c for c in report["cases"]}
        self.assertEqual(by_id["t-num"]["error"], "duplicate_result")
        self.assertEqual(by_id["t-obj"]["error"], "missing_result")
        # Mapping + wrapped shapes also accepted.
        ok_map = mod.score_final_suite(suite, {"t-num": "<final>42</final>",
                                               "t-obj": "<final>{\"a\": 1}</final>"})
        self.assertEqual(ok_map["totals"]["correct"], 2)
        ok_wrap = mod.score_final_suite(
            suite, {"results": [{"name": "t-num", "output_text": "<final>42</final>"},
                                {"name": "t-obj", "output_text": "<final>{\"a\": 1}</final>"}]})
        self.assertEqual(ok_wrap["totals"]["correct"], 2)

    def test_nonstring_results_explicit(self):
        suite = _mini_suite()
        report = mod.score_final_suite(suite, {"t-num": 42, "t-obj": "<final>{\"a\": 1}</final>"})
        self.assertIsNotNone(report["format_error"])
        self.assertTrue(all(c["error"] == report["format_error"] for c in report["cases"]))
        self.assertEqual(report["totals"]["correct"], 0)

    def test_conditional_counts(self):
        suite = _mini_suite()
        report = mod.score_final_suite(
            suite, {"t-num": "<think>cut off", "t-obj": "<final>{\"a\": 1}</final>"})
        by_id = {c["id"]: c for c in report["cases"]}
        self.assertEqual(by_id["t-num"]["error"], "thinking_parse_error")
        self.assertNotIn("capped_suspect", by_id["t-num"])
        self.assertNotIn("capped_suspect", report["totals"])
        self.assertEqual(report["totals"]["extracted"], 1)
        self.assertAlmostEqual(report["totals"]["conditional_rate"], 1.0)
        self.assertIn("f", report["by_family"])
        self.assertIn("c", report["by_category"])

    def test_conditional_rate_null_when_nothing_extracted(self):
        suite = _mini_suite()
        # Neither response yields an extracted answer: thinking + final-block
        # parse errors are explicit, not a measured zero-accuracy subset.
        report = mod.score_final_suite(
            suite, {"t-num": "<think>cut off", "t-obj": "<final>oops"})
        self.assertEqual(report["totals"]["extracted"], 0)
        self.assertIsNone(report["totals"]["conditional_rate"])
        by_id = {c["id"]: c for c in report["cases"]}
        self.assertEqual(by_id["t-num"]["error"], "thinking_parse_error")
        self.assertEqual(by_id["t-obj"]["error"], "parse_error")
        self.assertIsNone(by_id["t-obj"]["route"])

    def test_empty_suite_rates_null(self):
        suite = {"schema": mod.SCHEMA, "tasks": []}
        report = mod.score_final_suite(suite, {})
        self.assertEqual(report["totals"]["total"], 0)
        self.assertIsNone(report["totals"]["rate"])
        self.assertIsNone(report["totals"]["format_rate"])
        self.assertIsNone(report["totals"]["conditional_rate"])


class FinalSuiteTests(unittest.TestCase):
    def test_shape_counts_and_system(self):
        suite = mod.build_final_suite()
        self.assertEqual(suite["seed"], 20260923)
        self.assertEqual(suite["system"], EXPECTED_SYSTEM)
        self.assertEqual(suite["quality_max_tokens"], 2048)
        self.assertEqual(len(suite["tasks"]), 32)
        cats = Counter(t["category"] for t in suite["tasks"])
        self.assertEqual(set(cats), {"code", "reasoning", "tools", "general"})
        for cat, count in cats.items():
            self.assertEqual(count, 8, msg=cat)
        fams = Counter(t["family"] for t in suite["tasks"])
        expected_fams = {"code_stack_machine", "code_nested_transform",
                         "reasoning_clock_calendar", "reasoning_sets",
                         "tools_bind_arguments", "tools_dependency_order",
                         "general_graph_route", "general_fact_updates"}
        self.assertEqual(set(fams), expected_fams)
        for fam, count in fams.items():
            self.assertEqual(count, 4, msg=fam)
        ids = [t["id"] for t in suite["tasks"]]
        self.assertEqual(len(set(ids)), 32)
        for task in suite["tasks"]:
            for key in ("id", "category", "family", "prompt", "expected", "reference_inputs"):
                self.assertIn(key, task, msg=task["id"])
            self.assertTrue(task["reference_inputs"], msg=task["id"])
        self.assertIn("authored", suite["claim_limits"].lower())
        self.assertIn("bf16", suite["claim_limits"].lower())
        self.assertIn("no model-generated code", suite["claim_limits"].lower())

    def test_bounded_and_deterministic(self):
        first = mod.build_final_suite()
        second = mod.build_final_suite(seed=20260923)
        self.assertEqual(first, second)
        for task in first["tasks"]:
            self.assertLess(len(task["prompt"]), 3000, msg=task["id"])
            blob = json.dumps(task["expected"], ensure_ascii=False)
            self.assertLess(len(blob), 600, msg=task["id"])

    def test_reference_inputs_recompute_golds(self):
        import ast
        import re
        suite = mod.build_final_suite()
        by_fam = {}
        for task in suite["tasks"]:
            by_fam.setdefault(task["family"], []).append(task)
        # Stack machine: replay reference operations.
        for task in by_fam["code_stack_machine"]:
            ref = task["reference_inputs"]
            stack = list(ref["initial_stack"])
            for op in ref["operations"]:
                if op["op"] == "push":
                    stack.append(op["value"])
                elif op["op"] == "pop":
                    stack.pop()
                elif op["op"] == "add":
                    stack.append(stack.pop(-2) + stack.pop())
                else:
                    stack.append(stack.pop(-2) * stack.pop())
            self.assertEqual(task["expected"], stack, msg=task["id"])
            listed = re.findall(r"^\d+\. (push -?\d+|pop|add|multiply)",
                                task["prompt"], flags=re.M)
            self.assertEqual(len(listed), len(ref["operations"]), msg=task["id"])
        # Nested transform: exec the reference function source.
        for task in by_fam["code_nested_transform"]:
            ref = task["reference_inputs"]
            ns = {}
            exec(ref["function_source"], {}, ns)
            self.assertEqual(task["expected"], [ns["transform"](x) for x in ref["values"]],
                             msg=task["id"])
            self.assertIn(str(ref["values"]), task["prompt"], msg=task["id"])
        # Clock/calendar: recompute rollover.
        for task in by_fam["reasoning_clock_calendar"]:
            ref = task["reference_inputs"]
            total = ref["start_hour"] * 60 + ref["start_minute"] + ref["duration_minutes"]
            cycle = ref["weekday_cycle"]
            exp = {"weekday": cycle[(cycle.index(ref["start_weekday"]) + total // 1440) % 7],
                   "hour": (total % 1440) // 60, "minute": total % 60}
            self.assertEqual(task["expected"], exp, msg=task["id"])
        # Sets: union/neither consistent with regions.
        for task in by_fam["reasoning_sets"]:
            ref = task["reference_inputs"]
            regs = ref["regions"]
            total = sum(regs.values())
            self.assertEqual(ref["population"], total, msg=task["id"])
            self.assertEqual(task["expected"]["neither"], regs["neither"], msg=task["id"])
            self.assertEqual(task["expected"]["union"], total - regs["neither"], msg=task["id"])
        # Dependency order: validate topological + lexicographic tie-break.
        for task in by_fam["tools_dependency_order"]:
            ref = task["reference_inputs"]
            order, deps = task["expected"], ref["dependencies"]
            self.assertEqual(sorted(order), sorted(ref["tools"]), msg=task["id"])
            pos = {name: i for i, name in enumerate(order)}
            for tool, ds in deps.items():
                for dep in ds:
                    self.assertLess(pos[dep], pos[tool], msg=(task["id"], dep, tool))
            remaining = {t: set(deps[t]) for t in ref["tools"]}
            expect, done = [], set()
            while remaining:
                ready = sorted(t for t, ds in remaining.items() if ds <= done)
                expect.append(ready[0])
                done.add(ready[0])
                del remaining[ready[0]]
            self.assertEqual(order, expect, msg=task["id"])
        # Graph route: validate cost and minimality.
        for task in by_fam["general_graph_route"]:
            ref = task["reference_inputs"]
            weight = {(e["from"], e["to"]): e["weight"] for e in ref["edges"]}
            route = task["expected"]["route"]
            self.assertEqual(route[0], ref["start"], msg=task["id"])
            self.assertEqual(route[-1], ref["goal"], msg=task["id"])
            cost = sum(weight[(route[i], route[i + 1])] for i in range(len(route) - 1))
            self.assertEqual(task["expected"]["cost"], cost, msg=task["id"])
            adj = {}
            for e in ref["edges"]:
                adj.setdefault(e["from"], []).append(e["to"])
            best, stack = None, [(ref["start"], [ref["start"]], 0)]
            while stack:
                node, path, acc = stack.pop()
                if best is not None and acc > best[1]:
                    continue
                if node == ref["goal"]:
                    if best is None or acc < best[1] or (
                            acc == best[1] and tuple(path) < tuple(best[0])):
                        best = (list(path), acc)
                    continue
                for nxt in adj.get(node, []):
                    if nxt not in path:
                        stack.append((nxt, path + [nxt], acc + weight[(node, nxt)]))
            self.assertEqual((route, cost), best, msg=task["id"])
        # Fact updates: replay reference updates.
        for task in by_fam["general_fact_updates"]:
            ref = task["reference_inputs"]
            cur = dict(ref["initial"])
            for update in ref["updates"]:
                kind = update["kind"]
                if kind == "transfer":
                    cur[update["from"]] -= update["amount"]
                    cur[update["to"]] += update["amount"]
                elif kind in ("set", "move"):
                    key = update.get("entity", update.get("to", ""))
                    target = update.get("entity", "")
                    if kind == "move":
                        cur[target] = update["to"]
                    else:
                        cur[target] = update["value"]
                elif kind == "add":
                    cur[update["entity"]] += update["amount"]
                else:
                    self.fail(f"unknown update kind: {kind}")
            for key in ref["query"]:
                self.assertEqual(task["expected"][key], cur[key], msg=(task["id"], key))
        # Bind-arguments: provided overrides + defaults visible in prompt.
        for task in by_fam["tools_bind_arguments"]:
            ref = task["reference_inputs"]
            self.assertIn(ref["tool"], task["prompt"], msg=task["id"])
            self.assertIn(json.dumps(ref["provided_arguments"]), task["prompt"],
                          msg=task["id"])
            self.assertEqual(task["expected"]["name"], ref["tool"], msg=task["id"])
            self.assertEqual(task["expected"]["arguments"], ref["expected_arguments"],
                             msg=task["id"])
            self.assertNotIn("execute", task["expected"] and "" or "EXECUTE_SENTINEL")

    def test_tool_lines_are_valid_json(self):
        suite = mod.build_final_suite()
        tool_tasks = [t for t in suite["tasks"] if t["family"] == "tools_bind_arguments"]
        self.assertEqual(len(tool_tasks), 4)
        for task in tool_tasks:
            lines = [line for line in task["prompt"].splitlines()
                     if line.startswith("Tool: ")]
            self.assertEqual(len(lines), 1, msg=task["id"])
            parsed = json.loads(lines[0][len("Tool: "):])
            self.assertEqual(parsed["name"], task["reference_inputs"]["tool"],
                             msg=task["id"])

    def test_three_group_sets_state_pairwise_inclusion(self):
        suite = mod.build_final_suite()
        by_id = {t["id"]: t for t in suite["tasks"]}
        for task_id in ("reasoning_sets-02", "reasoning_sets-03"):
            self.assertIn("INCLUDE", by_id[task_id]["prompt"], msg=task_id)
            self.assertIn("all three", by_id[task_id]["prompt"], msg=task_id)

    def test_build_final_suite_seeds_0_to_99(self):
        for seed in range(100):
            suite = mod.build_final_suite(seed=seed)
            self.assertEqual(len(suite["tasks"]), 32, msg=f"seed={seed}")

    def test_cli_roundtrip_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            suite_path = tmp / "final.json"
            mod.main(["create-final", "--output", str(suite_path)])
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            self.assertEqual(len(suite["tasks"]), 32)
            with self.assertRaises(SystemExit):
                mod.main(["create-final", "--output", str(suite_path)])
            results = {t["id"]: "<final>" + json.dumps(t["expected"]) + "</final>"
                       for t in suite["tasks"]}
            results_path = tmp / "results.json"
            results_path.write_text(json.dumps(results), encoding="utf-8")
            report_path = tmp / "report.json"
            mod.main(["score", "--suite", str(suite_path), "--results", str(results_path),
                      "--output", str(report_path)])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["totals"]["correct"], 32)
            self.assertEqual(report["totals"]["format_ok"], 32)
            with self.assertRaises(SystemExit):
                mod.main(["score", "--suite", str(suite_path), "--results", str(results_path),
                          "--output", str(report_path)])


if __name__ == "__main__":
    unittest.main()
