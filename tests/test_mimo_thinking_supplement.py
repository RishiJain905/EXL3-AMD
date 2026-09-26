"""CPU-only checks for the MiMo thinking-supplement builder + scorer."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("mimo_validation_suite_under_test", "mimo_validation_suite.py")
sup = _load("mimo_thinking_supplement_under_test", "mimo_thinking_supplement.py")


def _mini_suite():
    return {
        "schema": base.SCHEMA,
        "seed": base.SEED,
        "system": base.SYSTEM,
        "quality_max_tokens": 2048,
        "claim_limits": "test",
        "tasks": [
            {"id": "t-direct", "category": "c", "family": "f1", "prompt": "p",
             "expected": {"a": 1}},
            {"id": "t-block", "category": "c", "family": "f2", "prompt": "p",
             "expected": [1, 2]},
            {"id": "t-strip", "category": "c", "family": "f3", "prompt": "p",
             "expected": 42},
            {"id": "t-wrong", "category": "c", "family": "f4", "prompt": "p",
             "expected": 7},
            {"id": "t-missing", "category": "c", "family": "f5", "prompt": "p",
             "expected": 1},
            {"id": "t-dup", "category": "c", "family": "f6", "prompt": "p",
             "expected": 2},
        ],
    }


class StripTests(unittest.TestCase):
    def test_direct_json_passthrough(self):
        for text in ('{"a": 1}', '  \n{"a": 1}  ', '```json\n{"a": 1}\n```'):
            answer, reasoning, error = sup.strip_thinking_block(text)
            self.assertIsNone(error)
            self.assertIsNone(reasoning)
            self.assertEqual(answer, text)
            value, parse_error = base.extract_json(answer)
            self.assertIsNone(parse_error)
            self.assertEqual(value, {"a": 1})

    def test_leading_block_and_whitespace(self):
        answer, reasoning, error = sup.strip_thinking_block("<think>scratch</think>{\"a\": 1}")
        self.assertIsNone(error)
        self.assertEqual((answer, reasoning), ('{"a": 1}', 7))
        answer, reasoning, error = sup.strip_thinking_block("\n <think>x</think>[1, 2]")
        self.assertIsNone(error)
        self.assertEqual((answer, reasoning), ("[1, 2]", 1))
        answer, reasoning, error = sup.strip_thinking_block("<think></think>42")
        self.assertIsNone(error)
        self.assertEqual((answer, reasoning), ("42", 0))

    def test_malformed_markers_rejected(self):
        vectors = {
            "Answer: <think>x</think>{\"a\": 1}": "thinking_block_misplaced",
            "<think>scratch {\"a\": 1}": "thinking_block_unclosed",
            "{\"a\": 1}</think>": "thinking_block_unbalanced",
            "<think>a<think>b</think></think>{}": "thinking_block_extra_markers",
            "<think>a</think>{\"x\": 1}<think>b</think>": "thinking_block_extra_markers",
            "<think>t</think>{\"k\": \"<think>\"}": "thinking_block_extra_markers",
        }
        for text, code in vectors.items():
            _, _, error = sup.strip_thinking_block(text)
            self.assertEqual(error, code, msg=text)
        _, _, error = sup.strip_thinking_block(123)
        self.assertEqual(error, "response_not_string")


class ScorerTests(unittest.TestCase):
    def test_mixed_end_to_end(self):
        suite = _mini_suite()
        before = copy.deepcopy(suite)
        results = [
            {"task_id": "t-direct", "response": '{"a": 1}'},
            {"task_id": "t-block", "response": "<think>work</think>[1, 2]"},
            {"task_id": "t-strip", "response": "<think>oops 42"},
            {"task_id": "t-wrong", "response": "<think>hmm</think>8"},
            {"task_id": "t-dup", "response": "2"},
            {"task_id": "t-dup", "response": "2"},
            {"task_id": "t-unknown", "response": "0"},
        ]
        results_before = copy.deepcopy(results)
        report = sup.score_suite_thinking(suite, results)
        self.assertEqual(suite, before)
        self.assertEqual(results, results_before)
        self.assertEqual(report["totals"], {**report["totals"], "passed": 2, "total": 6})
        self.assertEqual(report["missing_ids"], ["t-missing"])
        self.assertEqual(report["duplicate_ids"], ["t-dup"])
        self.assertEqual(report["unknown_ids"], ["t-unknown"])
        by_id = {c["id"]: c for c in report["cases"]}
        self.assertTrue(by_id["t-direct"]["passed"])
        self.assertIsNone(by_id["t-direct"]["reasoning_chars"])
        self.assertTrue(by_id["t-block"]["passed"])
        self.assertEqual(by_id["t-block"]["reasoning_chars"], 4)
        self.assertEqual(by_id["t-strip"]["error"], "thinking_parse_error")
        self.assertEqual(by_id["t-strip"]["strip_error"], "thinking_block_unclosed")
        self.assertEqual(by_id["t-wrong"]["error"], "mismatch")
        self.assertIsNone(by_id["t-wrong"]["strip_error"])
        self.assertEqual(by_id["t-missing"]["error"], "missing_result")
        self.assertEqual(by_id["t-dup"]["error"], "duplicate_result")
        self.assertEqual(report, sup.score_suite_thinking(suite, results))

    def test_final_json_strictness_delegated(self):
        suite = {"schema": base.SCHEMA, "tasks": [
            {"id": "only", "category": "c", "family": "f", "prompt": "p", "expected": {"x": 1}}]}
        bad = ['{"x": 1, "x": 2}', "<think>ok</think>{\"x\": 1, \"x\": 2}",
               "NaN", "<think>ok</think>NaN",
               "<think>t</think>see {\"x\": 1}", '{"x": 1} trailing',
               "<think>t</think>```json\n{\"x\": 1}\n```\n```\n2\n```"]
        for text in bad:
            report = sup.score_suite_thinking(suite, {"only": text})
            case = report["cases"][0]
            self.assertFalse(case["passed"], msg=text)
            self.assertEqual(case["error"], "parse_error", msg=text)
            self.assertIsNone(case["strip_error"], msg=text)
        good = ["```json\n{\"x\": 1}\n```", "<think>t</think>```json\n{\"x\": 1}\n```"]
        for text in good:
            report = sup.score_suite_thinking(suite, {"only": text})
            self.assertTrue(report["cases"][0]["passed"], msg=text)


class BuilderTests(unittest.TestCase):
    def test_subset_first_per_family_unmutated(self):
        suite = base.build_suite()
        before = copy.deepcopy(suite)
        selected = sup.select_thinking_subset(suite["tasks"])
        self.assertEqual(suite, before)
        self.assertEqual(len(selected), 16)
        self.assertTrue(all(t["id"].endswith("-00") for t in selected))
        self.assertEqual([t["family"] for t in selected],
                         [family for _, family, _ in base._FAMILIES])
        by_id = {t["id"]: t for t in suite["tasks"]}
        for task in selected:
            self.assertEqual(task["prompt"], by_id[task["id"]]["prompt"])
            self.assertEqual(task["expected"], by_id[task["id"]]["expected"])
        batch_a, batch_b = sup.split_batches(selected)
        self.assertEqual((batch_a, batch_b), (selected[:8], selected[8:]))
        self.assertEqual(len({t["id"] for t in batch_a} | {t["id"] for t in batch_b}), 16)

    def test_render_and_assemble_with_stub(self):
        class StubTokenizer:
            def __init__(self, too_long=False):
                self.too_long = too_long
                self.seen = []

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True):
                assert tokenize is False and add_generation_prompt is True
                assert enable_thinking is True
                self.seen.append(messages)
                return ("<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>"
                        "<|im_start|>user\n" + messages[1]["content"] + "<|im_end|>"
                        + sup.ASSISTANT_HEADER)

            def encode(self, text, add_special_tokens=False):
                assert add_special_tokens is False
                if self.too_long:
                    return [1] * 4096
                return [ord(c) % 251 for c in text[:32]]

            def __len__(self):
                return 248077

        stub = StubTokenizer()
        task = {"id": "fam-00", "family": "fam", "prompt": "Return 1.", "expected": 1}
        case = sup.render_thinking_case(stub, base.SYSTEM, task)
        self.assertEqual(case["id"], "fam-00")
        self.assertTrue(case["rendered"].endswith(sup.ASSISTANT_HEADER))
        self.assertNotIn("<think>", case["rendered"])
        self.assertEqual(case["answer"], "1")
        with self.assertRaises(ValueError):
            sup.render_thinking_case(StubTokenizer(too_long=True), base.SYSTEM, task)
        with self.assertRaises(ValueError):
            sup.render_thinking_case(stub, base.SYSTEM,
                                     {**task, "prompt": "evil <think> marker"})
        protocol = {"tokenizer_sha256": "t", "template_sha256": "j",
                    "stop_ids": [248046, 248044], "context": 4096, "cache": "f16",
                    "mtp": False, "valid_vocabulary_size": 248077,
                    "stored_vocabulary_size": 248320}
        built = sup.assemble_suite(base.build_suite(), [task])
        self.assertEqual(built["quality_max_tokens"], 2048)
        self.assertEqual(built["system"], base.SYSTEM)
        self.assertEqual(built["tasks"], [task])
        built["tasks"][0]["prompt"] = "mutated"
        self.assertEqual(task["prompt"], "Return 1.")
        assembled = sup.assemble_protocol(protocol, "suitesha", [case])
        self.assertEqual((assembled["thinking"], assembled["max_new_tokens"],
                          assembled["suite_file_sha256"], assembled["cases"]), (True, 2048, "suitesha", [case]))
        self.assertNotIn("teacher_forced_probes", assembled)
        self.assertIn("exploratory", assembled["scope"].lower())

    def test_score_cli_roundtrip_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            suite_path = tmp / "suite.json"
            suite_path.write_text(json.dumps(_mini_suite(), ensure_ascii=False), encoding="utf-8")
            results_path = tmp / "results.json"
            results = {"t-direct": '{"a": 1}', "t-block": "<think>r</think>[1, 2]",
                       "t-strip": "42", "t-wrong": "7", "t-missing": "1", "t-dup": "2"}
            results_path.write_text(json.dumps(results), encoding="utf-8")
            report_path = tmp / "report.json"
            sup.main(["score", "--suite", str(suite_path), "--results", str(results_path),
                      "--output", str(report_path)])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["totals"]["passed"], 6)
            with self.assertRaises(SystemExit):
                sup.main(["score", "--suite", str(suite_path), "--results", str(results_path),
                          "--output", str(report_path)])

    def test_build_refuses_existing_output_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                sup.main(["build", "--suite", "nope", "--protocol", "nope",
                          "--source", "nope", "--output", tmp])


if __name__ == "__main__":
    unittest.main()
