"""Scorer regression cases, independent of the held-out model responses."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cache_research_score', Path(__file__).resolve().parents[1] / 'kernels/exl3/score_cache_research.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
score = module.score


class ScoreTests(unittest.TestCase):
    def test_equivalent_arithmetic_object_preserves_semantics_but_not_array_format(self):
        result = score('```json\n{"additional_tokens_per_club":17,"tokens_left_over":2}\n```', 'reasoning', [17, 2])
        self.assertTrue(result['correct'])
        self.assertFalse(result['format_compliant'])
        self.assertFalse(score('{"additional_tokens_per_club":2,"tokens_left_over":17}', 'reasoning', [17, 2])['correct'])

    def test_equivalent_instruction_and_code_objects(self):
        self.assertTrue(score('{"gamma":"c","alpha":"a","beta":"b"}', 'instructions', ['a', 'b', 'c'])['correct'])
        self.assertTrue(score('{"output":[1,-2,3]}', 'code', [1, -2, 3])['correct'])
        self.assertFalse(score('{"result":[1,2,3]}', 'code', [1, -2, 3])['correct'])

    def test_duplicate_and_malformed_answers_do_not_award_nested_fragments(self):
        for text in ('{"q0":"bad","q0":"good"}', '```json\n[[1,2]]\n```', '{"output":[1,2],'):
            self.assertFalse(score(text, 'code', [1, 2])['correct'])
        self.assertFalse(score('{"q0":"bad","q0":"good"}', 'retrieval', {'q0': 'good'})['correct'])

    def test_last_complete_answer_controls_score(self):
        self.assertFalse(score('[1,2]\nCorrection: [1,3]', 'code', [1, 2])['correct'])
        self.assertTrue(score('Answer: [1,2]', 'code', [1, 2])['correct'])

    def test_strict_json_format_separate_from_correctness(self):
        result = score('[99,0]', 'reasoning', [17, 2])
        self.assertFalse(result['correct'])
        self.assertTrue(result['format_compliant'])
        result = score('{"q0":"good","extra":"ignored"}', 'retrieval', {'q0': 'good'})
        self.assertTrue(result['correct'])
        self.assertFalse(result['format_compliant'])

    def test_booleans_strings_and_nonfinite_are_not_numeric_answers(self):
        for text in ('[true,2]', '["1",2]', '[NaN,2]', '[Infinity,2]', '[1e999,2]'):
            self.assertFalse(score(text, 'reasoning', [1, 2])['correct'])
        self.assertFalse(score('[' + '9' * 400 + ',2]', 'reasoning', [1, 2])['correct'])


if __name__ == '__main__':
    unittest.main()
