"""Premise compilation must not silently mix options or truncate qualifications."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.query_premises import compile_queries


class QueryPremiseTests(unittest.TestCase):
    def test_exact_conditional_and_generic(self):
        self.assertEqual(compile_queries(['If you own a canoe, try a lake.', 'Try a lake.'],
            {'options': [{'index': 0, 'premises': ['If you own a canoe']},
                         {'index': 1, 'premises': []}]}), ['If you own a canoe', ''])

    def test_wrong_option_and_invented_text_rejected(self):
        for quote in ['you own a kayak', 'Try a lake.']:
            with self.assertRaises(ValueError):
                compile_queries(['If you own a canoe, try a lake.'],
                                {'options': [{'index': 0, 'premises': [quote]}]})

    def test_bad_shape_and_order_rejected(self):
        for entries in [[], [{'index': True, 'premises': []}],
                        [{'index': 1, 'premises': []}], [{'index': 0, 'premises': ''}],
                        [{'index': 0, 'premises': ['']}]]:
            with self.assertRaises(ValueError):
                compile_queries(['option'], {'options': entries})

    def test_over_budget_rejected_not_truncated(self):
        with self.assertRaises(ValueError):
            compile_queries(['x' * 241], {'options': [{'index': 0, 'premises': ['x' * 241]}]})

    def test_textual_validity_does_not_prove_semantic_scope(self):
        self.assertEqual(compile_queries(['Try a lake.'],
            {'options': [{'index': 0, 'premises': ['Try a lake.']}]}), ['Try a lake.'])


if __name__ == '__main__':
    unittest.main()
