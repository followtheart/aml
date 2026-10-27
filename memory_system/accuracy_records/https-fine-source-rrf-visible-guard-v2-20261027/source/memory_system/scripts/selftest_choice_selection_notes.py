"""Reject fabricated references and inadmissible choices in selection notes."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.choice_selection_notes import validate


class SelectionNotesTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {'s1': {'id': 's1', 'text': 'Original source'},
                        'r0': {'id': 'r0', 'text': 'Forget constraint'}}
        self.payload = {'answer': 'A', 'notes': [
            {'letter': letter, 'evidence': [sid], 'contextual_fit': '',
             'unsupported_or_prohibited_details': ''}
            for letter, sid in [('A', 's1'), ('C', 'r0')]]}

    def test_resolves_original_support_and_constraint(self):
        before = copy.deepcopy(self.catalog)
        result = validate(self.payload, ['A', 'C'], self.catalog)
        self.assertEqual(result['resolved_evidence']['A'], [self.catalog['s1']])
        self.assertEqual(result['resolved_evidence']['C'], [self.catalog['r0']])
        self.assertEqual(before, self.catalog)

    def test_rejects_ineligible_answer_even_with_valid_notes(self):
        self.payload['answer'] = 'B'
        with self.assertRaises(ValueError):
            validate(self.payload, ['A', 'C'], self.catalog)

    def test_rejects_unknown_evidence(self):
        self.payload['notes'][0]['evidence'] = ['s999']
        with self.assertRaises(ValueError):
            validate(self.payload, ['A', 'C'], self.catalog)

    def test_rejects_duplicate_or_missing_option(self):
        self.payload['notes'][1]['letter'] = 'A'
        with self.assertRaises(ValueError):
            validate(self.payload, ['A', 'C'], self.catalog)

    def test_rejects_generated_quote_instead_of_id(self):
        self.payload['notes'][0]['evidence'] = [{'source_id': 's1', 'quote': 'Invented'}]
        with self.assertRaises(ValueError):
            validate(self.payload, ['A', 'C'], self.catalog)


if __name__ == '__main__':
    unittest.main()
