"""Offline boundaries for optional oversized original-span support responses."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import ValidationError
from app import answer_choice as ac, choice_premises, config


class BoundedSupportTests(unittest.TestCase):
    def setUp(self):
        self.expected = {'A': 'A. Since you own a kayak you could go paddling. '
                         'Check the weather. Pick a route. Wear a buoyancy aid. '
                         'Start near shore. Keep it short. Pack water. Take a break. Return early.'}
        self.spans = choice_premises.option_spans('A', self.expected['A'])
        self.assertGreater(len(self.spans), 8)
        self.payload = {'options': [dict(letter='A', kind='personal', claims=[
            dict(span_id=s['id'], status='unsupported', premise_type='unknown',
                 reason='no_source', citations=[]) for s in self.spans])]}
        self.addCleanup(patch.stopall)
        patch.object(config, 'CHOICE_BOUNDED_SUPPORT_SPANS', True).start()

    def parse(self):
        return ac.parse_support_assessments(self.payload, self.expected)

    def test_preserves_claims_statuses_and_input(self):
        before = copy.deepcopy(self.payload)
        got = self.parse()
        self.assertEqual(len(got.options[0].claims), len(self.spans))
        self.assertTrue(all(c.status == 'unsupported' for c in got.options[0].claims))
        self.assertEqual(self.payload, before)

    def test_disabled_rejects(self):
        with patch.object(config, 'CHOICE_BOUNDED_SUPPORT_SPANS', False):
            with self.assertRaises(ValidationError): self.parse()

    def test_generation_schema_unchanged(self):
        self.assertEqual(ac.Assessments.model_json_schema()['$defs']['Assessment']
                         ['properties']['claims']['maxItems'], 8)

    def test_normal_response_exact(self):
        self.payload['options'][0]['claims'] = self.payload['options'][0]['claims'][:8]
        self.assertEqual(self.parse().model_dump(), ac.Assessments.model_validate(self.payload).model_dump())

    def test_wrong_option(self):
        self.payload['options'][0]['claims'][0]['span_id'] = 'B:0'
        with self.assertRaises(ValidationError): self.parse()

    def test_duplicate(self):
        self.payload['options'][0]['claims'][1]['span_id'] = 'A:0'
        with self.assertRaises(ValidationError): self.parse()

    def test_free_text_overflow_rejected(self):
        self.payload['options'][0]['claims'][0].pop('span_id')
        self.payload['options'][0]['claims'][0]['text'] = 'owning something else'
        with self.assertRaises(ValidationError): self.parse()

    def test_reference_text_mismatch(self):
        self.payload['options'][0]['claims'][0]['text'] = 'Incorrect text'
        with self.assertRaises(ValidationError): self.parse()

    def test_malformed_citation(self):
        self.payload['options'][0]['claims'][0]['citations'] = [dict(
            source_id='s0', quote='hello', basis='invented', subject='current_user')]
        with self.assertRaises(ValidationError): self.parse()

    def test_upper_bound(self):
        self.payload['options'][0]['claims'] *= 2
        with self.assertRaises(ValidationError): self.parse()

    def test_no_source_never_becomes_supported(self):
        got = ac.validate_assessments(self.payload, list(self.expected.values()), {})[0]
        self.assertEqual(got['status'], 'unsupported')
        self.assertTrue(all(c['status'] == 'unsupported' for c in got['claims']))


if __name__ == '__main__':
    unittest.main()
