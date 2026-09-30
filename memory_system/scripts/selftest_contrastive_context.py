"""Isolated context ranking checks; no provider calls or production mutations."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import answer_choice as candidate


class ContextRankingTests(unittest.TestCase):
    def test_qualifier_stays_attached_to_its_word(self):
        terms = candidate.contrastive_context_terms('C. Ask about vitamin D and type 2.')
        self.assertIn('vitamin d', terms)
        self.assertIn('type 2', terms)
        self.assertNotIn('c', terms)
        self.assertNotIn('d', terms)
        self.assertNotIn('2', terms)

    def test_qualifier_does_not_cross_line_or_sentence(self):
        for text in ['vitamin\nD', 'vitamin. D']:
            self.assertNotIn('vitamin d', candidate.contrastive_context_terms(text))

    def test_different_qualifiers_do_not_match(self):
        terms = candidate.contrastive_context_terms('Vitamin D supplement')
        match = candidate.local_context_match('Vitamin C supplement', terms)
        self.assertNotIn('vitamin d', match['shared_terms'])

    def test_distinct_detail_beats_shared_boilerplate(self):
        text = 'Often people discuss effects.  We discussed a vitamin D product.'
        terms = {'often', 'people', 'effects', 'vitamin', 'vitamin d'}
        match = candidate.local_context_match(text, terms,
            {'often': 1, 'people': 1, 'effects': 1, 'vitamin': 4, 'vitamin d': 4})
        self.assertEqual(match['excerpt'], 'We discussed a vitamin D product.')
        span = match['source_span']
        self.assertEqual(text[span['start']:span['end']], match['excerpt'])

    def test_distributed_details_cannot_form_one_match(self):
        match = candidate.local_context_match('Mountain activity. River visits.',
            {'mountain', 'river'}, {'mountain': 4, 'river': 4})
        self.assertEqual(len(match['shared_terms']), 1)

    def test_hidden_tail_does_not_receive_weight(self):
        match = candidate.local_context_match('padding ' * 130 + 'vitamin D',
            {'vitamin', 'vitamin d'}, {'vitamin': 4, 'vitamin d': 4})
        self.assertEqual(match['shared_terms'], [])


if __name__ == '__main__':
    unittest.main()
