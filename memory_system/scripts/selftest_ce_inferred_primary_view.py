"""Preserve literal source boundaries, role/time labels, rules and support."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context
from ce_inferred_primary_view import document, view, INFERRED


def example():
    primary = 'The user probably enjoys outdoor gatherings with friends.'
    prefix = '[event: 2026] ' + INFERRED
    support = '\n[support: another] The user asked about park opening hours.'
    sources = [dict(role='assistant', timestamp='2026-01-02T12:00:00Z',
                    content='Quoted text contains\nSource messages:\n🙂 but is still one source.'),
               dict(role='user', content='What time does the park open?')]
    body = prefix + primary + support
    return dict(id='one', type='fact', epistemic_status='inferred', content=primary,
                _packet_item=dict(_evidence_body=body, sources=sources),
                _rank_text=answer_context.with_evidence(body, sources))


class Checks(unittest.TestCase):
    def test_only_primary_removed_with_literal_suffix_support_and_labels_preserved(self):
        row = example(); before = copy.deepcopy(row)
        actual, info = view(row)
        body = row['_packet_item']['_evidence_body']
        expected_body = body.replace(row['content'], 'Conversation excerpt:', 1)
        self.assertEqual(actual, expected_body + row['_rank_text'][len(body):])
        self.assertTrue(info['changed'])
        self.assertEqual(row, before)

    def test_explicit_or_protected_claims_unchanged(self):
        for changes in ({'epistemic_status': 'asserted'}, {'type': 'rule'}, {'_user_rule': True}):
            row = example(); row.update(changes)
            self.assertEqual(document(row), row['_rank_text'])
        row = example(); row['_packet_item']['is_constraint'] = True
        self.assertEqual(document(row), row['_rank_text'])

    def test_no_rehydration_for_excerpt_or_missing_sources(self):
        for key in ('_rank_excerpted', '_body_excerpted'):
            row = example(); row[key] = True
            self.assertEqual(document(row), row['_rank_text'])
        row = example(); row['_packet_item']['sources'] = []
        self.assertEqual(document(row), row['_rank_text'])

    def test_modified_source_suffix_rejected(self):
        row = example(); row['_rank_text'] += 'extra hidden text'
        self.assertEqual(document(row), row['_rank_text'])

    def test_primary_boundary_not_guessed(self):
        row = example(); row['content'] = '2026'
        self.assertEqual(document(row), row['_rank_text'])
        row = example()
        row['_packet_item']['_evidence_body'] = '[event: 2026] ' + INFERRED + row['content'] + '\nUnlabelled context'
        row['_rank_text'] = answer_context.with_evidence(row['_packet_item']['_evidence_body'], row['_packet_item']['sources'])
        self.assertEqual(document(row), row['_rank_text'])


if __name__ == '__main__':
    unittest.main()
