"""Recall diversity and topic hypotheses must not become factual support."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, listwise_contract, retrieval_queries, schemas, store
from evaluate_recall_context import reference_turns


class TopicAnchors(unittest.TestCase):
    def build(self, option, proposed):
        plan = {'option_queries': [proposed], 'sub_queries': ['What training fits?']}
        with patch.object(config, 'RECALL_TOPIC_ANCHORS', True):
            specs = retrieval_queries.build('What training fits?', [option], plan)
        return specs, plan

    def test_activity_survives_embedded_identity(self):
        specs, plan = self.build('For mountain cycling, use a route matching your work as a teacher.',
                                 'your work as a teacher')
        topics = [q for q in specs if q['origin'] == 'option_topic']
        self.assertEqual(len(topics), 1)
        self.assertIn('mountain cycling', topics[0]['text'])
        self.assertEqual(topics[0]['coverage_ids'], [])
        self.assertNotIn('option_topic', [r['origin'] for r in plan['_coverage_requirements']])
        self.assertTrue(any('teacher' in q['text'] for q in specs if q['origin'] == 'option_premise'))

    def test_later_detail_after_premise_is_searchable(self):
        specs, _ = self.build('Since you prefer warm drinks, try a cup of rooibos with cinnamon.',
                              'Since you prefer warm drinks')
        self.assertTrue(any('rooibos' in q['text'] for q in specs))

    def test_negation_and_conditions_are_not_rewritten_as_facts(self):
        option = 'If you do not own a bicycle, consider renting one for gravel rides.'
        specs, _ = self.build(option, 'If you do not own a bicycle')
        for q in specs:
            if q['origin'] == 'option_topic':
                self.assertEqual(q['text'], option)
                self.assertFalse(q['coverage_ids'])

    def test_bounded_and_optional(self):
        options = [f'For activity{i}, consider equipment{i} and preparation{i}.' for i in range(20)]
        with patch.multiple(config, RECALL_TOPIC_ANCHORS=True, RECALL_TOPIC_ANCHOR_LIMIT=2):
            on = retrieval_queries.build('What next?', options, {'option_queries': [f'equipment{i}' for i in range(20)]})
        self.assertEqual(sum(q['origin'] == 'option_topic' for q in on), 2)
        self.assertTrue(all(len(q['text']) <= 240 for q in on))
        with patch.object(config, 'RECALL_TOPIC_ANCHORS', False):
            off = retrieval_queries.build('What next?', options, {'option_queries': [''] * 20})
        self.assertEqual(len(off), 1)


class SourceDiversity(unittest.TestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.addCleanup(self.st.conn.close)
        self.st.save_messages(schemas.AddRequest(request_id='r', user_id='u', session_id='s', messages=[
            schemas.Message(role='user', content='OrchidLedger observation one.'),
            schemas.Message(role='user', content='OrchidLedger observation two.'),
            schemas.Message(role='user', content='OrchidLedger private.', sensitivity='sensitive'),
            schemas.Message(role='user', content='OrchidLedger suppressed.', sensitivity='suppressed')]))
        self.ids = []
        for i in range(7):
            mid = self.st.insert_amu(user_id='u', session_id='s', content='OrchidLedger', type='fact')
            # Stable IDs make the crowding regression independent of UUID order.
            self.st.conn.execute('UPDATE amu SET id=? WHERE id=?', (f'm{i}', mid))
            self.st.link_sources(f'm{i}', 'r', [0 if i < 4 else i - 3])
            self.ids.append(f'm{i}')

    def test_diversity_before_per_query_cutoff(self):
        with patch.object(config, 'SOURCE_RECALL_DIVERSITY', False):
            old = self.st.source_search('u', 'OrchidLedger', 2)
        with patch.object(config, 'SOURCE_RECALL_DIVERSITY', True):
            new = self.st.source_search('u', 'OrchidLedger', 2)
        self.assertEqual([m['id'] for m in old], ['m0', 'm1'])
        self.assertEqual([m['id'] for m in new], ['m0', 'm4'])

    def test_privacy_history_and_fill_are_preserved(self):
        with patch.object(config, 'SOURCE_RECALL_DIVERSITY', True):
            rows = self.st.source_search('u', 'OrchidLedger', 20)
            self.assertEqual({m['id'] for m in rows}, set(self.ids[:5]))
            self.assertEqual(self.st.source_search('other', 'OrchidLedger', 20), [])
            self.st.conn.execute("UPDATE amu SET valid_to='2020-01-01' WHERE id='m4'")
            self.assertNotIn('m4', {m['id'] for m in self.st.source_search('u', 'OrchidLedger', 20)})
            self.assertIn('m4', {m['id'] for m in self.st.source_search('u', 'OrchidLedger', 20, include_history=True)})
            self.assertNotIn('m6', {m['id'] for m in self.st.source_search('u', 'OrchidLedger', 20, include_sensitive=True)})


class ContextContract(unittest.TestCase):
    def test_context_retained_without_support_or_score_promotion(self):
        state = listwise_contract.State(3, keyed=True)
        state.accept({'candidates': {'c0': {'priority': 50, 'relevance': 'context'},
                                    'c1': {'priority': 1, 'relevance': 'useful'},
                                    'c2': {'priority': 99, 'relevance': 'irrelevant'}}, 'groups': []})
        self.assertEqual(state.require_valid(), ([1, 0, 2], {2}, []))
        self.assertEqual(set(state.accepted_candidates['c0']), {'priority', 'relevance'})
        enums = listwise_contract.schema(1, keyed=True, context_label=True)['properties']['candidates']['properties']['c0']['properties']['relevance']['enum']
        self.assertIn('context', enums)


class ReferenceIdentity(unittest.TestCase):
    def test_repeated_question_cannot_borrow_another_historys_snippet(self):
        qa = dict(question='What should I try?', answer='A')
        row = dict(user_query="{'content': 'What should I try?'}", correct_answer='A',
                   chat_history_32k_link='history-a', chat_history_128k_link='history-a-long',
                   related_conversation_snippet='[{"role":"user","content":"I enjoy sailing."}]')
        self.assertEqual(reference_turns(qa, 'history-a', row), ['i enjoy sailing.'])
        with self.assertRaises(ValueError):
            reference_turns(qa, 'history-b', row)
        with self.assertRaises(ValueError):
            reference_turns(dict(qa, answer='B'), 'history-a', row)


if __name__ == '__main__':
    unittest.main()
