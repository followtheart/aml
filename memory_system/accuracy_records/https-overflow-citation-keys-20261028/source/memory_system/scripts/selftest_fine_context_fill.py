"""Residual capacity preserves accepted context and strict source/budget bounds."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, evidence_packet, fine_context_fill, fine_context_lexical, schemas, search_pipeline
from selftest_fine_context_lexical import candidate


class ResidualContextTests(unittest.TestCase):
    def setUp(self):
        self.base = [schemas.SearchItem(id='base', content='Accepted evidence.', packet_hash_version=4).model_dump()]
        self.fine = [candidate('one', 'Walking?'), candidate('two', 'Gardening?'),
                     candidate('large', 'Long story ' * 700), candidate('tail', 'Short question about rivers?'),
                     candidate('tea', 'A note mentioning chamomile tea.'), candidate('last', 'Question about mountains?')]
        self.kw = dict(user_id='u', top_k=10, byte_budget=9000, reserve_bytes=1000, query='Chamomile tea?', options=[])

    def test_prefix_and_full_sources_survive_with_later_small_units(self):
        unchanged = copy.deepcopy((self.base, self.fine))
        prefix, old = fine_context_lexical.propose(self.base, self.fine, **self.kw)
        result, trace = fine_context_fill.propose(self.base, self.fine, **self.kw)
        self.assertEqual(result[:len(prefix)], prefix)
        self.assertEqual(trace['residual']['appended_ids'], ['tail', 'last'])
        self.assertEqual(trace['residual']['reserve_bytes'], 1000-old['added_bytes'])
        self.assertEqual(trace['prefix_reserve_ids'], ['one', 'two', 'tea'])
        self.assertLessEqual(trace['added_bytes'], 1000)
        for item in result[len(prefix):]:
            source = next(c for c in self.fine if c['id'] == item['id'])
            self.assertEqual(item['sources'], source['_packet_item']['sources'])
            self.assertNotIn('Generated summary', item['content'])
            self.assertFalse(item['personal_evidence'])
            self.assertEqual(item['coverage_ids'], [])
            self.assertEqual(item['score'], source['_ce_score'])
        self.assertEqual((self.base, self.fine), unchanged)

    def test_each_capacity_bound_preserves_the_prefix(self):
        prefix, old = fine_context_lexical.propose(self.base, self.fine, **self.kw)
        for override in (dict(top_k=len(prefix)), dict(byte_budget=old['total_bytes']),
                         dict(reserve_bytes=old['added_bytes'])):
            result, trace = fine_context_fill.propose(self.base, self.fine, **dict(self.kw, **override))
            self.assertEqual(result, prefix)
            self.assertEqual(trace['residual']['appended_ids'], [])

    def test_invalid_sources_and_duplicate_user_events_cannot_fill(self):
        for kind in ('foreign', 'invisible', 'stale', 'unscored', 'partial', 'assistant', 'constraint', 'duplicate'):
            fine = copy.deepcopy(self.fine[:2] + self.fine[4:5])
            bad = candidate('bad', 'Another source')
            s = bad['_packet_item']['sources'][0]
            if kind == 'foreign': s['user_id'] = 'other'
            if kind == 'invisible': bad['_packet_item']['content'] = 'Summary only'
            if kind == 'stale': bad['view_status'] = 'stale'
            if kind == 'unscored': bad['_ce_score'] = None
            if kind == 'partial': s['content_span'] = dict(start=1, end=15, original_length=20)
            if kind == 'assistant': s['role'] = 'assistant'
            if kind == 'constraint': bad['_packet_item']['is_constraint'] = True
            if kind == 'duplicate': bad = dict(copy.deepcopy(fine[0]), id='bad')
            prefix, _ = fine_context_lexical.propose(self.base, fine, **self.kw)
            result, _ = fine_context_fill.propose(self.base, fine+[bad], **self.kw)
            self.assertEqual(result, prefix, kind)

    def test_search_entry_dependency_gates_and_hashes(self):
        anchor = dict(id='base', _packet_item=self.base[0])
        req = schemas.SearchRequest(user_id='u', query='Chamomile tea?', top_k=10, evidence_token_budget=9000)
        plan = dict(_cascade=dict(fine=dict(selected_ids=['base']+[c['id'] for c in self.fine])),
                    _selection=dict(selected_ids=['base']), _coverage_requirements=[])
        for reserve, lexical in ((False, False), (False, True), (True, False), (True, True)):
            with patch.multiple(config, PACKET_ADJACENT_CONTEXT=False, PACKET_FINE_CONTEXT_RESERVE=reserve,
                                PACKET_FINE_CONTEXT_5K=True, PACKET_FINE_CONTEXT_LEXICAL=lexical):
                with patch.object(config, 'PACKET_FINE_CONTEXT_FILL', False):
                    prefix, _, _ = search_pipeline._pack_evidence(None, req, copy.deepcopy(plan), [anchor], None, [anchor, *self.fine])
                with patch.object(config, 'PACKET_FINE_CONTEXT_FILL', True):
                    result, digest, manifest = search_pipeline._pack_evidence(None, req, copy.deepcopy(plan), [anchor], None, [anchor, *self.fine])
                if not (reserve and lexical):
                    self.assertEqual(result, prefix)
                else:
                    self.assertGreater(len(result), len(prefix))
                    self.assertEqual([dict(p, packet_hash=None) for p in result[:len(prefix)]],
                                     [dict(p, packet_hash=None) for p in prefix])
                    self.assertEqual(manifest['conversation_context']['policy'], fine_context_fill.POLICY)
                self.assertEqual(evidence_packet.digest(result), digest)
                self.assertTrue(all(p['packet_hash'] == digest for p in result))


if __name__ == '__main__':
    unittest.main()
