"""Context pairing must not invent provenance, factual support or spare capacity."""
import copy
import unittest
from unittest.mock import patch
from probe_adjacent_context import propose
from app import answer_context, config, evidence_packet, schemas, search_pipeline


class AdjacentContextTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(config, 'PACKET_FINE_CONTEXT_RESERVE', False))
        self.parent = dict(request_id='r', message_index=0, source_event_id='u0', user_id='u',
                           role='user', content='Why is the shaded trail cooler?')
        self.reply = dict(request_id='r', message_index=1, source_event_id='a1', user_id='u',
                          role='assistant', content='Trees provide shade.')
        self.anchor = dict(id='reply', content='Retained original reply.', sources=[self.reply])
        self.candidate = dict(id='pair', _packet_item=dict(id='pair', content='Untrusted generated summary.',
            personal_evidence=True, coverage_ids=['option:0'], sources=[self.parent, self.reply]))

    def run_proposal(self, **kwargs):
        return propose([self.anchor], [self.candidate], user_id='u', top_k=10, byte_budget=4096, **kwargs)

    def test_adds_only_exact_parent_without_supplier_summary_or_factual_marks(self):
        before = copy.deepcopy((self.anchor, self.candidate))
        packet, trace = self.run_proposal()
        self.assertEqual(packet[0], self.anchor)
        self.assertEqual(packet[1]['sources'], [self.parent])
        self.assertNotIn('Untrusted generated summary', packet[1]['content'])
        self.assertFalse(packet[1]['personal_evidence'])
        self.assertEqual(packet[1]['coverage_ids'], [])
        self.assertEqual(packet[1]['equivalent_ids'], [])
        self.assertEqual(trace['appended_ids'], ['pair'])
        self.assertEqual((self.anchor, self.candidate), before)

    def test_absent_or_stale_eligible_supplier_cannot_be_added(self):
        self.assertEqual(propose([self.anchor], [], user_id='u', top_k=10, byte_budget=4096)[0], [self.anchor])
        self.candidate['view_status'] = 'stale'
        self.assertEqual(self.run_proposal()[0], [self.anchor])

    def test_cross_user_nonadjacent_different_request_and_extra_sources_rejected(self):
        original = copy.deepcopy(self.parent)
        for field, value in (('user_id','another'), ('request_id','other'), ('message_index',4), ('role','assistant')):
            with self.subTest(field=field):
                self.parent.update(original)
                self.parent[field] = value
                self.assertEqual(self.run_proposal()[0], [self.anchor])
        self.parent.update(original)
        self.candidate['_packet_item']['sources'].append(dict(self.parent, source_event_id='extra', message_index=3))
        self.assertEqual(self.run_proposal()[0], [self.anchor])

    def test_clipped_parent_and_different_assistant_fragment_rejected(self):
        self.parent['content_span'] = dict(start=3, end=3+len(self.parent['content']), original_length=100)
        self.assertEqual(self.run_proposal()[0], [self.anchor])
        self.parent.pop('content_span')
        self.candidate['_packet_item']['sources'][1] = dict(self.reply, content='A different fragment.')
        self.assertEqual(self.run_proposal()[0], [self.anchor])

    def test_existing_parent_is_not_duplicated_and_budgets_preserve_anchor(self):
        packet = [self.anchor, dict(id='user', content=self.parent['content'], sources=[self.parent])]
        self.assertEqual(propose(packet,[self.candidate],user_id='u',top_k=10,byte_budget=4096)[0], packet)
        for top_k, byte_budget in ((1,4096),(10,len(self.anchor['content'].encode()))):
            result, trace = propose([self.anchor],[self.candidate],user_id='u',top_k=top_k,byte_budget=byte_budget)
            self.assertEqual(result,[self.anchor])
            self.assertFalse(trace['appended_ids'])
        self.assertEqual(self.run_proposal(limit=0)[0],[self.anchor])

    def test_real_packet_entry_updates_digest_and_accounting_without_support_upgrade(self):
        req = schemas.SearchRequest(user_id='u', query='Where could I walk?', top_k=10, evidence_token_budget=4096)
        anchor = dict(id='reply', _packet_item=dict(self.anchor, memory_type='episode', personal_evidence=False,
            coverage_ids=[], source_count=1, content=answer_context.with_evidence('Conversation excerpt:', [self.reply])))
        candidate = copy.deepcopy(self.candidate)
        plan = dict(_cascade=dict(fine=dict(selected_ids=['reply','pair'])), _coverage_requirements=[],
                    _selection=dict(selected_ids=['reply']))
        before = copy.deepcopy((anchor,candidate))
        with patch.object(config,'PACKET_ADJACENT_CONTEXT',False):
            base, base_hash, base_manifest = search_pipeline._pack_evidence(
                None,req,copy.deepcopy(plan),[anchor],'2026-10-23T00:00:00Z',[anchor,candidate])
        with patch.object(config,'PACKET_ADJACENT_CONTEXT',True):
            result, digest, manifest = search_pipeline._pack_evidence(
                None,req,plan,[anchor],'2026-10-23T00:00:00Z',[anchor,candidate])
        self.assertEqual([item['id'] for item in result],['reply','pair'])
        self.assertEqual(evidence_packet.digest(result),digest)
        self.assertTrue(all(item['packet_hash']==digest for item in result))
        self.assertEqual(dict(result[0],packet_hash=base_hash),base[0])
        self.assertEqual(manifest['token_upper_bound'],sum(len(item['content'].encode()) for item in result)+1)
        self.assertEqual(manifest['evidence_count'],base_manifest['evidence_count']+1)
        for key in ('covered_requirements','missing_requirements','constraint_count','constraint_bytes','core_bytes'):
            self.assertEqual(manifest[key],base_manifest[key])
        self.assertEqual(manifest['conversation_context'],plan['_cascade']['conversation_context'])
        self.assertEqual(manifest['conversation_context']['base_packet_hash'],base_hash)
        self.assertEqual(result[1]['personal_evidence'],False)
        self.assertEqual(result[1]['coverage_ids'],[])
        self.assertEqual((anchor,candidate),before)

    def test_context_outside_fine_or_temporal_scope_is_not_added_by_packet_entry(self):
        req = schemas.SearchRequest(user_id='u',query='Where could I walk?',top_k=10,evidence_token_budget=4096)
        anchor = dict(id='reply',_packet_item=self.anchor)
        candidate = copy.deepcopy(self.candidate)
        plan = dict(_cascade=dict(fine=dict(selected_ids=['reply'])),_coverage_requirements=[])
        with patch.object(config,'PACKET_ADJACENT_CONTEXT',True):
            packet,_,_ = search_pipeline._pack_evidence(None,req,copy.deepcopy(plan),[anchor],None,[anchor,candidate])
            self.assertEqual([p['id'] for p in packet],['reply'])
            plan['_cascade']['fine']['selected_ids'].append('pair')
            with patch.object(search_pipeline,'_foresight_filter',side_effect=lambda rows,_: [r for r in rows if r['id']!='pair']):
                packet,_,_ = search_pipeline._pack_evidence(None,req,plan,[anchor],None,[anchor,candidate])
            self.assertEqual([p['id'] for p in packet],['reply'])


if __name__ == '__main__':
    unittest.main()
