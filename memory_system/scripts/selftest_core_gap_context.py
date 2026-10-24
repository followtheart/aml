"""Core-gap packet entry: complete provenance, separate cap and dependency gates."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, core_gap_context as context, evidence_packet, schemas, search_pipeline
from selftest_fine_context_lexical import candidate


class CoreGapTests(unittest.TestCase):
    def setUp(self):
        self.base = schemas.SearchItem(id='base', content='Accepted evidence.', packet_hash_version=4).model_dump()
        self.fine = [candidate('one', 'Walking. ' + 'a'*1690),
                     candidate('two', 'Gardening. ' + 'b'*1690),
                     candidate('tea', 'Tea? ' + 'c'*1490)]
        self.extra = candidate('outside', 'Full original context. ' + 'd'*800)
        self.req = schemas.SearchRequest(user_id='u', query='Tea?', options=[], top_k=8, evidence_token_budget=12000)
        self.plan = dict(_coverage_requirements=[], _selection=dict(selected_ids=['base']),
            _cascade=dict(coarse=dict(selected_ids=['base','one','two','tea','outside']),
                fine=dict(selected_ids=['base','one','two','tea'], selection=dict(
                    decisions=[dict(id='outside', rank=1)]))))
        self.flags = dict(PACKET_CORE_GAP_CONTEXT=True, PACKET_FINE_CONTEXT_RESERVE=True,
            PACKET_FINE_CONTEXT_5K=True, PACKET_FINE_CONTEXT_LEXICAL=True,
            **{key:False for key in context.INCOMPATIBLE})

    def pack(self, *, flags=None, extra=None, plan=None, req=None):
        with patch.multiple(config, **dict(self.flags, **(flags or {}))):
            return search_pipeline._pack_evidence(None, req or self.req,
                copy.deepcopy(plan or self.plan), [dict(id='base',_packet_item=self.base)], None,
                [dict(id='base',_packet_item=self.base), *self.fine, extra or self.extra])

    def test_separate_allowance_preserves_every_original_item_and_source(self):
        frozen = copy.deepcopy((self.base, self.fine, self.extra, self.plan))
        before = self.pack(flags=dict(PACKET_CORE_GAP_CONTEXT=False))
        after = self.pack()
        self.assertEqual([p['id'] for p in before[0]], ['base','one','two','tea'])
        self.assertLess(5120-before[2]['conversation_context']['added_bytes'], len(after[0][-1]['content'].encode()))
        self.assertEqual([dict(p,packet_hash=None) for p in after[0][:-1]],
                         [dict(p,packet_hash=None) for p in before[0]])
        item = after[0][-1]
        self.assertEqual(item['sources'], self.extra['_packet_item']['sources'])
        self.assertFalse(item['personal_evidence'])
        self.assertEqual(item['coverage_ids'], [])
        self.assertNotIn('Generated summary', item['content'])
        self.assertEqual(after[2]['conversation_context']['reserve_bytes'], 9216)
        self.assertEqual(after[2]['conversation_context']['prefix_context'], before[2]['conversation_context'])
        self.assertEqual(evidence_packet.digest(after[0]), after[1])
        self.assertTrue(all(p['packet_hash']==after[1] for p in after[0]))
        self.assertEqual((self.base,self.fine,self.extra,self.plan), frozen)

    def test_rank_and_current_pool_boundaries(self):
        for kind in ('rank17','boolean_rank','unranked','not_coarse','already_fine','already_delivered'):
            plan = copy.deepcopy(self.plan)
            if kind=='rank17': plan['_cascade']['fine']['selection']['decisions'][0]['rank']=17
            if kind=='boolean_rank': plan['_cascade']['fine']['selection']['decisions'][0]['rank']=True
            if kind=='unranked': plan['_cascade']['fine']['selection']['decisions']=[]
            if kind=='not_coarse': plan['_cascade']['coarse']['selected_ids'].remove('outside')
            if kind=='already_fine': plan['_cascade']['fine']['selected_ids'].append('outside')
            extra = copy.deepcopy(self.extra)
            if kind=='already_delivered': extra['_packet_item']=copy.deepcopy(self.fine[0]['_packet_item'])
            with self.subTest(kind=kind):
                self.assertEqual(self.pack(plan=plan,extra=extra)[2]['conversation_context']['core_gap']['appended_ids'], [])

    def test_ineligible_sources_are_not_rehabilitated(self):
        for kind in ('foreign','stale','invisible','partial','assistant','constraint','unscored','nonfinite'):
            extra = copy.deepcopy(self.extra); source=extra['_packet_item']['sources'][0]
            if kind=='foreign': source['user_id']='another-user'
            if kind=='stale': extra['view_status']='stale'
            if kind=='invisible': extra['_packet_item']['content']='Summary only'
            if kind=='partial': source['content_span']=dict(start=1,end=10,original_length=900)
            if kind=='assistant': source['role']='assistant'
            if kind=='constraint': extra['_packet_item']['is_constraint']=True
            if kind=='unscored': extra['_ce_score']=None
            if kind=='nonfinite': extra['_ce_score']=float('nan')
            with self.subTest(kind=kind):
                self.assertEqual(self.pack(extra=extra)[2]['conversation_context']['core_gap']['appended_ids'], [])

    def test_path_dependency_must_already_be_delivered(self):
        for dependency, expected in [('missing',False),('one',True)]:
            extra=copy.deepcopy(self.extra);extra['_relation_path_ids']=['outside',dependency]
            self.assertEqual(bool(self.pack(extra=extra)[2]['conversation_context']['core_gap']['appended_ids']), expected)

    def test_item_global_bytes_and_extra_bytes_are_all_binding(self):
        original = self.pack(flags=dict(PACKET_CORE_GAP_CONTEXT=False))
        for req in (self.req.model_copy(update=dict(top_k=len(original[0]))),
                    self.req.model_copy(update=dict(evidence_token_budget=original[2]['token_upper_bound']))):
            self.assertEqual(self.pack(req=req)[:2], self.pack(req=req,flags=dict(PACKET_CORE_GAP_CONTEXT=False))[:2])
        self.assertEqual(self.pack(extra=candidate('outside', '界'*1400))[:2], original[:2])

    def test_optional_gates_and_scope(self):
        for key,value in [('PACKET_CORE_GAP_CONTEXT',False),('PACKET_FINE_CONTEXT_RESERVE',False),
                          ('PACKET_FINE_CONTEXT_5K',False),('PACKET_FINE_CONTEXT_LEXICAL',False),
                          *[(k,True) for k in context.INCOMPATIBLE]]:
            with patch.multiple(config, **dict(self.flags, **{key:value})):
                self.assertFalse(context.enabled())
        plan=copy.deepcopy(self.plan)
        plan['time_scope']={'from':'2025-01-01T00:00:00Z'}
        extra=copy.deepcopy(self.extra)
        extra.update(type='plan',temporal=dict(start='2020-01-01T00:00:00Z',end='2020-01-02T00:00:00Z'))
        self.assertEqual(self.pack(extra=extra,plan=plan)[2]['conversation_context']['core_gap']['appended_ids'], [])

    def test_bad_prefix_digest_rejected(self):
        packet, digest, manifest = self.pack(flags=dict(PACKET_CORE_GAP_CONTEXT=False))
        with self.assertRaises(ValueError):
            context.attach(packet, 'wrong', manifest, [self.extra], [], user_id='u', top_k=8)


if __name__ == '__main__':
    unittest.main()
