"""Composition must preserve source identity, budgets and default behavior."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import combined_query_context, config, evidence_packet, search_pipeline
from selftest_source_first_residual import SourceFirstResidualTests as Fixture
from selftest_fine_context_lexical import candidate


class CombinedContextTests(unittest.TestCase):
    def setUp(self):
        self.f=Fixture();self.f.setUp()
        self.f.flags.update(PACKET_SOURCE_FIRST_RESIDUAL=False,PACKET_MARGINAL_QUERY_CONTEXT=False,
                            PACKET_QUERY_CONTEXT_COMBINED=True)
        self.f.plan['_cascade']['fine']['selection']=dict(
            soft_reservation_policy=dict(max_ce_rank=20),decisions=[dict(id='outside',rank=15)])

    def test_pipeline_composes_two_distinct_gains_without_fact_promotion(self):
        f=self.f;plan=copy.deepcopy(f.plan)
        residual=candidate('residual','Another original source.')
        plan['_cascade']['coarse']['selected_ids'].append('residual')
        plan['_query_specs'][0]['option_index']=0
        plan['_routes'][0]['candidates'].append(dict(id='residual',_query_ranks={'0':2}))
        with patch.multiple(config,**f.flags),patch('socket.socket.connect',side_effect=AssertionError('Offline')):
            result,digest,manifest=search_pipeline._pack_evidence(None,f.req,plan,
                [dict(id='base',_packet_item=f.base)],None,[*f.fine,f.extra,residual])
        self.assertEqual([p['id'] for p in result],['base','outside','one','two','tea','residual'])
        self.assertEqual(result[1]['sources'],f.extra['_packet_item']['sources'])
        self.assertEqual(result[-1]['sources'],residual['_packet_item']['sources'])
        for p in result[1:]:
            self.assertFalse(p['personal_evidence']);self.assertEqual(p['coverage_ids'],[])
        self.assertEqual(evidence_packet.digest(result),digest)
        self.assertTrue(all(p['packet_hash']==digest for p in result))
        ctx=manifest['conversation_context']
        self.assertEqual(ctx['policy'],combined_query_context.POLICY)
        self.assertEqual(ctx['prefix_context']['source_seat']['appended_ids'],['outside'])
        self.assertEqual(ctx['combined_residual']['appended_ids'],['residual'])
        self.assertLessEqual(ctx['added_bytes'],5120)

    def test_source_budget_and_top_k_remain_hard_limits(self):
        f=self.f
        for req in (f.req.model_copy(update=dict(top_k=2)),
                    f.req.model_copy(update=dict(evidence_token_budget=50))):
            result,_,manifest=f.pack(req=req)
            self.assertEqual(result[0]['content'],f.base['content'])
            self.assertLessEqual(len(result),req.top_k)
            self.assertLessEqual(manifest['token_upper_bound'],req.evidence_token_budget)
        extra=candidate('outside','Very long text. '*800)
        self.assertNotIn('outside',[p['id'] for p in f.pack(extra=extra)[0]])

    def test_unsafe_or_incomplete_sources_cannot_enter_either_stage(self):
        f=self.f
        for kind in ('foreign','stale','partial','unscored','constraint'):
            extra=copy.deepcopy(f.extra);s=extra['_packet_item']['sources'][0]
            if kind=='foreign':s['user_id']='someone_else'
            elif kind=='stale':extra['view_status']='stale'
            elif kind=='partial':s['content_span']=dict(start=1,end=4,original_length=99)
            elif kind=='unscored':extra['_ce_score']=None
            elif kind=='constraint':extra['_packet_item']['is_constraint']=True
            with self.subTest(kind=kind):self.assertNotIn('outside',[p['id'] for p in f.pack(extra=extra)[0]])

    def test_missing_dependencies_and_other_experiments_do_not_compose(self):
        f=self.f
        for name,value in [('PACKET_QUERY_CONTEXT_COMBINED',False),('PACKET_FINE_CONTEXT_RESERVE',False),
            ('PACKET_FINE_CONTEXT_5K',False),('PACKET_FINE_CONTEXT_LEXICAL',False),('PACKET_FINE_CONTEXT_FILL',True),
            ('PACKET_SOURCE_QUERY_CONTEXT',True),('PACKET_MARGINAL_QUERY_CONTEXT',True),
            ('PACKET_SOURCE_FIRST_RESIDUAL',True),('PACKET_QUESTION_CONTEXT_CE',True)]:
            with self.subTest(flag=name):
                self.assertNotEqual(f.pack(flags={name:value})[2]['conversation_context']['policy'],combined_query_context.POLICY)


if __name__=='__main__':unittest.main()
