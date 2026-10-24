"""Marginal ordering, literal query provenance and packet boundary checks."""
import copy
import unittest

from selftest_source_first_residual import SourceFirstResidualTests
from app import evidence_packet,marginal_query_context as context,retrieval_queries


class MarginalQueryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=SourceFirstResidualTests()
        self.fixture.setUp()
        self.fixture.flags.update(PACKET_SOURCE_FIRST_RESIDUAL=False,PACKET_MARGINAL_QUERY_CONTEXT=True)

    def nominate(self,pool,packed=(),**changes):
        f=self.fixture
        kw=dict(routes=f.plan['_routes'],query_specs=f.plan['_query_specs'],query=f.req.query,
            options=f.req.options,user_id='u',coarse_ids=list(pool),selected_ids=[])
        kw.update(changes)
        return context.nominate(pool,packed,**kw)

    def test_pipeline_preserves_prefix_and_source_integrity(self):
        f=self.fixture
        before=f.pack(flags=dict(PACKET_MARGINAL_QUERY_CONTEXT=False))
        packet,digest,manifest=f.pack()
        self.assertEqual([p['id'] for p in packet],['base','one','two','tea','outside'])
        self.assertEqual([dict(p,packet_hash=None) for p in packet[:-1]],
                         [dict(p,packet_hash=None) for p in before[0]])
        self.assertEqual(packet[-1]['sources'],f.extra['_packet_item']['sources'])
        self.assertFalse(packet[-1]['personal_evidence']);self.assertEqual(packet[-1]['coverage_ids'],[])
        self.assertEqual(evidence_packet.digest(packet),digest)
        self.assertTrue(all(p['packet_hash']==digest for p in packet))
        ctx=manifest['conversation_context']
        self.assertEqual(ctx['policy'],context.POLICY)
        self.assertNotIn('source_first',ctx)
        self.assertEqual(ctx['marginal_query']['reserve_bytes'],5120-ctx['prefix_context']['added_bytes'])

    def test_exact_source_equivalence_discounts_duplicate_route_hits(self):
        f=self.fixture
        one=copy.deepcopy(f.extra);one['id']='duplicate'
        second=copy.deepcopy(f.fine[0]);second['id']='novel'
        specs=[dict(origin='question',text=f.req.query),dict(origin='option_topic',text=f.req.options[0],option_index=0)]
        route=[dict(channel='source_text',round=1,candidates=[
            dict(id='duplicate',_query_ranks={'0':1}),dict(id='novel',_query_ranks={'0':2,'1':2})])]
        got,trace=self.nominate({'duplicate':one,'novel':second},[dict(f.extra['_packet_item'],id='delivered')],routes=route,query_specs=specs)
        self.assertEqual([c['id'] for c in got],['novel'])
        self.assertEqual(trace['covered'],{'0':1.0,'1':0.0})
        self.assertEqual(trace['nominations'][0]['score'],.5)
        self.assertEqual(set(trace['nominations'][0]['gains']),{'1'})

    def test_only_exact_literal_queries_and_valid_ranks_qualify(self):
        f=self.fixture
        for kind in ('premise','wrong_topic','bad_index','followup','boolean','zero','negative','float','late_route'):
            specs=[dict(origin='option_topic',text=retrieval_queries.topic_anchor(f.req.options[0]),option_index=0)]
            route=[dict(channel='source_text',round=1,candidates=[dict(id='outside',_query_ranks={'0':1})])]
            if kind=='premise':specs[0]['origin']='option_premise'
            elif kind=='wrong_topic':specs[0]['text']='invented topic'
            elif kind=='bad_index':specs[0]['option_index']=True
            elif kind=='followup':specs[0]['origin']='grounded_followup'
            elif kind=='late_route':route[0]['round']=2
            else:route[0]['candidates'][0]['_query_ranks']['0']={'boolean':True,'zero':0,'negative':-1,'float':1.0}[kind]
            with self.subTest(kind=kind):
                self.assertEqual(self.nominate({'outside':f.extra},routes=route,query_specs=specs)[0],[])

    def test_option_duplicates_do_not_multiply_scores_and_ties_keep_route_order(self):
        f=self.fixture;other=copy.deepcopy(f.extra);other['id']='other'
        topic=dict(origin='option_topic',text=retrieval_queries.topic_anchor(f.req.options[0]),option_index=0)
        route=[dict(channel='source_text',round=1,candidates=[
            dict(id='other',_query_ranks={'0':2}),dict(id='outside',_query_ranks={'0':2,'1':2})])]
        got,trace=self.nominate({'outside':f.extra,'other':other},routes=route,query_specs=[topic,dict(topic)])
        self.assertEqual([c['id'] for c in got],['other','outside'])
        self.assertEqual([r['score'] for r in trace['nominations']],[.5,.5])

    def test_ineligible_sources_cannot_be_added(self):
        f=self.fixture
        for kind in ('foreign','stale','partial','invisible','unscored','nan','constraint'):
            extra=copy.deepcopy(f.extra);s=extra['_packet_item']['sources'][0]
            if kind=='foreign':s['user_id']='other-user'
            elif kind=='stale':extra['view_status']='stale'
            elif kind=='partial':s['content_span']=dict(start=0,end=4,original_length=100)
            elif kind=='invisible':extra['_packet_item']['content']='summary only'
            elif kind=='unscored':extra['_ce_score']=None
            elif kind=='nan':extra['_ce_score']=float('nan')
            elif kind=='constraint':extra['_packet_item']['is_constraint']=True
            with self.subTest(kind=kind):self.assertNotIn('outside',[p['id'] for p in f.pack(extra=extra)[0]])

    def test_item_byte_and_hash_guards(self):
        f=self.fixture;prefix=f.pack(flags=dict(PACKET_MARGINAL_QUERY_CONTEXT=False))
        for req in (f.req.model_copy(update=dict(top_k=len(prefix[0]))),
                    f.req.model_copy(update=dict(evidence_token_budget=prefix[2]['token_upper_bound']))):
            self.assertEqual(f.pack(req=req)[:2],f.pack(req=req,flags=dict(PACKET_MARGINAL_QUERY_CONTEXT=False))[:2])
        with self.assertRaises(ValueError):
            context.attach(prefix[0],'invalid',prefix[2],[f.extra],{},user_id='u',top_k=8)

    def test_dependencies_and_competing_experiments(self):
        for key,value in [('PACKET_MARGINAL_QUERY_CONTEXT',False),('PACKET_FINE_CONTEXT_RESERVE',False),
            ('PACKET_FINE_CONTEXT_5K',False),('PACKET_FINE_CONTEXT_LEXICAL',False),('PACKET_FINE_CONTEXT_FILL',True),
            ('PACKET_SOURCE_QUERY_CONTEXT',True),('PACKET_QUESTION_CONTEXT_CE',True),('PACKET_SOURCE_FIRST_RESIDUAL',True)]:
            with self.subTest(key=key):
                self.assertNotIn('marginal_query',self.fixture.pack(flags={key:value})[2]['conversation_context'])


if __name__=='__main__':unittest.main()
