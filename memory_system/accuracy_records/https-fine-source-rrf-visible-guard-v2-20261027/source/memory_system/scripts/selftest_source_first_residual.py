"""Original-question identity and residual-source provenance/capacity boundaries."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config,evidence_packet,schemas,search_pipeline,source_first_residual as context
from selftest_fine_context_lexical import candidate


class SourceFirstResidualTests(unittest.TestCase):
    def setUp(self):
        self.base=schemas.SearchItem(id='base',content='Accepted.',packet_hash_version=4).model_dump()
        self.fine=[candidate('one','Walking?'),candidate('two','Gardening?'),candidate('tea','Tea?')]
        self.extra=candidate('outside','Why do imported hybrid cars have different manufacturing histories?')
        self.extra['_ce_score']=.01
        self.req=schemas.SearchRequest(user_id='u',query='Tea?',options=['Personal claim.'],top_k=8,evidence_token_budget=9000)
        self.plan=dict(_coverage_requirements=[],_selection=dict(selected_ids=['base']),
            _cascade=dict(coarse=dict(selected_ids=['base','one','two','tea','outside']),
                fine=dict(selected_ids=['base','one','two','tea'])),
            _query_specs=[dict(origin='option_topic',text='Personal claim.'),dict(origin='question',text='Tea?')],
            _routes=[dict(channel='source_text',round=1,candidates=[dict(id='outside',_query_ranks={'1':1})])])
        self.flags=dict(PACKET_SOURCE_FIRST_RESIDUAL=True,PACKET_FINE_CONTEXT_RESERVE=True,
            PACKET_FINE_CONTEXT_5K=True,PACKET_FINE_CONTEXT_LEXICAL=True,PACKET_FINE_CONTEXT_FILL=False,
            PACKET_SOURCE_QUERY_CONTEXT=False,PACKET_QUESTION_CONTEXT_CE=False,PACKET_ADJACENT_CONTEXT=False)

    def pack(self,plan=None,extra=None,req=None,flags=None):
        with patch.multiple(config,**dict(self.flags,**(flags or {}))),patch('socket.socket.connect',side_effect=AssertionError('No network')):
            return search_pipeline._pack_evidence(None,req or self.req,copy.deepcopy(plan or self.plan),
                [dict(id='base',_packet_item=self.base)],None,[*self.fine,extra or self.extra])

    def test_preserves_three_seats_and_whole_original_sources(self):
        before=copy.deepcopy((self.base,self.fine,self.extra,self.plan))
        prefix=self.pack(flags=dict(PACKET_SOURCE_FIRST_RESIDUAL=False))
        packet,digest,manifest=self.pack()
        self.assertEqual([p['id'] for p in packet],['base','one','two','tea','outside'])
        self.assertEqual([dict(p,packet_hash=None) for p in packet[:-1]],
            [dict(p,packet_hash=None) for p in prefix[0]])
        self.assertEqual(packet[-1]['sources'],self.extra['_packet_item']['sources'])
        self.assertEqual(packet[-1]['score'],.01)
        self.assertFalse(packet[-1]['personal_evidence']);self.assertEqual(packet[-1]['coverage_ids'],[])
        self.assertNotIn('Generated summary',packet[-1]['content'])
        self.assertEqual(evidence_packet.digest(packet),digest)
        self.assertTrue(all(p['packet_hash']==digest for p in packet))
        self.assertEqual(manifest['token_upper_bound'],sum(len(p['content'].encode()) for p in packet)+len(packet)-1)
        self.assertEqual(manifest['conversation_context']['source_first']['reserve_bytes'],
            5120-prefix[2]['conversation_context']['added_bytes'])
        self.assertEqual((self.base,self.fine,self.extra,self.plan),before)

    def test_only_exact_original_question_first_hit_can_qualify(self):
        for kind in ('topic','subquery','followup_spec','changed_question','second_hit','boolean_rank','followup_route','missing_route','fine','selected','not_coarse'):
            plan=copy.deepcopy(self.plan);hit=plan['_routes'][0]['candidates'][0]
            if kind=='topic':hit['_query_ranks']={'0':1}
            elif kind=='subquery':plan['_query_specs'][1]['origin']='sub_queries'
            elif kind=='followup_spec':plan['_query_specs'][1]['origin']='grounded_followup'
            elif kind=='changed_question':plan['_query_specs'][1]['text']='Different question'
            elif kind=='second_hit':hit['_query_ranks']={'1':2}
            elif kind=='boolean_rank':hit['_query_ranks']={'1':True}
            elif kind=='followup_route':plan['_routes'][0]['round']=2
            elif kind=='missing_route':plan['_routes']=[]
            elif kind=='fine':plan['_cascade']['fine']['selected_ids'].append('outside')
            elif kind=='selected':plan['_selection']['selected_ids'].append('outside')
            elif kind=='not_coarse':plan['_cascade']['coarse']['selected_ids'].remove('outside')
            with self.subTest(kind=kind):
                self.assertEqual(self.pack(plan=plan)[2]['conversation_context']['source_first']['appended_ids'],[])

    def test_invalid_sources_never_become_context(self):
        for kind in ('foreign','stale','invisible','partial','assistant','constraint','unscored','nonfinite','duplicate'):
            extra=copy.deepcopy(self.extra);s=extra['_packet_item']['sources'][0]
            if kind=='foreign':s['user_id']='someone_else'
            elif kind=='stale':extra['view_status']='stale'
            elif kind=='invisible':extra['_packet_item']['content']='Summary only'
            elif kind=='partial':s['content_span']=dict(start=1,end=10,original_length=90)
            elif kind=='assistant':s['role']='assistant'
            elif kind=='constraint':extra['_packet_item']['is_constraint']=True
            elif kind=='unscored':extra['_ce_score']=None
            elif kind=='nonfinite':extra['_ce_score']=float('nan')
            elif kind=='duplicate':extra['_packet_item']=copy.deepcopy(self.fine[0]['_packet_item'])
            with self.subTest(kind=kind):
                self.assertNotIn('outside',[p['id'] for p in self.pack(extra=extra)[0]])

    def test_capacity_failure_keeps_original_packet_and_digest(self):
        original=self.pack(flags=dict(PACKET_SOURCE_FIRST_RESIDUAL=False))
        for req in (self.req.model_copy(update=dict(top_k=len(original[0]))),
                    self.req.model_copy(update=dict(evidence_token_budget=original[2]['token_upper_bound']))):
            prefix=self.pack(req=req,flags=dict(PACKET_SOURCE_FIRST_RESIDUAL=False))
            result=self.pack(req=req)
            self.assertEqual(result[:2],prefix[:2])
        extra=copy.deepcopy(self.extra);text='Huge source. '*600
        extra=candidate('outside',text)
        result=self.pack(extra=extra)
        self.assertEqual(result[:2],original[:2])

    def test_dependencies_and_other_context_experiments_gate_the_feature(self):
        for key,value in [('PACKET_SOURCE_FIRST_RESIDUAL',False),('PACKET_FINE_CONTEXT_RESERVE',False),
            ('PACKET_FINE_CONTEXT_5K',False),('PACKET_FINE_CONTEXT_LEXICAL',False),('PACKET_FINE_CONTEXT_FILL',True),
            ('PACKET_SOURCE_QUERY_CONTEXT',True),('PACKET_QUESTION_CONTEXT_CE',True)]:
            with self.subTest(key=key):
                self.assertNotIn('source_first',self.pack(flags={key:value})[2]['conversation_context'])

    def test_scope_and_hash_guards(self):
        plan=copy.deepcopy(self.plan);plan['time_scope']={'from':'2025-01-01T00:00:00Z'}
        extra=copy.deepcopy(self.extra)
        extra.update(type='plan',temporal=dict(start='2020-01-01T00:00:00Z',end='2020-01-02T00:00:00Z'))
        self.assertNotIn('outside',[p['id'] for p in self.pack(plan=plan,extra=extra)[0]])
        prefix=self.pack(flags=dict(PACKET_SOURCE_FIRST_RESIDUAL=False))
        with self.assertRaises(ValueError):
            context.attach(prefix[0],'wrong',prefix[2],[self.extra],{},user_id='u',top_k=8)


if __name__=='__main__':unittest.main()
