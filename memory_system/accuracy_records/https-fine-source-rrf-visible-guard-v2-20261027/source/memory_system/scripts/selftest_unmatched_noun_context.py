"""Original-source novelty, synonym boundaries and production packet integration."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from selftest_fine_context_lexical import candidate
from app import answer_context, config, evidence_packet, unmatched_noun_context as context, schemas, search_pipeline


class NounContextTests(unittest.TestCase):
    def setUp(self):
        self.base=schemas.SearchItem(id='base',content='Accepted evidence.',packet_hash_version=4).model_dump()
        self.fine=[candidate('one','Walking. '+'a'*900),candidate('two','Gardening. '+'b'*900)]
        self.extra=candidate('extra','Why do old movies imply major events?')
        self.extra['_packet_item']['sources']+=candidate('reply','This reflects censorship.',role='assistant')['_packet_item']['sources']
        self.refresh(self.extra)
        self.req=schemas.SearchRequest(user_id='u',query='Recommend films.',options=['A. A comedy.','B. A drama.'],top_k=8,evidence_token_budget=12000)
        self.plan=dict(_coverage_requirements=[],_selection=dict(selected_ids=['base']),
            _cascade=dict(coarse=dict(selected_ids=['base','one','two','extra']),fine=dict(selected_ids=['base','one','two'])))
        self.flags=dict(PACKET_UNMATCHED_NOUN_CONTEXT=True,PACKET_FINE_CONTEXT_RESERVE=True,
            PACKET_FINE_CONTEXT_5K=True,PACKET_FINE_CONTEXT_LEXICAL=True,**{k:False for k in context.INCOMPATIBLE})

    def refresh(self,c):
        c['_packet_item']['content']=answer_context.with_evidence('Original conversation:',c['_packet_item']['sources'])

    def pack(self,extras=None,plan=None,req=None,flags=None):
        with patch.multiple(config,**dict(self.flags,**(flags or {}))):
            return search_pipeline._pack_evidence(None,req or self.req,copy.deepcopy(plan or self.plan),
                [dict(id='base',_packet_item=self.base,_final=.9)],'2026-01-01T00:00:00Z',
                self.fine+([self.extra] if extras is None else extras))

    def test_whole_sources_original_prefix_and_no_fact_promotion(self):
        before=copy.deepcopy((self.base,self.fine,self.extra,self.plan))
        off=self.pack(flags=dict(PACKET_UNMATCHED_NOUN_CONTEXT=False));on=self.pack()
        self.assertEqual(on[0][-1]['id'],'extra')
        self.assertEqual(on[0][-1]['sources'],self.extra['_packet_item']['sources'])
        self.assertFalse(on[0][-1]['personal_evidence']);self.assertEqual(on[0][-1]['coverage_ids'],[])
        self.assertEqual([dict(p,packet_hash=None) for p in on[0][:-1]],[dict(p,packet_hash=None) for p in off[0]])
        self.assertEqual(on[2]['conversation_context']['prefix_context'],off[2]['conversation_context'])
        self.assertEqual(evidence_packet.digest(on[0]),on[1])
        self.assertEqual(before,(self.base,self.fine,self.extra,self.plan))

    def test_literal_matches_and_option_only_matches_do_not_qualify(self):
        for req in (self.req.model_copy(update=dict(query='Recommend movies.')),
                    self.req.model_copy(update=dict(query='Recommend tea.',options=['A. Films.','B. Movies.']))):
            self.assertFalse(self.pack(req=req)[2]['conversation_context']['unmatched_noun']['appended_ids'])

    def test_matching_source_itself_must_be_novel(self):
        self.fine[0]=copy.deepcopy(self.extra);self.fine[0]['id']='one'
        self.extra['_packet_item']['sources']+=candidate('other','I like tea.')['_packet_item']['sources']
        self.refresh(self.extra)
        self.assertFalse(self.pack()[2]['conversation_context']['unmatched_noun']['appended_ids'])

    def test_partial_matching_source_cannot_borrow_completeness(self):
        s=self.extra['_packet_item']['sources'][0]
        s['content_span']=dict(start=1,end=1+len(s['content']),original_length=100)
        self.extra['_packet_item']['sources']+=candidate('other','I like tea.')['_packet_item']['sources']
        self.refresh(self.extra)
        self.assertFalse(self.pack()[2]['conversation_context']['unmatched_noun']['appended_ids'])

    def test_duplicate_projections_do_not_change_corpus_weight(self):
        clone=copy.deepcopy(self.extra);clone['id']='clone';clone['_ce_score']=.01
        plan=copy.deepcopy(self.plan);plan['_cascade']['coarse']['selected_ids'].append('clone')
        a=self.pack();b=self.pack(extras=[self.extra,clone],plan=plan)
        self.assertEqual(a[0],b[0])
        x=a[2]['conversation_context']['unmatched_noun']['nomination']
        y=b[2]['conversation_context']['unmatched_noun']['nomination']
        self.assertEqual(x['corpus_observations'],y['corpus_observations'])
        self.assertEqual(x['nominations'][0]['score'],next(n['score'] for n in y['nominations'] if n['candidate_id']=='extra'))

    def test_ineligible_source_and_dependency_boundaries(self):
        for kind in ('foreign','hidden','stale','assistant','nan','constraint','dependency','scope'):
            c=copy.deepcopy(self.extra);plan=copy.deepcopy(self.plan)
            if kind=='foreign':c['_packet_item']['sources'][0]['user_id']='other'
            if kind=='hidden':c['_packet_item']['content']='Summary only.'
            if kind=='stale':c['view_status']='stale'
            if kind=='assistant':c['_packet_item']['sources'][0]['role']='assistant'
            if kind=='nan':c['_ce_score']=float('nan')
            if kind=='constraint':c['_packet_item']['is_constraint']=True
            if kind=='dependency':c['_relation_path_ids']=['extra','missing']
            if kind=='scope':
                plan['time_scope']={'from':'2025-01-01T00:00:00Z'}
                c.update(type='plan',temporal=dict(start='2020-01-01T00:00:00Z',end='2020-01-02T00:00:00Z'))
            with self.subTest(kind=kind):
                self.assertFalse(self.pack(extras=[c],plan=plan)[2]['conversation_context']['unmatched_noun']['appended_ids'])

    def test_item_and_byte_caps(self):
        off=self.pack(flags=dict(PACKET_UNMATCHED_NOUN_CONTEXT=False))
        for req in (self.req.model_copy(update=dict(top_k=len(off[0]))),
                    self.req.model_copy(update=dict(evidence_token_budget=off[2]['token_upper_bound']))):
            self.assertEqual(self.pack(req=req)[:2],self.pack(req=req,flags=dict(PACKET_UNMATCHED_NOUN_CONTEXT=False))[:2])
        self.extra['_packet_item']['sources'][1]['content']+='界'*2000;self.refresh(self.extra)
        self.assertFalse(self.pack()[2]['conversation_context']['unmatched_noun']['appended_ids'])

    def test_resource_integrity_failure_and_feature_gates(self):
        for k,v in self.flags.items():
            with patch.multiple(config,**dict(self.flags,**{k:not v})):
                self.assertFalse(context.enabled(),k)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip';p.write_bytes(b'changed corpus')
            with self.assertRaises(ValueError):context.NounConcepts(p)
        off=self.pack(flags=dict(PACKET_UNMATCHED_NOUN_CONTEXT=False))
        with patch.object(context,'lexicon',side_effect=FileNotFoundError):
            failed=self.pack()
        self.assertEqual(failed[:2],off[:2])
        self.assertEqual(failed[2]['conversation_context']['unmatched_noun']['nomination']['status'],'lexicon_unavailable')
        with self.assertRaises(ValueError):context.attach(off[0],'bad',off[2],[],{},user_id='u',top_k=8)


if __name__=='__main__':unittest.main()
