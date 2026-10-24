"""Role-constrained generation never substitutes for citation validation."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jsonschema
from app import answer_context,budget,config,listwise_recovery as recovery,llm,recovery_spans as spans


class RoleBoundRefs(unittest.TestCase):
    def setUp(self):
        self.req=SimpleNamespace(query='How could I relax?',options=['A. Try painting.'])
        self.cards=[dict(candidate_id='one',sources=[
            dict(source_id='local_user',role='user',text='I enjoy painting. It calms me.',request_id='r',message_index=0),
            dict(source_id='local_assistant',role='assistant',text='Painting can be calming.',request_id='r',message_index=1)]),
            dict(candidate_id='two',sources=[dict(source_id='local_other',role='user',text='I enjoy walking.',request_id='r',message_index=2)])]
        self.built=spans.build(self.req,self.cards,role_bound=True)

    def yes(self,kind='user_context',parts=None):
        return dict(useful=True,target_id='question',evidence_kind=kind,citation_parts=parts or ['s0.p0'])

    def no(self):
        return dict(useful=False,target_id='',evidence_kind='none',citation_parts=[])

    def payload(self,decision):
        return dict(c0=decision,c1=self.no())

    def test_valid_role_kinds_and_negative_are_schema_valid(self):
        for kind,parts in [('user_context',['s0.p0']),('topic_interest',['s0.p0']),
                           ('counterevidence',['s0.p0']),('assistant_context',['s1.p0']),
                           ('third_party_context',['s0.p0','s1.p0'])]:
            with self.subTest(kind=kind):
                jsonschema.validate(self.payload(self.yes(kind,parts)),self.built['schema'])
        jsonschema.validate(dict(c0=self.no(),c1=self.no()),self.built['schema'])

    def test_mixed_role_and_foreign_ownership_rejected_by_schema(self):
        for kind,parts in [('user_context',['s0.p0','s1.p0']),('topic_interest',['s1.p0']),
                           ('counterevidence',['s1.p0']),('assistant_context',['s0.p0']),
                           ('third_party_context',['s2.p0'])]:
            with self.subTest(kind=kind,parts=parts),self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(self.payload(self.yes(kind,parts)),self.built['schema'])

    def test_negative_fields_and_unknown_target_rejected(self):
        cases=[dict(self.no(),citation_parts=['s0.p0']),dict(self.no(),target_id='question'),
               dict(self.yes(),target_id='option:5'),dict(self.yes(),citation_parts=[])]
        for decision in cases:
            with self.subTest(decision=decision),self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(self.payload(decision),self.built['schema'])

    def test_provider_ignoring_schema_still_fails_existing_validator(self):
        restored,judgments=spans.decode_partial(self.payload(self.yes(parts=['s0.p0','s1.p0'])),self.built)
        self.assertEqual(restored,set())
        self.assertIn('not_user_source',judgments[0]['validation_errors'])
        self.assertEqual(len(judgments[0]['citations']),2,'Do not silently discard a bad citation')

    def test_role_correct_does_not_bypass_attribution(self):
        cards=copy.deepcopy(self.cards)
        cards[0]['sources'][0]['text']='Painting relaxes people.'
        built=spans.build(self.req,cards,role_bound=True)
        payload=self.payload(self.yes())
        jsonschema.validate(payload,built['schema'])
        restored,judgments=spans.decode_partial(payload,built)
        self.assertFalse(restored)
        self.assertIn('no_user_assertion',judgments[0]['validation_errors'])

    def test_source_text_owners_and_prompt_identical_to_control(self):
        before=copy.deepcopy(self.cards)
        control=spans.build(self.req,self.cards)
        for key in ('prompt','spans','owners','cards','candidate_map','targets'):
            self.assertEqual(control[key],self.built[key])
        self.assertEqual(before,self.cards)
        self.assertNotIn('anyOf',control['schema']['properties']['c0'])

    def test_no_user_source_has_no_user_kind_branch(self):
        built=spans.build(self.req,[dict(candidate_id='a',sources=self.cards[0]['sources'][1:])],role_bound=True)
        kinds=[k for b in built['schema']['properties']['c0']['anyOf'] for k in b['properties']['evidence_kind']['enum']]
        self.assertEqual(set(kinds),{'none','assistant_context','third_party_context'})

    def test_reservation_includes_schema_and_system_growth(self):
        with patch.multiple(config,LISTWISE_SPAN_REFS=True,LISTWISE_ROLE_BOUND_REFS=False):
            before=recovery.request_for(self.req,self.cards)
        with patch.multiple(config,LISTWISE_SPAN_REFS=True,LISTWISE_ROLE_BOUND_REFS=True):
            after=recovery.request_for(self.req,self.cards)
        delta=sum(len(after[k].encode())-len(before[k].encode()) for k in ('prompt','system'))
        delta+=len(json.dumps(after['schema']).encode())-len(json.dumps(before['schema']).encode())
        self.assertGreater(delta,0)
        self.assertEqual(after['reservation']-before['reservation'],delta)


class Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_real_recovery_uses_new_schema_and_respects_shared_budget(self):
        req=SimpleNamespace(query='How could I relax?',options=[])
        source=dict(request_id='r',message_index=0,source_event_id='s',role='user',content='I enjoy painting.')
        row=dict(id='a',content=source['content'],_rank_text=answer_context.with_evidence('',[source]),_packet_item=dict(sources=[source]))
        calls=[]
        async def model(prompt,**kwargs):
            budget.current.get().before_call()
            calls.append(kwargs['schema'])
            return dict(c0=dict(useful=True,target_id='question',evidence_kind='user_context',citation_parts=['s0.p0']))
        with patch.multiple(config,LISTWISE_SPAN_REFS=True,LISTWISE_ROLE_BOUND_REFS=True),patch.object(llm,'complete_json',model):
            with budget.scope(seconds=30,calls=1,tokens=40000) as limits:
                selected,trace=await recovery._recover_batch(req,[row],[],{0})
            self.assertEqual(selected,[row])
            self.assertEqual(trace['citation_format'],'source_span_refs_v2_role_bound')
            self.assertIn('anyOf',calls[0]['properties']['c0'])
            self.assertEqual((limits.calls,limits.reserved_calls,limits.reserved_tokens),(1,0,0))
            with budget.scope(seconds=30,calls=1,tokens=trace['request_bound']-1):
                selected,trace=await recovery._recover_batch(req,[row],[],{0})
            self.assertEqual(selected,[])
            self.assertEqual(len(calls),1)
            self.assertEqual(trace['omitted'][0]['reason'],'token_budget')


if __name__=='__main__':
    unittest.main()
