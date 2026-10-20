"""Offline output contract and actual cascade regressions; no model quality claims."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,cascade_rerank as cascade,config,cross_encoder,llm,schemas
from app.listwise_contract import State,schema


def cards():
    return {f'c{i}':dict(priority=i,relevance='useful') for i in range(3)}


class Contracts(unittest.TestCase):
    def test_required_fields_and_no_extra_candidates(self):
        spec=schema(3,keyed=True)['properties']['candidates']
        self.assertEqual(spec['required'],['c0','c1','c2'])
        self.assertFalse(spec['additionalProperties'])
        for value in [{}]:
            state=State(3,keyed=True);state.accept(dict(candidates=value,groups=[]))
            self.assertEqual(state.next_field(),'candidates')
            with self.assertRaises(ValueError):state.require_valid()

    def test_partial_repairs_accumulate_without_overwriting(self):
        state=State(3,keyed=True)
        state.accept(dict(candidates={'c0':dict(priority=90,relevance='uncertain'),
                                     'c1':dict(priority=True,relevance='irrelevant'),
                                     's0':dict(priority=0,relevance='irrelevant')},groups=[]))
        self.assertEqual(state.pending_candidate_keys(),['c1','c2'])
        self.assertEqual(state.ignored_candidate_keys,{'s0'})
        state.accept(dict(candidates={'c0':dict(priority=0,relevance='irrelevant'),
                                      'c1':dict(priority=1,relevance='useful')}),'candidates')
        self.assertEqual(state.pending_candidate_keys(),['c2'])
        state.accept(dict(candidates={'c2':dict(priority=2,relevance='useful')}),'candidates')
        self.assertEqual(state.require_valid(),([1,2,0],set(),[]))
        self.assertEqual(state.payload['candidates']['c0']['relevance'],'uncertain')

    def test_extra_candidate_key_does_not_invalidate_known_records(self):
        state=State(3,keyed=True)
        state.accept(dict(candidates=dict(cards(),s0=dict(priority=0,relevance='irrelevant')),groups=[]))
        self.assertEqual(state.require_valid(),([0,1,2],set(),[]))
        self.assertEqual(state.ignored_candidate_keys,{'s0'})

    def test_group_repair_disallows_groups_without_two_eligible_items(self):
        for eligible in ([], ['c1']):
            groups=schema(3,keyed=True,field='groups',group_keys=eligible)['properties']['groups']
            self.assertEqual(groups['maxItems'],0)
            self.assertTrue(groups['items']['items']['enum'])

    def test_ties_use_ce_order_and_uncertain_is_retained(self):
        value=cards()
        for v in value.values():v['priority']=5
        value['c1']['relevance']='uncertain';value['c2']['relevance']='irrelevant'
        state=State(3,keyed=True,tie_order=[2,1,0]);state.accept(dict(candidates=value,groups=[]))
        self.assertEqual(state.require_valid(),([2,1,0],{2},[]))

    def test_invalid_records_cannot_authorize_deletion(self):
        for bad in [dict(priority=True,relevance='useful'),dict(priority=1.5,relevance='useful'),
                    dict(priority=101,relevance='useful'),dict(priority=0,relevance='maybe'),
                    dict(priority=0,relevance='useful',source='s0'),{}]:
            state=State(3,keyed=True);value=cards();value['c1']=bad
            state.accept(dict(candidates=value,groups=[]))
            self.assertIsNone(state.ranking);self.assertIsNone(state.irrelevant)
            with self.assertRaises(ValueError):state.require_valid()

    def test_repair_cannot_modify_locked_fields(self):
        state=State(3);state.accept(dict(ranking=[0],irrelevant=[],groups=[]))
        before=copy.deepcopy(state.payload)
        with self.assertRaises(ValueError):state.accept(dict(ranking=[2,1,0],irrelevant=[0]),'ranking')
        self.assertEqual(state.payload,before)
        state.accept(dict(ranking=[2,1,0]),'ranking')
        self.assertEqual(state.require_valid(),([2,1,0],set(),[]))

    def test_new_relevance_revalidates_groups(self):
        state=State(3,keyed=True);state.accept(dict(candidates={},groups=[['c0','c1']]))
        value=cards();value['c0']['relevance']='irrelevant'
        state.accept(dict(candidates=value),'candidates')
        self.assertEqual(state.next_field(),'groups')
        state.accept(dict(groups=[]),'groups')
        self.assertEqual(state.require_valid(),([0,1,2],{0},[]))

    def test_source_ids_are_not_candidate_ids(self):
        state=State(3,keyed=True);state.accept(dict(candidates=cards(),groups=[['s0','c1']]))
        self.assertEqual(state.errors['groups']['code'],'group_invalid_indices_or_size')


class Runtime(unittest.IsolatedAsyncioTestCase):
    async def run_case(self,outputs,*,keyed=True,calls=8,max_repairs=2,after_response=None):
        self.requests=[];responses=iter(outputs)
        rows=[dict(id=f'm{i}',content=f'Evidence {i}',_rank_text=f'Evidence {i}',_fused=1-i*.1) for i in range(3)]
        async def ce(query,documents,**kwargs):
            budget.current.get().before_call();return [.9,.8,.7]
        async def provider(prompt,**kwargs):
            budget.current.get().before_call();self.requests.append(dict(prompt=prompt,**kwargs))
            result=next(responses)
            if isinstance(result,BaseException):raise result
            if after_response:
                after_response(budget.current.get())
            return copy.deepcopy(result)
        with patch.multiple(config,LISTWISE_KEYED_OUTPUT=keyed,RERANK_REPAIR_MAX_CALLS=max_repairs,
                CASCADE_COARSE_LIMIT=3,CASCADE_FINE_LIMIT=3,CASCADE_LLM_LIMIT=3,
                LISTWISE_SHARED_SOURCES=False,FINE_EVIDENCE_ADAPTIVE=False), \
             patch.object(cross_encoder,'rerank',side_effect=ce), \
             patch.object(llm,'complete_json',side_effect=provider), \
             patch('socket.socket.connect',side_effect=AssertionError('network forbidden')), \
             budget.scope(seconds=10,calls=calls,tokens=100000) as limits:
            plan={};result=await cascade.rank(schemas.SearchRequest(user_id='u',query='Evidence'),plan,rows)
            self.assertEqual((limits.reserved_tokens,limits.reserved_calls),(0,0))
            self.assertLessEqual(limits.calls,calls)
        return result,plan

    async def test_legacy_ranking_then_group_repair(self):
        result,plan=await self.run_case([dict(ranking=[0],irrelevant=[],groups=[[0]]),
            dict(ranking=[2,1,0]),dict(groups=[[0,1]])],keyed=False)
        self.assertEqual([list(r['schema']['properties']) for r in self.requests],
                         [['ranking','irrelevant','groups'],['ranking'],['groups']])
        self.assertIn('"missing": [1, 2]',self.requests[1]['system'])
        self.assertIn('"excluded_from_groups":[]',self.requests[2]['prompt'])
        self.assertNotIn('"ranking": [2, 1, 0]',self.requests[2]['prompt'])
        self.assertEqual([r['id'] for r in result if r['_cascade_selected']],['m2','m1','m0'])
        self.assertEqual(plan['_rerank_status'],'recovered')
        self.assertEqual(plan['_evidence_groups'],[['m0','m1']])

    async def test_keyed_missing_candidate_repair(self):
        values=cards();values['c1']['relevance']='uncertain'
        result,plan=await self.run_case([dict(candidates={},groups=[]),dict(candidates=values)])
        self.assertEqual(list(self.requests[1]['schema']['properties']),['candidates'])
        self.assertEqual(plan['_rerank_status'],'recovered')
        self.assertEqual(len([r for r in result if r['_cascade_selected']]),3)
        self.assertEqual(plan['_cascade']['listwise']['output_protocol'],'keyed_candidates_v1')

    async def test_repairs_request_only_unresolved_keys(self):
        value=cards();value['c0']['priority']=90
        result,plan=await self.run_case([
            dict(candidates={'c0':value['c0']},groups=[]),
            dict(candidates={'c1':value['c1']}),
            dict(candidates={'c2':value['c2']})])
        self.assertEqual(self.requests[1]['schema']['properties']['candidates']['required'],['c1','c2'])
        self.assertEqual(self.requests[2]['schema']['properties']['candidates']['required'],['c2'])
        self.assertIn('Requested candidate keys: ["c2"]',self.requests[2]['prompt'])
        self.assertEqual([r['id'] for r in result if r['_cascade_selected']],['m1','m2','m0'])
        self.assertEqual(plan['_rerank_status'],'recovered')

    async def test_missing_groups_repairs_only_groups(self):
        result,plan=await self.run_case([dict(candidates=cards()),dict(groups=[])])
        self.assertEqual(list(self.requests[1]['schema']['properties']),['groups'])
        self.assertNotIn('For each requested candidate key return priority',self.requests[1]['prompt'])
        self.assertEqual(plan['_rerank_status'],'recovered')
        self.assertEqual([r['id'] for r in result if r['_cascade_selected']],['m0','m1','m2'])

    async def test_group_failure_preserves_ranking_and_requests_deletion_review(self):
        values=cards();values['c0']['priority']=90;values['c0']['relevance']='irrelevant'
        result,plan=await self.run_case([dict(candidates=values,groups=[['c0','c1']]),
            dict(groups=[['c0','c1']])],max_repairs=1)
        self.assertEqual(plan['_cascade']['listwise']['status'],'groups_partial')
        self.assertEqual([r['id'] for r in result if r['_cascade_selected']],['m1','m2','m0'])
        self.assertEqual(plan['_cascade']['listwise']['primary_irrelevant_ids'],['m0'])
        self.assertEqual(self.requests[1]['schema']['properties']['groups']['items']['items']['enum'],['c1','c2'])
        self.assertEqual(plan['_rerank_status'],'partial')

    async def test_out_of_scope_group_repair_cannot_change_ranking(self):
        values=cards();values['c0']['priority']=90
        result,plan=await self.run_case([dict(candidates=values,groups=[['c0']]),
            dict(groups=[],candidates=cards())],max_repairs=1)
        self.assertEqual(plan['_cascade']['listwise']['status'],'groups_partial')
        self.assertEqual([r['id'] for r in result if r['_cascade_selected']],['m1','m2','m0'])

    async def test_call_exhaustion_preserves_candidates(self):
        result,plan=await self.run_case([dict(candidates={},groups=[])],calls=2)
        self.assertEqual(len(self.requests),1)
        self.assertEqual(plan['_cascade']['listwise']['status'],'fallback')
        self.assertEqual(len([r for r in result if r['_cascade_selected']]),3)

    async def test_group_repair_token_exhaustion_preserves_valid_order(self):
        values=cards();values['c0']['priority']=90
        def exhaust(limits):
            limits.tokens=limits.max_tokens
        result,plan=await self.run_case([dict(candidates=values,groups=[['c0']])],
                                       after_response=exhaust)
        self.assertEqual(len(self.requests),1)
        self.assertEqual(plan['_cascade']['listwise']['repair_calls'],0)
        self.assertEqual(plan['_cascade']['listwise']['status'],'groups_partial')
        self.assertEqual([r['id'] for r in result if r['_cascade_selected']],['m1','m2','m0'])

    async def test_keyed_fake_provider_matches_schema(self):
        req=schemas.SearchRequest(user_id='u',query='Evidence')
        rows=[dict(id=f'm{i}',content=f'Evidence {i}') for i in range(3)]
        with patch.object(config,'LISTWISE_KEYED_OUTPUT',True):
            request=cascade.listwise_request(req,{},rows)
            result=json.loads(llm.FakeLLM().complete(request['prompt']))
            state=State(3,keyed=True);state.accept(result);state.require_valid()
            repair=cascade.listwise_request(req,{},rows,repair=True,repair_field='groups',locked=result)
            self.assertEqual(json.loads(llm.FakeLLM().complete(repair['prompt'])),dict(groups=[]))
            self.assertGreater(request['reservation'],len(request['prompt'].encode()))

if __name__=='__main__':unittest.main(verbosity=2)
