"""Citation recovery must not promote claims before independent verification."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['AML_FAKE']='1'
from app import answer_choice as gate,budget,config,llm
if os.environ.get('ANSWER_CANDIDATE'):
    spec=importlib.util.spec_from_file_location('app.answer_candidate',os.environ['ANSWER_CANDIDATE'])
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        text='I cycled by the canal last weekend.'
        sources={'s0':dict(id='s0',role='user',text=text)}
        options=['A. Since you cycled by the canal last weekend, take a rest.']
        payload={'options':[dict(letter='A',kind='personal',claims=[dict(text='you cycled by the canal last weekend',status='unsupported',premise_type='experience',reason='no_source',citations=[])])]}
        entries=gate.validate_assessments(payload,options,sources)
        reply={'matches':[dict(claim_id='A:0',premise_type='experience',supported=True,citations=[dict(source_id='s0',quote=text,basis='self_report',subject='current_user')])]}
        return entries,sources,options,reply

    async def invoke(self,entries,sources,reply,diagnostics=None):
        mock=AsyncMock(side_effect=reply if isinstance(reply,Exception) else None,return_value=reply)
        with patch.object(llm,'complete_json',mock):
            result=await gate.recover_missing_citations(entries,sources,dict(question='How should I spend my weekend?'),diagnostics if diagnostics is not None else {})
        self.assertLessEqual(mock.await_count,1)
        return result,mock

    async def test_premise_verdict_and_option_tier_are_independent(self):
        entries,sources,options,reply=self.fixture();before=copy.deepcopy(entries)
        updated,_=await self.invoke(entries,sources,reply)
        self.assertEqual(entries,before)
        self.assertEqual(updated[0]['status'],'unsupported')
        self.assertEqual(updated[0]['claims'][0]['status'],'unsupported')
        self.assertTrue(gate._recoverable(updated[0]['claims'][0]))
        checks=gate.entailment_checks(updated,sources,options)
        self.assertNotIn('proposed_status', next(c for c in checks if c['claim_id']=='A:0'))
        for rejected in ['A:0','A:option',None]:
            verdict={'checks':[dict(claim_id=c['claim_id'],entailed=c['claim_id']!=rejected) for c in checks]}
            out=gate.validate_entailments(verdict,copy.deepcopy(updated),checks)
            expected = {None: 'supported', 'A:option': 'partial', 'A:0': 'unsupported'}
            self.assertEqual(out[0]['status'], expected[rejected])
            self.assertEqual(out[0]['claims'][0]['status'],
                             'unsupported' if rejected == 'A:0' else 'supported')

    async def test_invalid_source_quote_attribution_and_denial_stay_rejected(self):
        for kind in ['assistant','quote','third_party','denial']:
            entries,sources,_,reply=self.fixture()
            if kind=='assistant':sources['s0']['role']='assistant'
            if kind=='quote':reply['matches'][0]['citations'][0]['quote']='invented'
            if kind=='third_party':sources['s0']['text']='My colleague wrote: '+sources['s0']['text']
            if kind=='denial':sources['s0']['text']='I did not cycle by the canal.';reply['matches'][0]['citations'][0]['quote']=sources['s0']['text']
            with self.subTest(kind=kind):
                out,_=await self.invoke(entries,sources,reply)
                self.assertEqual(out,entries)

    async def test_known_premise_type_cannot_be_downgraded(self):
        entries,sources,_,reply=self.fixture()
        entries[0]['claims'][0].update(text='you own a telescope',premise_type='ownership')
        entries[0]['option']='A. Since you own a telescope, look at the sky.'
        sources['s0']['text']='Which telescope should I buy?'
        reply['matches'][0].update(premise_type='interest',citations=[dict(source_id='s0',quote=sources['s0']['text'],basis='topic_interest',subject='current_user')])
        out,_=await self.invoke(entries,sources,reply);self.assertEqual(out,entries)

    async def test_only_boundary_ellipsis_with_exact_remaining_source_is_normalized(self):
        entries,sources,_,reply=self.fixture()
        quote=reply['matches'][0]['citations'][0]['quote']
        for marker in ['...', '…']:
            reply['matches'][0]['citations'][0]['quote']=marker+' '+quote+' '+marker
            out,_=await self.invoke(entries,sources,reply)
            ref=out[0]['claims'][0]['citations'][0]
            self.assertEqual(ref['quote'],quote)
            self.assertEqual(ref['quote_normalization'],'boundary_ellipsis')
            self.assertTrue(ref['proposed_quote'].startswith(marker))
        for changed in ['I cycled ... last weekend.', '...I sailed by the canal last weekend.', '...']:
            reply['matches'][0]['citations'][0]['quote']=changed
            out,_=await self.invoke(entries,sources,reply)
            self.assertEqual(out,entries)

    async def test_no_target_no_provider_call(self):
        for mutation in ['unresolved','has_citation','rejected','supported','no_sources']:
            entries,sources,_,reply=self.fixture()
            if mutation=='unresolved':entries[0]['validation_status']='unresolved'
            if mutation=='has_citation':entries[0]['claims'][0]['citations']=[{}]
            if mutation=='rejected':entries[0]['claims'][0]['reason']='contradicted'
            if mutation=='supported':entries[0]['status']='supported'
            if mutation=='no_sources':sources={}
            out,mock=await self.invoke(entries,sources,reply)
            self.assertEqual(out,entries);self.assertEqual(mock.await_count,0)

    async def test_malformed_ids_and_timeout_preserve_original(self):
        for reply in [{'matches':[]},TimeoutError('test'),{'matches':[dict(claim_id='X:0',premise_type='unknown',supported=False,citations=[])]}]:
            entries,sources,_,_=self.fixture()
            out,_=await self.invoke(entries,sources,reply);self.assertEqual(out,entries)

    async def test_budget_reserves_downstream_capacity(self):
        entries,sources,_,reply=self.fixture()
        with budget.scope(seconds=60,calls=20,tokens=100000):
            out,mock=await self.invoke(entries,sources,reply)
        self.assertEqual(out,entries);self.assertEqual(mock.await_count,0)

    async def test_false_proposal_does_not_attach_and_secondaries_unchanged(self):
        entries,sources,_,reply=self.fixture()
        secondary=copy.deepcopy(entries[0]['claims'][0]);secondary['text']='you also own a boat'
        entries[0]['claims'].append(secondary)
        out,_=await self.invoke(entries,sources,reply)
        self.assertEqual(out[0]['claims'][1],secondary)
        reply['matches'][0]['supported']=False
        out,_=await self.invoke(entries,sources,reply);self.assertEqual(out,entries)

    async def test_complete_answer_flow_recovery_rejection_and_failure(self):
        from selftest_answer_choice import memory, seal, entailment_response
        for scenario in ['recover', 'rejected', 'malformed']:
            with self.subTest(scenario=scenario):
                text='I cycled by the canal last weekend.'
                packet=seal(memory(text))
                sources,_=gate.build_catalog(packet)
                qa=dict(question='What should I do after my ride?', options=[
                    'A. Since you cycled by the canal last weekend, take a rest.'],
                    gold_labels=['DO_NOT_INCLUDE_GOLD'])
                payload={'options':[dict(letter='A',kind='personal',claims=[dict(
                    text='you cycled by the canal last weekend',status='unsupported',
                    premise_type='experience',reason='no_source',citations=[])])]}
                stages=[]
                async def respond(prompt,**kw):
                    self.assertNotIn('DO_NOT_INCLUDE_GOLD',prompt)
                    stage=kw['stage'];stages.append(stage)
                    if stage=='eval.choice_support':return payload
                    if stage=='eval.choice_premise_scope':
                        return {'claims':[dict(claim_id='A:0',personal_fact=True)]}
                    if stage=='eval.choice_citation_recovery':
                        if scenario=='malformed':return {'matches':[]}
                        return {'matches':[dict(claim_id='A:0',premise_type='experience',supported=True,
                            citations=[dict(source_id=next(iter(sources)),quote=text,basis='self_report',subject='current_user')])]}
                    if stage=='eval.choice_entailment':
                        return entailment_response(prompt,{'A:0','A:option'} if scenario=='rejected' else set())
                    raise AssertionError('Unexpected stage: '+stage)
                diagnostics={}
                with patch.object(llm,'complete_json',AsyncMock(side_effect=respond)):
                    result=await gate.answer(qa,packet,diagnostics)
                self.assertEqual(result,'A' if scenario=='recover' else 'ABSTAIN')
                self.assertEqual(stages,['eval.choice_support'] +
                    (['eval.choice_premise_scope'] if hasattr(gate,'reclassify_uncited') else []) +
                    ['eval.choice_citation_recovery','eval.choice_entailment'])

if __name__=='__main__':unittest.main()
