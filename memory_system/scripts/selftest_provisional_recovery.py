"""Integration tests for provisional-positive recovery without relaxing evidence gates."""
import copy,os,sys,unittest
from pathlib import Path
from unittest.mock import patch,AsyncMock
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac

class RecoveryCandidateTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self,text='you enjoy astronomy',premise_type='interest',status='supported'):
        option='A. Since '+text+', try a related book.'
        sources={sid:dict(id=sid,role='user',text=body,timestamp=None,request_id=sid,message_index=0)
                 for sid,body in [('s0','I enjoy astronomy and own a telescope.'),('s1','Why do neutron stars spin so fast?')]}
        payload=dict(letter='A',kind='personal',claims=[dict(text=text,status=status,premise_type=premise_type,
                     reason='none',citations=[dict(source_id='s0',quote=sources['s0']['text'],basis='self_report',subject='current_user')])])
        entry=ac.validate_assessments({'options':[payload]},[option],sources)[0]
        self.assertEqual(entry['validation_status'],'valid')
        return entry,sources,dict(question='Suggest something useful.',options=[option])

    async def recover(self,entry,sources,qa,quote=None,premise_type='interest'):
        proposal=dict(matches=[dict(claim_id='A:0',premise_type=premise_type,supported=True,
            citations=[dict(source_id='s1',quote=quote or sources['s1']['text'],basis='topic_interest',subject='current_user')])])
        diag={}
        with patch.object(ac.llm,'complete_json',AsyncMock(return_value=proposal)) as calls:
            result=await ac.recover_missing_citations([entry],sources,qa,diag)
        return result,diag,calls

    async def test_positive_interest_is_provisional_until_independent_verification(self):
        entry,sources,qa=self.fixture();original=copy.deepcopy(entry)
        result,diag,calls=await self.recover(entry,sources,qa)
        self.assertEqual(calls.call_count,1)
        self.assertEqual(diag['answer_citation_recovery']['attached'],['A:0'])
        self.assertEqual([c['source_id'] for c in result[0]['claims'][0]['citations']],['s0','s1'])
        self.assertEqual(entry,original)
        checks=ac.entailment_checks(result,sources,qa['options'])
        denied={'checks':[dict(claim_id=c['claim_id'],entailed=False) for c in checks]}
        result=ac.validate_entailments(denied,result,checks)
        self.assertEqual(result[0]['claims'][0]['status'],'unsupported')
        self.assertEqual(ac.eligible_choices(result,set()),[])

    async def test_supported_strong_fact_types_do_not_add_recovery_calls(self):
        for premise_type in ('ownership','condition','habit','experience','occupation','location','unknown'):
            with self.subTest(premise_type=premise_type):
                entry,sources,qa=self.fixture('you own a telescope',premise_type)
                result,diag,calls=await self.recover(entry,sources,qa)
                self.assertEqual(calls.call_count,0)
                self.assertEqual(result,[entry])

    async def test_existing_unsupported_interest_still_recovers(self):
        entry,sources,qa=self.fixture(status='unsupported')
        result,diag,calls=await self.recover(entry,sources,qa)
        self.assertEqual(calls.call_count,1)
        self.assertEqual(result[0]['claims'][0]['status'],'unsupported')
        self.assertEqual(diag['answer_citation_recovery']['attached'],['A:0'])

    async def test_structural_or_attribution_errors_are_not_bypassed(self):
        for change in ('claim_error','entry_error','dropped_citation','contradiction'):
            entry,sources,qa=self.fixture();claim=entry['claims'][0]
            if change=='claim_error':claim['validation_errors']=['third_party_or_hypothetical']
            elif change=='entry_error':entry['validation_errors']=['support_unresolved']
            elif change=='dropped_citation':claim['dropped_citations']=[dict(source_id='s2')]
            else:claim['reason']='contradicted'
            result,diag,calls=await self.recover(entry,sources,qa)
            self.assertEqual(calls.call_count,0,change)
            self.assertEqual(result,[entry])

    async def test_nonliteral_quote_is_rejected(self):
        entry,sources,qa=self.fixture();result,diag,calls=await self.recover(entry,sources,qa,quote='I study astronomy daily.')
        self.assertEqual(calls.call_count,1)
        self.assertEqual(diag['answer_citation_recovery']['attached'],[])
        self.assertEqual(result[0]['claims'][0]['citations'],entry['claims'][0]['citations'])

    async def test_mislabeled_ownership_cannot_use_topic_question_as_proof(self):
        entry,sources,qa=self.fixture('you own a telescope','interest')
        result,diag,calls=await self.recover(entry,sources,qa)
        self.assertEqual(calls.call_count,1)
        self.assertEqual(diag['answer_citation_recovery']['attached'],[])
        self.assertEqual(diag['answer_citation_recovery']['rejected'],['A:0'])
        self.assertEqual(result[0]['claims'][0]['citations'],entry['claims'][0]['citations'])

if __name__=='__main__':unittest.main()
