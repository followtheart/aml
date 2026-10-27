"""Regression boundaries for source-grounded reader recovery."""
import copy,os,sys,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac

class QuoteRetryTests(unittest.IsolatedAsyncioTestCase):
 def fixture(self):
  sources={'s0':dict(id='s0',role='user',text='Why do stars have different colors?'),'s1':dict(id='s1',role='user',text='How do neutron stars form?')}
  option='A. Given your interest in astronomy, try this book.'
  claim=dict(text='Given your interest in astronomy,',status='unsupported',premise_type='interest',reason='none',citations=[dict(source_id='s0',quote='Given your interest in astronomy,',basis='self_report',subject='current_user')])
  entry=ac.validate_assessments({'options':[dict(letter='A',kind='personal',claims=[claim])]},[option],sources)[0]
  assert entry['validation_status']=='valid' and entry['claims'][0]['dropped_citations']
  proposal={'matches':[dict(claim_id='A:0',premise_type='interest',supported=True,citations=[dict(source_id='s1',quote=sources['s1']['text'],basis='topic_interest',subject='current_user')])]}
  return entry,sources,dict(question='Recommend a book.',options=[option]),proposal
 async def run_recovery(self,e,s,q,p):
  d={}
  with patch.object(ac.llm,'complete_json',AsyncMock(return_value=p)) as calls:r=await ac.recover_missing_citations([e],s,q,d)
  return r,d,calls.call_count
 async def test_new_exact_citation_requires_independent_entailment(self):
  e,s,q,p=self.fixture();before=copy.deepcopy(e);r,d,n=await self.run_recovery(e,s,q,p)
  self.assertEqual(n,1);self.assertEqual(e,before);claim=r[0]['claims'][0]
  self.assertEqual(claim['status'],'unsupported');self.assertEqual(claim['dropped_citations'],before['claims'][0]['dropped_citations']);self.assertEqual([c['source_id'] for c in claim['citations']],['s1'])
  checks=ac.entailment_checks(r,s,q['options']);self.assertIn('A:0',[c['claim_id'] for c in checks]);r=ac.validate_entailments({'checks':[dict(claim_id=c['claim_id'],entailed=False) for c in checks]},r,checks);self.assertEqual(ac.eligible_choices(r,set()),[])
 async def test_attribution_structural_and_mixed_failures_stay_blocked(self):
  for errors in ([],['third_party_or_hypothetical'],['quote_not_in_source','not_user_support'],['unknown_source']):
   e,s,q,p=self.fixture();e['claims'][0]['dropped_citations'][0]['validation_errors']=errors
   r,d,n=await self.run_recovery(e,s,q,p);self.assertEqual(n,0,errors)
 async def test_invalid_claim_or_entry_and_contradiction_stay_blocked(self):
  for where in ('claim','entry','contradicted'):
   e,s,q,p=self.fixture()
   if where=='claim':e['claims'][0]['validation_errors']=['no_valid_user_support']
   elif where=='entry':e['validation_errors']=['support_unresolved']
   else:e['claims'][0]['reason']='contradicted'
   r,d,n=await self.run_recovery(e,s,q,p);self.assertEqual(n,0,where)
 async def test_missing_or_nonuser_source_and_wrong_subject_stay_blocked(self):
  for change in ('missing','assistant','subject'):
   e,s,q,p=self.fixture()
   if change=='missing':del s['s0']
   elif change=='assistant':s['s0']['role']='assistant'
   else:e['claims'][0]['dropped_citations'][0]['subject']='third_party'
   r,d,n=await self.run_recovery(e,s,q,p);self.assertEqual(n,0,change)
 async def test_new_nonliteral_quote_not_repaired_or_promoted(self):
  e,s,q,p=self.fixture();p['matches'][0]['citations'][0]['quote']='I study astronomy every evening.'
  r,d,n=await self.run_recovery(e,s,q,p);self.assertEqual(n,1);self.assertEqual(r[0]['claims'][0]['citations'],[])
 async def test_question_cannot_anchor_ownership(self):
  e,s,q,p=self.fixture();e['claims'][0].update(text='you own a telescope',premise_type='ownership');p['matches'][0]['premise_type']='ownership'
  r,d,n=await self.run_recovery(e,s,q,p);self.assertEqual(n,1);self.assertEqual(d['answer_citation_recovery']['attached'],[])
if __name__=='__main__':unittest.main()
