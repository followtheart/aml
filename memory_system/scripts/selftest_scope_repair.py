"""Bounded partial scope repair preserves classifications and downstream budget."""
import copy
import unittest
from unittest.mock import AsyncMock, patch
import selftest_answer_choice  # Configure offline dependencies first.
from app import answer_choice as gate, llm, budget

def decision(cid,value):return dict(claim_id=cid,personal_fact=value)
class ScopeRepairTests(unittest.IsolatedAsyncioTestCase):
 def fixture(self):
  texts=['You could walk.','You own a canoe.']
  return [dict(letter='A',option='A. '+ ' '.join(texts),kind='personal',status='unsupported',primary_claim=0,validation_status='valid',validation_errors=[],claims=[dict(text=t,status='unsupported',reason='no_source',citations=[],validation_errors=[]) for t in texts])]
 async def run_case(self,*responses):
  original=self.fixture();before=copy.deepcopy(original);d={};mock=AsyncMock(side_effect=responses)
  with patch.object(llm,'complete_json',mock):result=await gate.reclassify_uncited(original,d)
  self.assertEqual(original,before);return result,original,d,mock
 async def test_missing_only_repair_and_index_mapping(self):
  r,o,d,m=await self.run_case({'claims':[decision('A:1',True)]},{'claims':[decision('A:0',False)]})
  self.assertEqual([c['text'] for c in r[0]['claims']],['You own a canoe.']);self.assertEqual(r[0]['status'],'unsupported');self.assertEqual(d['answer_premise_scope']['claim_index_map'],{'A:0':None,'A:1':'A:0'})
  self.assertEqual(m.call_args.kwargs['schema']['$defs']['Scope']['properties']['claim_id']['enum'],['A:0'])
 async def test_accepted_decision_cannot_be_overwritten(self):
  r,o,d,m=await self.run_case({'claims':[decision('A:1',True)]},{'claims':[decision('A:1',False),decision('A:0',False)]})
  self.assertEqual([c['text'] for c in r[0]['claims']],['You own a canoe.'])
 async def test_duplicate_decision_requires_repair(self):
  r,o,d,m=await self.run_case({'claims':[decision('A:0',False),decision('A:0',True),decision('A:1',True)]},{'claims':[decision('A:0',False)]})
  self.assertEqual(m.await_count,2);self.assertEqual(len(r[0]['claims']),1)
 async def test_unknown_ids_cannot_remove_claims(self):
  r,o,d,m=await self.run_case({'claims':[decision('X:0',False)]},{'claims':[decision('X:0',False)]})
  self.assertIs(r,o);self.assertEqual(d['answer_premise_scope']['status'],'kept_original')
 async def test_failed_retry_retains_original(self):
  r,o,d,m=await self.run_case({'claims':[decision('A:0',False)]},TimeoutError())
  self.assertIs(r,o);self.assertEqual(m.await_count,2)
 async def test_budget_declines_retry_after_first_call(self):
  o=self.fixture();d={}
  with budget.scope(seconds=240,calls=6,tokens=128000) as b:
   async def response(*args,**kwargs):b.before_call();return {'claims':[decision('A:0',False)]}
   with patch.object(llm,'complete_json',AsyncMock(side_effect=response)) as m:r=await gate.reclassify_uncited(o,d)
  self.assertIs(r,o);self.assertEqual(m.await_count,1);self.assertEqual(d['answer_premise_scope']['error_type'],'BudgetExceeded')
 async def test_complete_response_needs_no_retry(self):
  r,o,d,m=await self.run_case({'claims':[decision('A:0',False),decision('A:1',True)]});self.assertEqual(m.await_count,1)

if __name__ == "__main__":
    unittest.main()
