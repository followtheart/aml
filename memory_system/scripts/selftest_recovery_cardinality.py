"""Request-specific recovery output count; evidence validators remain strict."""
import copy,json,unittest
from types import SimpleNamespace
from unittest.mock import patch
from app import config,listwise_recovery as recovery,recovery_spans
class RecoveryCardinality(unittest.TestCase):
 def setUp(self):
  self.req=SimpleNamespace(query='Which hobby suits me?',options=['A. Try cycling.','B. Try reading.'])
  self.cards=[dict(candidate_id='m'+str(i),sources=[dict(source_id='s0',role='user',text=text)]) for i,text in enumerate(['I enjoy cycling.','I enjoy reading.'])]
 def test_singleton_count_and_candidate_id(self):
  with patch.object(config,'LISTWISE_SPAN_REFS',False):r=recovery.candidate_contract(recovery.request_for(self.req,self.cards[:1]),self.cards[:1])
  d=r['schema']['properties']['decisions'];self.assertEqual((d['minItems'],d['maxItems']),(1,1));self.assertEqual(r['schema']['$defs']['Decision']['properties']['candidate_id']['enum'],['m0'])
  self.assertIn('Return exactly 1 decision object(s)',r['prompt']);self.assertIn('Do not add separate negative decisions',r['prompt'])
 def test_independent_request_schemas_and_unchanged_sources(self):
  before=copy.deepcopy(self.cards)
  with patch.object(config,'LISTWISE_SPAN_REFS',False):a=recovery.candidate_contract(recovery.request_for(self.req,self.cards[:1]),self.cards[:1]);b=recovery.candidate_contract(recovery.request_for(self.req,self.cards),self.cards)
  self.assertEqual(a['schema']['properties']['decisions']['maxItems'],1);self.assertEqual(b['schema']['properties']['decisions']['maxItems'],2);self.assertEqual(self.cards,before)
 def test_conflicting_duplicate_decisions_are_not_merged(self):
  positive=dict(candidate_id='m0',useful=True,target_id='option:0',target_quote='cycling',evidence_kind='user_context',citations=[dict(source_id='s0',quote='I enjoy cycling.')]);negative=dict(candidate_id='m0',useful=False,target_id='',target_quote='',evidence_kind='none',citations=[])
  with self.assertRaises(ValueError):recovery.validate(dict(decisions=[positive,negative]),self.cards[:1],recovery.targets_for(self.req))
 def test_bad_quote_cannot_restore_candidate(self):
  d=dict(candidate_id='m0',useful=True,target_id='option:0',target_quote='cycling',evidence_kind='user_context',citations=[dict(source_id='s0',quote='I own a bicycle.')])
  restored,judgments=recovery.validate(dict(decisions=[d]),self.cards[:1],recovery.targets_for(self.req));self.assertEqual(restored,set());self.assertIn('quote_not_in_candidate_source',judgments[0]['validation_errors'])
 def test_span_reference_format_is_unchanged(self):
  with patch.object(config,'LISTWISE_SPAN_REFS',True):r=recovery.request_for(self.req,self.cards)
  expected=recovery_spans.build(self.req,self.cards);self.assertEqual(r['prompt'],expected['prompt']);self.assertEqual(r['schema'],expected['schema'])

class ContractBudgetFlow(unittest.IsolatedAsyncioTestCase):
 async def run_case(self,limited):
  from app import budget,llm
  req=SimpleNamespace(query='Which hobby?',options=['A. Cycling.'])
  source=dict(role='user',content='I enjoy cycling.',request_id='r',message_index=0)
  row=dict(id='m0',content=source['content'],_rank_text=source['content'],_packet_item=dict(sources=[source]))
  cards=[dict(candidate_id='m0',sources=recovery.sources_for(row))]
  with patch.object(config,'LISTWISE_SPAN_REFS',False):base=recovery.request_for(req,cards)
  seen=[]
  async def provider(prompt,**kwargs):
   budget.current.get().before_call();seen.append(prompt)
   return dict(decisions=[dict(candidate_id='m0',useful=False,target_id='',target_quote='',evidence_kind='none',citations=[])])
  with patch.object(config,'LISTWISE_SPAN_REFS',False),patch.object(llm,'complete_json',provider),budget.scope(seconds=60,calls=1,tokens=base['reservation'] if limited else 64000) as b:
   result,trace=await recovery.recover(req,[row],[],[0])
  self.assertEqual(result,[]);self.assertEqual(trace['candidate_ids'],['m0']);self.assertEqual((b.calls,b.reserved_calls,b.reserved_tokens),(1,0,0))
  return base,seen,trace
 async def test_insufficient_contract_tokens_keeps_original_request(self):
  base,seen,trace=await self.run_case(True);self.assertEqual(seen,[base['prompt']]);self.assertEqual(trace['output_contract']['status'],'kept_original_budget')
 async def test_available_budget_applies_count_contract(self):
  _,seen,trace=await self.run_case(False);self.assertEqual(trace['output_contract']['status'],'applied');self.assertIn('Return exactly 1 decision object(s)',seen[0])

if __name__=='__main__':unittest.main()
