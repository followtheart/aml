"""Regression boundaries for source-grounded reader recovery."""
import copy,os,sys,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac

class SupportWrapperTests(unittest.IsolatedAsyncioTestCase):
 def test_relocation_preserves_records_and_input(self):
  payload={'options':{'A':{'kind':'generic','claims':[]}},'B':{'kind':'generic','claims':[]}};before=copy.deepcopy(payload)
  self.assertEqual(ac.canonical_support_payload(payload,['A','B']),{'options':dict(A=payload['options']['A'],B=payload['B'])});self.assertEqual(payload,before)
 def test_duplicate_and_unknown_keys_are_not_resolved_by_guessing(self):
  for payload in ({'options':{'A':{}},'A':{}},{'options':{'A':{}},'extra':{}},{'options':{'Z':{}},'B':{}}):self.assertEqual(ac.canonical_support_payload(payload,['A','B']),payload)
 def test_array_and_scalar_wrappers_unchanged(self):
  for payload in ({'options':[{'letter':'A'}],'B':{}},{'options':None,'B':{}},[]):self.assertEqual(ac.canonical_support_payload(payload,['A','B']),payload)
 def test_embedded_letter_is_still_invalid(self):
  payload={'options':{'A':{'kind':'generic','claims':[]}},'B':{'letter':'C','kind':'generic','claims':[]}}
  rows=ac.assessment_rows(ac.canonical_support_payload(payload,['A','B']));self.assertIn('invalid_map_value',rows[1]);self.assertEqual(rows[1]['letter'],'B')
 async def test_valid_siblings_are_frozen_and_only_missing_option_is_repaired(self):
  options=['A. Try reading.','B. Try walking.','C. Try painting.'];payload={'options':{'A':{'kind':'generic','claims':[]}},'B':{'kind':'generic','claims':[]}};repair={'options':[dict(letter='C',kind='generic',claims=[])]};diagnostics={}
  with patch.object(ac.llm,'complete_json',AsyncMock(side_effect=[payload,repair])) as calls:
   result=await ac.assess_support('Options:\n'+'\n'.join(options),ac.Assessments.model_json_schema(),options,{},diagnostics)
  self.assertEqual(calls.call_count,2);self.assertEqual([e['letter'] for e in result],['A','B','C']);self.assertIn('Assess ONLY these option letters, exactly once each: C',calls.call_args_list[1].args[0])
 async def test_bad_citation_still_requires_repair(self):
  options=['A. Since you own a boat, try sailing.','B. Try walking.'];payload={'options':{'A':dict(kind='personal',claims=[dict(text='you own a boat',status='supported',premise_type='ownership',reason='none',citations=[dict(source_id='s0',quote='I own a boat.',basis='self_report',subject='current_user')])])},'B':dict(kind='generic',claims=[])}
  repair={'options':[dict(letter='A',kind='personal',claims=[dict(text='you own a boat',status='unsupported',premise_type='ownership',reason='no_source',citations=[])])]};diagnostics={};sources={'s0':dict(id='s0',role='user',text='How does sailing work?')}
  with patch.object(ac.llm,'complete_json',AsyncMock(side_effect=[payload,repair])) as calls:result=await ac.assess_support('Options:\n'+'\n'.join(options),ac.Assessments.model_json_schema(),options,sources,diagnostics)
  self.assertEqual(calls.call_count,2);self.assertEqual(result[0]['claims'][0]['status'],'unsupported');self.assertEqual(result[0]['claims'][0]['citations'],[])
if __name__=='__main__':unittest.main()
