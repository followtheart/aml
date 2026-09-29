"""Actual repair requests bind exact spans to pending options and preserve accepted siblings."""
import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,choice_premises,config,llm,prompts

class RepairSpanTests(unittest.IsolatedAsyncioTestCase):
 async def invoke(self,enabled,wrong_span=False):
  options=['A. Since you already own a kayak, try a quiet lake.','B. Take a walk.']
  sources={'s0':dict(id='s0',role='user',text='I own a kayak.',request_id='r',message_index=0),
           's1':dict(id='s1',role='assistant',text='I own a kayak.',request_id='r',message_index=1)}
  witnesses={'A':[dict(source_id='s1',role='assistant',quote='I own a kayak.')]}
  d=dict(answer_witness_prefill=witnesses);calls=[]
  text=next(s['text'] for s in choice_premises.option_spans('A',options[0]) if s['id']=='A:1')
  async def provider(prompt,*args,**kw):
   calls.append(dict(stage=kw['stage'],prompt=prompt,schema=kw['schema']))
   if len(calls)==1:
    return dict(options={'A':dict(kind='personal',claims=[dict(text=text,status='supported',premise_type='ownership',reason='none',citations=[dict(source_id='s1',quote='I own a kayak.',basis='self_report',subject='current_user')])]),'B':dict(kind='generic',claims=[])})
   claim=dict(status='unsupported',premise_type='ownership',reason='no_source',citations=[])
   claim.update({'span_id':'B:1' if wrong_span else 'A:1'} if enabled else {'text':text})
   return dict(options=[dict(letter='A',kind='personal',claims=[claim])])
  prompt=prompts.render('12_choice_support.txt',question='Any ideas?',question_date='',task_instructions='',options='\n'.join(options),witnesses=ac.choice_witness.render(witnesses),sources=json.dumps(ac._cards(sources),ensure_ascii=False))
  with patch.object(config,'CHOICE_REPAIR_SPAN_REFS',enabled),patch.object(config,'CHOICE_REPAIR_CONTEXT',True),patch.object(config,'CHOICE_ALLOW_INFERRED',False),patch.object(llm,'complete_json',provider):
   entries=await ac.assess_support(prompt,ac.Assessments.model_json_schema(),options,sources,d)
  self.assertEqual(len(calls),2);self.assertEqual(entries[1]['letter'],'B');self.assertEqual(entries[1]['status'],'generic')
  return entries,d,calls
 async def test_enabled_source_error_uses_span_contract(self):
  entries,d,calls=await self.invoke(True);c=calls[1]['schema']['$defs']['Claim']
  self.assertNotIn('text',c['properties']);self.assertIn('span_id',c['required']);self.assertTrue(all(s.startswith('A:') for s in c['properties']['span_id']['enum']))
  self.assertEqual(d['answer_repair_span_contract'][0]['pending'],['A']);self.assertTrue(d['answer_repair_span_contract'][0]['extended']);self.assertEqual(entries[0]['claims'][0]['status'],'unsupported')
 async def test_disabled_preserves_existing_request(self):
  _,d,calls=await self.invoke(False);self.assertIn('text',calls[1]['schema']['$defs']['Claim']['properties']);self.assertNotIn('answer_repair_span_contract',d)
 async def test_other_option_span_rejected_without_erasing_sibling(self):
  entries,d,_=await self.invoke(True,True);self.assertEqual(entries[0]['validation_status'],'unresolved');self.assertEqual(d['answer_support_validation']['unresolved_options'],['A'])
 async def test_exact_span_decodes_to_same_claim(self):
  baseline,_,_=await self.invoke(False);candidate,_,_=await self.invoke(True);self.assertEqual(baseline,candidate)
if __name__=='__main__':unittest.main()
