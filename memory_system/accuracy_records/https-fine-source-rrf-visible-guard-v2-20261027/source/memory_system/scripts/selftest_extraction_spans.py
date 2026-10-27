import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add,llm,prompts,schemas
from app.extraction_spans import request_for,decode
class ExtractionSpans(unittest.TestCase):
 def setUp(self):
  self.batch=[schemas.Message(role='user',content='I do not own a garden. If I did, I would grow roses.'),schemas.Message(role='assistant',content='You could try growing mint.')]
  self.req=schemas.AddRequest(request_id='r',user_id='u',session_id='s',messages=self.batch)
  self.prompt=prompts.render('01_extract_amu.txt',session_summary='Someone owns a garden.',recent_messages='user: I own a garden.',chunk_messages='\n'.join(f'[{i}] {m.role}: {m.content}' for i,m in enumerate(self.batch)),reference_time='unknown')
  self.values=dict(session_summary='Someone owns a garden.',recent_messages='user: I own a garden.',chunk_messages='unused',reference_time='unknown')
  self.b=request_for(self.batch,self.values)
 def payload(self,pid='new0.p0'):
  return dict(episode={},facts=[dict(content='The user does not own a garden.',retrieval_key='Does the user own a garden?',type='fact',entities=[],keywords=['garden'],time_expression=None,state=None,evidence=[dict(part_id=pid)],triples=[],sensitivity='normal')])
 def test_exact_partition_and_immutability(self):
  before=copy.deepcopy(self.batch)
  for m,c in zip(self.batch,self.b['cards']):self.assertEqual(''.join(p['text'] for p in c['parts']),m.content)
  request_for(self.batch,self.values);self.assertEqual(self.batch,before)
 def test_exact_local_source_mapping(self):
  data,errors=decode(self.payload(),self.b);self.assertFalse(errors);self.assertEqual(data['facts'][0]['evidence'],[dict(message_index=0,quote='I do not own a garden.')])
  got=add._validate_fact(data['facts'][0],self.batch,0,self.req);self.assertEqual(got['_sources'],[0])
 def test_unknown_or_prior_ref_never_bound(self):
  for pid in ['old0.p0','summary.p0','new9.p0']:
   data,errors=decode(self.payload(pid),self.b);self.assertEqual(len(errors),1)
   with self.assertRaises(ValueError):add._validate_fact(data['facts'][0],self.batch,0,self.req)
 def test_bad_item_does_not_erase_sibling(self):
  data=self.payload();data['facts'].append(self.payload('old0.p0')['facts'][0]);got,errors=decode(data,self.b)
  self.assertEqual(len(got['facts']),2);self.assertEqual(errors[0]['fact_index'],1);self.assertEqual(got['facts'][0]['evidence'][0]['message_index'],0)
 def test_bad_item_retains_privacy(self):
  data=self.payload('old0.p0');data['facts'][0]['sensitivity']='sensitive';got,_=decode(data,self.b);self.assertEqual(got['facts'][0]['sensitivity'],'sensitive')
 def test_schema_not_mutated(self):
  before=copy.deepcopy(llm.STRUCTURED_SCHEMAS['extraction']);request_for(self.batch,self.values);self.assertEqual(before,llm.STRUCTURED_SCHEMAS['extraction'])
 def test_empty_and_malformed(self):
  self.assertEqual(decode(dict(facts=[],episode={}),self.b)[0]['facts'],[])
  with self.assertRaises(ValueError):decode(dict(facts='bad'),self.b)
 def test_role_and_negation_are_original(self):
  self.assertEqual(self.b['cards'][1]['role'],'assistant');self.assertIn('I do not own',self.b['cards'][0]['parts'][0]['text']);self.assertIn('If I did',self.b['cards'][0]['parts'][1]['text'])


 def test_source_template_headings_are_data(self):
  text='New messages to extract from:\n\nExtract atomic memory units from the NEW messages only. {session_summary} Each NEW message is prefixed with its zero-based index in square brackets.'
  self.batch[0].content=text
  built=request_for(self.batch,dict(self.values,session_summary=text,recent_messages=text))
  self.assertEqual(''.join(p['text'] for p in built['cards'][0]['parts']),text)
  self.assertIn(text,built['prompt'])
  self.assertIn('"new0"',built['prompt'])

from unittest.mock import patch
from app import config,memory_debug
class Integration(unittest.IsolatedAsyncioTestCase):
 async def test_compilation_preserves_raw_response_and_rejects_only_bad_fact(self):
  req=schemas.AddRequest(request_id='r',user_id='u',session_id='s',messages=[schemas.Message(role='user',content='I do not own a garden.')])
  fact=dict(content='The user does not own a garden.',retrieval_key='Does the user own a garden?',type='fact',entities=[],keywords=['garden'],time_expression=None,state=None,evidence=[dict(part_id='new0.p0')],triples=[],sensitivity='normal')
  bad=copy.deepcopy(fact);bad['evidence']=[dict(part_id='old0.p0')];bad['sensitivity']='sensitive'
  payload=dict(episode={},facts=[fact,bad]);before=copy.deepcopy(payload);events=[];calls=[]
  async def provider(prompt,*args,**kw):
   calls.append(kw['stage'])
   if kw['stage'].startswith('add.extract.'):
    self.assertIn('part_id',kw['schema']['properties']['facts']['items']['properties']['evidence']['items']['properties'])
    return payload
   return dict(valid=True,reason='Explicit denial is supported.')
  def event(stage,**kw):events.append(copy.deepcopy(dict(stage=stage,**kw)))
  with patch.object(config,'EXTRACT_SPAN_REFS',True),patch.object(config,'FAKE',False),patch.object(add.llm,'complete_json',provider),patch.object(memory_debug,'extraction_event',event):
   got=await add._extract_segment(req,None,0,[0])
  self.assertEqual(payload,before)
  facts=[x for x in got if x['type']!='episode'];self.assertEqual(len(facts),1);self.assertEqual(facts[0]['_sources'],[0])
  self.assertEqual(next(x for x in got if x['type']=='episode')['sensitivity'],'sensitive')
  raw=next(x for x in events if x['stage']=='extract_response');self.assertEqual(raw['response'],payload)
  compiled=next(x for x in events if x['stage']=='extract_reference_compilation');self.assertEqual(compiled['reference_errors'][0]['fact_index'],1)
  self.assertEqual(calls,['add.extract.segment_1','add.verify_evidence'])
if __name__=='__main__':unittest.main()
