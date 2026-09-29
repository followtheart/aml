import copy,unittest,sys,os
from pathlib import Path
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from app.recovery_spans import build,decode,decode_partial
class SpanRefs(unittest.TestCase):
 def setUp(self):
  self.req=SimpleNamespace(query='How could I relax?',options=['A. Try painting.','B. Take a walk.'])
  self.cards=[dict(candidate_id='one',sources=[dict(source_id='s0',role='user',text='I enjoy painting. It calms me.',request_id='r',message_index=0)]),dict(candidate_id='two',sources=[dict(source_id='s0',role='user',text='I enjoy walking.',request_id='r',message_index=1)])]
 def yes(self,pid='s0.p0'):return dict(useful=True,target_id='question',evidence_kind='user_context',citation_parts=[pid])
 def no(self):return dict(useful=False,target_id='',evidence_kind='none',citation_parts=[])
 def test_roundtrip_and_no_mutation(self):
  before=copy.deepcopy(self.cards);b=build(self.req,self.cards);r,j=decode(dict(c0=self.yes(),c1=self.no()),b);self.assertEqual(r,{'one'});self.assertEqual(j[0]['citations'][0]['quote'],'I enjoy painting.');self.assertEqual(self.cards,before)
 def test_local_id_collision_rejected(self):
  b=build(self.req,self.cards)
  with self.assertRaises(ValueError):decode(dict(c0=self.yes('s1.p0'),c1=self.no()),b)
 def test_shared_source_mapping(self):
  self.cards[1]['sources']=[dict(self.cards[0]['sources'][0],source_id='local7')];b=build(self.req,self.cards);r,j=decode(dict(c0=self.yes(),c1=self.yes()),b);self.assertEqual(r,{'one','two'});self.assertEqual(j[1]['citations'][0]['source_id'],'local7')
 def test_attribution_check_preserved(self):
  self.cards[0]['sources'][0]['role']='assistant';b=build(self.req,self.cards);r,j=decode(dict(c0=self.yes(),c1=self.no()),b);self.assertFalse(r);self.assertIn('not_user_source',j[0]['validation_errors'])
 def test_unknown_target(self):
  b=build(self.req,self.cards);d=self.yes();d['target_id']='option:9'
  with self.assertRaises(ValueError):decode(dict(c0=d,c1=self.no()),b)
 def test_negative_nonempty_rejected(self):
  b=build(self.req,self.cards);d=self.no();d['citation_parts']=['s0.p0'];r,j=decode(dict(c0=d,c1=self.no()),b);self.assertFalse(r);self.assertFalse(j[0]['valid'])
 def test_missing_key(self):
  b=build(self.req,self.cards)
  with self.assertRaises(ValueError):decode(dict(c0=self.yes()),b)
 def test_whitespace_roundtrip(self):
  self.cards[0]['sources'][0]['text']=' \nI enjoy painting.\n\n It calms me.  ';b=build(self.req,self.cards);parts=[p['quote'] for p in b['spans'].values() if p['global_source']=='s0'];self.assertEqual(''.join(parts),self.cards[0]['sources'][0]['text'])
 def test_one_bad_decision_preserves_other_valid(self):
  b=build(self.req,self.cards);a=self.yes('s1.p0');z=self.yes('s1.p0');ids,rows=decode_partial(dict(c0=a,c1=z),b);self.assertEqual(ids,{'two'});self.assertFalse(rows[0]['valid']);self.assertTrue(rows[1]['valid'])
 def test_missing_decision_is_unreviewed(self):
  b=build(self.req,self.cards);ids,rows=decode_partial(dict(c0=self.yes()),b);self.assertEqual(ids,{'one'});self.assertFalse(rows[1]['valid'])
 def test_unknown_key_rejected(self):
  b=build(self.req,self.cards)
  with self.assertRaises(ValueError):decode_partial(dict(c0=self.yes(),c1=self.no(),unknown=self.no()),b)
 def test_no_invalid_promoted(self):
  b=build(self.req,self.cards);ids,rows=decode_partial(dict(c0=self.yes('not_there'),c1=self.yes('s0.p0')),b);self.assertEqual(ids,set());self.assertTrue(all(not x['valid'] for x in rows))

from unittest.mock import patch
from app import answer_context,budget,config,listwise_recovery as lr,llm
class RecoveryIntegration(unittest.IsolatedAsyncioTestCase):
 async def test_flagged_recovery_preserves_good_items_and_budget(self):
  def row(mid,index,text):
   source=dict(request_id='case',message_index=index,source_event_id='event'+str(index),role='user',content=text)
   return dict(id=mid,content=text,_rank_text=answer_context.with_evidence(text,[source]),_packet_item=dict(sources=[source]))
  keep=row('keep',0,'I enjoy tea.');a=row('a',1,'I enjoy painting.');b=row('b',2,'I enjoy hiking.')
  req=SimpleNamespace(query='How could I relax?',options=['A. Paint.','B. Hike.'])
  seen=[]
  async def model(prompt,**kw):
   budget.current.get().before_call();seen.append(kw['schema'])
   return dict(c0=dict(useful=True,target_id='question',evidence_kind='user_context',citation_parts=['s0.p0']),c1=dict(useful=True,target_id='question',evidence_kind='user_context',citation_parts=['s0.p0']))
  with patch.object(config,'LISTWISE_SPAN_REFS',True),patch.object(llm,'complete_json',model),budget.scope(seconds=30,calls=1,tokens=40000) as limits:
   selected,trace=await lr.recover(req,[a,b],[keep],{0,1})
  self.assertEqual([x['id'] for x in selected],['keep','a']);self.assertEqual(trace['status'],'partial');self.assertEqual(trace['restored_ids'],['a']);self.assertEqual(trace['unreviewed_ids'],['b']);self.assertEqual(trace['citation_format'],'source_span_refs_v1');self.assertEqual(limits.calls,1);self.assertEqual((limits.reserved_calls,limits.reserved_tokens),(0,0));self.assertEqual(seen[0]['required'],['c0','c1'])
 def test_disabled_request_retains_legacy_contract(self):
  req=SimpleNamespace(query='How could I relax?',options=[])
  cards=[dict(candidate_id='a',sources=[dict(source_id='s0',role='user',text='I enjoy painting.')])]
  with patch.object(config,'LISTWISE_SPAN_REFS',False):request=lr.request_for(req,cards)
  self.assertIn('decisions',request['schema']['properties']);self.assertNotIn('c0',request['schema']['properties'])
if __name__=='__main__':unittest.main()
