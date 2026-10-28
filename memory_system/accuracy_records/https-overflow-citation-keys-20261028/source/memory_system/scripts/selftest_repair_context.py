import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.repair_context import triggered_witnesses,triggered_feedback,targets
class Tests(unittest.TestCase):
 def setUp(self):
  self.s={f's{i}':dict(role='assistant' if i==7 else 'user',request_id='r',message_index=i,text=f'Text {i}') for i in [2,4,6,7]}
  self.f={'A':json.dumps(dict(claims=[dict(dropped_citations=[dict(source_id='s7',quote='Text 7',errors=['not_user_source'])])]))}
  self.w={'A':[dict(source_id='s7',quote='Text 7')],'B':[dict(source_id='s7',quote='Text 7')]}
 def call(self):return triggered_witnesses(self.w,['A','B'],self.s,self.f)
 def test_scoped_and_bounded(self):
  r=self.call();self.assertEqual([x['source_id'] for x in r['A']],['s6','s4']);self.assertEqual(r['B'],self.w['B'])
 def test_no_error_identity(self):
  self.f={};self.assertEqual(self.call(),self.w)
 def test_different_error_identity(self):
  self.f={'A':json.dumps(dict(source_id='s7',errors=['quote_not_in_source']))};self.assertEqual(self.call(),self.w)
 def test_role_authoritative(self):
  self.s['s7']['declared']='user';self.assertEqual(self.call(),self.w)
 def test_unknown_source(self):
  self.s.pop('s7');self.assertEqual(self.call(),self.w)
 def test_request_boundary(self):
  self.s['s6']['request_id']='other';self.assertEqual([x['source_id'] for x in self.call()['A']],['s4'])
 def test_size_and_time(self):
  self.s['s6']['message_index']=8;self.s['s4']['text']='é'*801;self.assertEqual(self.call()['A'],[])
 def test_no_mutation(self):
  before=copy.deepcopy((self.w,self.s,self.f));self.call();self.assertEqual((self.w,self.s,self.f),before)
 def test_feedback_only_rejected_quote(self):
  feedback=dict(required_options=['A','B'],accepted_options=['C'],invalid_options=self.f|{'B':'unchanged'},previous_invalid_options=[dict(letter='A',claims=[dict(source_id='s7',quote='bad'),dict(source_id='s4',quote='good')]),dict(letter='B',claims=[dict(source_id='s7',quote='preserve')])])
  before=copy.deepcopy(feedback);r=triggered_feedback(feedback,self.s,self.f)
  self.assertEqual(feedback,before);self.assertNotIn('quote',r['previous_invalid_options'][0]['claims'][0]);self.assertEqual(r['previous_invalid_options'][0]['claims'][1]['quote'],'good');self.assertEqual(r['previous_invalid_options'][1],before['previous_invalid_options'][1]);self.assertIn('not_user_source',r['invalid_options']['A'])
 def test_duplicate_and_no_pending(self):
  self.w['A']=[dict(source_id='s6',quote='Text 6')];self.assertEqual(len(self.call()['A']),2);self.assertEqual(triggered_witnesses(self.w,[],self.s,self.f),{})
 def test_unaffected_feedback_identity(self):
  f=dict(required_options=['B'],accepted_options=[],invalid_options={'B':'no source'},previous_invalid_options=[dict(letter='B',source_id='s7',quote='unchanged')]);self.assertEqual(triggered_feedback(f,self.s,self.f),f)
if __name__=='__main__':unittest.main()
