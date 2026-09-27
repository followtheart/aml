import copy, unittest
from app import choice_witness as gate

class LengthIsolation(unittest.TestCase):
 def setUp(self):
  self.payload={'options':[{'letter':'A','witnesses':[{'source_id':'s0','quote':'x'*321}]},{'letter':'B','witnesses':[{'source_id':'s1','quote':'I enjoy photography.'}]}]}
  self.sources={'s0':{'text':'x'*321,'role':'user'},'s1':{'text':'I enjoy photography.','role':'user'}}
  self.options=['A. Since you enjoy art, paint.','B. Since you enjoy photography, take photos.']
 def test_preserves_valid_sibling_and_input(self):
  original=copy.deepcopy(self.payload);trace={};r=gate.validate(self.payload,self.options,self.sources,trace)
  self.assertEqual(set(r),{'B'});self.assertEqual(r['B'][0]['quote'],'I enjoy photography.');self.assertEqual(self.payload,original)
  self.assertEqual(trace['rejected_witnesses'],[dict(letter='A',source_id='s0',reason='quote_too_long')])
 def test_all_rejected_still_fails(self):
  self.payload['options'][1]['witnesses']=[]
  with self.assertRaises(ValueError):gate.validate(self.payload,self.options,self.sources)
 def test_other_schema_error_stays_fatal(self):
  self.payload['options'][1]['witnesses'][0]['extra']=True
  with self.assertRaises(ValueError):gate.validate(self.payload,self.options,self.sources)
 def test_duplicate_option_stays_fatal(self):
  self.payload['options'][1]['letter']='A'
  with self.assertRaises(ValueError):gate.validate(self.payload,self.options,self.sources)
 def test_cross_source_quote_not_rescued(self):
  self.payload['options'][1]['witnesses'][0]['source_id']='s0'
  with self.assertRaises(ValueError):gate.validate(self.payload,self.options,self.sources)
 def test_maximum_bound_preserved(self):
  self.payload['options'][0]['witnesses'][0]['quote']='x'*320
  self.assertIn('A',gate.validate(self.payload,self.options,self.sources))

if __name__ == '__main__':
 unittest.main()
