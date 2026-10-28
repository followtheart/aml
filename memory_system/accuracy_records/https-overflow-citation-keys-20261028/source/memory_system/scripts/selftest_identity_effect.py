import importlib.util,sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));R=Path(__file__).resolve().parent
from app import answer_choice as m
class Tests(unittest.TestCase):
 def claim(self,t,o):
  i=o.index(t);return dict(text=t,option_span=dict(start=i,end=i+len(t)),validation_errors=[],reason='none',status='unsupported',citations=[],premise_type='interest')
 def test_advice_effect(self):
  for t in ['so guests get a sense of both your history and personality.','so visitors can get a glimpse of your personal style.']:
   o='A. Try arranging new photos, '+t;self.assertEqual(m._nonasserted_scope_reason(self.claim(t,o),o),'generic_identity_advice_effect')
 def test_specific_identity_not_removed(self):
  for t in ['so guests get a sense of your history as a surgeon.','so guests get a sense of your love of jazz.','so guests get a sense of your personality and your lifelong hobby.']:
   o='A. Try arranging photos, '+t;self.assertIsNone(m._nonasserted_scope_reason(self.claim(t,o),o))
 def test_existing_history_not_removed(self):
  t='so guests get a sense of your personality.'
  for o in ['A. You already display photos, '+t,'A. Try a lamp. Your photos are displayed '+t,'A. Since you own these pictures, try displaying them '+t]:self.assertIsNone(m._nonasserted_scope_reason(self.claim(t,o),o))
 def test_invalid_evidence_not_removed(self):
  t='so guests get a sense of your personality.';o='A. Try arranging photos, '+t;c=self.claim(t,o);c['reason']='contradicted';self.assertIsNone(m._nonasserted_scope_reason(c,o))
 def test_hidden_fact_outside_effect_defers(self):
  t='so guests get a sense of both your history and personality'
  for ending in [' because you own a gallery.', ' because you visited Peru last year.', '. You own a gallery.']:
   o='A. Try arranging photos '+t+ending
   self.assertIsNone(m._nonasserted_scope_reason(self.claim(t,o),o))
 def test_whole_option_still_checked(self):
  t='so guests get a sense of your personality';o='A. Try new photos '+t+' because you own a gallery.'
  e=dict(letter='A',kind='personal',status='unsupported',primary_claim=0,option=o,claims=[self.claim(t,o)],validation_status='valid',validation_errors=[],removed_claims=[])
  es=m.normalize_nonasserted_scopes([e],{});checks=m.entailment_checks(es,{},[o]);self.assertEqual(checks[0]['option'],o)
  verified=m.validate_entailments(dict(checks=[dict(claim_id='A:option',entailed=False)]),es,checks);self.assertEqual(m.eligible_choices(verified,set()),[])
if __name__=='__main__':unittest.main()
