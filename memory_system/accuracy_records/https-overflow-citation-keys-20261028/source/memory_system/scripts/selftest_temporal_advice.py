import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from app import answer_choice as ac
temporal_advice = ac.temporal_advice_claim

def rewrite(entries):
    diagnostics = {}
    result = ac.normalize_nonasserted_scopes(entries, diagnostics)
    return result, diagnostics['answer_scope_normalization']['removed']
class TemporalAdviceTests(unittest.TestCase):
 def claim(self,text,option):
  a=option.index(text);return dict(text=text,option_span=dict(start=a,end=a+len(text)),reason='no_source',validation_errors=[])
 def accepts(self,prefix,body='it can help to pause.'):
  option='A. '+prefix+', '+body
  return temporal_advice(self.claim(prefix,option),option)
 def test_indefinite_scenario_with_explicit_advice(self):
  self.assertTrue(self.accepts('After a difficult day'))
  self.assertTrue(self.accepts('After any exhausting shift','you could take a break.'))
  self.assertTrue(self.accepts('After an argument','you may want some quiet time.'))
 def test_specific_or_historical_event_is_not_removed(self):
  for prefix in ('After that difficult day','After your difficult day','After a difficult day yesterday','After a meeting with Karen','After a difficult day at your job','After a difficult day last month','After a meeting in 2024','After a recent meeting','After a meeting at NASA'):
   self.assertFalse(self.accepts(prefix),prefix)
 def test_asserted_body_is_not_advice(self):
  self.assertFalse(self.accepts('After a difficult day','you started a new job.'))
 def test_claim_must_stay_in_antecedent(self):
  option='A. After a difficult day, it can help to pause.';claim=self.claim(option[3:],option)
  self.assertFalse(temporal_advice(claim,option))
 def test_invalid_span_and_claim_are_not_removed(self):
  option='A. After a difficult day, it can help to pause.';claim=self.claim('After a difficult day',option)
  for change in (dict(option_span=None),dict(validation_errors=['quote_not_in_source']),dict(reason='contradicted')):
   bad=dict(claim,**change);self.assertFalse(temporal_advice(bad,option))
 def test_complete_option_keeps_unextracted_history_visible(self):
  option='A. After a difficult day, you could relax because you own a sauna.';claim=self.claim('After a difficult day',option);claim.update(status='unsupported',premise_type='unknown',citations=[])
  entry=dict(letter='A',kind='personal',status='unsupported',primary_claim=0,option=option,claims=[claim],validation_status='valid',validation_errors=[],removed_claims=[]);original=copy.deepcopy(entry)
  entries,removed=rewrite([entry]);self.assertEqual(entry,original);self.assertEqual(len(removed),1)
  checks=ac.entailment_checks(entries,{},[option]);self.assertEqual(checks[0]['option'],option)
  validated=ac.validate_entailments({'checks':[dict(claim_id='A:option',entailed=False)]},entries,checks);self.assertEqual(ac.eligible_choices(validated,set()),[])
if __name__=='__main__':unittest.main()
