import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.citation_text import original_profile_json_quote as original_quote
class TokenBoundaryTests(unittest.TestCase):
 def fix(self,text,quote,declared='persona'):return original_quote(dict(text=text,declared=declared),quote)
 def test_restores_literal_span(self):
  text='  "role": {\n  "title": "Safety Advocate",\n';r=self.fix(text,'"role":{"title":"Safety Advocate"');self.assertEqual(text[r['start']:r['end']],r['quote'])
 def test_string_value_whitespace_is_not_normalized(self):self.assertIsNone(self.fix('"title": "Safety Advocate"','"title":"SafetyAdvocate"'))
 def test_words_case_negation_and_punctuation_unchanged(self):
  for q in ['"role":"not teacher"','"role":"Teacher"','"role"="teacher"']:
   self.assertIsNone(self.fix('"role": "teacher"',q))
 def test_user_or_assistant_text_not_repaired(self):
  for role in ['user','assistant',None]:self.assertIsNone(self.fix('"role": "teacher"','"role":"teacher"',role))
 def test_ambiguous_rejected(self):self.assertIsNone(self.fix('"role": "teacher", "role": "teacher"','"role":"teacher"'))
 def test_exact_unchanged(self):self.assertIsNone(self.fix('"role": "teacher"','"role": "teacher"'))
 def test_unclosed_string_rejected(self):self.assertIsNone(self.fix('"role": "teacher','"role":"teacher'))
 def test_escaped_quotes_remain_literal(self):
  self.assertIsNotNone(self.fix('"note": "say \\"no\\" now"','"note":"say \\"no\\" now"'))
 def test_separate_numeric_tokens_cannot_be_merged(self):
  self.assertIsNone(self.fix('"count": 1 2','"count":12'))
 def test_escaped_value_spelling_cannot_be_changed(self):
  self.assertIsNone(self.fix('"value": "a"','"value":"\\u0061"'))
 def test_scalar_value_and_sign_preserved(self):
  for text,quote in [('"count": -12','"count":12'),('"ready": false','"ready":true'),('"count": 1.0','"count":1')]:self.assertIsNone(self.fix(text,quote))

import copy
from app import answer_choice as ac

class GateTests(unittest.TestCase):
 def setUp(self):
  self.sources={'s0':dict(role='user',declared='persona',text='  "occupation": {\n  "job_title": "Teacher",\n')}
  self.citation=ac.Citation(source_id='s0',quote='"occupation":{"job_title":"Teacher"',basis='self_report',subject='current_user')
 def test_literal_span_and_inputs_preserved(self):
  before=copy.deepcopy((self.sources,self.citation.model_dump()));c=ac._check_citation(self.citation,'you are a teacher',self.sources,'occupation')
  self.assertTrue(c['valid']);self.assertEqual(c['quote_normalization'],'persona_json_layout');span=c['source_span'];self.assertEqual(self.sources['s0']['text'][span['start']:span['end']],c['quote']);self.assertEqual((self.sources,self.citation.model_dump()),before)
 def test_unknown_subject_not_repaired(self):
  c=ac._check_citation(self.citation.model_copy(update={'subject':'unknown'}),'you are a teacher',self.sources,'occupation');self.assertIn('not_current_user',c['validation_errors'])
 def test_occupation_cannot_establish_other_experience(self):
  c=ac._check_citation(self.citation,'you climbed Mount Everest',self.sources,'experience');self.assertIn('occupation_profile_does_not_prove_event',c['validation_errors'])
 def test_different_value_stays_invalid(self):
  c=ac._check_citation(self.citation.model_copy(update={'quote':'"occupation":{"job_title":"Pilot"'}),'you are a pilot',self.sources,'occupation');self.assertIn('quote_not_in_source',c['validation_errors'])
if __name__=='__main__':unittest.main()
