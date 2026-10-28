import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac
from app.citation_text import original_quote
from app import config
from unittest.mock import patch

class TypographyTests(unittest.TestCase):
 def setUp(self):
  self.addCleanup(patch.stopall)
  patch.object(config,'CHOICE_TYPOGRAPHIC_QUOTES',True).start()
 def test_disabled_keeps_strict_match(self):
  with patch.object(config,'CHOICE_TYPOGRAPHIC_QUOTES',False):
   got=self.check('I’ve owned a kayak.','user',"I've owned a kayak.")
  self.assertIn('quote_not_in_source',got['validation_errors']);self.assertNotIn('quote_normalization',got)
 def test_ascii_apostrophe(self):
  text='Prefix. I’ve been reading. End.';r=original_quote(text,"I've been reading.")
  self.assertEqual(r,dict(quote='I’ve been reading.',start=8,end=26));self.assertEqual(text[r['start']:r['end']],r['quote'])
 def test_reverse_and_double_quotes(self):
  self.assertEqual(original_quote('I said "yes".', 'I said “yes”.')['quote'],'I said "yes".')
  self.assertEqual(original_quote("I've read it.",'I’ve read it.')['quote'],"I've read it.")
 def test_exact_match_unchanged_even_with_folded_duplicate(self):
  self.assertIsNone(original_quote("I've read it. I’ve read it.","I've read it."))
 def test_ambiguous_normalized_match_rejected(self):
  self.assertIsNone(original_quote('I’ve read it. I’ve read it.',"I've read it."))
 def test_words_negation_space_case_and_dashes_not_repaired(self):
  for quote in ["I've never read it.","I've  read it.","i've read it.","I've read-it."]:
   self.assertIsNone(original_quote('I’ve read it.',quote))
 def test_no_cross_source_or_empty_match(self):
  self.assertIsNone(original_quote('',"I've read it."));self.assertIsNone(original_quote('I’ve read it.',''))
 def test_punctuation_alone_not_repaired(self):self.assertIsNone(original_quote('‘',"'"))
 def test_other_unicode_not_folded(self):self.assertIsNone(original_quote('Ｉ’ve read it.',"I've read it."))
 def test_preserves_input_and_original_offsets(self):
  sources={'s0':dict(role='user',text='Here: I’ve owned a kayak for years.')}
  citation=ac.Citation(source_id='s0',quote="I've owned a kayak for years.",basis='self_report',subject='current_user');before=copy.deepcopy((sources,citation.model_dump()))
  checked=ac._check_citation(citation,'you own a kayak',sources,'ownership')
  self.assertTrue(checked['valid']);self.assertEqual(checked['quote'],'I’ve owned a kayak for years.')
  self.assertEqual(checked['source_span'],dict(start=6,end=35));self.assertEqual((sources,citation.model_dump()),before)
 def test_assistant_stays_rejected(self):
  checked=self.check('I’ve owned a kayak.','assistant',"I've owned a kayak.")
  self.assertIn('not_user_source',checked['validation_errors'])
 def test_third_party_stays_rejected(self):
  checked=self.check('My sister wrote: I’ve owned a kayak.','user',"I've owned a kayak.")
  self.assertFalse(checked['valid']);self.assertIn('different_person',checked['validation_errors'])
 def test_denial_stays_rejected(self):
  checked=self.check('I don’t own a kayak.','user',"I don't own a kayak.")
  self.assertIn('denied_event',checked['validation_errors'])
 def test_left_quote_is_not_a_supported_apostrophe(self):
  checked=self.check('I don‘t own a kayak.','user',"I don't own a kayak.")
  self.assertFalse(checked['valid']);self.assertIn('quote_not_in_source',checked['validation_errors'])
 def test_unknown_subject_stays_rejected(self):
  checked=self.check('I’ve owned a kayak.','user',"I've owned a kayak.",subject='unknown')
  self.assertIn('not_current_user',checked['validation_errors'])
 def check(self,text,role,quote,subject='current_user'):
  return ac._check_citation(ac.Citation(source_id='s0',quote=quote,basis='self_report',subject=subject),'you own a kayak',{'s0':dict(role=role,text=text)},'ownership')
if __name__=='__main__':unittest.main()
