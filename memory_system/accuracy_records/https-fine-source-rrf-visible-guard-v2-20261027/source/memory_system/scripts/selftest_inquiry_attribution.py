"""Attribution entrance preserves topic interest without inventing stronger facts."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from app import answer_choice as ac
class CitationInquiryTests(unittest.TestCase):
 def cite(self,text,claim,premise_type,basis='self_report',**extra):
  sources={'s0':dict(role='user',text=text,**extra)}
  c=ac.Citation(source_id='s0',quote=text,basis=basis,subject='current_user')
  return ac._check_citation(c,claim,sources,premise_type)
 def test_pure_inquiry_is_still_interest(self):
  c=self.cite("I've been wondering—why do potters use different glazes?",'you are interested in pottery','interest')
  self.assertTrue(c['valid']);self.assertEqual(c['basis'],'topic_interest');self.assertEqual(c['strength_gap'],0)
 def test_pure_inquiry_cannot_anchor_an_experience(self):
  c=self.cite("I'm curious: why do people attend retreats?",'you attended a retreat','experience')
  self.assertTrue(c['valid']);self.assertEqual(c['basis'],'topic_interest');self.assertEqual(c['strength_gap'],1)
  claim=dict(status='unsupported',reason='no_source',validation_errors=[],citations=[c]);self.assertFalse(ac._recoverable(claim))
 def test_pure_inquiry_cannot_establish_ownership_or_diagnosis(self):
  for claim,kind in [('you own a greenhouse','ownership'),('you have chronic headaches','condition')]:
   c=self.cite('I wonder how plants grow.',claim,kind);self.assertTrue(c['valid']);self.assertEqual(c['basis'],'context');self.assertFalse(c['anchor'])
 def test_context_label_cannot_promote_question(self):
  c=self.cite('I wonder how plants grow.','you worked in a greenhouse','experience','context');self.assertEqual(c['basis'],'context');self.assertFalse(c['anchor'])
 def test_real_personal_clause_is_preserved(self):
  for text,claim,kind in [('I wonder why my greenhouse gets cold.','you own a greenhouse','ownership'),("I've been wondering why I felt dizzy yesterday.",'you felt dizzy yesterday','experience'),('I wonder why plants grow. I own a greenhouse.','you own a greenhouse','ownership')]:
   c=self.cite(text,claim,kind);self.assertTrue(c['valid']);self.assertEqual(c['basis'],'self_report');self.assertEqual(c['strength_gap'],0)
 def test_persona_and_received_attribution_unchanged(self):
  c=self.cite('I wonder how plants grow.','you like gardening','interest',declared='persona');self.assertEqual(c['basis'],'self_report')
  c=self.cite('Jordan wrote: I wonder why I felt dizzy.','you felt dizzy yesterday','experience');self.assertFalse(c['valid']);self.assertIn('third_party_or_hypothetical',c['validation_errors'])
 def test_exact_quote_requirement_remains(self):
  c=ac.Citation(source_id='s0',quote='I wonder why plants grow.',basis='self_report',subject='current_user')
  self.assertFalse(ac._check_citation(c,'you like gardening',{'s0':dict(text='Different text',role='user')},'interest')['valid'])
class MixedEvidenceTests(unittest.TestCase):
 cite = CitationInquiryTests.cite
 def test_context_question_alone_cannot_recover_strong_fact(self):
  ref=self.cite('I wonder how morning stretches affect mood.','you practice stretching daily','habit')
  claim=dict(status='unsupported',reason='no_source',validation_errors=[],citations=[ref])
  self.assertFalse(ac._recoverable(claim))
 def test_context_can_accompany_a_separate_direct_anchor(self):
  context=self.cite('I wonder how morning stretches affect mood.','you practice stretching daily','habit')
  direct=self.cite('I start each morning with slow stretches.','you practice stretching daily','habit')
  claim=dict(status='unsupported',reason='no_source',validation_errors=[],citations=[context,direct])
  self.assertFalse(context['anchor']);self.assertTrue(direct['anchor']);self.assertTrue(ac._recoverable(claim))

if __name__=='__main__':unittest.main()

