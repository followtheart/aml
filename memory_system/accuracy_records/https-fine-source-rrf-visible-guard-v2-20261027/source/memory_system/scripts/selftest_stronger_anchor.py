"""Stronger direct anchors must preserve valid topical references and source boundaries."""
import copy
import unittest
import selftest_answer_choice  # Configure offline dependencies first.
from app import answer_choice as gate

class StrongerAnchorTests(unittest.TestCase):
 def fixture(self):
  text='Since you are passionate about surfing when near the coast'
  sources={'weak':dict(id='weak',role='user',text='Why do coastal travel guides discuss surfing?'),'direct':dict(id='direct',role='user',text='I waxed my board, though the real joy came in catching those early waves.')}
  prior=gate._check_citation(gate.Citation(source_id='weak',quote=sources['weak']['text'],basis='topic_interest',subject='current_user'),text,sources,'interest')
  assert prior['valid']
  claim=dict(text=text,status='supported',reason='none',premise_type='interest',citations=[prior],validation_errors=[])
  entries=[dict(letter='A',kind='personal',status='supported',validation_status='valid',validation_errors=[],claims=[claim])]
  return entries,sources
 def test_adds_stronger_anchor_preserving_prior_and_status(self):
  e,s=self.fixture();prior=copy.deepcopy(e[0]['claims'][0]['citations'][0]);d={}
  gate.recover_direct_board_wave_citations(e,s,d);c=e[0]['claims'][0]
  self.assertEqual(c['citations'][0],prior);self.assertEqual(c['citations'][1]['source_id'],'direct');self.assertTrue(c['citations'][1]['valid']);self.assertEqual(c['status'],'supported');self.assertEqual(c['citation_recovery'],'awaiting_entailment')
 def test_idempotent(self):
  e,s=self.fixture();gate.recover_direct_board_wave_citations(e,s,{});before=copy.deepcopy(e);gate.recover_direct_board_wave_citations(e,s,{});self.assertEqual(e,before)
 def test_no_stronger_evidence_no_change(self):
  e,s=self.fixture();del s['direct'];before=copy.deepcopy(e);gate.recover_direct_board_wave_citations(e,s,{});self.assertEqual(e,before)
 def test_assistant_cannot_supply_anchor(self):
  e,s=self.fixture();s['direct']['role']='assistant';before=copy.deepcopy(e);gate.recover_direct_board_wave_citations(e,s,{});self.assertEqual(e,before)
 def test_invalid_prior_not_enriched(self):
  e,s=self.fixture();e[0]['claims'][0]['citations'][0]['valid']=False;before=copy.deepcopy(e);gate.recover_direct_board_wave_citations(e,s,{});self.assertEqual(e,before)
 def test_reference_limit_retained(self):
  e,s=self.fixture();e[0]['claims'][0]['citations']*=3;before=copy.deepcopy(e);gate.recover_direct_board_wave_citations(e,s,{});self.assertEqual(e,before)

if __name__ == "__main__":
    unittest.main()
