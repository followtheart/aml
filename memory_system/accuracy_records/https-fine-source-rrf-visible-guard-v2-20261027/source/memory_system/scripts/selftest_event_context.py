"""Regression boundaries for source-grounded reader recovery."""
import copy,os,sys,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac

def enrich(checks):
    diagnostics = {}
    values = ac.enrich_event_context(checks, diagnostics)
    return values, diagnostics['answer_event_context']['added']

class TemporalContextTests(unittest.TestCase):
 def check(self,claim='you taught during the spring term',kind='experience'):
  return dict(claim_id='A:0',check_type='premise',claim=claim,premise_type=kind,sources=[dict(id='s0',quote='I teach history.',context='I teach history.',role='user')])
 def test_qualifier_is_exact_and_sources_and_identity_stay_unchanged(self):
  check=self.check();before=copy.deepcopy(check);values,added=enrich([check]);self.assertEqual(check,before);self.assertEqual(values[0]['required_event_context'],['during the spring term']);self.assertEqual(values[0]['sources'],before['sources']);self.assertEqual(values[0]['claim_id'],'A:0');self.assertEqual(added[0]['claim_id'],'A:0')
 def test_experience_without_qualifier_unchanged(self):
  c=self.check('you taught history');self.assertEqual(enrich([c]),([c],[]))
 def test_other_types_are_unchanged(self):
  for kind in ('interest','habit','condition','ownership','occupation','location','unknown'):
   c=self.check(kind=kind);self.assertEqual(enrich([c]),([c],[]))
 def test_complete_options_are_not_rewritten(self):
  c=self.check();c['check_type']='option';self.assertEqual(enrich([c]),([c],[]))
 def test_qualifiers_end_at_clause_punctuation(self):
  c=self.check('you taught during spring; you also mentored after class.');v,_=enrich([c]);self.assertEqual(v[0]['required_event_context'],['during spring','after class'])
 def test_no_new_checks_or_inferred_facts(self):
  c=self.check();v,_=enrich([c]);self.assertEqual(len(v),1);self.assertNotIn('verdict',v[0]);self.assertNotIn('entailed',v[0]);self.assertEqual(enrich([]),([],[]))
if __name__=='__main__':unittest.main()
