import copy,importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));R=Path(__file__).resolve().parent
from app import answer_choice as m
class Gate(unittest.TestCase):
 def checks(self):return [dict(claim_id='A:0',check_type='premise',premise_type='experience',claim='you taught classes during remote schooling',required_event_context=['during remote schooling'],sources=[dict(id='s0',role='user',quote='I teach classes.',context='I teach classes.')],context_neighbors=[])]
 def test_source_and_literal_context_preserved(self):
  checks=self.checks();old=copy.deepcopy(checks);diag={};expanded=m.add_event_subchecks(checks,diag);self.assertEqual(checks,old);child=expanded[-1];self.assertIn(child['claim'],child['parent_claim']);self.assertEqual(child['sources'],checks[0]['sources']);self.assertEqual(child['context_neighbors'],[])
 def test_failed_frame_blocks_otherwise_supported_parent(self):
  d={};checks=m.add_event_subchecks(self.checks(),d);v={'checks':[{'claim_id':'A:0','entailed':True},{'claim_id':'A:0:event:0','entailed':False}]};result=m.apply_event_subcheck_results(v,checks,d);self.assertFalse(result['checks'][0]['entailed']);self.assertTrue(v['checks'][0]['entailed'])
 def test_passed_frame_never_promotes_unsupported_parent(self):
  d={};checks=m.add_event_subchecks(self.checks(),d);v={'checks':[{'claim_id':'A:0','entailed':False},{'claim_id':'A:0:event:0','entailed':True}]};self.assertFalse(m.apply_event_subcheck_results(v,checks,d)['checks'][0]['entailed'])
 def test_no_personal_premise_is_not_passing_frame(self):
  d={};checks=m.add_event_subchecks(self.checks(),d);payload={'checks':[{'claim_id':c['claim_id'],'verdict':'no_personal_premise'} for c in checks]};self.assertFalse(m._typed_verdicts(payload,checks)['checks'][-1]['entailed'])
 def test_noun_while_does_not_create_event_frame(self):
  checks=self.checks();checks[0].update(claim='you took a while to process regret',required_event_context=['while to process regret']);d={};self.assertEqual(m.add_event_subchecks(checks,d),checks);self.assertEqual(d['answer_event_subchecks']['skipped'][0]['reason'],'noun_while')
 def test_expansion_limits_keep_original_checks(self):
  checks=self.checks();checks[0]['required_event_context']=['during remote schooling']*10;d={};result=m.add_event_subchecks(checks,d);self.assertEqual(result[:1],checks);self.assertEqual(len(result),9);self.assertLessEqual(d['answer_event_subchecks']['added_bytes'],12000)
 def test_positive_source_rule_cannot_override_failed_frame(self):
  checks=self.checks();checks[0].update(claim='you keep an eye on your cholesterol during pregnancy',required_event_context=['during pregnancy'],sources=[dict(id='s0',role='user',quote='I keep an eye on my results from routine checkups. What does cholesterol mean?',context='I keep an eye on my results from routine checkups. What does cholesterol mean?')]);d={};checks=m.add_event_subchecks(checks,d);v={'checks':[{'claim_id':c['claim_id'],'entailed':False} for c in checks]};v=m.apply_direct_source_entailment_rules(v,checks,d);self.assertTrue(v['checks'][0]['entailed']);v=m.apply_event_subcheck_results(v,checks,d);self.assertFalse(v['checks'][0]['entailed'])
 def test_whole_parent_is_not_duplicated(self):
  checks=self.checks();checks[0]['claim']='After a difficult parent meeting';checks[0]['required_event_context']=['After a difficult parent meeting'];d={};self.assertEqual(m.add_event_subchecks(checks,d),checks);self.assertEqual(d['answer_event_subchecks']['skipped'][0]['reason'],'whole_parent')
 def test_exhausted_remaining_budget_preserves_original_checks(self):
  from app import budget
  checks=self.checks();d={}
  with budget.scope(seconds=60,calls=4,tokens=25000) as b:
   b.tokens=6000
   result=m.add_event_subchecks(checks,d)
  self.assertEqual(result,checks);self.assertEqual(d['answer_event_subchecks']['extra_limit'],0);self.assertEqual(d['answer_event_subchecks']['skipped'][0]['reason'],'expansion_budget')
 def test_reserved_tokens_are_unavailable_to_optional_checks(self):
  from app import budget
  checks=self.checks();d={}
  with budget.scope(seconds=60,calls=4,tokens=30000) as b:
   b.reserve_tokens(11000)
   result=m.add_event_subchecks(checks,d)
   b.release_tokens(11000)
  self.assertEqual(result,checks);self.assertEqual(d['answer_event_subchecks']['extra_limit'],0)
if __name__=='__main__':unittest.main()
