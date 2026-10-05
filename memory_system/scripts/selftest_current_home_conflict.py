import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from app.answer_choice import current_home_conflicts as check
class BoundaryTests(unittest.TestCase):
 def test_literal_current_conflict(self):
  q='I am hosting a reunion at my home (located at 10 Pine Street). Could you help draft a plan?'
  options=['A. Prepare your home at 12 Oak Street.', 'B. Prepare your home at 10 Pine Street.']
  out=check(q,options);self.assertEqual([x['letter'] for x in out],['A'])
  for x in out:
   self.assertEqual(q[x['question_span']['start']:x['question_span']['end']],x['question_address'])
   self.assertEqual(options[0][x['option_span']['start']:x['option_span']['end']],x['option_address'])
 def test_matching_address_and_abbreviation_preserved(self):
  self.assertEqual(check('My home is at 12 Oak Street.',['A. Prepare your home at 12 Oak St.']),[])
 def test_no_current_address_is_not_negative_evidence(self):self.assertEqual(check('I live in Denver.',['A. Prepare your home at 12 Oak Street.']),[])
 def test_other_person_and_new_venue_defer(self):
  self.assertEqual(check('My brother lives at 10 Pine Street.',['A. Prepare your home at 12 Oak Street.']),[])
  self.assertEqual(check('My home is at 10 Pine Street.',['A. Meet at a venue at 12 Oak Street.']),[])
 def test_relocation_and_hypothetical_defer(self):
  q='My home is at 10 Pine Street.'
  for o in ['A. Imagine your home at 12 Oak Street.','A. Prepare your new home at 12 Oak Street.','A. Move to your home at 12 Oak Street.','A. Build your home at 12 Oak Street.']:self.assertEqual(check(q,[o]),[])
  self.assertEqual(check(q+' I am moving.',['A. Prepare your home at 12 Oak Street.']),[])
 def test_ambiguous_or_negated_current_address_defer(self):
  for q in ['My home is not at 10 Pine Street.','My home is at 10 Pine Street; my other home is at 12 Oak Street.']:
   self.assertEqual(check(q,['A. Prepare your home at 99 Elm Road.']),[])
 def test_different_street_same_number_not_rejected(self):self.assertEqual(check('My home is at 12 Oak Street.',['A. Prepare your home at 12 Elm Road.']),[])
 def test_nonasserted_scope_regressions(self):
  rows=ADVERSARIAL_CASES
  for x in rows:
   with self.subTest(case=x['case']):self.assertEqual(bool(check(x['question'],[x['option']])),x['expected_conflict'])
 def test_modal_reporting_and_past_options_defer(self):
  q='My home is at 10 Pine Street.'
  for o in ['A. Assume your home at 12 Oak Street is available.','A. Your home at 12 Oak Street would work.','A. Previously, your home at 12 Oak Street hosted parties.', 'A. The sign says your home at 12 Oak Street.', 'A. Label the sample “your home at 12 Oak Street”.']:
   with self.subTest(option=o):self.assertEqual(check(q,[o]),[])
 def test_quoted_question_without_reporting_keyword(self):
  for q in ['“My home is at 10 Pine Street.”', '"My home is at 10 Pine Street."', "'My home is at 10 Pine Street.'", '`My home is at 10 Pine Street.`']:
   with self.subTest(question=q):self.assertEqual(check(q,['A. Prepare your home at 12 Oak Street.']),[])
 def test_unrelated_quotation_does_not_hide_conflict(self):
  q='My home is at 10 Pine Street. The theme is “family”.'
  self.assertEqual([x['letter'] for x in check(q,['A. Prepare your home at 12 Oak Street.'])],['A'])


import re
from unittest.mock import AsyncMock,patch
from app import answer_choice as m,config,llm
ADVERSARIAL_CASES=[{'case': 'conditional', 'question': 'My home is at 10 Pine Street.', 'option': 'A. If your home at 12 Oak Street were available, it could host the event.', 'conflicts': [{'letter': 'A', 'reason': 'explicit_current_home_number_conflict', 'question_address': '10 Pine Street', 'question_span': {'start': 14, 'end': 28}, 'option_address': '12 Oak Street', 'option_span': {'start': 19, 'end': 32}}], 'expected_conflict': False}, {'case': 'reported_error', 'question': 'My home is at 10 Pine Street.', 'option': 'A. Correct the invitation that mistakenly lists your home at 12 Oak Street.', 'conflicts': [{'letter': 'A', 'reason': 'explicit_current_home_number_conflict', 'question_address': '10 Pine Street', 'question_span': {'start': 14, 'end': 28}, 'option_address': '12 Oak Street', 'option_span': {'start': 61, 'end': 74}}], 'expected_conflict': False}, {'case': 'question_quoted', 'question': 'The draft incorrectly says "my home is at 10 Pine Street". Help fix it.', 'option': 'A. Prepare your home at 12 Oak Street.', 'conflicts': [{'letter': 'A', 'reason': 'explicit_current_home_number_conflict', 'question_address': '10 Pine Street', 'question_span': {'start': 42, 'end': 56}, 'option_address': '12 Oak Street', 'option_span': {'start': 24, 'end': 37}}], 'expected_conflict': False}, {'case': 'literal_conflict', 'question': 'My home is at 10 Pine Street.', 'option': 'A. Prepare your home at 12 Oak Street.', 'conflicts': [{'letter': 'A', 'reason': 'explicit_current_home_number_conflict', 'question_address': '10 Pine Street', 'question_span': {'start': 14, 'end': 28}, 'option_address': '12 Oak Street', 'option_span': {'start': 24, 'end': 37}}], 'expected_conflict': True}]
class FlowTests(unittest.IsolatedAsyncioTestCase):
 async def run_case(self,q,option,expect_block,reject_a=False):
  qa=dict(question=q,options=[option,'B. Arrange seating for your guests.']);diag={};stages=[]
  async def respond(prompt,*args,**kw):
   stage=kw['stage'];stages.append(stage)
   if stage=='eval.choice_support':return {'options':{x:dict(kind='generic',claims=[]) for x in ['A','B']}}
   if stage=='eval.choice_entailment':
    checks=json.loads(re.search(r'<checks>\s*(.*?)\s*</checks>',prompt,re.S)[1])
    return {'checks':[dict(claim_id=x['claim_id'],verdict='unsupported_personal' if reject_a and x['claim_id']=='A:option' else 'no_personal_premise') for x in checks]}
   if stage in ['eval.choice_select','eval.choice_select_review']:
    self.assertNotIn('A',diag['answer_eligible_options']) if expect_block else None
    return {'answer':'B' if expect_block or reject_a else 'A'}
   raise AssertionError('Unexpected stage: '+stage)
  with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')),patch.object(config,'CHOICE_ENTAILMENT_REVIEW',False),patch.object(config,'CHOICE_ADAPTIVE_INTEREST_RETRY',False),patch.object(m.choice_witness,'select',AsyncMock(return_value={})),patch.object(llm,'complete_json',respond):
   answer=await m.answer(qa,[],diag)
  self.assertEqual(diag['answer_blocked_options'],['A'] if expect_block else [])
  self.assertEqual(answer,'B' if expect_block or reject_a else 'A')
  self.assertIn('eval.choice_entailment',stages)
  return diag
 async def test_generic_misclassification_cannot_bypass_conflict(self):
  d=await self.run_case('My home is at 10 Pine Street.','A. Prepare your home at 12 Oak Street.',True)
  self.assertEqual(d['answer_eligible_options'],['B'])
 async def test_same_address_does_not_promote_failed_whole_option(self):
  d=await self.run_case('My home is at 10 Pine Street.','A. Prepare your home at 10 Pine Street.',False,True)
  self.assertEqual(d['answer_eligible_options'],['B'])
 async def test_nonasserted_cases_preserve_existing_eligibility(self):
  for row in ADVERSARIAL_CASES:
   if not row['expected_conflict']:
    with self.subTest(case=row['case']):
     d=await self.run_case(row['question'],row['option'],False)
     self.assertEqual(d['answer_eligible_options'],['A','B'])
if __name__=='__main__':unittest.main()
