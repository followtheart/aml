"""Ignore generic usage overlaps while preserving specific-context selection review."""
import json
import re
import unittest
from unittest.mock import AsyncMock, patch
from selftest_answer_choice import OfflineCase, memory, seal
from app import llm

class ContextFilterTests(OfflineCase,unittest.IsolatedAsyncioTestCase):
 async def check_case(self,source,options,expected,review_expected):
  async def provider(prompt,**kwargs):
   stage=kwargs['stage']
   if stage=='eval.choice_witness':return {'options':[dict(letter=k,witnesses=[]) for k in 'AB']}
   if stage=='eval.choice_support':return {'options':[dict(letter=k,kind='generic',claims=[]) for k in 'AB']}
   if stage=='eval.choice_entailment':
    checks=json.loads(re.search(r'<checks>\s*(.*?)\s*</checks>',prompt,re.S)[1]);return {'checks':[dict(claim_id=c['claim_id'],verdict='no_personal_premise') for c in checks]}
   if stage=='eval.choice_select':return {'answer':'A'}
   if stage=='eval.choice_select_review':return {'answer':'B'}
   raise AssertionError(stage)
  mock=AsyncMock(side_effect=provider);d={}
  with patch.object(llm,'complete_json',mock):
   answer=await self.gate.answer(dict(question='What could I try this weekend?',options=options),seal(memory(source)),d)
  self.assertEqual(answer,expected);self.assertEqual('answer_selection_review' in d,review_expected)
 async def test_generic_usage_overlap_does_not_override_first_selection(self):
  await self.check_case('Which camera works well for everyday use?', ['A. Try a colorful linen jacket.','B. Use a wool coat that works well for outings.'],'A',False)
 async def test_specific_topic_context_still_triggers_review(self):
  await self.check_case('I enjoy botanical gardens and bird photography.', ['A. Try a quiet stroll.','B. Visit botanical gardens and try bird photography.'],'B',True)

if __name__ == "__main__":
    unittest.main()
