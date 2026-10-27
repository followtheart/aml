import copy,importlib.util,sys,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
sys.path.insert(0,str(Path.cwd()))
from app import budget,config
from app import answer_choice as ac
class Tests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        entries=[dict(letter='A',status='supported',claims=[dict(status='supported',citations=[dict(anchor=True)])])]
        checks=[dict(claim_id='A:option'),dict(claim_id='A:0')]
        verdicts=dict(checks=[dict(claim_id='A:option',entailed=False),dict(claim_id='A:0',entailed=True)])
        return entries,checks,verdicts
    async def run_review(self,entries,checks,verdicts,mock,diag):
        with patch.object(config,'CHOICE_ENTAILMENT_REVIEW',True),patch.object(ac,'_judge',mock):
            return await ac.review_entailments(verdicts,checks,entries,{},dict(question='Help me plan.'),diag)
    async def test_rejected_premise_cannot_be_restored(self):
        e,c,v=self.fixture();v['checks'][1]['entailed']=False;m=AsyncMock()
        self.assertEqual(await self.run_review(e,c,v,m,{}),v);m.assert_not_awaited()
    async def test_only_disputed_option_reviewed_and_inputs_preserved(self):
        e,c,v=self.fixture();old=copy.deepcopy((e,c,v));d={}
        m=AsyncMock(return_value=dict(checks=[dict(claim_id='A:option',entailed=True)]))
        out=await self.run_review(e,c,v,m,d)
        self.assertTrue(all(x['entailed'] for x in out['checks']));self.assertEqual((e,c,v),old)
        prompt=m.call_args.args[0];self.assertIn('A:option',prompt);self.assertNotIn('A:0',prompt)
        self.assertEqual(d['answer_entailment_review']['restored_check_ids'],['A:option'])
    async def test_unanchored_or_unchecked_secondary_blocks_review(self):
        for claim in [dict(status='supported',citations=[]),dict(status='supported',citations=[dict(anchor=True)])]:
            e,c,v=self.fixture();e[0]['claims'].append(claim);m=AsyncMock()
            self.assertEqual(await self.run_review(e,c,v,m,{}),v);m.assert_not_awaited()
    async def test_rejected_context_may_be_reviewed_but_never_restored(self):
        e,c,v=self.fixture();e.append(dict(letter='B',status='supported',claims=[dict(status='supported',citations=[dict(anchor=True)])]))
        c.extend([dict(claim_id='B:option'),dict(claim_id='B:0')])
        v['checks'].extend([dict(claim_id='B:option',entailed=False),dict(claim_id='B:0',entailed=False)])
        m=AsyncMock(return_value=dict(checks=[dict(claim_id=x,entailed=True) for x in ['A:option','B:option','B:0']]))
        d={};out=await self.run_review(e,c,v,m,d)
        self.assertEqual(d['answer_entailment_review']['restored_check_ids'],['A:option'])
        self.assertTrue(next(x['entailed'] for x in out['checks'] if x['claim_id']=='A:option'))
        self.assertFalse(any(x['entailed'] for x in out['checks'] if x['claim_id'].startswith('B:')))
        self.assertIn('B:0',m.call_args.args[0])
    async def test_failed_review_preserves_first_verdict(self):
        e,c,v=self.fixture();d={};m=AsyncMock(side_effect=ValueError('malformed'))
        self.assertEqual(await self.run_review(e,c,v,m,d),v)
        self.assertEqual(d['answer_entailment_review']['status'],'kept_first_verdict')
    async def test_insufficient_budget_preserves_first_verdict(self):
        e,c,v=self.fixture();d={};m=AsyncMock()
        with budget.scope(seconds=240,calls=2,tokens=128000):
            self.assertEqual(await self.run_review(e,c,v,m,d),v)
        m.assert_not_awaited();self.assertEqual(d['answer_entailment_review']['status'],'skipped_budget')
if __name__=='__main__':unittest.main()
