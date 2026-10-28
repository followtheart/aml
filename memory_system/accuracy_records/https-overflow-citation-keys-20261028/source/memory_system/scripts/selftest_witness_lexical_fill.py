"""Additional lexical hints cannot replace existing quotes or assert support."""
import copy
import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice,choice_witness,config


class StopBeforeSupport(Exception):
    pass


class LexicalFill(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.options=['A. Since you enjoy painting with watercolors, join a painting class.']
        self.sources={sid:dict(id=sid,role=role,text=text,item_ids=[]) for sid,role,text in [
            ('s0','user','I enjoy walking.'),('s1','user','I enjoy painting with watercolors.'),
            ('s2','assistant','Painting can be relaxing.'),('s3','user','I sometimes sketch and paint.') ]}
        self.original={'A':[dict(source_id='s0',role='user',quote='I enjoy walking.',first_party=True)]}

    def test_fills_without_displacing_existing_first_party_hint(self):
        old=copy.deepcopy(self.original)
        result,trace=choice_witness.fill_unused(self.options,self.sources,self.original)
        self.assertEqual(self.original,old)
        self.assertEqual(result['A'][:1],old['A'])
        self.assertIn('s1',[w['source_id'] for w in result['A']])
        self.assertNotIn('s2',[w['source_id'] for w in result['A']])
        self.assertLessEqual(len(result['A']),3)
        self.assertGreater(trace['added_count'],0)
        for w in result['A'][1:]:
            self.assertIn(w['quote'],self.sources[w['source_id']]['text'])
            self.assertNotIn('supported',w)

    def test_full_seats_remain_unchanged(self):
        full={'A':[dict(self.original['A'][0],source_id=sid) for sid in ['s0','s1','s3']]}
        result,trace=choice_witness.fill_unused(self.options,self.sources,full)
        self.assertEqual(result,full);self.assertEqual(trace['added_count'],0)

    def test_no_second_quote_from_same_source(self):
        current={'A':[dict(source_id='s1',quote='I enjoy painting with watercolors.',role='user')]}
        result,_=choice_witness.fill_unused(self.options,self.sources,current)
        self.assertEqual(sum(w['source_id']=='s1' for w in result['A']),1)

    def test_bad_or_wrong_role_lexical_candidates_are_rejected(self):
        for sid,quote in [('s1','I love painting.'),('s2','Painting can be relaxing.'),('missing','I enjoy painting.')]:
            with self.subTest(sid=sid),patch.object(choice_witness,'build',return_value={'A':[dict(source_id=sid,quote=quote,first_party=True)]}):
                got,trace=choice_witness.fill_unused(self.options,self.sources,self.original)
                self.assertEqual(got,self.original);self.assertEqual(trace['added_count'],0)

    async def test_integration_after_coverage_and_default_off(self):
        for enabled in [False,True]:
            diag={}
            with patch.multiple(config,CHOICE_WITNESS_LEXICAL_FILL=enabled,CHOICE_COVERAGE_WITNESSES=True),\
                    patch.object(answer_choice,'build_catalog',return_value=(self.sources,{})),\
                    patch.object(answer_choice,'add_current_question'),\
                    patch.object(choice_witness,'select',AsyncMock(return_value={})),\
                    patch.object(choice_witness,'supplement_from_coverage',return_value=self.original),\
                    patch.object(answer_choice,'assess_support',AsyncMock(side_effect=StopBeforeSupport)),\
                    self.assertRaises(StopBeforeSupport):
                await answer_choice._answer(dict(question='What hobby could I try?',options=self.options),[],diag)
            self.assertEqual(diag['answer_witness_prefill']['A'][:1],self.original['A'])
            self.assertEqual('answer_witness_lexical_fill'in diag,enabled)
            if not enabled:self.assertEqual(diag['answer_witness_prefill'],self.original)

    async def test_reused_strict_hints_not_expanded_again(self):
        diag={}
        with patch.object(config,'CHOICE_WITNESS_LEXICAL_FILL',True),\
                patch.object(answer_choice,'build_catalog',return_value=(self.sources,{})),\
                patch.object(answer_choice,'add_current_question'),\
                patch.object(choice_witness,'fill_unused') as fill,\
                patch.object(answer_choice,'assess_support',AsyncMock(side_effect=StopBeforeSupport)),\
                self.assertRaises(StopBeforeSupport):
            await answer_choice._answer(dict(question='What hobby?',options=self.options),[],diag,witness_prefill=self.original)
        fill.assert_not_called()
        self.assertEqual(diag['answer_witness_prefill'],self.original)


if __name__=='__main__':unittest.main()
