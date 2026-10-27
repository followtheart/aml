"""Context review needs co-located words in the exact visible excerpt."""
import copy
import unittest
from unittest.mock import AsyncMock, patch
from app import answer_choice as gate, personal_evidence


class LocalContextTests(unittest.TestCase):
    def test_distributed_words_do_not_form_a_joint_match(self):
        text='I read about school. I traveled through Kansas. I enjoy community events.'
        match=gate.local_context_match(text, {'school','kansas','events'})
        self.assertEqual(len(match['shared_terms']),1)

    def test_sentence_and_profile_line_preserve_exact_span(self):
        for text,terms in [
            ('Other topic.  Films from that era had strict rules. More discussion.', {'films','era'}),
            ('"devices": ["Laptop"],\n  "social_media": ["Facebook for family updates"]', {'facebook','family'}),
        ]:
            with self.subTest(text=text):
                match=gate.local_context_match(text,terms)
                self.assertEqual(set(match['shared_terms']),terms)
                span=match['source_span']
                self.assertEqual(text[span['start']:span['end']],match['excerpt'])

    def test_hidden_tail_and_empty_text_do_not_score(self):
        text='padding '*130+'films era'
        match=gate.local_context_match(text,{'films','era'})
        self.assertEqual(match['shared_terms'],[])
        self.assertEqual(gate.local_context_match('',{'films'})['shared_terms'],[])


class LocalContextFlowTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, source, options, first, review=None):
        entries=[dict(letter=option[0],option=option,kind='generic',status='generic',claims=[],
                      validation_errors=[],validation_status='valid') for option in options]
        sources={'s0':dict(id='s0',role='user',text=source)}
        calls=[]
        async def judge(prompt,schema,validate,stage,diag,**kw):
            calls.append(stage)
            if stage=='eval.choice_select':return validate({'answer':first})
            self.assertEqual(stage,'eval.choice_select_review')
            self.assertIsNotNone(review)
            return validate({'answer':review})
        diag={}
        with patch.object(gate,'build_catalog',return_value=(sources,{})), \
             patch.object(gate.choice_witness,'select',AsyncMock(return_value={})), \
             patch.object(gate,'assess_support',AsyncMock(return_value=copy.deepcopy(entries))), \
             patch.object(gate,'recover_missing_citations',AsyncMock(return_value=copy.deepcopy(entries))), \
             patch.object(gate,'entailment_checks',return_value=[]), \
             patch.object(gate,'constraint_pairs',return_value=[]),patch.object(gate,'_judge',judge):
            result=await gate._answer(dict(question='Any suggestions?',options=options),[],diag)
        return result,calls,diag

    async def test_irrelevant_long_context_cannot_overturn_initial_choice(self):
        result,calls,_=await self.invoke(
            'I read about school. I traveled through Kansas. I enjoy community events.',
            ['A. Try classic clothes.','B. Wear a Kansas scarf at school events.'],'A')
        self.assertEqual(result,'A')
        self.assertEqual(calls,['eval.choice_select'])

    async def test_local_context_still_reaches_review(self):
        source='Other topic. Were films from that era restricted? More discussion.'
        result,calls,diag=await self.invoke(source,
            ['A. Try a new book.','B. Watch films from that era.'],'A','B')
        self.assertEqual(result,'B')
        self.assertEqual(calls,['eval.choice_select','eval.choice_select_review'])
        row=diag['answer_selection_review']['context_evidence'][0]
        span=row['source_span']
        self.assertEqual(row['excerpt'],source[span['start']:span['end']])
        self.assertTrue(set(row['shared_terms'])<=personal_evidence.terms(row['excerpt']))


if __name__=='__main__':unittest.main()
