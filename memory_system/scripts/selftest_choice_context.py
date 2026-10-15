"""Query-focused context must preserve provenance, gates and answer budgets."""
import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, patch

from selftest_answer_choice import assessment, entailment_response, memory, seal
from app import answer_choice, budget, choice_context, config, llm


class QueryContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        patch.multiple(config, FAKE=False, CHOICE_QUERY_CONTEXT=True,
                       CHOICE_SEMANTIC_WITNESSES=False).start()
        self.sources = {
            's0': dict(id='s0', role='user', text='My colleague needs help with a difficult conversation.'),
            's1': dict(id='s1', role='assistant', text='Try writing a plan.'),
        }
        self.payload = dict(sources=[dict(source_id='s0', relation='situation')])

    async def test_query_only_ids_recover_unchanged_original_source(self):
        before = copy.deepcopy(self.sources)
        mock = AsyncMock(return_value=self.payload)
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            focused = await choice_context.select('How can I respond?', self.sources, diagnostics)
        self.assertEqual(self.sources, before)
        self.assertEqual(focused[0]['text'], self.sources['s0']['text'])
        self.assertNotIn(self.sources['s1']['text'], mock.call_args.args[0])
        self.assertTrue(diagnostics['answer_context_focus']['question_only'])
        self.assertIn(self.sources['s0']['text'], choice_context.render(focused))

    def test_unknown_duplicate_and_assistant_ids_are_rejected(self):
        for ids in [('unknown',), ('s0', 's0'), ('s1',)]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                choice_context.validate(dict(sources=[dict(source_id=s, relation='situation')
                                                       for s in ids]), self.sources)

    async def test_disabled_and_insufficient_budget_make_no_call(self):
        mock = AsyncMock()
        with patch.object(llm, 'complete_json', mock):
            with patch.object(config, 'CHOICE_QUERY_CONTEXT', False):
                self.assertEqual(await choice_context.select('Question?', self.sources, {}), [])
            with budget.scope(seconds=60, calls=2, tokens=128000):
                d = {}
                self.assertEqual(await choice_context.select('Question?', self.sources, d), [])
                self.assertEqual(d['answer_context_focus']['reason'], 'answer_budget')
        mock.assert_not_awaited()

    async def test_provider_failure_releases_reserved_calls(self):
        with budget.scope(seconds=60, calls=5, tokens=128000) as usage:
            with patch.object(llm, 'complete_json', AsyncMock(side_effect=ValueError('bad output'))):
                d = {}
                self.assertEqual(await choice_context.select('Question?', self.sources, d), [])
            self.assertEqual(usage.reserved_calls, 0)
            self.assertEqual(d['answer_context_focus']['status'], 'fallback')

    async def test_cancellation_is_not_swallowed(self):
        with budget.scope(seconds=60, calls=5, tokens=128000) as usage:
            with patch.object(llm, 'complete_json', AsyncMock(side_effect=asyncio.CancelledError)):
                with self.assertRaises(asyncio.CancelledError):
                    await choice_context.select('Question?', self.sources, {})
            self.assertEqual(usage.reserved_calls, 0)

    async def test_focus_never_reopens_a_rejected_personal_option(self):
        text = 'I sometimes wonder about growing orchids.'
        packet = seal(memory(text))
        qa = dict(question='What should I do this weekend?',
                  options=['A. Since you own a greenhouse, expand it.', 'B. Take a walk.'])
        support = assessment('you own a greenhouse', 's0', text)
        stages = []

        async def respond(prompt, **kwargs):
            stage = kwargs['stage']
            stages.append(stage)
            if stage == 'eval.choice_support':
                return support
            if stage == 'eval.choice_entailment':
                return entailment_response(prompt, {'A:0', 'A:option'})
            raise AssertionError('Unexpected stage: ' + stage)

        with patch.object(llm, 'complete_json', respond):
            diagnostics = {}
            selected = await answer_choice.answer(qa, packet, diagnostics)
        self.assertEqual(selected, 'B')
        self.assertEqual(diagnostics['answer_eligible_options'], ['B'])
        self.assertNotIn('eval.choice_context', stages)

    async def test_eligible_selection_receives_original_focus(self):
        text = self.sources['s0']['text']
        qa = dict(question='How should I respond?',
                  options=['A. Listen attentively.', 'B. Spend time outdoors.'])
        stages = []

        async def respond(prompt, **kwargs):
            stage = kwargs['stage']
            stages.append(stage)
            if stage == 'eval.choice_support':
                return dict(options=[dict(letter=letter, kind='generic', claims=[]) for letter in 'AB'])
            if stage == 'eval.choice_entailment':
                return entailment_response(prompt)
            if stage == 'eval.choice_context':
                self.assertNotIn(qa['options'][0], prompt)
                self.assertNotIn(qa['options'][1], prompt)
                return self.payload
            if stage in ('eval.choice_select', 'eval.choice_select_review'):
                self.assertIn('Question-focused original context', prompt)
                self.assertIn(text, prompt)
                return dict(answer='A')
            raise AssertionError('Unexpected stage: ' + stage)

        with patch.object(llm, 'complete_json', respond):
            diagnostics = {}
            selected = await answer_choice.answer(qa, seal(memory(text)), diagnostics)
        self.assertEqual(selected, 'A')
        self.assertEqual(diagnostics['answer_eligible_options'], ['A', 'B'])
        self.assertLess(stages.index('eval.choice_context'), stages.index('eval.choice_select'))


if __name__ == '__main__':
    unittest.main()
