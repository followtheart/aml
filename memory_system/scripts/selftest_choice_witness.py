"""Offline semantic-witness, provenance, fallback and answer-flow regressions."""
import asyncio
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from selftest_answer_choice import memory, seal, assessment, entailment_response
from app import answer_choice, budget, choice_witness, config, llm


MORNING = ('I still start the day with twenty quiet minutes on the mat, some slow '
           'stretches, and a few deep breaths before the kids tumble into the kitchen.')
OPTIONS = ['A. Since you already practice yoga and meditation daily, try a new routine.',
           'B. Take a quiet break.']


def result(sid='s0', quote=MORNING):
    return dict(options=[dict(letter='A', witnesses=[dict(source_id=sid, quote=quote)]),
                         dict(letter='B', witnesses=[])])


class SemanticWitnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        patch.multiple(config, FAKE=False, CHOICE_SEMANTIC_WITNESSES=True).start()
        self.packet = seal(memory('Can you improve this personal email?\n\n' + MORNING))
        self.sources, _ = answer_choice.build_catalog(self.packet)
        self.diagnostics = {}

    async def select(self):
        return await choice_witness.select('How can I start a calm morning?', OPTIONS,
                                           self.sources, self.diagnostics)

    async def test_zero_keyword_overlap_is_sent_and_selected_verbatim(self):
        self.assertNotIn('A', choice_witness.build(OPTIONS, self.sources))
        original = copy.deepcopy(self.sources)
        model = AsyncMock(return_value=result())
        with patch.object(llm, 'complete_json', model):
            witnesses = await self.select()
        self.assertEqual(witnesses['A'][0]['quote'], MORNING)
        self.assertEqual(witnesses['A'][0]['match_method'], 'semantic')
        self.assertTrue(witnesses['A'][0]['first_party'])
        self.assertEqual(self.sources, original)
        prompt = model.call_args.args[0]
        cards = json.loads(prompt.split('<sources>\n')[1].split('\n</sources>')[0])
        self.assertEqual(cards[0]['text'], self.sources['s0']['text'])
        self.assertEqual(model.call_args.kwargs['attempts'], 1)
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['status'], 'ok')

    async def test_real_quote_inside_long_source_is_not_keyword_prefiltered(self):
        self.sources['s0']['text'] = ('Unrelated context. ' * 80) + MORNING
        with patch.object(llm, 'complete_json', AsyncMock(return_value=result())):
            witnesses = await self.select()
        self.assertEqual(witnesses['A'][0]['quote'], MORNING)

    async def test_provider_failure_falls_back_and_releases_budget(self):
        self.sources['s0']['text'] = 'I practice yoga and meditation daily.'
        expected = choice_witness.build(OPTIONS, self.sources)
        with budget.scope(seconds=60, calls=9, tokens=128000) as limits:
            async def fail(*args, **kwargs):
                self.assertEqual(limits.reserved_calls, 8)
                self.assertGreater(limits.reserved_tokens, 0)
                limits.before_call()
                raise llm.LLMError('provider unavailable')
            with patch.object(llm, 'complete_json', fail):
                self.assertEqual(await self.select(), expected)
            self.assertEqual((limits.calls, limits.reserved_calls, limits.reserved_tokens), (1, 0, 0))
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['reason'], 'semantic_error')
        self.assertEqual(self.diagnostics['answer_calls'][0]['error_type'], 'LLMError')

    async def test_malformed_ids_quotes_or_option_coverage_fall_back_once(self):
        missing = result(); missing['options'].pop()
        duplicate = result(); duplicate['options'].append(copy.deepcopy(duplicate['options'][0]))
        invented_role = result(); invented_role['options'][0]['witnesses'][0]['role'] = 'user'
        invalid = [result(sid='unknown'), result(quote='I practice yoga daily.'),
                   result(quote=' '), result(quote='x' * 321), missing, duplicate,
                   invented_role]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.diagnostics = {}
                model = AsyncMock(return_value=payload)
                with patch.object(llm, 'complete_json', model):
                    await self.select()
                self.assertEqual(model.await_count, 1)
                self.assertEqual(self.diagnostics['answer_witness_retrieval']['status'], 'fallback')

    async def test_bad_quote_in_another_option_does_not_erase_valid_semantic_match(self):
        payload = result()
        payload['options'][0]['witnesses'] *= 2
        payload['options'][1]['witnesses'] = [dict(source_id='s0', quote=MORNING + '"')]
        with patch.object(llm, 'complete_json', AsyncMock(return_value=payload)):
            witnesses = await self.select()
        self.assertEqual(len(witnesses['A']), 1)
        self.assertEqual(witnesses['A'][0]['quote'], MORNING)
        self.assertNotIn('B', witnesses)
        trace = self.diagnostics['answer_witness_retrieval']
        self.assertEqual(trace['status'], 'partial')
        self.assertEqual([r['reason'] for r in trace['rejected_witnesses']],
                         ['duplicate_source', 'quote_not_in_source'])

    async def test_cross_source_quote_is_rejected(self):
        self.sources['s1'] = dict(id='s1', role='user', text='I collect stamps.')
        with patch.object(llm, 'complete_json', AsyncMock(return_value=result(sid='s1'))):
            await self.select()
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['status'], 'fallback')

    async def test_timeout_is_bounded_but_cancellation_propagates(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(1)
        with patch.object(choice_witness, 'TIMEOUT_SECONDS', .01), \
                patch.object(llm, 'complete_json', slow):
            await self.select()
        self.assertEqual(self.diagnostics['answer_calls'][0]['error_type'], 'TimeoutError')
        with budget.scope(seconds=60, calls=9, tokens=128000) as limits:
            with patch.object(llm, 'complete_json', AsyncMock(side_effect=asyncio.CancelledError)), \
                    self.assertRaises(asyncio.CancelledError):
                await self.select()
            self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))

    async def test_budget_shortage_preserves_answer_capacity_without_calling(self):
        for settings in (dict(calls=4, tokens=128000), dict(calls=9, tokens=1000)):
            with self.subTest(settings=settings), budget.scope(seconds=60, **settings) as limits:
                with patch.object(llm, 'complete_json', AsyncMock()) as model:
                    await self.select()
                model.assert_not_awaited()
                self.assertEqual((limits.calls, limits.reserved_calls, limits.reserved_tokens), (0, 0, 0))
                self.assertEqual(self.diagnostics['answer_witness_retrieval']['reason'], 'answer_budget')

    async def test_prompt_limit_is_explicit_and_never_clips_sources(self):
        with patch.object(choice_witness, 'MAX_PROMPT_BYTES', 10), \
                patch.object(llm, 'complete_json', AsyncMock()) as model:
            await self.select()
        model.assert_not_awaited()
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['reason'], 'prompt_budget')

    async def test_empty_disabled_and_fake_modes_make_no_provider_call(self):
        for setting, reason in ((dict(CHOICE_SEMANTIC_WITNESSES=False), 'disabled'),
                                (dict(FAKE=True), 'offline_fake')):
            with patch.multiple(config, **setting), patch.object(llm, 'complete_json', AsyncMock()) as model:
                await self.select()
            model.assert_not_awaited()
            self.assertEqual(self.diagnostics['answer_witness_retrieval']['reason'], reason)
        self.sources = {}
        with patch.object(llm, 'complete_json', AsyncMock()) as model:
            self.assertEqual(await self.select(), {})
        model.assert_not_awaited()
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['status'], 'not_needed')

    async def test_semantic_selection_keeps_authoritative_source_roles(self):
        self.sources['s0']['role'] = 'assistant'
        with patch.object(llm, 'complete_json', AsyncMock(return_value=result())):
            witnesses = await self.select()
        self.assertEqual(witnesses['A'][0]['role'], 'assistant')
        self.assertFalse(witnesses['A'][0]['first_party'])
        self.sources['s0'].update(role='user', declared='persona')
        with patch.object(llm, 'complete_json', AsyncMock(return_value=result())):
            witnesses = await self.select()
        self.assertEqual(witnesses['A'][0]['role'], 'persona')

    async def test_empty_semantic_results_rescue_matching_first_party_quote(self):
        self.sources['s0']['text'] = 'I practice yoga and meditation daily.'
        payload = dict(options=[dict(letter=letter, witnesses=[]) for letter in 'AB'])
        with patch.object(llm, 'complete_json', AsyncMock(return_value=payload)):
            witnesses = await self.select()
        self.assertEqual(witnesses['A'][0]['source_id'], 's0')
        self.assertTrue(witnesses['A'][0]['first_party'])
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['status'], 'ok')

    async def test_empty_semantic_results_remain_empty_without_lexical_first_party_match(self):
        self.sources = {'s99': dict(id='s99', role='user', text='I enjoy cooking soup.')}
        payload = dict(options=[dict(letter=letter, witnesses=[]) for letter in 'AB'])
        with patch.object(llm, 'complete_json', AsyncMock(return_value=payload)):
            self.assertEqual(await self.select(), {})

    def test_first_party_lexical_rescue_precedes_assistant_context(self):
        options = ['A. Since you’re keeping an eye on your cholesterol, choose a snack.']
        sources = {
            's_user': dict(id='s_user', role='user', text=(
                'Why would cholesterol numbers change noticeably between two routine checkups?')),
            's_assistant': dict(id='s_assistant', role='assistant', text=(
                'Cholesterol numbers can shift between routine checkups.')),
        }
        semantic = {'A': [dict(source_id='s_assistant', role='assistant',
                               quote=sources['s_assistant']['text'], first_party=False,
                               match_method='semantic')]}
        rescued = choice_witness.supplement_first_party(options, sources, semantic)
        self.assertEqual([w['source_id'] for w in rescued['A']], ['s_user', 's_assistant'])
        self.assertEqual(rescued['A'][0]['role'], 'user')

    def test_semantic_first_party_match_does_not_get_reordered(self):
        semantic = {'A': [dict(source_id='s0', role='user', quote=MORNING,
                               first_party=True, match_method='semantic')]}
        rescued = choice_witness.supplement_first_party(OPTIONS, self.sources, semantic)
        self.assertEqual(rescued, semantic)

    async def test_answer_receives_semantic_witness_but_entailment_can_reject_it(self):
        qa = dict(question='What morning routine would help?', options=OPTIONS)
        support = assessment('you already practice yoga', 's0', MORNING)
        stages = []

        async def respond(prompt, **kwargs):
            stage = kwargs['stage']; stages.append(stage)
            if stage == 'eval.choice_witness':
                return result()
            if stage == 'eval.choice_support':
                hints = prompt.split('<witnesses>')[1].split('</witnesses>')[0]
                self.assertIn(MORNING, hints)
                return support
            if stage == 'eval.choice_entailment_review':
                return entailment_response(prompt, {'A:0', 'A:option'})
            self.assertEqual(stage, 'eval.choice_entailment')
            return entailment_response(prompt, {'A:0', 'A:option'})

        with patch.object(llm, 'complete_json', respond):
            selected = await answer_choice.answer(qa, self.packet, self.diagnostics)
        self.assertEqual(selected, 'B')
        self.assertEqual(stages, ['eval.choice_witness', 'eval.choice_support',
                                  'eval.choice_entailment'])

    async def test_semantic_witness_does_not_bypass_interest_or_third_party_gates(self):
        for text, basis, expected_error in (
                ('How do yoga and meditation affect stress?', 'topic_interest', 'interest_does_not_prove_trait'),
                ('Jordan wrote: I practice yoga daily.', 'self_report', 'third_party_or_hypothetical'),
                ('I never practice yoga daily.', 'self_report', 'denied_event')):
            with self.subTest(text=text):
                self.sources['s0']['text'] = text
                with patch.object(llm, 'complete_json', AsyncMock(return_value=result(quote=text))):
                    witnesses = await self.select()
                ref = answer_choice.Citation(source_id='s0', quote=witnesses['A'][0]['quote'],
                                             basis=basis, subject='current_user')
                checked = answer_choice._check_citation(ref, 'you practice yoga daily', self.sources, 'habit')
                self.assertIn(expected_error, checked['validation_errors'])

    async def test_standalone_semantic_call_leaves_eight_calls_for_answer_repairs(self):
        text = 'I enjoy photography.'
        packet = seal(memory(text), memory('Please forget that I enjoy photography.', mid='rule', constraint=True))
        qa = dict(question='What should I do?', options=['A. Since you enjoy photography, take photos.',
                                                       'B. Rest.', 'C. Take a walk.'])
        support = assessment('you enjoy photography', 's0', text)
        support['options'].append(dict(letter='C', kind='generic', claims=[]))
        calls, scopes = [], []

        async def respond(prompt, **kwargs):
            active = budget.current.get(); active.before_call(); scopes.append(active)
            stage = kwargs['stage']; calls.append(stage)
            if stage == 'eval.choice_witness':
                return dict(options=[dict(letter=l, witnesses=[]) for l in 'ABC'])
            if not stage.endswith('.repair'):
                return {}
            stage = stage.removesuffix('.repair')
            if stage == 'eval.choice_support':
                return support
            if stage == 'eval.choice_entailment':
                return entailment_response(prompt)
            if stage == 'eval.choice_constraints':
                return dict(decisions=[dict(pair_id='A:r0', violates=True)])
            self.assertEqual(stage, 'eval.choice_select')
            return dict(answer='C')

        with patch.object(llm, 'complete_json', respond):
            self.assertEqual(await answer_choice.answer(qa, packet, self.diagnostics), 'C')
        self.assertEqual(len(calls), 9)
        self.assertEqual(scopes[0].calls, 9)
        self.assertGreaterEqual(scopes[0].max_calls, scopes[0].calls)
        self.assertEqual((scopes[0].reserved_calls, scopes[0].reserved_tokens), (0, 0))


if __name__ == '__main__':
    unittest.main()
