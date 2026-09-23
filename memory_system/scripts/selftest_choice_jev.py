"""Offline regressions for independent JEV support judgment and LLM review."""
import asyncio
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from selftest_answer_choice import memory, seal, assessment, entailment_response
from app import answer_choice, budget, choice_jev, config, jev, llm

_real_decide = jev.decide


QUOTE = 'I use Facebook to stay in touch with my family.'
CLAIM = 'you use Facebook to stay in touch with your family'
OPTIONS = ['A. Since ' + CLAIM + ', share an update.', 'B. Take a quiet break.']


def decision(label='supported', confidence=.95):
    labels = ['supported', 'contradicted', 'insufficient', 'not_a_premise']
    return dict(answers={'A:0': dict(type='choice', choice=label, confidence=confidence,
        probabilities={key: float(key == label) for key in labels})}, model='typesafe/jev-1.13-dated',
        usage={'input_tokens': 100, 'output_tokens': 20})


class JevSupportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        patch.multiple(config, FAKE=False, CHOICE_JEV_SUPPORT=True, JEV_API_KEY='test-only',
                       CHOICE_JEV_MIN_CONFIDENCE=.8, CHOICE_SEMANTIC_WITNESSES=False).start()
        self.sources, _ = answer_choice.build_catalog(seal(memory(QUOTE)))
        self.payload = assessment(CLAIM, 's0', QUOTE)
        self.entries = answer_choice.validate_assessments(self.payload, OPTIONS, self.sources)
        self.diag = {}
        self.model = AsyncMock(return_value=copy.deepcopy(self.payload))
        patch.object(llm, 'complete_json', self.model).start()
        self.decider = AsyncMock(return_value=decision())
        patch.object(jev, 'decide', self.decider).start()

    async def reconcile(self):
        return await choice_jev.reconcile(
            dict(question='What should I do?', options=OPTIONS), self.entries,
            answer_choice._cards(self.sources), {}, self.diag, judge=answer_choice._judge,
            schema=answer_choice.Assessments.model_json_schema(),
            validate=lambda result, options: answer_choice.validate_assessments(result, options, self.sources))

    async def test_agreement_keeps_entries_without_review_and_jev_has_no_llm_status(self):
        before = copy.deepcopy(self.entries)
        self.assertEqual(await self.reconcile(), before)
        self.model.assert_not_awaited()
        state = self.decider.call_args.args[0]
        self.assertNotIn('proposed_status', json.dumps(state))
        self.assertNotIn('reason', json.dumps(state))
        self.assertEqual(state['sources'][0]['text'], QUOTE)
        self.assertEqual(self.diag['answer_jev_support']['status'], 'agreed')

    async def test_unsupported_can_be_recovered_only_by_llm_cited_review(self):
        self.entries[0]['claims'][0].update(status='unsupported', reason='no_source')
        self.entries[0]['status'] = 'unsupported'
        self.model.return_value = {'options': [self.payload['options'][0]]}
        original = copy.deepcopy(self.entries)
        result = await self.reconcile()
        self.assertEqual(result[0]['status'], 'supported')
        self.assertEqual(result[1], original[1])
        self.assertEqual(self.entries, original)
        self.assertEqual(self.model.call_args.kwargs['stage'], 'eval.choice_jev_review')
        self.assertEqual(self.diag['answer_jev_support']['review_letters'], ['A'])

    async def test_disagreement_demotes_with_review_and_keeps_other_options(self):
        self.decider.return_value = decision('contradicted')
        revised = copy.deepcopy(self.payload['options'][0])
        revised['claims'][0].update(status='unsupported', reason='contradicted', citations=[])
        self.model.return_value = {'options': [revised]}
        result = await self.reconcile()
        self.assertEqual(result[0]['status'], 'unsupported')
        self.assertEqual(result[1], self.entries[1])

    async def test_low_confidence_agreement_is_reviewed(self):
        self.decider.return_value = decision(confidence=.3)
        self.model.return_value = {'options': [self.payload['options'][0]]}
        await self.reconcile()
        self.model.assert_awaited_once()

    async def test_invented_citation_cannot_promote_unsupported(self):
        self.entries[0]['claims'][0]['status'] = 'unsupported'
        self.entries[0]['status'] = 'unsupported'
        revised = copy.deepcopy(self.payload['options'][0])
        revised['claims'][0]['citations'][0]['quote'] = 'I am a professional Facebook manager.'
        self.model.return_value = {'options': [revised]}
        self.assertEqual((await self.reconcile())[0]['status'], 'unsupported')

    async def test_failure_is_observable_and_preserves_initial_result(self):
        for error in (TimeoutError(), ValueError('bad response'), budget.BudgetExceeded('budget')):
            self.decider.side_effect = error
            self.assertEqual(await self.reconcile(), self.entries)
            self.assertEqual(self.diag['answer_jev_support']['status'], 'fallback')
        self.model.assert_not_awaited()

    async def test_cancel_propagates_and_releases_reservations(self):
        self.decider.side_effect = asyncio.CancelledError
        with budget.scope(seconds=240, calls=10, tokens=128000) as limits:
            with self.assertRaises(asyncio.CancelledError):
                await self.reconcile()
            self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))

    async def test_disabled_fake_missing_key_and_insufficient_budget_never_dispatch(self):
        for settings in ({'CHOICE_JEV_SUPPORT': False}, {'FAKE': True}, {'JEV_API_KEY': ''}):
            with patch.multiple(config, **settings):
                self.assertEqual(await self.reconcile(), self.entries)
        with budget.scope(seconds=240, calls=4, tokens=128000):
            self.assertEqual(await self.reconcile(), self.entries)
        self.decider.assert_not_awaited()

    async def test_invalid_review_option_coverage_keeps_first_result(self):
        self.decider.return_value = decision('insufficient')
        self.model.return_value = self.payload  # Review must assess only A, never B.
        self.assertEqual(await self.reconcile(), self.entries)
        self.assertEqual(self.diag['answer_jev_support']['status'], 'kept_initial')

    async def test_review_failure_or_cancel_releases_budget(self):
        self.decider.return_value = decision('insufficient')
        for error in (TimeoutError(), llm.LLMError('unavailable'), asyncio.CancelledError()):
            self.model.side_effect = error
            with budget.scope(seconds=240, calls=10, tokens=128000) as limits:
                if isinstance(error, asyncio.CancelledError):
                    with self.assertRaises(asyncio.CancelledError):
                        await self.reconcile()
                else:
                    self.assertEqual(await self.reconcile(), self.entries)
                self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))

    async def test_review_cannot_use_assistant_only_quote_as_personal_support(self):
        self.sources['s0']['role'] = 'assistant'
        self.entries[0]['claims'][0]['status'] = 'unsupported'
        self.entries[0]['status'] = 'unsupported'
        self.model.return_value = {'options': [self.payload['options'][0]]}
        self.assertEqual((await self.reconcile())[0]['status'], 'unsupported')

    async def test_real_transport_drives_review_with_shared_budget(self):
        self.entries[0]['claims'][0]['status'] = 'unsupported'
        self.entries[0]['status'] = 'unsupported'
        client_class = httpx.AsyncClient
        seen = []
        def response(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json=decision())
        def client(**kwargs):
            return client_class(transport=httpx.MockTransport(response), **kwargs)
        async def review(*args, **kwargs):
            limits = budget.current.get()
            self.assertEqual(limits.calls, 1)
            self.assertEqual(limits.reserved_calls, limits.max_calls - 2)
            limits.before_call()
            return {'options': [self.payload['options'][0]]}
        self.model.side_effect = review
        with patch.object(jev, 'decide', _real_decide), \
                patch.object(httpx, 'AsyncClient', side_effect=client), \
                patch.multiple(config, PROVIDER_RPM=0, PROVIDER_SOFT_TPM=0):
            with budget.scope(seconds=240, calls=10, tokens=128000) as limits:
                self.assertEqual((await self.reconcile())[0]['status'], 'supported')
                self.assertEqual((limits.calls, limits.reserved_calls, limits.reserved_tokens), (2, 0, 0))
        self.assertEqual(seen[0]['model'], 'typesafe/jev-1.13')
        self.assertEqual(set(seen[0]['questions']), {'A:0'})

    async def test_review_cannot_relabel_personal_premise_as_generic(self):
        self.decider.return_value = decision('not_a_premise')
        initial = copy.deepcopy(self.payload)
        initial['options'][0]['claims'][0].update(status='unsupported', reason='no_source', citations=[])
        async def respond(prompt, **kwargs):
            if kwargs['stage'] == 'eval.choice_support':
                return initial
            if kwargs['stage'] == 'eval.choice_jev_review':
                return {'options': [dict(letter='A', kind='generic', claims=[])]}
            if kwargs['stage'] == 'eval.choice_entailment':
                return entailment_response(prompt)
            raise AssertionError(kwargs['stage'])
        self.model.side_effect = respond
        with patch.object(config, 'CHOICE_ENTAILMENT_REVIEW', False):
            selected = await answer_choice.answer(dict(question='What should I do?', options=OPTIONS),
                                                   seal(memory(QUOTE)), self.diag)
        self.assertEqual(selected, 'B')

    async def test_review_cannot_erase_unsupported_core_to_gain_partial_tier(self):
        options = ['A. Since you own a car and you love cycling, drive to a bike trail.',
                   'B. Take a quiet break.']
        source = 'I love cycling.'
        initial = assessment('you love cycling', 's0', source)
        initial['options'][0]['claims'].insert(0, dict(text='you own a car',
            status='unsupported', premise_type='ownership', reason='no_source', citations=[]))
        revised = assessment('you love cycling', 's0', source)
        revised['options'].pop()
        jev_result = decision()
        jev_result['answers']['A:1'] = copy.deepcopy(jev_result['answers']['A:0'])
        self.decider.return_value = jev_result
        async def respond(prompt, **kwargs):
            if kwargs['stage'] == 'eval.choice_support':
                return initial
            if kwargs['stage'] == 'eval.choice_jev_review':
                return revised
            if kwargs['stage'] == 'eval.choice_entailment':
                return entailment_response(prompt, rejected=('A:option',))
            raise AssertionError(kwargs['stage'])
        self.model.side_effect = respond
        with patch.object(config, 'CHOICE_ENTAILMENT_REVIEW', False):
            selected = await answer_choice.answer(dict(question='What should I do?', options=options),
                                                   seal(memory(source)), self.diag)
        self.assertEqual(selected, 'B')
        self.assertEqual(self.diag['answer_jev_support']['status'], 'kept_initial')

    async def test_answer_flow_runs_review_before_entailment(self):
        initial = copy.deepcopy(self.payload)
        initial['options'][0]['claims'][0].update(status='unsupported', citations=[], reason='no_source')
        stages = []
        async def respond(prompt, **kwargs):
            stages.append(kwargs['stage'])
            if kwargs['stage'] == 'eval.choice_support':
                return initial
            if kwargs['stage'] == 'eval.choice_jev_review':
                return {'options': [self.payload['options'][0]]}
            if kwargs['stage'] == 'eval.choice_entailment':
                self.assertIn(QUOTE, prompt)
                return entailment_response(prompt)
            raise AssertionError(kwargs['stage'])
        self.model.side_effect = respond
        selected = await answer_choice.answer(dict(question='What should I do?', options=OPTIONS),
                                               seal(memory(QUOTE)), self.diag)
        self.assertEqual(selected, 'A')
        self.assertEqual(stages, ['eval.choice_support', 'eval.choice_jev_review', 'eval.choice_entailment'])


if __name__ == '__main__':
    unittest.main()
