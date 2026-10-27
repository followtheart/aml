import asyncio
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as answer, budget, config, llm, source_scope


class SourceScopeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.multiple(config, CHOICE_SOURCE_SCOPE=True, FAKE=False).start()
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        self.text = 'En mi sala tengo fotos de mi familia.'
        self.sources = {'s': dict(id='s', role='user', text=self.text, item_ids=[])}
        self.keys = [('s', self.text)]

    def payload(self, scope='self_report', polarity='affirmative', id='0'):
        return dict(judgments=[dict(id=id, scope=scope, polarity=polarity)])

    async def test_explicit_approval_only(self):
        for scope, polarity, expected in [('self_report', 'affirmative', set(self.keys)),
                ('self_report', 'negative', set()), ('third_party', 'affirmative', set()),
                ('hypothetical', 'affirmative', set()), ('question', 'unclear', set()),
                ('ambiguous', 'affirmative', set())]:
            with patch.object(llm, 'complete_json', AsyncMock(return_value=self.payload(scope, polarity))):
                self.assertEqual(await source_scope.classify(self.keys, self.sources, {}), expected)

    async def test_bad_ids_fail_closed(self):
        for payload in [self.payload(id='unknown'), dict(judgments=[]),
                        dict(judgments=self.payload()['judgments'] * 2)]:
            with patch.object(llm, 'complete_json', AsyncMock(return_value=payload)):
                diagnostics = {}
                self.assertEqual(await source_scope.classify(self.keys, self.sources, diagnostics), set())
                self.assertEqual(diagnostics['answer_source_scope']['status'], 'fallback')

    async def test_nonuser_nonexact_and_disabled_never_call_provider(self):
        with patch.object(llm, 'complete_json', AsyncMock()) as provider:
            await source_scope.classify([('s', 'invented quote')], self.sources, {})
            await source_scope.classify(self.keys, {'s': dict(self.sources['s'], role='assistant')}, {})
            with patch.object(config, 'CHOICE_SOURCE_SCOPE', False):
                await source_scope.classify(self.keys, self.sources, {})
        provider.assert_not_awaited()

    async def test_input_and_answer_budget_skip(self):
        long = 'x' * 25000
        with patch.object(llm, 'complete_json', AsyncMock()) as provider:
            await source_scope.classify([('s', long)], {'s': dict(role='user', text=long)}, {})
            with budget.scope(seconds=120, calls=4):
                await source_scope.classify(self.keys, self.sources, {})
        provider.assert_not_awaited()

    async def test_failure_releases_reservations_and_cancellation_propagates(self):
        with budget.scope(seconds=120, calls=5) as limits:
            with patch.object(llm, 'complete_json', AsyncMock(side_effect=RuntimeError('offline'))):
                self.assertEqual(await source_scope.classify(self.keys, self.sources, {}), set())
            self.assertEqual(limits.reserved_calls, 0)
            with patch.object(llm, 'complete_json', AsyncMock(side_effect=asyncio.CancelledError)):
                with self.assertRaises(asyncio.CancelledError):
                    await source_scope.classify(self.keys, self.sources, {})
            self.assertEqual(limits.reserved_calls, 0)

    async def test_other_citation_guards_cannot_be_overridden(self):
        citation = answer.Citation(source_id='s', quote=self.text,
                                    basis='self_report', subject='current_user')
        self.assertFalse(answer._check_citation(citation, 'your family photos', self.sources, 'ownership')['valid'])
        self.assertTrue(answer._check_citation(citation, 'your family photos', self.sources,
                                             'ownership', scope_verified=True)['valid'])
        assistant = {'s': dict(self.sources['s'], role='assistant')}
        self.assertIn('not_user_source', answer._check_citation(citation, 'your family photos',
                      assistant, 'ownership', scope_verified=True)['validation_errors'])
        bad = citation.model_copy(update={'quote': 'missing'})
        self.assertIn('quote_not_in_source', answer._check_citation(bad, 'your family photos',
                      self.sources, 'ownership', scope_verified=True)['validation_errors'])

    async def test_recovery_attaches_without_promoting_support(self):
        claim = dict(text='your family photos', premise_type='ownership', status='unsupported',
                     reason='no_source', citations=[], dropped_citations=[], validation_errors=[])
        entries = [dict(letter='D', option='ANSWER_OPTION_CANARY', kind='personal', primary_claim=0,
                        status='unsupported', validation_status='valid', validation_errors=[], claims=[claim])]
        before = copy.deepcopy(entries)
        proposal = answer.CitationRecovery(claim_id='D:0', premise_type='interest', supported=True,
                    citations=[answer.Citation(source_id='s', quote=self.text,
                                               basis='self_report', subject='current_user')])
        with patch.object(answer, '_judge', AsyncMock(return_value=[proposal])), \
                patch.object(llm, 'complete_json', AsyncMock(return_value=self.payload())) as provider:
            result = await answer.recover_missing_citations(entries, self.sources,
                                                           {'question': 'QUESTION_CANARY'}, {})
        self.assertEqual(entries, before)
        recovered = result[0]['claims'][0]
        self.assertEqual(recovered['status'], 'unsupported')
        self.assertEqual(recovered['premise_type'], 'ownership')
        self.assertEqual(recovered['citation_recovery'], 'awaiting_entailment')
        self.assertTrue(recovered['citations'][0]['valid'])
        prompt = provider.await_args.args[0]
        self.assertNotIn('ANSWER_OPTION_CANARY', prompt)
        self.assertNotIn('QUESTION_CANARY', prompt)

    async def test_scope_approval_preserves_attribution_hypothetical_and_denial_rejections(self):
        for text, quote, expected in [
                ('Jordan wrote: I own a car.', 'I own a car.', 'third_party_or_hypothetical'),
                ('Imagine that I own a car.', 'I own a car.', 'third_party_or_hypothetical'),
                ('I do not own a car.', 'I do not own a car.', 'denied_event')]:
            sources = {'s': dict(id='s', role='user', text=text)}
            citation = answer.Citation(source_id='s', quote=quote,
                                        basis='self_report', subject='current_user')
            checked = answer._check_citation(citation, 'you own a car', sources,
                                             'ownership', scope_verified=True)
            self.assertIn(expected, checked['validation_errors'])


if __name__ == '__main__':
    unittest.main()
