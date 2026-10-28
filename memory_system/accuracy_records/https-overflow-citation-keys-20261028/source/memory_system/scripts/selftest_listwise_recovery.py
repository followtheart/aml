"""Offline behavioral regressions for source-verified listwise deletion review.

Run from the repository root with:
  memory_system/.venv/Scripts/python.exe memory_system/scripts/selftest_listwise_recovery.py
"""
import copy
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import time
import unittest
import httpx
from unittest.mock import patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_read_text = Path.read_text
with patch.object(Path, 'read_text', lambda path, *a, **k:
                  '' if path.name == '.env' else _read_text(path, *a, **k)):
    from app import answer_context, budget, cascade_rerank as cascade, config, cross_encoder, llm, schemas


RIDE_SOURCE = 'I used to ride there a lot before an old leg injury started acting up again.'
QUESTION = 'How can I cope with losing a physical activity I used to enjoy?'
OPTIONS = ['A. Return gently to cycling after an old leg injury.',
           'B. Start a new collection.']


def evidence(mid, text, *, role='user', summary=None):
    sources = [dict(request_id='listwise-fixture', message_index=int(mid[1:]),
                    source_event_id='source-' + mid, role=role, content=text,
                    timestamp='2026-01-01T00:00:00Z')]
    body = answer_context.with_evidence(summary or text, sources)
    return dict(id=mid, content=summary or text, _rank_text=body, _fused=.9,
                _packet_item=dict(id=mid, content=body, sources=sources))


def prompt_rows(prompt):
    rows = re.findall(r'^(\d+): (.*)$', prompt, re.M)
    if [int(i) for i, _ in rows] != list(range(len(rows))) or not rows:
        raise AssertionError('Listwise prompt must expose contiguous document indices')
    return [text for _, text in rows]


def decision(mid, quote, *, kind='user_context', source_id='s0',
             target_id='question', target_quote=QUESTION):
    return dict(candidate_id=mid, useful=True, target_id=target_id,
                target_quote=target_quote, evidence_kind=kind,
                citations=[dict(source_id=source_id, quote=quote)])


def selected_ids(rows):
    return [r['id'] for r in rows if r.get('_cascade_selected')]


class ListwiseRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def test_validation_diagnostics_distinguish_contract_failures(self):
        valid = dict(ranking=[0, 1, 2, 3], irrelevant=[], groups=[])
        cases = [
            (None, 'response_not_object'),
            (dict(valid, ranking=[0, 0, 2, 3]), 'ranking_invalid_indices'),
            (dict(valid, ranking=[0, 1, 2]), 'ranking_incomplete'),
            (dict(valid, irrelevant=[4]), 'irrelevant_invalid_indices'),
            (dict(valid, groups=None), 'groups_not_array'),
            (dict(valid, groups=[[0]]), 'group_invalid_indices_or_size'),
            (dict(valid, irrelevant=[1], groups=[[0, 1]]), 'group_contains_irrelevant'),
            (dict(valid, groups=[[0, 1], [1, 2]]), 'groups_overlap'),
        ]
        for payload, code in cases:
            with self.subTest(code=code), self.assertRaises(ValueError) as caught:
                cascade.validate(payload, 4)
            self.assertEqual(caught.exception.validation_code, code)
        self.assertEqual(cascade.validate(valid, 4), ([0, 1, 2, 3], set(), []))

    def setUp(self):
        settings = patch.multiple(config, LISTWISE_SPAN_REFS=False, LISTWISE_KEYED_OUTPUT=False, RERANK_REPAIR_MAX_CALLS=1, CASCADE_COARSE_LIMIT=50, CASCADE_FINE_LIMIT=12,
            CASCADE_LLM_LIMIT=10, CE_BATCH_SIZE=8, CE_MAX_REQUEST_BYTES=90000,
            CE_MAX_DOCUMENT_BYTES=12000, CE_TIMEOUT_SECONDS=1., CE_DEADLINE_SECONDS=10.,
            RERANK_CONCURRENCY=2, RERANK_MAX_PROMPT_BYTES=24000, create=True)
        settings.start()
        self.addCleanup(settings.stop)
        network = patch('socket.socket.connect', side_effect=AssertionError('No network in offline regression'))
        network.start()
        self.addCleanup(network.stop)

    async def run_case(self, rows, decisions=(), *, deleted=None, calls=8,
                       primary=None, recovery_error=None, after_primary=None,
                       query=QUESTION, options=None, ce_score=.95, requirements=(),
                       primary_sequence=None):
        self.stages, self.prompts, self.ce_inputs = [], {}, []
        self.systems = {}
        sequence = iter(primary_sequence) if primary_sequence is not None else None
        async def ce(query, documents, **kwargs):
            budget.current.get().before_call()
            self.ce_inputs.extend(documents)
            return [ce_score] * len(documents)
        async def classify(prompt, **kwargs):
            budget.current.get().before_call()
            stage = kwargs['stage']
            self.stages.append(stage)
            self.prompts[stage] = prompt
            self.systems[stage] = kwargs.get('system', '')
            if stage == 'search.listwise.recovery':
                if recovery_error:
                    raise recovery_error
                return copy.deepcopy(decisions if isinstance(decisions, dict)
                                     else dict(decisions=list(decisions)))
            if stage not in ('search.listwise', 'search.listwise.repair', 'search.listwise.retry'):
                raise AssertionError('Unexpected model stage: ' + stage)
            if sequence is not None:
                result = next(sequence)
                if isinstance(result, BaseException):
                    raise result
                return copy.deepcopy({k: result[k] for k in kwargs['schema']['properties'] if k in result})
            count = len(prompt_rows(prompt))
            result = primary if primary is not None else dict(ranking=list(range(count)),
                irrelevant=list(range(count)) if deleted is None else list(deleted), groups=[])
            if after_primary:
                after_primary(budget.current.get())
            return copy.deepcopy({k: result[k] for k in kwargs['schema']['properties'] if k in result})
        plan = {'_coverage_requirements': list(requirements)}
        with budget.scope(seconds=30, calls=calls, tokens=64000) as limits, \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=classify):
            self.limits = limits
            result = await cascade.rank(schemas.SearchRequest(user_id='u', query=query,
                options=OPTIONS if options is None else options), plan, rows)
        self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))
        self.assertLessEqual(limits.calls, limits.max_calls)
        self.assertLessEqual(limits.tokens, limits.max_tokens)
        self.assertLessEqual(self.stages.count('search.listwise.recovery'), 1)
        return result, plan

    async def test_all_irrelevant_cannot_erase_verified_positive_user_evidence(self):
        row = evidence('m0', RIDE_SOURCE)
        original = copy.deepcopy(row)
        result, plan = await self.run_case([row], [decision('m0', RIDE_SOURCE)])
        self.assertEqual(self.ce_inputs, [original['_rank_text']])
        self.assertEqual(row['_packet_item'], original['_packet_item'])
        self.assertIn('m0', selected_ids(result),
                      f'Positive original user evidence was erased: stages={self.stages}; trace={plan["_cascade"]["listwise"]}')

    async def test_keyed_irrelevant_still_requires_source_verified_review(self):
        row = evidence('m0', RIDE_SOURCE)
        with patch.object(config, 'LISTWISE_KEYED_OUTPUT', True):
            result, plan = await self.run_case([row], [decision('m0', RIDE_SOURCE)],
                primary_sequence=[dict(candidates=dict(c0=dict(priority=10, relevance='irrelevant')), groups=[])])
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertEqual(plan['_cascade']['listwise']['recovery']['restored_ids'], ['m0'])
        self.assertEqual(self.stages, ['search.listwise', 'search.listwise.recovery'])

    async def test_keyed_uncertain_is_kept_without_deletion_review(self):
        with patch.object(config, 'LISTWISE_KEYED_OUTPUT', True):
            result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)],
                primary_sequence=[dict(candidates=dict(c0=dict(priority=10, relevance='uncertain')), groups=[])])
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertEqual(self.stages, ['search.listwise'])
        self.assertEqual(plan['_cascade']['listwise']['primary_irrelevant_ids'], [])

    async def test_partial_deletion_appends_restored_in_input_order_without_displacing_selected(self):
        rows = [evidence('m0', RIDE_SOURCE), evidence('m1', 'I enjoy swimming.'),
                evidence('m2', 'I enjoy hiking on weekends.')]
        result, plan = await self.run_case(rows, [decision('m2', rows[2]['content']),
            decision('m0', RIDE_SOURCE)], deleted=[0, 2])
        self.assertEqual(selected_ids(result), ['m1', 'm0', 'm2'])

    async def test_true_false_review_does_not_keep_every_deleted_candidate(self):
        rows = [evidence('m0', RIDE_SOURCE), evidence('m1', 'I sorted receipts.')]
        negative = dict(candidate_id='m1', useful=False, target_id='', target_quote='',
                        evidence_kind='none', citations=[])
        result, plan = await self.run_case(rows, [decision('m0', RIDE_SOURCE), negative])
        self.assertEqual(selected_ids(result), ['m0'])

    async def test_no_deletion_does_not_spend_a_recovery_call(self):
        result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)], deleted=[])
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertEqual(self.stages, ['search.listwise'])

    async def test_unresolved_fallback_is_bounded_and_does_not_override_valid_negative(self):
        query = 'How can I follow emerging cryptocurrency markets?'
        rows = [evidence('m0', 'I follow stock news.'),
                evidence('m1', 'I ask about stablecoins and digital currencies.'),
                evidence('m2', 'I ask about cryptocurrency regulation.')]
        invalid = decision('m1', rows[1]['content'], target_quote='not in the question')
        negative = dict(candidate_id='m2', useful=False, target_id='', target_quote='',
                        evidence_kind='none', citations=[])
        with patch.object(config, 'LISTWISE_UNREVIEWED_LIMIT', 1):
            result, plan = await self.run_case(rows, [invalid, negative], deleted=[1, 2],
                query=query, requirements=[dict(id='question', text=query)])
        self.assertEqual(selected_ids(result), ['m0', 'm1'])
        recovery = plan['_cascade']['listwise']['recovery']
        self.assertEqual(recovery['provisional_ids'], ['m1'])
        self.assertFalse(recovery['restored_ids'])
        self.assertEqual(recovery['status'], 'partial')
        self.assertTrue(any(x['id'] == 'm2' and x['reason'] == 'irrelevant_confirmed'
                            for x in plan['_cascade']['omitted']))
        self.assertFalse(any(x['id'] == 'm1' and x['reason'] == 'irrelevant_confirmed'
                             for x in plan['_cascade']['omitted']))

    async def test_invalid_primary_permutation_cannot_authorize_a_deleted_review(self):
        result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)],
            primary=dict(ranking=[True], irrelevant=[0], groups=[]))
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertEqual(self.stages, ['search.listwise', 'search.listwise.repair'])
        self.assertEqual([a['validation_code'] for a in plan['_cascade']['listwise']['attempts']],
                         ['ranking_invalid_indices', 'ranking_invalid_indices'])

    async def test_ranking_repair_receives_missing_and_repeated_indices(self):
        result, plan = await self.run_case([evidence('m0', RIDE_SOURCE), evidence('m1', 'I swim.')],
            primary_sequence=[dict(ranking=[0, 0], irrelevant=[], groups=[]),
                              dict(ranking=[1, 0], irrelevant=[], groups=[])])
        system = self.systems['search.listwise.repair']
        self.assertIn('"missing": [1]', system)
        self.assertIn('"repeated": [0]', system)
        self.assertEqual(plan['_cascade']['listwise']['status'], 'recovered')
        self.assertEqual(plan['_rerank_errors'], [])
        self.assertEqual(len(selected_ids(result)), 2)

    async def test_failed_group_repair_preserves_order_and_reviews_deletions(self):
        result, plan = await self.run_case(
            [evidence('m0', RIDE_SOURCE), evidence('m1', 'I swim.'), evidence('m2', 'I hike.')],
            primary_sequence=[
                dict(ranking=[2, 0, 1], irrelevant=[0], groups=[[0, 1]]),
                dict(ranking=[0, 0, 1], irrelevant=[], groups=[[0, 1]])])
        trace = plan['_cascade']['listwise']
        self.assertEqual(trace['status'], 'groups_partial')
        self.assertEqual(selected_ids(result), [trace['candidate_ids'][i] for i in [2, 1, 0]])
        self.assertEqual(trace['primary_irrelevant_ids'], [trace['candidate_ids'][0]])
        self.assertEqual(plan['_evidence_groups'], [])
        self.assertEqual(plan['_rerank_status'], 'partial')
        self.assertIn('search.listwise.recovery', self.stages)
        self.assertIn(trace['candidate_ids'][0], trace['recovery']['unreviewed_ids'])

    async def test_group_failure_cannot_bypass_negative_review(self):
        result, plan = await self.run_case(
            [evidence('m0', RIDE_SOURCE), evidence('m1', 'I swim.')],
            [dict(candidate_id='m0', useful=False, target_id='', target_quote='',
                  evidence_kind='none', citations=[])],
            primary_sequence=[dict(ranking=[1, 0], irrelevant=[0], groups=[[0, 1]]),
                              dict(groups=[[0, 1]])])
        trace = plan['_cascade']['listwise']
        self.assertEqual(trace['status'], 'groups_partial')
        self.assertEqual(trace['candidate_contract_status'], 'valid')
        self.assertEqual(trace['groups_status'], 'invalid')
        self.assertEqual(selected_ids(result), ['m1'])
        self.assertEqual(trace['recovery']['confirmed_irrelevant_ids'], ['m0'])
        self.assertEqual(plan['_rerank_errors'][0]['stage'], 'listwise_groups')

    async def test_wrapped_500_retries_once_and_recovers(self):
        failure = llm.LLMError('provider failed')
        failure.__cause__ = httpx.HTTPStatusError('server failure',
            request=httpx.Request('POST', 'https://fixture.invalid'),
            response=httpx.Response(500))
        _, plan = await self.run_case([evidence('m0', RIDE_SOURCE)],
            primary_sequence=[failure, dict(ranking=[0], irrelevant=[], groups=[])])
        self.assertEqual(self.stages, ['search.listwise', 'search.listwise.retry'])
        self.assertEqual(plan['_rerank_status'], 'recovered')
        self.assertEqual(plan['_rerank_errors'], [])

    async def test_transport_retry_is_bounded_and_preserves_candidates(self):
        failure = httpx.ConnectError('connection failed')
        result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)],
            primary_sequence=[failure, failure])
        self.assertEqual(self.stages, ['search.listwise', 'search.listwise.retry'])
        self.assertEqual(plan['_cascade']['listwise']['status'], 'fallback')
        self.assertEqual(selected_ids(result), ['m0'])

    async def test_transport_retry_cannot_exceed_call_budget(self):
        result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)], calls=2,
            primary_sequence=[httpx.ConnectError('connection failed')])
        self.assertEqual(self.stages, ['search.listwise'])
        self.assertEqual(plan['_cascade']['listwise']['status'], 'fallback')
        self.assertEqual(selected_ids(result), ['m0'])

    async def test_permanent_provider_failure_is_not_retried(self):
        failure = httpx.HTTPStatusError('unauthorized',
            request=httpx.Request('POST', 'https://fixture.invalid'),
            response=httpx.Response(401))
        _, plan = await self.run_case([evidence('m0', RIDE_SOURCE)],
            primary_sequence=[failure])
        self.assertEqual(self.stages, ['search.listwise'])
        self.assertEqual(plan['_cascade']['listwise']['status'], 'fallback')

    async def test_transport_retry_obeys_listwise_deadline(self):
        with patch.object(config, 'RERANK_DEADLINE_SECONDS', .03):
            result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)],
                primary_sequence=[httpx.ConnectError('connection failed')])
        self.assertEqual(self.stages, ['search.listwise'])
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertEqual(plan['_cascade']['listwise']['status'], 'fallback')

    async def test_group_repair_success_can_use_valid_groups(self):
        _, plan = await self.run_case([evidence('m0', RIDE_SOURCE), evidence('m1', 'I swim.')],
            primary_sequence=[dict(ranking=[1, 0], irrelevant=[], groups=[[0]]),
                              dict(ranking=[1, 0], irrelevant=[], groups=[[0, 1]])])
        trace = plan['_cascade']['listwise']
        self.assertEqual(trace['status'], 'recovered')
        self.assertEqual(plan['_evidence_groups'], [trace['candidate_ids']])
        self.assertIn('group_invalid_indices_or_size', self.systems['search.listwise.repair'])

    async def test_retry_token_exhaustion_does_not_erase_valid_ranking(self):
        def exhaust(limits):
            limits.tokens = limits.max_tokens
        result, plan = await self.run_case(
            [evidence('m0', RIDE_SOURCE), evidence('m1', 'I swim.')],
            primary=dict(ranking=[1, 0], irrelevant=[], groups=[[0]]),
            after_primary=exhaust)
        trace = plan['_cascade']['listwise']
        self.assertEqual(self.stages, ['search.listwise'])
        self.assertEqual(trace['status'], 'groups_partial')
        self.assertEqual(selected_ids(result), list(reversed(trace['candidate_ids'])))

    async def test_rules_bypass_and_never_supply_ordinary_evidence(self):
        rule = evidence('m1', 'Please forget that I enjoy cycling.')
        rule['_user_rule'] = True
        result, plan = await self.run_case([rule, evidence('m0', RIDE_SOURCE)],
                                       [decision('m0', RIDE_SOURCE)])
        self.assertEqual(selected_ids(result), ['m1', 'm0'])
        self.assertNotIn(rule['_rank_text'], self.ce_inputs)
        self.assertNotIn(rule['content'], self.prompts['search.listwise.recovery'])
        self.assertEqual(rule['_score_kind'], 'constraint')

    async def test_third_party_and_hypothetical_cannot_be_relabelled_user_context(self):
        for text in ['My friend used to ride there before her old leg injury returned.',
                     'If I used to ride there before an old leg injury, what could help?']:
            with self.subTest(text=text):
                result, plan = await self.run_case([evidence('m0', text)], [decision('m0', text)])
                self.assertEqual(selected_ids(result), ['m0'])
                self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])

    async def test_assistant_cannot_be_relabelled_user_context(self):
        result, plan = await self.run_case([evidence('m0', RIDE_SOURCE, role='assistant')],
                                       [decision('m0', RIDE_SOURCE)])
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])

    async def test_sentence_initial_relation_attribution_cannot_launder_quoted_first_person(self):
        quote = 'I used to ride before my old leg injury.'
        for prefix in ('My friend wrote: ', 'My friend says: ', 'Our colleague replied: ',
                       'Our colleague writes: ', 'Jordan said: '):
            with self.subTest(prefix=prefix):
                result, plan = await self.run_case([evidence('m0', prefix + quote)], [decision('m0', quote)])
                self.assertEqual(selected_ids(result), ['m0'])
                self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])

    async def test_counterevidence_and_background_preserve_qualifiers_without_promoting_coverage(self):
        cases = [('I have never owned a bicycle.', 'user', 'counterevidence'),
                 ('My friend used to cycle before her injury.', 'user', 'third_party_context'),
                 ('A fictional character used to ride there.', 'assistant', 'assistant_context')]
        for text, role, kind in cases:
            with self.subTest(kind=kind):
                row = evidence('m0', text, role=role)
                original = copy.deepcopy(row)
                result, plan = await self.run_case([row], [decision('m0', text, kind=kind)])
                self.assertEqual(selected_ids(result), ['m0'])
                self.assertEqual(row['_rank_text'], original['_rank_text'])
                self.assertEqual(row['_packet_item'], original['_packet_item'])
                self.assertFalse(row.get('_supported_coverage_ids'))
                self.assertFalse(row.get('_personal_evidence'))

    async def test_topic_question_supports_interest_without_ownership_or_diagnosis(self):
        text = 'Why do vinyl LPs have a warmer sound than digital recordings?'
        query = 'What music-related activity might interest me?'
        row = evidence('m0', text)
        result, plan = await self.run_case([row],
            [decision('m0', text, kind='topic_interest', target_quote=query)], query=query)
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertFalse(row.get('_supported_coverage_ids'))
        self.assertFalse(row.get('_personal_evidence'))
        self.assertEqual(row['_packet_item']['sources'][0]['content'], text)

    async def test_topic_question_cannot_be_relabelled_user_experience(self):
        text = 'Why do vinyl LPs have a warmer sound than digital recordings?'
        result, plan = await self.run_case([evidence('m0', text)], [decision('m0', text)])
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])

    async def test_third_party_question_is_not_current_user_topic_interest(self):
        text = 'My friend asks why vinyl LPs have a warmer sound than digital recordings.'
        result, plan = await self.run_case([evidence('m0', text)],
                                       [decision('m0', text, kind='topic_interest')])
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])

    async def test_recovery_requires_verbatim_candidate_local_source_and_target(self):
        bad = [dict(quote='I rode there before my injury.'), dict(source_id='s900'),
               dict(target_id='option:7'), dict(target_quote='a nonexistent query premise')]
        for changes in bad:
            with self.subTest(changes=changes):
                kwargs = dict(changes)
                result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)],
                    [decision('m0', kwargs.pop('quote', RIDE_SOURCE), **kwargs)])
                self.assertEqual(selected_ids(result), ['m0'])
                self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])

    async def test_citation_cannot_borrow_another_candidates_source(self):
        rows = [evidence('m0', RIDE_SOURCE), evidence('m1', 'I collect stamps.')]
        result, plan = await self.run_case(rows, [decision('m1', RIDE_SOURCE)], deleted=[1])
        self.assertEqual(selected_ids(result), ['m0', 'm1'])
        self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])

    async def test_unsourced_summary_or_hidden_source_cannot_become_original_evidence(self):
        for hidden in (False, True):
            with self.subTest(hidden=hidden):
                row = evidence('m0', RIDE_SOURCE)
                if hidden:
                    row['_rank_text'] = 'A generic physical activity summary.'
                else:
                    row['_packet_item']['sources'] = []
                result, plan = await self.run_case([row], [decision('m0', RIDE_SOURCE)])
                self.assertEqual(selected_ids(result), ['m0'])
                self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])

    async def test_invalid_identity_bool_and_duplicate_response_never_resurrect(self):
        valid = decision('m0', RIDE_SOURCE)
        invalid = [[dict(valid, candidate_id='m999')], [dict(valid, useful='true')],
                   [valid, valid], [dict(valid, candidate_id=0)]]
        for payload in invalid:
            with self.subTest(payload=payload):
                result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)], payload)
                self.assertEqual(selected_ids(result), ['m0'])
                self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])
                self.assertEqual(self.stages.count('search.listwise.recovery'), 1)

    async def test_model_failure_has_no_unverified_high_score_resurrection_or_format_retry(self):
        for failure in (TimeoutError('review timeout'), ValueError('invalid review')):
            with self.subTest(failure=type(failure).__name__):
                result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)],
                    recovery_error=failure)
                self.assertEqual(selected_ids(result), ['m0'])
                self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])
                self.assertEqual(self.stages.count('search.listwise.recovery'), 1)
                self.assertTrue(plan['_rerank_errors'])

    async def test_call_and_token_exhaustion_skip_review_without_exceeding_shared_budget(self):
        for tokens in (False, True):
            with self.subTest(tokens=tokens):
                def spend(limits):
                    limits.tokens = limits.max_tokens - 1
                result, plan = await self.run_case([evidence('m0', RIDE_SOURCE)],
                    [decision('m0', RIDE_SOURCE)], calls=8 if tokens else 2,
                    after_primary=spend if tokens else None)
                self.assertEqual(selected_ids(result), ['m0'])
                self.assertFalse(plan['_cascade']['listwise']['recovery']['restored_ids'])
                self.assertNotIn('search.listwise.recovery', self.stages)
                self.assertTrue(plan['_rerank_errors'])

    async def test_cancellation_propagates_and_releases_reservations(self):
        with self.assertRaises(asyncio.CancelledError):
            await self.run_case([evidence('m0', RIDE_SOURCE)],
                recovery_error=asyncio.CancelledError())
        self.assertEqual((self.limits.reserved_calls, self.limits.reserved_tokens), (0, 0))

    async def test_review_preserves_full_source_and_does_not_use_a_ce_probability_threshold(self):
        text = 'I cycled every weekend. ' + ('My route followed the river. ' * 90) + RIDE_SOURCE
        row = evidence('m0', text, summary='Cycling experience with an old injury.')
        original = copy.deepcopy(row)
        result, plan = await self.run_case([row], [decision('m0', RIDE_SOURCE)], ce_score=-3.5)
        self.assertEqual(selected_ids(result), ['m0'])
        self.assertEqual(row['_rank_text'], original['_rank_text'])
        self.assertEqual(row['_packet_item'], original['_packet_item'])
        self.assertIn(text, self.prompts['search.listwise.recovery'])
        self.assertEqual(row['_final'], -3.5)


class MeaningfulStatementTests(unittest.TestCase):
    def validate(self, *, quote=RIDE_SOURCE, target_quote=QUESTION, kind='user_context'):
        from app import listwise_recovery
        cards = [dict(candidate_id='m0', sources=[dict(source_id='s0', role='user',
                                                     text=RIDE_SOURCE)])]
        payload = dict(decisions=[decision('m0', quote, kind=kind, target_quote=target_quote)])
        restored, judgments = listwise_recovery.validate(payload, cards, {'question': QUESTION})
        self.assertEqual(restored, set())
        self.assertFalse(judgments[0]['valid'])
        self.assertTrue(judgments[0]['validation_errors'])

    def test_whitespace_quote_and_target_are_not_a_source_grounded_statement(self):
        self.validate(quote=' ', target_quote=' ', kind='third_party_context')

    def test_whitespace_source_quote_is_rejected_even_with_valid_target(self):
        self.validate(quote=' ', kind='third_party_context')

    def test_whitespace_target_is_rejected_even_with_valid_source_quote(self):
        self.validate(target_quote=' ')

    def test_pronoun_only_quote_is_not_a_meaningful_user_assertion(self):
        self.validate(quote='I')

    def test_pronoun_only_target_is_not_a_meaningful_evidence_association(self):
        self.validate(target_quote='I')


class RecoveryBoundsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings = patch.object(config, 'LISTWISE_SPAN_REFS', False)
        settings.start()
        self.addCleanup(settings.stop)
        network = patch('socket.socket.connect', side_effect=AssertionError('No network in offline regression'))
        network.start()
        self.addCleanup(network.stop)

    async def run_review(self, rows, *, selected=(), deadline=None, responder=None):
        from app import listwise_recovery
        self.calls, self.cards, self.prompts = [], [], []
        async def review(prompt, **kwargs):
            budget.current.get().before_call()
            self.calls.append(kwargs)
            self.prompts.append(prompt)
            self.assertEqual(kwargs['stage'], 'search.listwise.recovery')
            cards = json.loads(re.search(r'<candidates>\s*(.*?)\s*</candidates>', prompt, re.S)[1])
            self.cards.extend(cards)
            if responder:
                return await responder(cards)
            return dict(decisions=[decision(c['candidate_id'], c['sources'][0]['text']) for c in cards])
        with budget.scope(seconds=30, calls=8, tokens=128000) as limits, \
                patch.object(llm, 'complete_json', side_effect=review):
            self.limits = limits
            result, trace = await listwise_recovery._recover_batch(
                schemas.SearchRequest(user_id='u', query=QUESTION, options=OPTIONS),
                rows, list(selected), set(range(len(rows))), deadline=deadline)
        self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))
        self.assertLessEqual(len(self.calls), 1)
        return result, trace

    async def test_review_admits_at_most_ten_whole_candidates_and_reports_unreviewed(self):
        rows = [evidence(f'm{i}', f'I rode route {i} before my injury.') for i in range(13)]
        result, trace = await self.run_review(rows)
        self.assertEqual(len(self.cards), 10)
        self.assertEqual([r['id'] for r in result], [r['id'] for r in rows[:10]])
        self.assertEqual(set(trace['unreviewed_ids']), {'m10', 'm11', 'm12'})
        self.assertTrue(trace['errors'])

    async def test_oversized_source_is_skipped_whole_and_smaller_source_remains_complete(self):
        from app import listwise_recovery
        large = 'I rode across town. ' * 500
        rows = [evidence('m0', large), evidence('m1', RIDE_SOURCE)]
        originals = copy.deepcopy(rows)
        req = schemas.SearchRequest(user_id='u', query=QUESTION, options=OPTIONS)
        small = dict(candidate_id='m1', sources=listwise_recovery.sources_for(rows[1]))
        cap = len(listwise_recovery.candidate_contract(listwise_recovery.request_for(req, [small]), [small])['prompt'].encode())
        self.assertLess(cap, listwise_recovery.MAX_PROMPT_BYTES)
        with patch.object(config, 'RERANK_MAX_PROMPT_BYTES', cap):
            result, trace = await self.run_review(rows)
        self.assertEqual([r['id'] for r in result], ['m1'])
        self.assertEqual(self.cards[0]['sources'][0]['text'], RIDE_SOURCE)
        self.assertNotIn(large[:200], self.prompts[0])
        self.assertEqual(rows, originals)
        self.assertIn('m0', trace['unreviewed_ids'])
        self.assertIn(dict(candidate_id='m0', reason='prompt_budget'), trace['omitted'])

    async def test_already_represented_deletion_spends_no_model_call(self):
        accepted = evidence('m0', RIDE_SOURCE)
        duplicate = copy.deepcopy(accepted)
        duplicate['id'] = 'm1'
        result, trace = await self.run_review([duplicate], selected=[accepted])
        self.assertEqual(result, [accepted])
        self.assertEqual(self.calls, [])
        self.assertEqual(trace['represented'], [dict(candidate_id='m1', represented_by=['m0'])])

    async def test_expired_deadline_skips_model_and_keeps_previously_selected(self):
        accepted = evidence('m1', 'I enjoy swimming.')
        result, trace = await self.run_review([evidence('m0', RIDE_SOURCE)],
            selected=[accepted], deadline=time.monotonic() - 1)
        self.assertEqual(result, [accepted])
        self.assertEqual(self.calls, [])
        self.assertEqual(trace['status'], 'budget_skipped')
        self.assertTrue(trace['errors'])

    async def test_live_short_deadline_cancels_hanging_review_and_releases_every_reservation(self):
        finished = []
        async def hung(cards):
            try:
                await asyncio.sleep(10)
            finally:
                finished.append(True)
        result, trace = await self.run_review([evidence('m0', RIDE_SOURCE)],
            deadline=time.monotonic() + .03, responder=hung)
        self.assertEqual(result, [])
        self.assertEqual(finished, [True])
        self.assertLessEqual(self.calls[0]['timeout'], .03)
        self.assertEqual(trace['status'], 'error')
        self.assertTrue(trace['errors'])

    async def test_transport_retry_cannot_spend_a_second_review_call(self):
        async def retry(cards):
            budget.current.get().before_call()
            raise AssertionError('A retry must be locally rejected before dispatch')
        result, trace = await self.run_review([evidence('m0', RIDE_SOURCE)], responder=retry)
        self.assertEqual(result, [])
        self.assertEqual(self.limits.calls, 1)
        self.assertEqual(trace['status'], 'error')
        self.assertEqual(trace['error_type'], 'BudgetExceeded')

    async def test_inconsistent_original_span_cannot_produce_source_evidence(self):
        row = evidence('m0', RIDE_SOURCE)
        row['_packet_item']['sources'][0]['content_span'] = dict(start=2, end=9, original_length=100)
        result, trace = await self.run_review([row])
        self.assertEqual(result, [])
        self.assertEqual(self.calls, [])
        self.assertIn('m0', trace['unreviewed_ids'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
