"""Boundary tests for byte-overflow review and immutable primary decisions."""
import copy
import re
import time
import unittest
from unittest.mock import patch

from selftest_listwise_recovery import evidence, decision, QUESTION, OPTIONS
from app import budget, config, listwise_overflow as overflow, llm, schemas


class OverflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.req = schemas.SearchRequest(user_id='u', query=QUESTION, options=OPTIONS)
        p = patch.multiple(config, LISTWISE_SPAN_REFS=False, LISTWISE_OVERFLOW_EXTEND=False,
                           LISTWISE_OVERFLOW_BATCHED=False, RERANK_MAX_PROMPT_BYTES=24000)
        p.start()
        self.addCleanup(p.stop)
        p = patch('socket.socket.connect', side_effect=AssertionError('Offline only'))
        p.start()
        self.addCleanup(p.stop)
        self.rows = [evidence(f'm{i}', f'I enjoyed cycling in city number {i}.') for i in range(6)]
        self.omissions = [dict(id=c['id'], stage='listwise', reason='prompt_budget') for c in self.rows[1:]]

    async def run_review(self, payload, *, calls=2, tokens=64000, deadline=None, selected=None, submitted=None):
        stages = []
        async def complete(_prompt, **kwargs):
            budget.current.get().before_call()
            stages.append(kwargs['stage'])
            if isinstance(payload, Exception):
                raise payload
            return copy.deepcopy(payload)
        before = copy.deepcopy(self.rows)
        with budget.scope(seconds=30, calls=calls, tokens=tokens) as limits, patch.object(llm, 'complete_json', side_effect=complete):
            result, trace = await overflow.review(self.req, self.rows,
                self.rows[:1] if submitted is None else submitted,
                self.rows[:1] if selected is None else selected, self.omissions, deadline=deadline)
        self.assertEqual(before, self.rows)
        self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))
        self.assertLessEqual(len(stages), 1)
        if stages:
            self.assertEqual(stages, ['search.listwise.overflow'])
        return result, trace, stages

    def payload(self, useful=(), invalid=()):
        decisions = []
        for c in self.rows[1:5]:
            mid = c['id']
            if mid in useful or mid in invalid:
                d = decision(mid, c['content'], kind='assistant_context' if mid in invalid else 'user_context')
            else:
                d = dict(candidate_id=mid, useful=False, target_id='', target_quote='', evidence_kind='none', citations=[])
            decisions.append(d)
        return dict(decisions=decisions)

    async def test_only_valid_positive_appended_and_order_preserved(self):
        result, trace, _ = await self.run_review(self.payload(useful=['m2'], invalid=['m1']))
        self.assertEqual([c['id'] for c in result], ['m0','m2'])
        self.assertEqual(trace['appended_ids'], ['m2'])
        self.assertIn('m1', trace['review']['unreviewed_ids'])

    async def test_failure_does_not_accept_unreviewed_overflow(self):
        result, trace, _ = await self.run_review(RuntimeError('provider failure'))
        self.assertEqual(result, self.rows[:1])
        self.assertEqual(trace['appended_ids'], [])

    async def test_call_token_deadline_bounds_prevent_dispatch(self):
        for kwargs in (dict(calls=0), dict(tokens=1), dict(deadline=time.monotonic()-1)):
            with self.subTest(kwargs=kwargs):
                result, trace, stages = await self.run_review({}, **kwargs)
                self.assertEqual(result, self.rows[:1])
                self.assertEqual(stages, [])

    def test_only_byte_omissions_and_no_primary_rejections(self):
        self.omissions[1]['reason']='token_budget'
        self.omissions[2]['reason']='listwise_limit'
        pending, _, _ = overflow.propose(self.req, self.rows, self.rows[:2], self.rows[:1], self.omissions)
        self.assertEqual([c['id'] for c in pending], ['m4','m5'])

    async def test_partial_positive_dependency_group_not_appended(self):
        self.rows[1]['_relation_path_ids']=['m1','m2']
        result, trace, _ = await self.run_review(self.payload(useful=['m1']))
        self.assertEqual(result, self.rows[:1])
        self.assertEqual(trace['dependency_blocked_ids'], ['m1'])

    def test_dependency_deleted_by_primary_blocks_admission(self):
        self.rows[2]['_relation_path_ids']=['m2','m1']
        pending, _, decisions = overflow.propose(self.req, self.rows, self.rows[:2], self.rows[:1], self.omissions)
        self.assertNotIn('m2', [c['id'] for c in pending])
        self.assertIn(dict(id='m2', group_ids=['m1','m2'], reason='unavailable_dependency'), decisions)

    def test_prompt_and_count_limits_keep_complete_units(self):
        pending, _, _ = overflow.propose(self.req, self.rows, self.rows[:1], self.rows[:1], self.omissions)
        self.assertEqual(len(pending), 4)
        with patch.object(config, 'RERANK_MAX_PROMPT_BYTES', 100):
            pending, _, _ = overflow.propose(self.req, self.rows, self.rows[:1], self.rows[:1], self.omissions)
        self.assertEqual(pending, [])

    def test_hidden_sources_do_not_qualify(self):
        self.rows[1]['_rank_text']='source text not visible'
        pending, _, _ = overflow.propose(self.req, self.rows, self.rows[:1], self.rows[:1], self.omissions)
        self.assertNotIn('m1',[c['id'] for c in pending])

    def large_rows(self):
        self.rows = [evidence(f'm{i}', f'I enjoyed cycling {i}. ' + ('detail ' * 450)) for i in range(8)]
        self.omissions = [dict(id=c['id'], stage='listwise', reason='prompt_budget') for c in self.rows[1:]]

    async def test_extension_preserves_baseline_and_dispatches_above_old_byte_cap(self):
        self.large_rows()
        baseline, _, _ = overflow.propose(self.req, self.rows, self.rows[:1], self.rows[:1], self.omissions)
        payload = dict(decisions=[dict(candidate_id=c['id'], useful=False, target_id='',
            target_quote='', evidence_kind='none', citations=[]) for c in self.rows[1:]])
        with patch.object(config, 'LISTWISE_OVERFLOW_EXTEND', True):
            _, trace, stages = await self.run_review(payload)
        self.assertEqual(trace['baseline_ids'], [c['id'] for c in baseline])
        self.assertEqual(trace['proposed_ids'][:len(baseline)], trace['baseline_ids'])
        self.assertGreater(len(trace['extra_ids']), 0)
        self.assertLessEqual(len(trace['extra_ids']), 2)
        self.assertEqual(trace['review']['candidate_ids'], trace['proposed_ids'])
        self.assertGreater(trace['review']['input_bytes'], 12000)
        self.assertLessEqual(trace['review']['input_bytes'], 16000)
        self.assertEqual(len(stages), 1)

    def test_extension_token_limit_keeps_baseline(self):
        from app import listwise_recovery
        baseline, _, _ = overflow.propose(self.req, self.rows, self.rows[:1], self.rows[:1], self.omissions)
        cards = [dict(candidate_id=c['id'], sources=listwise_recovery.sources_for(c)) for c in baseline]
        bound = listwise_recovery.candidate_contract(listwise_recovery.request_for(self.req, cards), cards)['reservation']
        with budget.scope(seconds=30, calls=1, tokens=bound), patch.object(config, 'LISTWISE_OVERFLOW_EXTEND', True):
            pending, _, decisions, metadata = overflow.proposal(self.req, self.rows, self.rows[:1], self.rows[:1], self.omissions)
        self.assertEqual(pending, baseline)
        self.assertEqual(metadata['extra_ids'], [])
        self.assertIn('token_budget', [d['reason'] for d in decisions if d.get('phase') == 'extension'])

    def test_extension_dependency_group_cannot_exceed_two_extra_units(self):
        self.rows.extend(evidence(f'm{i}', f'I enjoyed cycling in city number {i}.') for i in range(6, 9))
        self.omissions = [dict(id=c['id'], stage='listwise', reason='prompt_budget') for c in self.rows[1:]]
        self.rows[5]['_relation_path_ids'] = ['m5', 'm6', 'm7']
        self.rows[6]['_relation_path_ids'] = ['m5', 'm6', 'm7']
        self.rows[7]['_relation_path_ids'] = ['m5', 'm6', 'm7']
        with patch.object(config, 'LISTWISE_OVERFLOW_EXTEND', True):
            pending, _, _, metadata = overflow.proposal(self.req, self.rows, self.rows[:1], self.rows[:1], self.omissions)
        self.assertEqual(metadata['baseline_ids'], ['m1','m2','m3','m4'])
        self.assertEqual(metadata['extra_ids'], ['m8'])
        self.assertEqual([c['id'] for c in pending], ['m1','m2','m3','m4','m8'])

    def test_extension_global_prompt_limit_and_primary_rejections(self):
        self.large_rows()
        with patch.multiple(config, LISTWISE_OVERFLOW_EXTEND=True, RERANK_MAX_PROMPT_BYTES=12000):
            pending, _, _, metadata = overflow.proposal(self.req, self.rows, self.rows[:2], self.rows[:1], self.omissions)
        self.assertNotIn('m1', [c['id'] for c in pending])
        self.assertEqual(metadata['extra_ids'], [])
        self.assertEqual(metadata['prompt_limit_bytes'], 12000)

    def test_overlapping_paths_and_primary_groups_pack_as_disjoint_components(self):
        from app import evidence_packet
        groups = overflow.merge_groups([['m0','m1'], ['m3','m4'], ['m1','m2'], ['m2','m3']], self.rows)
        self.assertEqual(groups, [['m0','m1','m2','m3','m4']])
        packed, _, _ = evidence_packet.pack_ranked([c['_packet_item'] for c in self.rows],
            top_k=10, token_budget=64000, groups=groups)
        self.assertEqual(len(packed),6)

    async def batched_review(self, *, count=15, calls=12, fail_batch=None, invalid=(),
                             negative=(), dependent=False, exhaust_tokens=False, exhaust_deadline=False):
        self.rows = [evidence(f'm{i}', f'I enjoyed cycling in city number {i}.') for i in range(count)]
        if dependent:
            self.rows[5]['_relation_path_ids'] = ['m1', 'm5']
        omissions = [dict(id=c['id'], stage='listwise', reason='prompt_budget') for c in self.rows[1:]]
        dispatched = []
        async def complete(prompt, **kwargs):
            active = budget.current.get()
            active.before_call()
            mids = re.findall(r'"candidate_id": "(m\d+)"', prompt)
            self.assertEqual(len(mids), len(set(mids)))
            dispatched.append(mids)
            if len(dispatched) == fail_batch:
                raise RuntimeError('provider failure')
            if exhaust_tokens:
                active.tokens = active.max_tokens
            if exhaust_deadline:
                active.deadline = time.monotonic()
            return dict(decisions=[
                dict(candidate_id=mid, useful=False, target_id='', target_quote='', evidence_kind='none', citations=[])
                if mid in negative else decision(mid, self.rows[int(mid[1:])]['content'],
                    kind='assistant_context' if mid in invalid else 'user_context') for mid in mids])
        before = copy.deepcopy(self.rows)
        with patch.object(config, 'LISTWISE_OVERFLOW_BATCHED', True), \
                patch.object(llm, 'complete_json', side_effect=complete), \
                budget.scope(seconds=30, calls=calls, tokens=128000) as limits:
            result, trace = await overflow.review(self.req, self.rows, self.rows[:1], self.rows[:1], omissions)
        self.assertEqual(before, self.rows)
        self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))
        return result, trace, dispatched

    async def test_batches_add_unseen_evidence_without_retrying_invalid_or_negative(self):
        result, trace, batches = await self.batched_review(invalid=['m1'], negative=['m2'])
        self.assertEqual(batches, [[f'm{i}' for i in range(a,a+4)] for a in (1,5,9)])
        self.assertEqual([c['id'] for c in result], ['m0']+[f'm{i}' for i in range(3,13)])
        self.assertEqual(trace['baseline_ids'], ['m1','m2','m3','m4'])
        self.assertEqual(trace['review']['unreviewed_ids'], ['m1'])
        self.assertEqual(len(trace['review']['candidate_ids']), 12)
        self.assertEqual(trace['status'], 'partial')
        self.assertNotIn('input_bytes', trace['review'])

    async def test_later_batch_failure_preserves_earlier_additions(self):
        result, trace, batches = await self.batched_review(fail_batch=2)
        self.assertEqual(len(batches), 2)
        self.assertEqual([c['id'] for c in result], [f'm{i}' for i in range(5)])
        self.assertEqual(trace['batches'][1]['status'], 'error')

    async def test_batches_share_call_and_token_budget(self):
        for kw in (dict(calls=1), dict(exhaust_tokens=True), dict(exhaust_deadline=True)):
            with self.subTest(kw=kw):
                result, _, batches = await self.batched_review(**kw)
                self.assertEqual(len(batches), 1)
                self.assertEqual([c['id'] for c in result], [f'm{i}' for i in range(5)])

    async def test_later_dependencies_cannot_restore_rejected_earlier_candidate(self):
        result, _, batches = await self.batched_review(negative=['m1'], dependent=True)
        self.assertNotIn('m5', [mid for b in batches for mid in b])
        self.assertNotIn('m5', [c['id'] for c in result])

    async def test_one_batch_compatible_when_no_other_candidates(self):
        result, trace, batches = await self.batched_review(count=4)
        self.assertEqual(batches, [['m1','m2','m3']])
        self.assertEqual([c['id'] for c in result], ['m0','m1','m2','m3'])
        self.assertEqual(trace['status'], 'recovered')


if __name__ == '__main__':
    unittest.main()
