"""Coarse extension preserves source units and existing downstream budgets."""
import copy
import re
import unittest
from unittest.mock import patch

from selftest_fine_selection import candidate, source, documents_in, permutation
from app import budget, cascade_rerank as cascade, config, cross_encoder, llm, schemas


class CoarseExtensionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        p = patch.multiple(config, COARSE_EVIDENCE_EXTEND=False, CASCADE_COARSE_LIMIT=12,
            CASCADE_FINE_LIMIT=6, CASCADE_LLM_LIMIT=5, CE_BATCH_SIZE=8,
            CE_MAX_DOCUMENT_BYTES=10000, CE_MAX_REQUEST_BYTES=90000,
            CE_RETRY_BATCHES=0, RERANK_MAX_PROMPT_BYTES=24000, RERANK_REPAIR_MAX_CALLS=0,
            FINE_EVIDENCE_ADAPTIVE=False, LISTWISE_KEYED_OUTPUT=False, LISTWISE_STAGED=False,
            LISTWISE_SHARED_SOURCES=False, LISTWISE_OVERFLOW_REVIEW=False)
        p.start(); self.addCleanup(p.stop)
        p = patch('socket.socket.connect', side_effect=AssertionError('Offline'))
        p.start(); self.addCleanup(p.stop)
        self.rows = [candidate(f'r{i}', 1-i*.01, [source(f's{i}', f'Complete original detail {i}.')]) for i in range(24)]

    def test_disabled_reproduces_selection_and_enabled_preserves_whole_prefix(self):
        before = copy.deepcopy(self.rows)
        baseline, old = cascade.coarse_shortlist(self.rows, [])
        self.assertEqual([c['id'] for c in baseline], [f'r{i}' for i in range(12)])
        self.assertEqual(old['extra_ids'], [])
        with patch.object(config, 'COARSE_EVIDENCE_EXTEND', True):
            expanded, trace = cascade.coarse_shortlist(self.rows, [])
        self.assertEqual(expanded[:12], baseline)
        self.assertEqual(len(expanded),20)
        self.assertEqual(len(set(c['id'] for c in expanded)),20)
        self.assertEqual(trace['extra_ids'], [f'r{i}' for i in range(12,20)])
        self.assertEqual(self.rows, before)

    def test_small_pool_never_duplicates_original_candidates(self):
        with patch.object(config, 'COARSE_EVIDENCE_EXTEND', True):
            for size in (0,4,13):
                selected, trace = cascade.coarse_shortlist(self.rows[:size], [])
                self.assertEqual(len(selected), size)
                self.assertEqual(len(trace['extra_ids']), max(0,size-12))

    async def run_case(self, *, tokens=200000, calls=20, seconds=30):
        documents_seen=[]
        async def ce(query, documents, **kwargs):
            limits=budget.current.get()
            bound=cross_encoder.request_size(query, documents)
            limits.reserve_tokens(bound)
            try:
                limits.before_call()
                documents_seen.extend(documents)
                return [1.0 if 'Unit r15:' in d else .5 for d in documents]
            finally:
                limits.release_tokens(bound)
        async def main(prompt, **kwargs):
            budget.current.get().before_call()
            return permutation(len(documents_in(prompt)))
        plan={}
        with budget.scope(seconds=seconds,calls=calls,tokens=tokens) as limits, \
                patch.object(config, 'COARSE_EVIDENCE_EXTEND', True), \
                patch.object(cross_encoder,'rerank',side_effect=ce), \
                patch.object(llm,'complete_json',side_effect=main):
            result=await cascade.rank(schemas.SearchRequest(user_id='u',query='Personal context'),plan,copy.deepcopy(self.rows))
        self.assertEqual((limits.reserved_calls,limits.reserved_tokens),(0,0))
        self.assertLessEqual(limits.calls,calls)
        return result,plan['_cascade'],documents_seen,limits

    async def test_new_candidate_is_scored_and_can_pass_fine_and_main(self):
        _, trace, docs, _ = await self.run_case()
        self.assertEqual(len(docs),20)
        self.assertEqual(trace['cross_encoder']['scored_count'],20)
        self.assertIn('r15',trace['fine']['selected_ids'])
        self.assertIn('r15',trace['listwise']['selected_ids'])
        self.assertLessEqual(len(trace['fine']['selected_ids']),6)
        self.assertLessEqual(len(trace['listwise']['candidate_ids']),5)
        self.assertIn(self.rows[15]['_rank_text'],docs)

    async def test_oversize_extra_is_rejected_without_truncation(self):
        self.rows[15]['_rank_text'] += 'x'*11000
        _, trace, docs, _ = await self.run_case()
        self.assertFalse(any('Unit r15:' in d for d in docs))
        self.assertIn(dict(id='r15',error='input_budget'),trace['cross_encoder']['errors'])

    async def test_shared_call_and_token_limits_apply_to_extras(self):
        for kwargs in (dict(calls=1),dict(tokens=1)):
            _, trace, docs, limits = await self.run_case(**kwargs)
            self.assertEqual(docs,[])
            self.assertEqual(trace['cross_encoder']['scored_count'],0)
            if kwargs.get('tokens')==1:
                self.assertEqual(limits.calls,0)

    async def test_expired_shared_deadline_prevents_dispatch(self):
        with self.assertRaises(TimeoutError):
            await self.run_case(seconds=-1)


if __name__ == '__main__':
    unittest.main()
