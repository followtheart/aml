"""Deletion authorization, bounded review batches and compact repair regressions."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank, config, listwise_recovery as recovery, llm


def rows(count):
    return [dict(id=f'm{i}', content=f'I enjoy activity {i}.', _rank_text=f'I enjoy activity {i}.',
        _packet_item=dict(sources=[dict(role='user', content=f'I enjoy activity {i}.',
                                      request_id='test', message_index=i)])) for i in range(count)]


class Retention(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, count, *, max_batches=3, max_cards=2, calls=8, bad=False):
        candidates = rows(count); before = copy.deepcopy(candidates)
        seen = []
        async def provider(prompt, **kw):
            budget.current.get().before_call()
            keys = kw['schema']['required']; seen.append(keys)
            return {key: dict(useful=False, target_id='', evidence_kind='none',
                              citation_parts=['not_a_source'] if bad else []) for key in keys}
        with patch.multiple(config, LISTWISE_SPAN_REFS=True, LISTWISE_REVIEW_MAX_BATCHES=max_batches), \
             patch.object(recovery, 'MAX_CANDIDATES', max_cards), \
             patch.object(llm, 'complete_json', provider), \
             budget.scope(seconds=30, calls=calls, tokens=100000) as limits:
            selected, trace = await recovery.recover(SimpleNamespace(query='Which hobby suits me?', options=[]),
                                                     candidates, [], set(range(count)))
            self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))
        self.assertEqual(candidates, before)
        self.assertLessEqual(len(seen), max_batches)
        return selected, trace, seen

    async def test_batches_review_every_card_without_duplicates(self):
        selected, trace, seen = await self.run_case(5)
        self.assertEqual(selected, [])
        self.assertEqual([len(k) for k in seen], [2, 2, 1])
        self.assertEqual(trace['candidate_ids'], ['m0','m1','m2','m3','m4'])
        self.assertEqual(trace['confirmed_irrelevant_ids'], trace['candidate_ids'])
        self.assertEqual(trace['unreviewed_ids'], [])
        self.assertEqual(trace['errors'], [])

    async def test_batch_cap_retains_unreviewed_not_confirmed_negatives(self):
        selected, trace, seen = await self.run_case(5, max_batches=1)
        self.assertEqual([c['id'] for c in selected], ['m2','m3','m4'])
        self.assertEqual(trace['provisional_ids'], ['m2','m3','m4'])
        self.assertEqual(trace['status'], 'partial')
        self.assertEqual(trace['confirmed_irrelevant_ids'], ['m0','m1'])

    async def test_call_budget_cannot_silently_delete_tail(self):
        selected, trace, seen = await self.run_case(5, calls=1)
        self.assertEqual(len(seen), 1)
        self.assertEqual([c['id'] for c in selected], ['m2','m3','m4'])
        self.assertTrue(trace['errors'])

    async def test_invalid_negative_citations_do_not_authorize_deletion(self):
        selected, trace, seen = await self.run_case(5, bad=True)
        self.assertEqual(len(selected), 5)
        self.assertEqual(trace['confirmed_irrelevant_ids'], [])
        self.assertEqual(trace['restored_ids'], [])
        self.assertEqual(len(trace['unreviewed_ids']), 5)
        # Earlier capacity deferrals are not final causes after later attempts.
        self.assertEqual(trace['omitted'], [])

    async def test_no_source_retained_without_factual_support(self):
        candidate = dict(id='m0', content='A summary without sources')
        with patch.object(llm, 'complete_json', side_effect=AssertionError('No model call')):
            selected, trace = await recovery.recover(SimpleNamespace(query='hobby', options=[]),
                                                     [candidate], [], {0})
        self.assertEqual(selected, [candidate])
        self.assertEqual(trace['restored_ids'], [])
        self.assertEqual(trace['provisional_ids'], ['m0'])

    def test_group_repair_omits_priority_records_and_preserves_exclusions(self):
        candidates = rows(24)
        locked = {'candidates': {f'c{i}': dict(priority=99, relevance='irrelevant' if i == 3 else 'useful')
                                 for i in range(24)}}
        req = SimpleNamespace(query='Which hobby?', options=[])
        request = cascade_rerank.listwise_request(req, {}, candidates, keyed=True,
                                                 repair=True, repair_field='groups', locked=locked)
        self.assertIn('"excluded_from_groups":["c3"]', request['prompt'])
        self.assertNotIn('"priority": 99', request['prompt'])
        self.assertEqual(list(request['schema']['properties']), ['groups'])

if __name__ == '__main__':
    unittest.main()
