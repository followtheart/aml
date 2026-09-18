"""Behavioral regressions for graph fusion and coarse -> CE -> listwise ranking.

All external scoring is controlled; these fixtures do not claim model quality.
"""
import copy
import importlib
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
os.environ['AML_MEMORY_DEBUG_LOG'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, llm, schemas


def documents_in(prompt):
    """Read the public numbered-candidate representation, not batch internals."""
    rows = re.findall(r'^(\d+): (.*)$', prompt, re.M)
    if not rows:
        raise AssertionError('A nonempty listwise request must expose numbered candidates')
    indices = [int(index) for index, _ in rows]
    if indices != list(range(len(rows))):
        raise AssertionError('Candidate indices must be contiguous and unambiguous')
    return [text for _, text in rows]


def evidence_number(text):
    match = re.search(r'Evidence (\d+)\b', text)
    if not match:
        raise AssertionError(f'Missing fixture evidence marker in {text!r}')
    return int(match.group(1))


def permutation(count, reverse=False, irrelevant=(), groups=()):
    order = list(range(count))
    return dict(ranking=list(reversed(order)) if reverse else order,
                irrelevant=list(irrelevant), groups=[list(group) for group in groups])


def selected(rows):
    return [row for row in rows if row.get('_cascade_selected')]


class GraphFusionTests(unittest.TestCase):
    def setUp(self):
        self.fusion = importlib.import_module('app.graph_fusion')

    def plan(self, routes):
        return {'entities': ['Alpha'], '_used_queries': ['Alpha'],
                '_query_specs': [{'text': 'Alpha', 'coverage_ids': ['question']}],
                '_coverage_requirements': [{'id': 'question', 'text': 'Alpha'}],
                '_routes': [dict(channel='vector', family='vector', weight=1., candidates=route)
                            for route in routes]}

    def test_repeating_routes_cannot_amplify_the_same_observation(self):
        route = [dict(id='a', content='Alpha fact'), dict(id='b', content='Beta fact')]
        once = self.fusion.fuse([copy.deepcopy(route)], self.plan([route]))
        many_routes = [copy.deepcopy(route) for _ in range(20)]
        repeated = self.fusion.fuse(many_routes, self.plan(many_routes))
        self.assertEqual([row['id'] for row in once], [row['id'] for row in repeated])
        self.assertEqual(len(repeated), 2)
        for first, second in zip(once, repeated):
            self.assertAlmostEqual(first['_fused'], second['_fused'])

    def test_directed_bridge_receives_support_only_along_the_forward_path(self):
        routes = [[dict(id='seed', content='Alpha knows Beta')],
                  [dict(id='bridge', content='Beta founded Gamma'),
                   dict(id='noise', content='Unrelated discussion')]]
        forward = [dict(subject='Alpha', relation='knows', object='Beta', amu_id='seed'),
                   dict(subject='Beta', relation='founded', object='Gamma', amu_id='bridge')]
        reverse = [dict(subject='Beta', relation='knows', object='Alpha', amu_id='seed'),
                   dict(subject='Gamma', relation='founded', object='Beta', amu_id='bridge')]
        def scores(triples):
            result = self.fusion.fuse(copy.deepcopy(routes), self.plan(routes), triples=triples)
            return {row['id']: row['_fused'] for row in result}
        self.assertGreater(scores(forward)['bridge'], scores(reverse)['bridge'])

    def test_duplicate_candidate_retains_all_coverage_and_bridge_metadata(self):
        routes = [[dict(id='same', content='Alpha observation', _coverage_ids=['a'], _bridge_ids=['s1'])],
                  [dict(id='same', content='Alpha observation', _coverage_ids=['b'], _bridge_ids=['s2'])]]
        result = self.fusion.fuse(routes, self.plan(routes))
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0]['_coverage_ids']), {'a', 'b'})
        self.assertEqual(set(result[0]['_bridge_ids']), {'s1', 's2'})
        self.assertEqual(result[0]['content'], 'Alpha observation')


class CascadeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cascade = importlib.import_module('app.cascade_rerank')
        self.ce = importlib.import_module('app.cross_encoder')
        self.settings = patch.multiple(config, CASCADE_COARSE_LIMIT=50,
            CASCADE_FINE_LIMIT=12, CASCADE_LLM_LIMIT=10, create=True)
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def req(self):
        return schemas.SearchRequest(user_id='u', query='Evidence', top_k=50)

    def rows(self, count=70):
        return [dict(id=f'm{i}', content=f'Evidence {i}', _rank_text=f'Evidence {i}',
                     _fused=1. - i / (count + 1), _coverage_ids=[]) for i in range(count)]

    async def ce_scores(self, query, documents, **kwargs):
        return [.2 + evidence_number(text) / 100 for text in documents]

    async def listwise_identity(self, prompt, *args, **kwargs):
        return permutation(len(documents_in(prompt)))

    async def test_stage_pools_contract_before_the_expensive_listwise_call(self):
        ce_inputs, llm_inputs = [], []
        async def ce(query, documents, **kwargs):
            ce_inputs.append(list(documents))
            return await self.ce_scores(query, documents, **kwargs)
        async def rank(prompt, *args, **kwargs):
            self.assertEqual(kwargs['stage'], 'search.listwise')
            rows = documents_in(prompt)
            llm_inputs.append(rows)
            return permutation(len(rows))
        with patch.object(self.ce, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=rank):
            result = await self.cascade.rank(self.req(), {}, self.rows())
        self.assertTrue(ce_inputs)
        ce_documents = [text for batch in ce_inputs for text in batch]
        self.assertEqual(len(ce_documents), 50)
        self.assertEqual(len({evidence_number(t) for t in ce_documents}), 50)
        self.assertTrue(all(len(batch) <= config.CE_BATCH_SIZE for batch in ce_inputs))
        self.assertEqual(len(llm_inputs), 1)
        self.assertEqual(len(llm_inputs[0]), 10)
        self.assertEqual(len(selected(result)), 10)
        self.assertTrue({evidence_number(t) for t in llm_inputs[0]} <=
                        {evidence_number(t) for t in ce_documents})

    async def test_cross_encoder_can_promote_a_lower_coarse_candidate(self):
        seen = []
        async def rank(prompt, *args, **kwargs):
            seen.extend(documents_in(prompt))
            return permutation(len(seen))
        with patch.object(self.ce, 'rerank', side_effect=self.ce_scores), \
                patch.object(llm, 'complete_json', side_effect=rank):
            result = await self.cascade.rank(self.req(), {}, self.rows())
        self.assertIn(49, {evidence_number(t) for t in seen})
        self.assertNotIn(0, {evidence_number(t) for t in seen})
        self.assertEqual(selected(result)[0]['id'], 'm49')

    async def test_listwise_permutation_changes_order_without_fabricating_scores(self):
        submitted = []
        async def rank(prompt, *args, **kwargs):
            submitted.extend(evidence_number(t) for t in documents_in(prompt))
            return permutation(len(submitted), reverse=True)
        with patch.object(self.ce, 'rerank', side_effect=self.ce_scores), \
                patch.object(llm, 'complete_json', side_effect=rank):
            result = selected(await self.cascade.rank(self.req(), {}, self.rows(8)))
        self.assertEqual([row['id'] for row in result], [f'm{i}' for i in reversed(submitted)])
        for row in result:
            self.assertAlmostEqual(row['_final'], .2 + evidence_number(row['content']) / 100)
            self.assertIn('_listwise_rank', row)

    async def test_bad_permutations_fall_back_after_at_most_one_repair(self):
        for bad in ([0, 0, 2, 3], [0, 1, 2], [0, 1, 2, 999], [0, 1, 2, -1], [0, 1, 2, True]):
            with self.subTest(ranking=bad):
                calls = []
                async def rank(prompt, *args, **kwargs):
                    calls.append(kwargs['stage'])
                    return dict(ranking=bad, irrelevant=[], groups=[])
                with patch.object(self.ce, 'rerank', side_effect=self.ce_scores), \
                        patch.object(llm, 'complete_json', side_effect=rank):
                    result = selected(await self.cascade.rank(self.req(), {}, self.rows(4)))
                self.assertEqual(calls, ['search.listwise', 'search.listwise.repair'])
                self.assertEqual([row['id'] for row in result], ['m3', 'm2', 'm1', 'm0'])
                self.assertTrue(all(row.get('_final') is not None for row in result))

    async def test_explicit_irrelevant_item_is_not_refilled_from_the_tail(self):
        rejected = []
        async def rank(prompt, *args, **kwargs):
            texts = documents_in(prompt)
            rejected.append(f'm{evidence_number(texts[0])}')
            return permutation(len(texts), irrelevant=[0])
        with patch.object(self.ce, 'rerank', side_effect=self.ce_scores), \
                patch.object(llm, 'complete_json', side_effect=rank):
            result = selected(await self.cascade.rank(self.req(), {}, self.rows(20)))
        self.assertNotIn(rejected[0], {row['id'] for row in result})
        self.assertEqual(len(result), 9)

    async def test_evidence_groups_map_input_indices_to_original_ids(self):
        expected, plan = [], {}
        async def rank(prompt, *args, **kwargs):
            texts = documents_in(prompt)
            expected.extend(f'm{evidence_number(t)}' for t in texts[:2])
            return permutation(len(texts), reverse=True, groups=[[0, 1]])
        with patch.object(self.ce, 'rerank', side_effect=self.ce_scores), \
                patch.object(llm, 'complete_json', side_effect=rank):
            await self.cascade.rank(self.req(), plan, self.rows(4))
        self.assertIn(set(expected), [set(group) for group in plan['_evidence_groups']])

    async def test_cross_encoder_failure_retains_unknown_coarse_evidence(self):
        with patch.object(self.ce, 'rerank', AsyncMock(side_effect=ConnectionError('offline'))), \
                patch.object(llm, 'complete_json', side_effect=self.listwise_identity):
            result = selected(await self.cascade.rank(self.req(), {}, self.rows(4)))
        self.assertTrue(result)
        self.assertTrue(all(row.get('_final') is None for row in result))
        self.assertTrue(all(row.get('_score_kind') for row in result))
        self.assertEqual(result[0]['id'], 'm0')

    async def test_listwise_timeout_retains_cross_encoder_order_and_scores(self):
        with patch.object(self.ce, 'rerank', side_effect=self.ce_scores), \
                patch.object(llm, 'complete_json', AsyncMock(side_effect=TimeoutError('deadline'))) as call:
            result = selected(await self.cascade.rank(self.req(), {}, self.rows(4)))
        self.assertLessEqual(call.call_count, 2)
        self.assertEqual([row['id'] for row in result], ['m3', 'm2', 'm1', 'm0'])
        self.assertTrue(all(row.get('_final') is not None for row in result))

    async def test_rule_and_rare_option_survive_all_stage_caps(self):
        rows = self.rows()
        rows[-1].update(type='rule', _user_rule=True)
        rows[-2]['_coverage_ids'] = ['option:1']
        # Model the preparation stage's source-witnessed reservation, not a
        # bare lexical topic hit (which must no longer force a low-score item).
        rows[-2]['_supported_coverage_ids'] = ['option:1']
        plan = {'_coverage_requirements': [{'id': 'option:1', 'text': 'Rare option'}]}
        async def ce(query, documents, **kwargs):
            return [.001 if evidence_number(t) == 68 else .8 for t in documents]
        with patch.object(self.ce, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=self.listwise_identity):
            result = selected(await self.cascade.rank(self.req(), plan, rows))
        self.assertTrue({'m68', 'm69'} <= {row['id'] for row in result})

    async def test_empty_pool_does_not_call_models_or_divide_by_zero(self):
        with patch.object(self.ce, 'rerank', AsyncMock()) as ce, \
                patch.object(llm, 'complete_json', AsyncMock()) as rank:
            result = await self.cascade.rank(self.req(), {}, [])
        self.assertEqual(result, [])
        ce.assert_not_awaited()
        rank.assert_not_awaited()


if __name__ == '__main__':
    unittest.main(verbosity=2)
