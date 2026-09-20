"""Recall-layer bounds: graph seeds, batched source hydration, rule lane cap.

Run with ``python memory_system/scripts/selftest_recall_bounds.py``.
All provider behavior is deterministic; no credentials or network are used.
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

os.environ['AML_FAKE'] = '1'
os.environ['AML_MEMORY_DEBUG_LOG'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, graph, schemas, search_pipeline as search, store


def _triples():
    return [dict(subject='Alpha', relation='knows', object='Beta', amu_id='seed'),
            dict(subject='Beta', relation='founded', object='Gamma', amu_id='m2'),
            dict(subject='Gamma', relation='located_in', object='Delta', amu_id='m3')]


class GraphSeedExclusion(unittest.TestCase):
    def test_seed_memory_does_not_consume_the_graph_slot(self):
        # One slot: the seed's own teleport mass used to win it.
        without = graph.ppr_recall(_triples(), ['Alpha'], top_n=1, seed_amu_ids=['seed'])
        self.assertEqual(without, ['seed'])
        recalled = graph.ppr_recall(_triples(), ['Alpha'], top_n=1, seed_amu_ids=['seed'], exclude_ids=['seed'])
        self.assertEqual(len(recalled), 1)
        self.assertNotIn('seed', recalled)

    def test_exclusion_keeps_reachable_bridges(self):
        recalled = graph.ppr_recall(_triples(), ['Alpha'], seed_amu_ids=['seed'], exclude_ids=['seed'])
        self.assertEqual(set(recalled), {'m2', 'm3'})


class StoreReads(unittest.TestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    def memory(self, content, **kwargs):
        return self.st.insert_amu(user_id='u', session_id='s', content=content,
                                  embedding=np.array([1., 0.]), **kwargs)

    def test_triples_for_user_filters_validity_in_sql(self):
        live = self.memory('Live fact')
        stale = self.memory('Stale fact', valid_to='2020-01-01T00:00:00Z')
        self.st.insert_triple('u', 'user', 'likes', 'tea', live)
        self.st.insert_triple('u', 'user', 'liked', 'coffee', stale)
        current = {t['amu_id'] for t in self.st.triples_for_user('u', include_history=False)}
        self.assertEqual(current, {live})
        historical = {t['amu_id'] for t in self.st.triples_for_user('u')}
        self.assertEqual(historical, {live, stale})

    def test_batch_sources_match_the_single_reader(self):
        first, second, lonely = self.memory('First'), self.memory('Second'), self.memory('Lonely')
        self.st.save_messages(schemas.AddRequest(request_id='r1', user_id='u', session_id='s', messages=[
            schemas.Message(role='user', content='I like tea.'),
            schemas.Message(role='assistant', content='Noted.'),
            schemas.Message(role='user', content='And coffee.')]))
        self.st.link_sources(first, 'r1', [0, 2])
        self.st.link_sources(second, 'r1', [1])
        grouped = self.st.sources_for_amus([first, second, lonely, first])
        self.assertEqual(set(grouped), {first, second, lonely})
        for mid in (first, second, lonely):
            self.assertEqual(grouped[mid], self.st.sources_for_amu(mid))
        self.assertEqual([s['message_index'] for s in grouped[first]], [0, 2])
        self.assertEqual(grouped[lonely], [])


class LexicalHydration(unittest.TestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.req = schemas.SearchRequest(user_id='u', query='OrchidLedger')

    def tearDown(self):
        self.st.conn.close()

    def memory(self, content, **kwargs):
        return self.st.insert_amu(user_id='u', session_id='s', content=content,
                                  embedding=np.array([1., 0.]), **kwargs)

    def test_lexical_recall_hydrates_hits_without_per_candidate_reads(self):
        ids = [self.memory(f'OrchidLedger detail {i}') for i in range(5)]
        self.st.save_messages(schemas.AddRequest(request_id='r1', user_id='u', session_id='s', messages=[
            schemas.Message(role='user', content='OrchidLedger was founded in Hangzhou.')]))
        for mid in ids[:3]:
            self.st.link_sources(mid, 'r1', [0])
        plan = {'_include_history': False, '_include_sensitive': False}
        specs = [{'text': 'OrchidLedger'}]
        with patch.object(self.st, 'sources_for_amu', side_effect=AssertionError('per-candidate read')):
            merged = search._lexical_recall(self.st, self.req, plan, specs)
        self.assertEqual({c['id'] for c in merged[0]}, set(ids))
        self.assertGreaterEqual(plan['_source_cache_primed'], 5)
        self.assertEqual(plan['_direct_coverage'][0]['independent_sources'], 1)
        # Primed entries carry the same shape the per-candidate path builds.
        full, sources = search._candidate_sources(self.st, self.req, plan, {'id': ids[0]})
        self.assertEqual(full[0]['id'], ids[0])
        self.assertEqual([(s['request_id'], s['message_index']) for s in sources], [('r1', 0)])

    def test_priming_skips_stores_without_the_batch_reader(self):
        class Legacy:
            pass

        plan = {}
        search._prime_source_cache(Legacy(), self.req, plan, [{'id': 'x'}])
        self.assertNotIn('_source_cache_primed', plan)


class RuleLaneBounds(unittest.TestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.req = schemas.SearchRequest(user_id='u', query='tea')

    def tearDown(self):
        self.st.conn.close()

    def rule(self, content, **kwargs):
        return self.st.insert_amu(user_id='u', session_id='s', content=content, type='rule',
                                  embedding=np.array([1., 0.]), **kwargs)

    def rows(self, ids, score=0):
        return [dict(m, _score=score) for mid in ids for m in self.st.get_amus_by_ids([mid])]

    def test_rule_lane_keeps_constraints_and_caps_the_rest(self):
        plain = [self.rule(f'Always answer briefly about topic {i}.') for i in range(5)]
        forget = self.rule('Please forget that I ever liked tea.')
        ordered = self.rows(plain) + self.rows([forget])
        with patch.multiple(config, RECALL_RULE_LIMIT=2):
            kept = search._bounded_rules(self.st, self.req, {}, ordered)
        self.assertEqual([c['id'] for c in kept], plain[:2] + [forget])

    def test_rule_lane_checks_sources_only_for_rules_that_fit(self):
        inferred = [self.rule(f'Inferred rule {i}', epistemic_status='inferred') for i in range(4)]
        self.st.save_messages(schemas.AddRequest(request_id='r1', user_id='u', session_id='s', messages=[
            schemas.Message(role='user', content='Rule one.'), schemas.Message(role='user', content='Rule two.'),
            schemas.Message(role='assistant', content='Rule three.')]))
        self.st.link_sources(inferred[0], 'r1', [0])
        self.st.link_sources(inferred[1], 'r1', [1])
        self.st.link_sources(inferred[2], 'r1', [2])
        reads = []
        original = self.st.sources_for_amu

        def counting(mid):
            reads.append(mid)
            return original(mid)

        plan = {}
        with patch.object(self.st, 'sources_for_amu', side_effect=counting), \
                patch.multiple(config, RECALL_RULE_LIMIT=2):
            kept = search._bounded_rules(self.st, self.req, plan, self.rows(inferred))
        # Two user-sourced rules fill the lane; the rest are never read.
        self.assertEqual([c['id'] for c in kept], inferred[:2])
        self.assertEqual(set(reads), {inferred[0], inferred[1]})
        reasons = {d['id']: d['reason'] for d in plan['_rule_recall']['dropped']}
        self.assertEqual(reasons[inferred[2]], 'rule_recall_limit')
        self.assertEqual(reasons[inferred[3]], 'rule_recall_limit')

    def test_inferred_rule_without_user_source_is_dropped(self):
        inferred = self.rule('Inferred rule', epistemic_status='inferred')
        self.st.save_messages(schemas.AddRequest(request_id='r1', user_id='u', session_id='s', messages=[
            schemas.Message(role='assistant', content='Assistant guess.')]))
        self.st.link_sources(inferred, 'r1', [0])
        plan = {}
        kept = search._bounded_rules(self.st, self.req, plan, self.rows([inferred]))
        self.assertEqual(kept, [])
        self.assertEqual(plan['_rule_recall']['dropped'][0]['reason'], 'inferred_without_user_source')

    def test_zero_limit_keeps_every_rule(self):
        plain = [self.rule(f'Rule {i}') for i in range(20)]
        with patch.multiple(config, RECALL_RULE_LIMIT=0):
            kept = search._bounded_rules(self.st, self.req, {}, self.rows(plain))
        self.assertEqual(len(kept), 20)


if __name__ == '__main__':
    unittest.main(verbosity=1)
