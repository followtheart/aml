"""Offline regressions for the ULM lifecycle: segmentation, MemCell/MemScene
consolidation, novelty gate, heat/forgetting, single-pass retrieval, foresight filter."""
import json
import math
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

os.environ['AML_FAKE'] = '1'
os.environ['AML_MEMORY_DEBUG_LOG'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline as add, config, llm, scenes, schemas, search_pipeline as search, segment, store
from app.embeddings import embed


def msgs(*texts, ts=1684972800000):
    return [schemas.Message(role='user', content=t, timestamp=ts + i) for i, t in enumerate(texts)]


def request(rid, *texts, session='s'):
    return schemas.AddRequest(request_id=rid, user_id='u', session_id=session, messages=msgs(*texts))


async def score_all(prompt, *args, **kwargs):
    if 'sufficiency verifier' in prompt:
        return {'sufficient': True, 'confidence': .9, 'missing': '', 'follow_up_queries': []}
    ids = re.findall(r'^(amu_[^:]+):', prompt, re.M)
    return {'scores': [{'id': aid, 'relevance': 0.9, 'keep': True} for aid in ids]}


class SegmentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_topic_shift_creates_boundary_and_max_size_holds(self):
        texts = ['Alice plays tennis on Monday', 'Alice tennis coach Monday practice',
                 'quantum chromodynamics lattice gauge', 'lattice gauge quantum field']
        vecs = await embed(texts)
        groups = segment.segment_indices(vecs, max_size=6, min_size=1, drop=0.35)
        self.assertEqual(groups, [[0, 1], [2, 3]])
        groups = segment.segment_indices(vecs, max_size=1, min_size=1)
        self.assertEqual(groups, [[0], [1], [2], [3]])
        self.assertEqual(segment.segment_indices(vecs[:0], max_size=3), [])


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'search.jsonl'
        self.setting = patch.object(config, 'SEARCH_DEBUG_LOG', str(self.path))
        self.setting.start()

    def tearDown(self):
        self.st.conn.close()
        self.setting.stop()
        self.tmp.cleanup()

    def trace(self):
        return json.loads(self.path.read_text().splitlines()[-1])

    async def run_search(self, req, plan=None, side_effect=score_all):
        with patch.object(search, '_understand', AsyncMock(return_value=plan or {
                'intent': 'fact', 'entities': ['Alice']})), patch.object(
                search.llm, 'complete_json', side_effect=side_effect):
            return await search.run_search(self.st, req)

    async def test_memcell_has_episode_and_facts_in_one_scene(self):
        await add.run_add(self.st, request('r1', 'Alice plays tennis.', 'Alice tennis coach is Bob.'))
        amus = self.st.get_amus('u')
        types = sorted(a['type'] for a in amus)
        self.assertEqual(types.count('episode'), 1)
        self.assertGreaterEqual(types.count('fact'), 2)
        self.assertEqual(len({a['scene_id'] for a in amus}), 1)
        self.assertEqual(len({a['cell_id'] for a in amus}), 1)
        scene = self.st.list_scenes('u')[0]
        self.assertEqual(scene['cell_count'], 1)
        self.assertIn('tennis', ' '.join(scene['keywords']).lower() + scene['summary'].lower())

    async def test_related_cell_joins_scene_and_unrelated_opens_new_scene(self):
        await add.run_add(self.st, request('r1', 'Alice plays tennis with Bob.'))
        await add.run_add(self.st, request('r2', 'Alice plays tennis with Bob on Monday.', session='s2'))
        self.assertEqual(len(self.st.list_scenes('u')), 1)
        await add.run_add(self.st, request('r3', 'quantum chromodynamics lattice gauge theory.'))
        self.assertEqual(len(self.st.list_scenes('u')), 2)

    async def test_novelty_gate_skips_governance_llm_for_duplicates(self):
        await add.run_add(self.st, request('r1', 'Alice lives in Paris.'))
        before = len([a for a in self.st.get_amus('u') if a['type'] != 'episode'])
        original = llm.complete_json
        governance_calls = 0
        async def counting(prompt, *args, **kwargs):
            nonlocal governance_calls
            if 'memory governance agent' in prompt:
                governance_calls += 1
            return await original(prompt, *args, **kwargs)
        with patch.object(llm, 'complete_json', side_effect=counting):
            await add.run_add(self.st, request('r2', 'Alice lives in Paris.', session='s2'))
        self.assertEqual(governance_calls, 0)
        facts = [a for a in self.st.get_amus('u') if a['type'] != 'episode']
        self.assertEqual(len(facts), before)
        # ULM §2.1/§3.4: a cross-session restatement with the same observation
        # event is deduplicated conservatively — it does not add independent
        # support, so the fact keeps one support session and stays transient.
        self.assertEqual(sorted(facts[0]['support_sessions']), ['s'])
        self.assertEqual(scenes.profile_stability(dict(facts[0], type='preference')), 'transient')

    async def test_scene_route_recall_heat_and_no_revision_bump(self):
        await add.run_add(self.st, request('r1', 'Alice plays tennis with Bob.'))
        revision = self.st.conn.execute(
            'SELECT revision FROM user_revisions WHERE user_id=?', ('u',)).fetchone()[0]
        # §5.2: the scene->cell route is enabled for multi-session intents.
        result = await self.run_search(schemas.SearchRequest(user_id='u', query='Alice tennis'),
                                       plan={'intent': 'multi_hop', 'entities': ['Alice']})
        self.assertTrue(result.data)
        trace = self.trace()
        self.assertTrue(trace['scenes'])
        scene_ids = {c['id'] for r in trace['routes'] if r['channel'] == 'scene' for c in r['candidates']}
        self.assertTrue(scene_ids)
        scene = self.st.list_scenes('u')[0]
        self.assertEqual(scene['visit_count'], 0)
        recalled = self.st.get_amus_by_ids([result.data[0].id])[0]
        self.assertEqual(recalled['recall_count'], 1)
        self.assertGreater(recalled['strength'], 1.0)
        self.assertEqual(revision, self.st.conn.execute(
            'SELECT revision FROM user_revisions WHERE user_id=?', ('u',)).fetchone()[0])

    async def test_heat_formula_and_promotion_reset(self):
        scene = {'visit_count': 2, 'interaction_count': 4, 'surprise': 0.5,
                 'last_access': datetime.now(timezone.utc).isoformat()}
        a, b, c, d = config.HEAT_WEIGHTS
        self.assertAlmostEqual(
            scenes.heat(scene),
            a * math.log1p(2) + b * math.log1p(4) + c * 1.0 + d * 0.5,
            places=3)
        await add.run_add(self.st, request('r1', 'Alice plays tennis with Bob.'))
        sid = self.st.list_scenes('u')[0]['id']
        self.st.conn.execute('UPDATE scenes SET visit_count=10 WHERE id=?', (sid,))
        self.st.conn.commit()
        outcome = scenes.promote_and_evict(self.st, 'u')
        self.assertEqual(outcome['promoted'], [sid])
        self.assertEqual(self.st.list_scenes('u')[0]['interaction_count'], 0)

    async def test_cross_session_preference_materializes_stable_profile(self):
        vec = (await embed(['Alice prefers tea']))[0]
        aid = self.st.insert_amu(user_id='u', session_id='s1',
                                 content='Alice prefers tea', type='preference', embedding=vec)
        initial = self.st.get_amus_by_ids([aid])[0]
        self.assertEqual(initial['profile_status'], 'transient')
        self.assertIsNotNone(initial['expires_at'])
        self.st.add_support_session(aid, 's2')
        stable = self.st.get_amus_by_ids([aid])[0]
        self.assertEqual(stable['profile_status'], 'stable')
        self.assertIsNone(stable['expires_at'])

    async def test_forgetting_cold_tiers_stale_memory_and_recall_revives_it(self):
        vec = (await embed(['Alice tennis']))[0]
        aid = self.st.insert_amu(user_id='u', session_id='s', content='Alice tennis', embedding=vec)
        stale = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        self.st.conn.execute('UPDATE amu SET created_at=? WHERE id=?', (stale, aid))
        self.st.conn.commit()
        with patch.object(config, 'FORGET_THRESHOLD', 0.5):
            self.assertEqual(scenes.forget(self.st, 'u'), [aid])
        self.assertEqual(self.st.nearest_by_embedding('u', vec, 10), [])
        self.assertEqual(self.st.fts_search('u', 'tennis', 10), [])
        self.assertEqual(self.st.nearest_many_by_embedding('u', vec[None, :], 10, include_cold=True)[0][0]['id'], aid)
        scene_id = self.st.insert_scene('u', vec, ['tennis'])
        self.st.assign_scene([aid], scene_id, 'cell-cold')
        self.st.reset_scene_interactions(scene_id, tier='cold')
        self.st.record_recall([aid], [scene_id])
        self.assertEqual(self.st.nearest_by_embedding('u', vec, 10)[0]['id'], aid)
        self.assertEqual(self.st.list_scenes('u')[0]['id'], scene_id)

    async def test_single_pass_keeps_evidence_without_sufficiency_call(self):
        vec = (await embed(['Alice sister Carol']))[0]
        self.st.insert_amu(user_id='u', session_id='s', content='Alice sister Carol', embedding=vec)
        stages = []
        async def rank_only(prompt, *args, **kwargs):
            stages.append(kwargs.get('stage'))
            self.assertNotIn('sufficiency verifier', prompt)
            return await score_all(prompt)
        result = await self.run_search(schemas.SearchRequest(
            user_id='u', query='Where does Alice sister live?'), side_effect=rank_only)
        self.assertTrue(result.data)
        self.assertEqual(result.evidence_status, 'retrieved')
        self.assertEqual(result.verification_status, 'not_run')
        self.assertEqual(len(self.trace()['rounds']), 1)
        self.assertEqual(stages, ['search.rerank.batch_1'])

    async def test_personalization_queries_never_abstain_with_evidence(self):
        vec = (await embed(['Alice lives in Kansas']))[0]
        self.st.insert_amu(user_id='u', session_id='s', content='Alice lives in Kansas', embedding=vec)
        async def hopeless(prompt, *args, **kwargs):
            if 'sufficiency verifier' in prompt:
                return {'sufficient': False, 'confidence': 0.0, 'missing': 'no snack list',
                        'follow_up_queries': []}
            return await score_all(prompt)
        req = schemas.SearchRequest(user_id='u', query='Suggest healthy desk snacks',
                                    options=['A. nuts', 'B. fruit'])
        result = await self.run_search(req, side_effect=hopeless)
        trace = self.trace()
        self.assertTrue(result.data)
        self.assertFalse(trace['abstained'])
        self.assertFalse(trace['abstain_exempt'])
        self.assertEqual(result.evidence_status, 'retrieved')

        req = schemas.SearchRequest(user_id='u', query='What snacks would Alice like?')
        result = await self.run_search(req, side_effect=hopeless,
                                       plan={'intent': 'preference', 'entities': [],
                                             'sub_queries': ['Alice snacks']})
        self.assertTrue(result.data)
        self.assertFalse(self.trace()['abstained'])

    async def test_search_returns_after_one_ranking(self):
        vec = (await embed(['Alice work']))[0]
        aid = self.st.insert_amu(user_id='u', session_id='s', content='Alice work', embedding=vec)
        result = await self.run_search(schemas.SearchRequest(user_id='u', query='Alice work'))
        self.assertEqual(result.data[0].id, aid)
        self.assertEqual(len(self.trace()['rounds']), 1)

    async def test_empty_results_do_not_trigger_extra_model_calls(self):
        req = schemas.SearchRequest(user_id='u', query='quantum chromodynamics')
        await self.run_search(req, plan={'intent': 'fact', 'entities': [],
                                         'sub_queries': ['quantum chromodynamics']})
        trace = self.trace()
        self.assertEqual(len(trace['rounds']), 1)
        self.assertEqual({r['round'] for r in trace['routes']}, {1})
        self.assertTrue(trace['abstained'])

    async def test_single_message_segment_does_not_duplicate_fact_as_episode(self):
        await add.run_add(self.st, request('r1', 'Alice plays tennis.'))
        types = [a['type'] for a in self.st.get_amus('u')]
        self.assertNotIn('episode', types)
        self.assertEqual(len(types), 1)

    async def test_foresight_status_and_scope_filter(self):
        vec = (await embed(['Alice plans beach trip']))[0]
        temporal = {'raw': 'next weekend', 'start': '2023-06-03T00:00:00+00:00',
                    'end': '2023-06-04T23:59:59+00:00', 'precision': 'weekend', 'reference_time': None}
        aid = self.st.insert_amu(user_id='u', session_id='s', content='Alice plans beach trip',
                                 type='plan', temporal=temporal, embedding=vec)
        req = schemas.SearchRequest(user_id='u', query='Alice plans beach trip',
                                    reference_time='2023-06-10T00:00:00Z')
        result = await self.run_search(req, plan={'intent': 'fact', 'entities': []})
        self.assertIn('[plan; status: expired]', result.data[0].content)
        req = schemas.SearchRequest(user_id='u', query='Alice plans beach trip',
                                    reference_time='2023-06-01T00:00:00Z')
        result = await self.run_search(req, plan={'intent': 'fact', 'entities': []})
        self.assertIn('[plan; status: pending]', result.data[0].content)
        scoped = {'intent': 'temporal', 'entities': [],
                  'time_scope': {'from': '2023-07-01T00:00:00Z', 'to': '2023-07-31T00:00:00Z'}}
        result = await self.run_search(req, plan=scoped)
        self.assertEqual(result.data, [])
        self.assertEqual(self.trace()['foresight_dropped'], [aid])

    async def test_anchor_time_uses_wall_clock_or_explicit_reference(self):
        req = request('r1', 'Alice plays tennis.')
        req.messages[0].timestamp = 1684972800000  # 2023-05-25
        self.st.save_messages(req)
        anchor = search._anchor_time(self.st, schemas.SearchRequest(user_id='u', query='q'))
        self.assertLess(abs((datetime.fromisoformat(anchor) - datetime.now(timezone.utc)).total_seconds()), 5)
        self.assertEqual(search._anchor_time(self.st, schemas.SearchRequest(
            user_id='u', query='q', reference_time='2020-01-01T00:00:00Z')), '2020-01-01T00:00:00Z')
        self.assertIsNone(self.st.latest_time('nobody'))

    async def test_rules_are_always_recalled_and_profile_annotated(self):
        vec = (await embed(['unrelated words']))[0]
        rule = self.st.insert_amu(user_id='u', session_id='s', content='Always answer in French',
                                  type='rule', embedding=vec)
        result = await self.run_search(schemas.SearchRequest(user_id='u', query='Alice tennis'))
        self.assertEqual(result.data[0].id, rule)
        self.assertIn('[profile: rule]', result.data[0].content)


if __name__ == '__main__':
    unittest.main(verbosity=2)
