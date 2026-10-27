"""Porter recall must preserve source eligibility, read snapshots and old coarse rows."""
import copy
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, porter_recall, schemas, store


class PorterViewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.st = store.Store(str(Path(self.directory.name) / 'memory.db'))
        self.addCleanup(self.st.conn.close)

    def add(self, suffix, *, user='u', sensitive='normal'):
        request = schemas.AddRequest(request_id='r' + suffix, user_id=user, session_id='s', messages=[
            schemas.Message(role='user', content='I enjoy running trails.', sensitivity=sensitive)])
        self.st.save_messages(request)
        mid = self.st.insert_amu(user_id=user, session_id='s', content='The user runs trails.')
        self.st.link_sources(mid, request.request_id, [0])
        return mid

    def test_stemming_is_retrieval_only_and_parent_snapshot_remains_read_only(self):
        mid = self.add('a')
        original = list(map(tuple, self.st.conn.execute('SELECT rowid,* FROM amu_fts')))
        with self.st.snapshot('u') as snapshot:
            self.assertFalse(snapshot.fts_search('u', 'running', 20))
            self.assertFalse(snapshot.source_search('u', 'runs', 20))
            with porter_recall.lexical_view(snapshot, 'u') as lexical:
                self.assertEqual([c['id'] for c in lexical.fts_search('u', 'running', 20)], [mid])
                self.assertEqual([c['id'] for c in lexical.source_search('u', 'runs', 20)], [mid])
                self.assertEqual(lexical.sources_for_amu(mid)[0]['content'], 'I enjoy running trails.')
                self.assertEqual(list(map(tuple, lexical.conn.execute('SELECT rowid,* FROM amu_fts'))), original)
                with self.assertRaises(sqlite3.OperationalError):
                    lexical.conn.execute("DELETE FROM amu")
            self.assertEqual(snapshot.conn.execute('PRAGMA query_only').fetchone()[0], 1)
            self.assertFalse(snapshot.fts_search('u', 'running', 20))
        self.assertEqual(list(map(tuple, self.st.conn.execute('SELECT rowid,* FROM amu_fts'))), original)

    def test_all_original_source_filters_and_user_boundary_apply(self):
        normal = self.add('normal')
        sensitive = self.add('private', sensitive='sensitive')
        self.add('suppressed', sensitive='suppressed')
        self.add('other', user='other')
        past, retracted, stale, cold = [self.add(name) for name in ('past', 'retracted', 'stale', 'cold')]
        for mid, column, value in ((past, 'valid_to', '2020-01-01'), (retracted, 'resolution_status', 'retracted'),
                                   (stale, 'view_status', 'stale'), (cold, 'tier', 'cold')):
            self.st.conn.execute(f'UPDATE amu SET {column}=? WHERE id=?', (value, mid))
        self.st.conn.commit()
        with self.st.snapshot('u') as snapshot, porter_recall.lexical_view(snapshot, 'u') as lexical:
            for history, private, include_cold, expected in (
                (False, False, False, {normal}), (False, True, False, {normal, sensitive}),
                (True, False, False, {normal, past}), (False, False, True, {normal, cold}),
            ):
                found = lexical.source_search('u', 'runs', 40, include_history=history,
                    include_sensitive=private, include_cold=include_cold)
                self.assertEqual({c['id'] for c in found}, expected)
            with self.assertRaises(ValueError):
                lexical.source_search('other', 'runs', 40)
            with self.assertRaises(ValueError):
                with porter_recall.lexical_view(snapshot, 'other'):
                    pass

    def test_historical_projection_does_not_read_latest_claim_text(self):
        mid = self.add('history')
        self.st.conn.execute("UPDATE amu SET recorded_from='2020-01-01T00:00:00Z' WHERE id=?", (mid,))
        self.st.conn.commit()
        import numpy as np
        self.st.replace_fact(mid, dict(content='The user swims daily.', type='fact'),
                             np.ones(config.EMBED_DIM, dtype=np.float32))
        with self.st.snapshot('u') as latest, porter_recall.lexical_view(latest, 'u') as lexical:
            self.assertFalse(lexical.fts_search('u', 'running', 20))
        with self.st.snapshot('u', as_of='2025-01-01T00:00:00Z') as historical:
            with porter_recall.lexical_view(historical, 'u') as lexical:
                self.assertEqual([c['id'] for c in lexical.fts_search('u', 'running', 20)], [mid])
                self.assertFalse(lexical.fts_search('u', 'swimming', 20))

    def test_cancelled_budget_cannot_build_or_return_optional_recall(self):
        self.add('cancel')
        with budget.scope(seconds=0, calls=1, tokens=1):
            with self.assertRaises(TimeoutError):
                with porter_recall.lexical_view(self.st, 'u'):
                    self.fail('Expired work yielded a usable index')


class CoarsePreservationTests(unittest.TestCase):
    def test_cascade_does_not_readmit_new_heads_into_old_coarse_quota(self):
        from app import cascade_rerank
        original = [dict(id=f'old{i}', content=f'Original passage {i}', _fused=1-i/100) for i in range(65)]
        extra = [dict(id='new', content='New passage', _fused=10)]
        with patch.multiple(config, RECALL_PORTER_SUPPLEMENT=True, COARSE_EVIDENCE_EXTEND=True,
                            CASCADE_COARSE_LIMIT=50):
            base, old = cascade_rerank.coarse_shortlist(original, [])
            selected, info = cascade_rerank.recall_shortlist(original + extra, [], {'_porter_supplement_ids': ['new']})
        self.assertEqual(selected, base + extra)
        self.assertEqual(info['baseline_ids'], old['baseline_ids'])
        self.assertEqual(info['extra_ids'], old['extra_ids'])
        self.assertNotIn('new', info['baseline_ids'])

    def test_extension_preserves_old_rows_eligibility_and_bounded_new_work(self):
        original = [dict(id=f'old{i}', content=f'Original evidence {i}', _fused=1-i/100) for i in range(58)]
        metadata = dict(limit=58, baseline_ids=[c['id'] for c in original[:50]],
                        extra_ids=[c['id'] for c in original[50:]], selected_ids=[c['id'] for c in original],
                        output_count=58, selection={'original': True})
        before = copy.deepcopy((original, metadata))
        extra = [original[-1]] + [dict(id=f'new{i}', content=f'New {i}') for i in range(25)]
        selected, trace = porter_recall.extend_coarse(original, metadata, extra)
        self.assertEqual(selected[:58], before[0])
        self.assertEqual((original, metadata), before)
        self.assertEqual(len(selected), 78)
        self.assertEqual(trace['baseline_ids'], metadata['baseline_ids'])
        self.assertEqual(trace['extra_ids'], metadata['extra_ids'])
        self.assertEqual(trace['supplemental_ids'], [f'new{i}' for i in range(20)])
        self.assertEqual(trace['selected_ids'], [c['id'] for c in selected])

    def test_disabled_path_does_no_database_or_index_work(self):
        with patch.object(config, 'RECALL_PORTER_SUPPLEMENT', False), patch.object(porter_recall, 'lexical_view') as view:
            plan, fused, candidates = {}, [], []
            result = porter_recall.augment(None, None, plan, [], fused, candidates)
            self.assertIs(result[0], fused)
            self.assertIs(result[1], candidates)
            self.assertEqual(plan, {})
            view.assert_not_called()

    def test_failed_optional_index_leaves_original_candidates_and_relations_intact(self):
        plan = dict(_query_specs=[dict(origin='question', text='running')], _routes=[], _fusion_edges=[{'old': True}])
        original = copy.deepcopy(plan)
        fused, candidates = [dict(id='original')], [dict(id='original', _rank_text='Original')]
        with patch.multiple(config, RECALL_PORTER_SUPPLEMENT=True, SOURCE_RECALL_DIVERSITY=True,
                SOURCE_RECALL_NOVELTY=False, COARSE_EVIDENCE_EXTEND=True, FINE_COARSE_CORE_QUOTA=True,
                FINE_EVIDENCE_ADAPTIVE=True), patch.object(porter_recall, 'lexical_view', side_effect=sqlite3.OperationalError):
            result = porter_recall.augment(None, schemas.SearchRequest(user_id='u', query='running'),
                                           plan, [], fused, candidates)
        self.assertIs(result[0], fused)
        self.assertIs(result[1], candidates)
        self.assertEqual(plan['_routes'], original['_routes'])
        self.assertEqual(plan['_fusion_edges'], original['_fusion_edges'])
        self.assertNotIn('_porter_supplement_ids', plan)
        self.assertEqual(plan['_porter_recall']['status'], 'error')


if __name__ == '__main__':
    unittest.main()
