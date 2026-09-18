"""Offline storage boundaries for graph fusion over an explicit candidate pool."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_read_text = Path.read_text
with patch.object(Path, 'read_text', lambda path, *a, **kw:
                  '' if path.name == '.env' else _read_text(path, *a, **kw)):
    from app import config, evidence_packet, schemas, search_pipeline, store


class GraphFusionStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='aml-graph-fusion-')
        self.st = store.Store(str(Path(self.directory.name) / 'memory.sqlite3'))

    def tearDown(self):
        self.st.conn.close()
        self.directory.cleanup()

    def memory(self, name, user='u', count=1, **kwargs):
        aid = self.st.insert_amu(user_id=user, session_id='s-' + user, content=name, **kwargs)
        for number in range(count):
            self.st.insert_triple(user, name, 'related_to', f'object-{number}', aid)
        return aid

    def rows(self, ids, **kwargs):
        return self.st.graph_rows_for_candidates('u', ids, **kwargs)

    def test_candidate_pool_excludes_neighbor_and_unrelated_memories(self):
        first = self.memory('Alpha')
        self.memory('object-0')  # A real graph neighbor still needs explicit admission.
        self.memory('Unrelated')
        rows = self.rows([first])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['amu_id'], first)
        self.assertEqual((rows[0]['subject'], rows[0]['relation'], rows[0]['object']),
                         ('Alpha', 'related_to', 'object-0'))

    def test_forged_triple_owner_cannot_expose_another_users_amu(self):
        own = self.memory('Own')
        foreign = self.memory('Private foreign', user='v')
        # Deliberately corrupt both directions; joins must verify both owners.
        self.st.conn.execute('INSERT INTO triples(user_id,subject,relation,object,amu_id) '
                             'VALUES (?,?,?,?,?)', ('u', 'Private leak', 'is', 'secret', foreign))
        self.st.conn.execute('INSERT INTO triples(user_id,subject,relation,object,amu_id) '
                             'VALUES (?,?,?,?,?)', ('v', 'Wrong owner', 'is', 'secret', own))
        self.st.conn.commit()
        rows = self.rows([own, foreign], include_history=True, include_sensitive=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['amu_id'], own)
        self.assertEqual(rows[0]['subject'], 'Own')

    def test_history_and_sensitive_filters_match_candidate_visibility(self):
        current = self.memory('Current')
        historical = self.memory('Historical', valid_to='2024-01-01T00:00:00Z')
        sensitive = self.memory('Sensitive', sensitivity='sensitive')
        ids = [current, historical, sensitive]
        self.assertEqual({r['amu_id'] for r in self.rows(ids)}, {current})
        self.assertEqual({r['amu_id'] for r in self.rows(ids, include_history=True)},
                         {current, historical})
        self.assertEqual({r['amu_id'] for r in self.rows(ids, include_sensitive=True)},
                         {current, sensitive})
        self.assertEqual({r['amu_id'] for r in self.rows(ids,
                         include_history=True, include_sensitive=True)}, set(ids))

    def test_suppressed_stale_and_retracted_are_always_excluded(self):
        visible = self.memory('Visible')
        suppressed = self.memory('Suppressed', sensitivity='suppressed')
        stale = self.memory('Stale')
        retracted = self.memory('Retracted', resolution_status='retracted')
        self.st.conn.execute("UPDATE amu SET view_status='stale' WHERE id=?", (stale,))
        self.st.conn.commit()
        self.assertEqual({r['amu_id'] for r in self.rows(
            [visible, suppressed, stale, retracted], include_history=True, include_sensitive=True)},
            {visible})

    def test_per_candidate_cap_prevents_large_memory_from_consuming_pool(self):
        ids = [self.memory('Many', count=30), self.memory('Few A', count=2),
               self.memory('Few B', count=2)]
        rows = self.rows(ids, per_candidate=2, limit=6)
        self.assertEqual(len(rows), 6)
        self.assertEqual({mid: sum(row['amu_id'] == mid for row in rows) for mid in ids},
                         {mid: 2 for mid in ids})

    def test_global_limit_preserves_a_turn_for_each_candidate(self):
        ids = [self.memory('Many', count=30), self.memory('Few A', count=2),
               self.memory('Few B', count=2)]
        rows = self.rows(ids, per_candidate=8, limit=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual({row['amu_id'] for row in rows}, set(ids))

    def test_empty_missing_duplicate_and_zero_budget_inputs(self):
        aid = self.memory('Candidate', count=3)
        self.assertEqual(self.rows([]), [])
        self.assertEqual(self.rows(['amu_does_not_exist']), [])
        self.assertEqual(self.rows([aid], per_candidate=0), [])
        self.assertEqual(self.rows([aid], limit=0), [])
        rows = self.rows([aid, aid, aid], per_candidate=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row['id'] for row in rows}), 2)

    def test_snapshot_rejects_cross_user_argument_and_candidate_ids(self):
        own = self.memory('Own')
        foreign = self.memory('Foreign', user='v')
        with self.st.snapshot('u') as snapshot:
            with self.assertRaises(ValueError):
                snapshot.graph_rows_for_candidates('v', [foreign])
            rows = snapshot.graph_rows_for_candidates('u', [own, foreign])
            self.assertEqual({row['amu_id'] for row in rows}, {own})

    def test_actual_candidate_queries_use_amu_scoped_triples_index(self):
        aid = self.memory('Candidate', count=2)
        for number in range(20):
            self.memory('Noise ' + str(number), count=3)
        observed = []
        self.st.conn.set_trace_callback(observed.append)
        try:
            self.rows([aid], per_candidate=2)
        finally:
            self.st.conn.set_trace_callback(None)
        candidate_queries = [sql for sql in observed if sql.lstrip().upper().startswith('SELECT')
                             and ('FROM TRIPLES' in sql.upper() or 'JOIN TRIPLES' in sql.upper())]
        self.assertTrue(candidate_queries, 'no candidate triple query was observed')
        composite = []
        for row in self.st.conn.execute('PRAGMA index_list(triples)'):
            name = row['name']
            columns = [part['name'] for part in self.st.conn.execute('PRAGMA index_info(' + name + ')')]
            if columns[:3] == ['user_id', 'amu_id', 'id']:
                composite.append(name)
        self.assertTrue(composite, 'missing (user_id, amu_id, id) triples index')
        for sql in candidate_queries:
            plan = [row['detail'] for row in self.st.conn.execute('EXPLAIN QUERY PLAN ' + sql)]
            self.assertTrue(any(name in detail for name in composite for detail in plan), plan)

    def test_atomic_group_admission_preserves_relative_listwise_order(self):
        items = [dict(id=mid, content='Evidence ' + mid) for mid in ('a', 'b', 'c')]
        packed, _, manifest = evidence_packet.pack_ranked(
            items, top_k=3, token_budget=2048, groups=[['a', 'c']])
        self.assertEqual([item['id'] for item in packed], ['a', 'b', 'c'])
        self.assertEqual(manifest['evidence_groups'], [['a', 'c']])

    def test_intervening_source_duplicate_cannot_break_an_already_admitted_group(self):
        source = dict(request_id='source-r', message_index=0, role='user',
                      content='Acme is based in Oslo.', content_span=dict(start=0, end=22))
        items = [dict(id='a', content='Alice founded Acme.'),
                 dict(id='b', content='Acme location', sources=[dict(source)]),
                 dict(id='c', content='Quoted location', memory_type='episode', sources=[dict(source)])]
        packed, packet_hash, manifest = evidence_packet.pack_ranked(
            items, top_k=3, token_budget=4096, groups=[['a', 'c']])
        self.assertEqual([item['id'] for item in packed], ['a', 'c'])
        self.assertEqual(manifest['evidence_groups'], [['a', 'c']])
        self.assertEqual(packed[1]['sources'][0]['content'], source['content'])
        self.assertEqual(evidence_packet.digest(packed), packet_hash)
        self.assertTrue(any(item['id'] == 'b' for item in manifest['omitted']))

    def test_post_ranking_scope_filter_drops_whole_group_without_failing_search(self):
        rows = [dict(id=mid, content='Evidence ' + mid, type='fact',
                     _packet_item=dict(id=mid, content='Evidence ' + mid))
                for mid in ('a', 'b', 'c')]
        rows[2].update(type='plan', temporal=dict(start='2030-01-01', end='2030-01-02'))
        plan = dict(_cascade={}, _evidence_groups=[['a', 'c']],
                    time_scope={'from': '2024-01-01', 'to': '2024-12-31'})
        req = schemas.SearchRequest(user_id='u', query='What happened in 2024?', top_k=3)
        packed, _, manifest = search_pipeline._pack_evidence(
            self.st, req, plan, rows, '2026-01-01T00:00:00Z')
        self.assertEqual([item['id'] for item in packed], ['b'])
        self.assertEqual(manifest['evidence_groups'], [])

    def test_graph_metadata_reads_obey_candidate_budget_before_hydration(self):
        for index in range(24):
            self.memory('Candidate ' + str(index), count=0)
        rows = self.st.get_amus('u')
        plan = dict(query='candidate', intent='fact', _used_queries=['candidate'],
                    _routes=[{'channel': 'vector'}])
        req = schemas.SearchRequest(user_id='u', query='candidate')
        with patch.object(config, 'GRAPH_FUSION_MAX_CANDIDATES', 16), patch.object(
                self.st, 'graph_rows_for_candidates', wraps=self.st.graph_rows_for_candidates) as read:
            fused = search_pipeline._fuse_candidates(self.st, req, plan, [rows])
        self.assertLessEqual(len(fused), 16)
        admitted_ids = {mid for call in read.call_args_list for mid in call.args[1]}
        self.assertLessEqual(len(admitted_ids), 16,
                             'fusion queried metadata for candidates outside its graph budget')


if __name__ == '__main__':
    unittest.main()
