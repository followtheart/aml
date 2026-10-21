"""Offline snapshot integrity and paired comparison regression tests."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experience_replay as replay


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.run=self.root/'run';self.run.mkdir()
        prepared=self.root/'prepared.jsonl'
        prepared.write_text(json.dumps(dict(conversation_id='c',qa=[dict(id='q',question='test',answer='A')]))+'\n')
        trace=dict(search_id='s',ranked=[dict(id='m',_ce_score=0.5)],
                   cascade=dict(cross_encoder=dict(status='ok'),coarse=dict(selected_ids=['m'])))
        (self.run/'server-search-debug.jsonl').write_text(json.dumps(trace)+'\n')
        results=self.run/'completed/data/results';results.mkdir(parents=True)
        (results/'personamem-v2-32k.jsonl').write_text(json.dumps(dict(search_id='s',conversation_id='c',qa_id='q'))+'\n')
        replay.write(self.run/'isolated-manifest.json',dict(server_exit_code=-15,audit_exit_code=0,
                     data=str(prepared),data_sha256=replay.digest(prepared)))
        self.conn=sqlite3.connect(self.run/'memory.db')
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('CREATE TABLE example(value TEXT)')
        self.conn.execute("INSERT INTO example VALUES ('committed WAL data')");self.conn.commit()
        self.snapshot=self.root/'snapshot'

    def tearDown(self):
        self.conn.close();self.temp.cleanup()

    def test_snapshot_captures_wal_and_detects_tampering(self):
        replay.freeze(self.run,self.snapshot)
        db=sqlite3.connect(self.snapshot/'memory.db')
        self.assertEqual(db.execute('SELECT value FROM example').fetchone()[0],'committed WAL data');db.close()
        (self.snapshot/'cases.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'integrity mismatch'):replay.verify(self.snapshot)

    def test_partial_run_and_changed_dataset_are_rejected(self):
        path=self.run/'isolated-manifest.json';manifest=json.loads(path.read_text())
        manifest.pop('server_exit_code');replay.write(path,manifest)
        with self.assertRaisesRegex(ValueError,'completed'):replay.freeze(self.run,self.snapshot)
        manifest['server_exit_code']=-15;replay.write(path,manifest)
        (self.root/'prepared.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError,'dataset changed'):replay.freeze(self.run,self.snapshot)

    def test_snapshot_refuses_overwrite_and_missing_ce_scores(self):
        replay.freeze(self.run,self.snapshot)
        with self.assertRaises(FileExistsError):replay.freeze(self.run,self.snapshot)
        trace=dict(ranked=[dict(id='m')],cascade=dict(cross_encoder=dict(status='ok'),coarse=dict(selected_ids=['m'])))
        with self.assertRaisesRegex(ValueError,'CE score'):replay.validate_trace(trace)

    async def test_same_inputs_repetitions_and_alternating_order(self):
        replay.freeze(self.run,self.snapshot)
        calls=[]
        async def arm(case,staged,**kwargs):
            calls.append((replay.fingerprint(case),staged))
            return dict(cascade=dict(coarse=dict(selected_ids=['m']),fine=dict(selected_ids=['m']),
                                     listwise=dict(candidate_ids=['m'])))
        with patch.object(replay.config,'FAKE',False),patch.object(replay,'run_arm',arm):
            result=await replay.compare(self.snapshot,self.root/'out',execute=True,repeats=2)
        self.assertEqual([x[1] for x in calls],[False,True,True,False])
        self.assertEqual(len({x[0] for x in calls}),1)
        self.assertEqual(result['comparable'],2)
        self.assertIsNone(result['mean_score_delta'])
        self.assertFalse(result['automatic_promotion'])
        replay.verify(self.snapshot)

    async def test_admission_drift_and_provider_error_are_not_quality_improvements(self):
        replay.freeze(self.run,self.snapshot)
        async def arm(case,staged,**kwargs):
            return dict(cascade=dict(coarse=dict(selected_ids=['m']),fine=dict(selected_ids=['m']),
                                     listwise=dict(candidate_ids=[] if staged else ['m'])))
        with patch.object(replay.config,'FAKE',False),patch.object(replay,'run_arm',arm):
            result=await replay.compare(self.snapshot,self.root/'out',execute=True,repeats=1)
        self.assertEqual(result['invalid_pairs'],1)
        async def fail(case,staged,**kwargs):raise RuntimeError('provider failed')
        with patch.object(replay.config,'FAKE',False),patch.object(replay,'run_arm',fail):
            result=await replay.compare(self.snapshot,self.root/'errors',execute=True,repeats=1,answer=True)
        self.assertIsNone(result['mean_score_delta'])
        self.assertEqual(result['invalid_pairs'],1)

    async def test_requires_explicit_execution_real_provider_and_valid_case(self):
        replay.freeze(self.run,self.snapshot)
        with self.assertRaisesRegex(ValueError,'requires'):await replay.compare(self.snapshot,self.root/'out')
        with patch.object(replay.config,'FAKE',False):
            with self.assertRaisesRegex(ValueError,'Unknown case'):
                await replay.compare(self.snapshot,self.root/'out',execute=True,selected=['missing'])


if __name__=='__main__':unittest.main(verbosity=2)
