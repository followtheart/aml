"""Recovery refuses changed input, partial writes and already-scored runs."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from resume_isolated_https_ingestion import preflight


class Recovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.failed=self.root/'failed';self.failed.mkdir()
        self.input=self.root/'input.jsonl';self.plan=self.root/'plan.json'
        data=dict(conversation_id='c',dataset='normalized',sessions=[[dict(role='user',content=f'message {i}',timestamp=None) for i in range(21)]],qa=[dict(id='q')])
        self.input.write_text(json.dumps(data)+'\n')
        self.plan.write_text(json.dumps(dict(input=str(self.input),sha256=hashlib.sha256(self.input.read_bytes()).hexdigest(),expected_qa=1)))
        self.run_id='a'*32;uid=f'https:{self.run_id}:1:normalized:c';session=uid+':session:0';request=session+':chunk:0'
        (self.failed/'output.log').write_text(f'run_id={self.run_id} url=https://localhost:18443\n')
        (self.failed/'isolated-manifest.json').write_text(json.dumps(dict(client_exit_code=1,server_exit_code=-15,files={},data=str(self.input))))
        (self.failed/'server-memory-debug.jsonl').write_text(json.dumps(dict(request_id=request))+'\n')
        c=sqlite3.connect(self.failed/'memory.db')
        c.execute('CREATE TABLE requests (request_id TEXT,user_id TEXT,session_id TEXT,created_at TEXT)')
        c.execute('CREATE TABLE source_messages (request_id TEXT,message_index INTEGER,role TEXT,content TEXT,timestamp TEXT)')
        c.execute('INSERT INTO requests VALUES (?,?,?,?)',(request,uid,session,'now'))
        c.executemany('INSERT INTO source_messages VALUES (?,?,?,?,?)',[(request,i,'user',f'message {i}',None) for i in range(20)])
        c.commit();c.close()

    def test_valid_prefix(self):
        r=preflight(self.failed,self.plan)
        self.assertEqual((r['committed_chunks'],r['total_chunks'],r['messages']),(1,2,20))

    def interrupted_manifest(self):
        path=self.failed/'isolated-manifest.json'
        manifest=json.loads(path.read_text())
        manifest.pop('client_exit_code');manifest.pop('server_exit_code')
        manifest['server_pid']=123456789
        path.write_text(json.dumps(manifest))

    def test_missing_exit_records_require_explicit_mode(self):
        self.interrupted_manifest()
        with self.assertRaisesRegex(ValueError,'terminal failed'):
            preflight(self.failed,self.plan)

    def test_interrupted_recovery_checks_processes(self):
        self.interrupted_manifest()
        with patch('resume_isolated_https_ingestion.verify_interrupted', return_value={'server_pid_absent':True}) as check:
            result=preflight(self.failed,self.plan,allow_interrupted=True)
        check.assert_called_once()
        self.assertTrue(result['interruption']['server_pid_absent'])

    def test_live_server_refused(self):
        self.interrupted_manifest()
        with patch('resume_isolated_https_ingestion.os.kill', return_value=None):
            with self.assertRaisesRegex(ValueError,'still exists'):
                preflight(self.failed,self.plan,allow_interrupted=True)

    def test_live_client_refused_even_when_server_missing(self):
        self.interrupted_manifest()
        environment=dict(os.environ, AML_EXPERIMENT_DIR=str(self.failed.resolve()))
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],env=environment)
        try:
            with self.assertRaisesRegex(ValueError,'run process still exists'):
                preflight(self.failed,self.plan,allow_interrupted=True)
        finally:
            child.terminate();child.wait(timeout=5)

    def test_absent_processes_allow_valid_interrupted_prefix(self):
        self.interrupted_manifest()
        result=preflight(self.failed,self.plan,allow_interrupted=True)
        self.assertTrue(result['interruption']['exit_codes_unknown'])
        self.assertEqual(result['interruption']['matching_processes'],0)

    def test_changed_input_rejected(self):
        self.input.write_text(self.input.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'hash changed'):preflight(self.failed,self.plan)

    def test_uncommitted_partial_message_rejected(self):
        c=sqlite3.connect(self.failed/'memory.db');c.execute("INSERT INTO source_messages VALUES ('partial',0,'user','extra',NULL)");c.commit();c.close()
        with self.assertRaisesRegex(ValueError,'Uncommitted'):preflight(self.failed,self.plan)

    def test_scored_run_rejected(self):
        p=self.failed/'completed/data/results/personamem-v2-32k.jsonl';p.parent.mkdir(parents=True);p.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'Scoring already began'):preflight(self.failed,self.plan)

    def test_mutated_source_rejected(self):
        c=sqlite3.connect(self.failed/'memory.db');c.execute("UPDATE source_messages SET content='changed' WHERE message_index=0");c.commit();c.close()
        with self.assertRaisesRegex(ValueError,'differs from input'):preflight(self.failed,self.plan)


if __name__=='__main__':unittest.main()
