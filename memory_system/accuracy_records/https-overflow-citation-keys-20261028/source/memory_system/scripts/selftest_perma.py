"""Offline PERMA mapping, date precision, leakage and HTTPS audit regressions."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import perma, eval_data, eval_scoring
from https_eval import search_request
from audit_temporal_https import audit


def fixture(stage=2):
    row=dict(task_id='SD-Books-task-1',type=stage,preferences=['ORACLE_PREF'],affinity_links=['ORACLE_LINK'],
             context=[[[{'role':'user','content':'I prefer short novels.','hidden':'ORACLE_TURN'},
                        {'role':'assistant','content':'Try a novella.'}],'2023-01-02 Morning']])
    meta=dict(task_id=row['task_id'],task_type=stage,scope='overall',question_date='2023-01-02',
              question='Recommend a book.',task_goal='ORACLE_GOAL',options='\n'.join(f'{chr(65+i)}: Option {i}\nsecond line' for i in range(8)),gold_label='H')
    return row,meta


class Tests(unittest.TestCase):
    def conv(self,stage=2):return perma.convert(*fixture(stage),'user108','clean','pinned')

    def test_all_eight_options_and_exact_scoring(self):
        c=eval_data.validate(self.conv());q=c['qa'][0]
        self.assertEqual(len(q['options']),8);self.assertEqual(q['gold_labels'],['H'])
        self.assertIn('\nsecond line',q['options'][0]);self.assertEqual(eval_scoring.choice_score('H',['H'],'single_choice'),1)

    def test_fake_eight_option_reader_has_no_schema_errors(self):
        prediction, score, diagnostics = asyncio.run(eval_scoring.evaluate(self.conv(1)['qa'][0], []))
        self.assertNotIn('error_stage', diagnostics)
        self.assertIn(prediction, list('ABCDEFGH'))

    def test_type1_is_truly_empty_memory(self):
        c=eval_data.validate(self.conv(1));self.assertEqual(c['sessions'],[])
        self.assertEqual(list(eval_data.ingestion_chunks(c)),[])
        c['sessions']=[[dict(role='user',content='Injected',timestamp=1)]]
        with self.assertRaisesRegex(ValueError,'no ingested history'):eval_data.validate(c)

    def test_empty_memory_not_generalized_to_other_datasets(self):
        c=self.conv(1);c['dataset']='normalized'
        with self.assertRaises(ValueError):eval_data.validate(c)

    def test_same_day_preserved_future_rejected(self):
        c=self.conv();self.assertEqual(c['sessions'][0][0]['timestamp'],perma.date_ms('2023-01-02 Morning'))
        row,meta=fixture();row['context'][0][1]='2023-01-03 Morning'
        with self.assertRaisesRegex(ValueError,'after the question day'):perma.convert(row,meta,'user108','clean','pinned')

    def test_original_roles_content_and_no_oracle_leak(self):
        c=self.conv();self.assertEqual([m['role'] for m in c['sessions'][0]],['user','assistant'])
        self.assertEqual(c['sessions'][0][0]['content'],'I prefer short novels.')
        self.assertNotIn('ORACLE_',json.dumps(c))
        q=c['qa'][0];q['answer']='GOLD_ONLY';q['gold_labels']=['GOLD_ONLY']
        self.assertNotIn('GOLD_ONLY',eval_scoring.answer_prompt(q,[]))
        args=SimpleNamespace(answer_top_k=10,evidence_token_budget=4000,no_sensitive=False)
        self.assertNotIn('GOLD_ONLY',search_request(args,'uid',q).model_dump_json())

    def test_identity_options_and_date_rejections(self):
        row,meta=fixture();meta['task_type']=3
        with self.assertRaisesRegex(ValueError,'identity'):perma.convert(row,meta,'user108','clean','pinned')
        for text in ['A: one\nA: two','A: one\nC: two','A: one\nB: ']:
            with self.assertRaises(ValueError):perma.options(text)
        with self.assertRaises(ValueError):perma.date_ms('2023-01-01 Night')

    def test_snapshot_identity_includes_variant_and_stage(self):
        a=self.conv();b=self.conv(3);self.assertNotEqual(a['conversation_id'],b['conversation_id'])
        c=perma.convert(*fixture(),'user108','noisy','pinned');self.assertNotEqual(a['conversation_id'],c['conversation_id'])

    def test_download_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rel='tasks/user108/input_data_c.json';f=root/rel;f.parent.mkdir(parents=True);f.write_text('{}')
            index=dict(dataset='perma',histories=[rel],files={rel:{'sha256':'wrong'}},variant='clean',revision='x')
            (root/'index.json').write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):list(perma.conversations(root/'index.json'))

    def test_zero_ingestion_https_audit_and_score_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.conv(1);inp=root/'input.jsonl';inp.write_text(json.dumps(c)+'\n')
            manifest=dict(client_exit_code=0,data_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),evaluation=dict(AML_EVAL_CONVS=1,AML_EVAL_LIMIT=1,AML_EVAL_CHUNK_MESSAGES=20,AML_EVAL_CHUNK_WORDS=2000))
            (root/'isolated-manifest.json').write_text(json.dumps(manifest));(root/'exit-code').write_text('0')
            db=sqlite3.connect(root/'memory.db');db.execute('create table requests(request_id)');db.execute('create table source_messages(request_id)');db.close()
            uid=f"https:test:1:perma:{c['conversation_id']}"
            row=dict(conversation_id=c['conversation_id'],qa_id=c['qa'][0]['id'],run_id='test',user_id=uid,dataset='perma',protocol=perma.PROTOCOL,transport='https',fake=False,prediction='H',score=1,search_id='search',category='single/checkpoint-1')
            out=root/'completed/data/results/perma.jsonl';out.parent.mkdir(parents=True);out.write_text(json.dumps(row)+'\n')
            memory=root/'memory.jsonl';memory.write_text('');search=root/'search.jsonl';search.write_text(json.dumps(dict(user_id=uid,search_id='search'))+'\n')
            self.assertEqual(audit(root,inp,memory,search,'perma')['mean_score'],1)
            row['score']=0;out.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'score mismatch'):audit(root,inp,memory,search,'perma')


if __name__=='__main__':unittest.main(verbosity=2)
