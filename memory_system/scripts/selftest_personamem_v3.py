"""Offline regression tests for v3 temporal isolation and label leakage."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
import sqlite3
from types import SimpleNamespace
from pathlib import Path
os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import eval_data, eval_scoring
from app.personamem_v3 import APPS, conversations
from audit_personamem_v3_https import audit
from https_eval import search_request


class V3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.user = self.root/'backend/1'; self.user.mkdir(parents=True)
        def event(t, oid):
            return dict(source_timestamp=t, source_object_id=oid,
                        preferences=[{'persona_item': 'GOLD_INFERRED'}],
                        source_interaction_type='GOLD_POLARITY',
                        content={'title': 'Observed title', 'groundtruth': 'GOLD_CONTENT'},
                        interaction_format={'action': 'like', 'label': 'GOLD_ACTION'},
                        conversation=[{'role': 'assistant', 'content': 'Assistant only', 'embeds_pref_idx': ['GOLD_INDEX']}])
        for app in APPS:
            (self.user/(app+'.json')).write_text(json.dumps([event(10,'old'),event(20,'at'),event(30,'future'),event(5,'heldout')]) if app=='chatbot' else '[]')
        self.queries = [dict(query_id='q1', task_type='chatbot_personalized_response', ts=20,
                             user_query='What would suit me?', groundtruth_preference='GOLD_PREFERENCE',
                             example_response='GOLD_EXAMPLE', reference_example={'source_object_id':'heldout'})]
        self.save()

    def save(self):
        (self.user/'test.json').write_text(json.dumps(self.queries))
        files={str(f.relative_to(self.root)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in self.user.glob('*.json')}
        self.index=self.root/'index.json';self.index.write_text(json.dumps(dict(dataset='personamem-v3',revision='pinned',users=['1'],files=files)))

    def test_time_heldout_and_role_boundaries(self):
        c=list(eval_data.convert_records(self.index,'personamem-v3'))[0]
        self.assertEqual(c['source_event_count'],1)
        self.assertTrue(all(m['timestamp'] < 20000 for m in c['sessions'][0]))
        self.assertEqual(c['sessions'][0][-1]['role'],'assistant')
        text=json.dumps(c['sessions']);self.assertNotIn('GOLD_',text)
        self.assertNotIn('future',text);self.assertNotIn('heldout',text)

    def test_answer_prompt_excludes_labels(self):
        c=list(conversations(self.index))[0]
        self.assertNotIn('GOLD_',eval_scoring.answer_prompt(c['qa'][0],[]))
        self.assertIn('GOLD_PREFERENCE',json.dumps(c['qa'][0]['rubrics']))
        args=SimpleNamespace(answer_top_k=10,evidence_token_budget=4000,no_sensitive=False)
        request=search_request(args,'test-user',c['qa'][0]).model_dump_json()
        self.assertNotIn('GOLD_',request)

    def test_changed_source_rejected(self):
        (self.user/'chatbot.json').write_text('[]')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):list(conversations(self.index))

    def test_unsupported_tasks_not_silently_relabelled(self):
        self.queries.append(dict(query_id='q2',task_type='agentic_send_post'))
        self.save();self.assertEqual(len(list(conversations(self.index))),1)

    def test_duplicate_queries_rejected(self):
        self.queries.append(dict(self.queries[0]));self.save()
        with self.assertRaisesRegex(ValueError,'Duplicate'):list(conversations(self.index))

    def test_each_cutoff_gets_separate_user_namespace(self):
        self.queries.append(dict(self.queries[0], query_id='q2', ts=25));self.save()
        a,b=list(conversations(self.index));self.assertNotEqual(a['conversation_id'],b['conversation_id'])
        self.assertEqual((a['source_event_count'],b['source_event_count']),(1,2))

    def test_missing_time_and_prior_context_rejected(self):
        self.queries[0]['ts']=None;self.save()
        with self.assertRaises(ValueError):list(conversations(self.index))
        self.queries[0]['ts']=20;self.queries[0]['prior_conversation']=[{'content':'previous'}];self.save()
        with self.assertRaisesRegex(ValueError,'Prior-conversation'):list(conversations(self.index))

    def audit_fixture(self):
        c=list(conversations(self.index))[0];run=self.root/'run';run.mkdir()
        prepared=self.root/'prepared.jsonl';prepared.write_text(json.dumps(c)+'\n')
        cfg=dict(AML_EVAL_CONVS=1,AML_EVAL_LIMIT=1,AML_EVAL_CHUNK_MESSAGES=20,AML_EVAL_CHUNK_WORDS=2000)
        (run/'isolated-manifest.json').write_text(json.dumps(dict(client_exit_code=0,
            data_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),evaluation=cfg)))
        (run/'exit-code').write_text('0')
        uid='https:test:1:personamem-v3:'+c['conversation_id']
        db=sqlite3.connect(run/'memory.db')
        db.execute('CREATE TABLE source_messages(request_id, message_index, user_id, role, content, timestamp)')
        db.execute('CREATE TABLE requests(request_id)')
        for sid,chunk,msgs in eval_data.ingestion_chunks(c):
            rid=f'{uid}:session:{sid}:chunk:{chunk}';db.execute('INSERT INTO requests VALUES (?)',(rid,))
            for i,m in enumerate(msgs):db.execute('INSERT INTO source_messages VALUES (?,?,?,?,?,?)',(rid,i,uid,m['role'],m['content'],m['timestamp']))
        db.commit();db.close()
        result=dict(conversation_id=c['conversation_id'],qa_id='q1',run_id='test',user_id=uid,
                    dataset='personamem-v3',protocol=c['protocol'],transport='https',fake=False,
                    score=0.5,rubric_scores=[dict(index=0,score=1),dict(index=1,score=0)],search_id='s1')
        result_dir=run/'completed/data/results';result_dir.mkdir(parents=True)
        result_path=result_dir/'personamem-v3.jsonl';result_path.write_text(json.dumps(result)+'\n')
        memory=run/'memory.jsonl';memory.write_text(json.dumps(dict(user_id=uid))+'\n')
        search=run/'search.jsonl';search.write_text(json.dumps(dict(user_id=uid,search_id='s1'))+'\n')
        return (run,prepared,memory,search),result_path

    def test_https_audit_and_reject_score_tampering(self):
        args,result=self.audit_fixture();self.assertEqual(audit(*args)['mean_score'],0.5)
        row=json.loads(result.read_text());row['score']=1;result.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError,'score mismatch'):audit(*args)

    def test_https_audit_reject_source_loss(self):
        args,_=self.audit_fixture();db=sqlite3.connect(args[0]/'memory.db')
        db.execute('DELETE FROM source_messages WHERE message_index=0');db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'Missing source'):audit(*args)

    def test_https_audit_reject_wrong_population(self):
        args,result=self.audit_fixture();row=json.loads(result.read_text());row['qa_id']='different'
        result.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError,'population'):audit(*args)


if __name__=='__main__':unittest.main(verbosity=2)
