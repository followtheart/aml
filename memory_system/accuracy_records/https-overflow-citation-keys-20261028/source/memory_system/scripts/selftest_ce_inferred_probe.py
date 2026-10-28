"""A failed batch stays missing; later planned batches still persist once."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import probe_ce_inferred_primary as probe


class BatchChecks(unittest.IsolatedAsyncioTestCase):
    def test_csv_binary_batches_preserve_utf8_crlf_multiline_and_final_record(self):
        text = 'id,text\r\n1,"🙂 first\r\nsecond, quoted"\r\n2,last'
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'test.csv'
            path.write_bytes(text.encode('utf-8'))
            for size in (1, 7, 23, 4*1024*1024):
                self.assertEqual(list(probe.csv_batches(path, size)),
                    [{'id':'1','text':'🙂 first\r\nsecond, quoted'}, {'id':'2','text':'last'}])

    async def test_timeout_preserves_other_batches_and_has_no_retry(self):
        seen=[]
        async def score(query, docs, **kwargs):
            self.assertEqual(probe.config.RATE_LIMIT_RETRIES,0)
            limits=probe.budget.current.get();limits.before_call()
            seen.append(docs[0])
            if docs[0]=='middle':raise TimeoutError('synthetic timeout')
            limits.tokens+=7
            return [0.5]
        job=dict(qa_id='test',conversation_id='synthetic',
            documents={'candidate':['first','middle','last']},queries={'candidate':'question'},
            batches={'candidate':[[0],[1],[2]]},t={'versions':{'settings':{}}})
        with tempfile.TemporaryDirectory() as name:
            out=Path(name)
            with patch.object(probe.cross_encoder,'rerank',new=score):
                event=await probe.score_arm(job,'candidate',out)
            rows=[json.loads(s) for s in (out/'batches.jsonl').read_text().splitlines()]
            self.assertEqual(seen,['first','middle','last'])
            self.assertEqual([r['status'] for r in rows],['complete','failed','complete'])
            self.assertEqual(event['scores'],[0.5,None,0.5])
            self.assertEqual(event['failed_batches'],[1])
            self.assertEqual((event['calls'],event['tokens']),(3,14))
            self.assertEqual(json.loads((out/'responses.jsonl').read_text()),event)

    async def test_failed_arm_never_gets_archived_or_zero_scores(self):
        job=dict(changed=True,coarse=[{'_ce_score':0.1}],
            t={'cascade':{'admission_budget':{'token_headroom':100}}})
        events=dict(control=dict(status='failed',scores=[None],tokens=0),
                    candidate=dict(status='complete',scores=[0.5],tokens=7))
        with patch.object(probe.common.base,'replay',side_effect=lambda j,s:dict(scores=s)):
            variants,delta=probe.selections(job,events)
        self.assertIsNone(variants['control'])
        self.assertIsNone(variants['candidate_cost_adjusted'])
        self.assertIsNone(delta)
        self.assertEqual(variants['candidate']['scores'],[0.5])
        self.assertEqual(variants['archived']['scores'],[0.1])


if __name__=='__main__':unittest.main()
