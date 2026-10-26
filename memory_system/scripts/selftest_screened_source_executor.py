"""Exercise the real CE request adapter with an offline HTTP response fixture."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config
from batched_records import json_records
from run_screened_source_ce import body_digest, execute
from source_nomination import wire_body
import hashlib
import httpx


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def trial(self, fail_second=False, usage=10):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(search_id='s',conversation_id='c',qa_id='q',ce_query='question',
                ce_model='qwen3-rerank',ce_api_format='cohere',
                scheduling=dict(eligible=True,main_reservation=1000,ce_deadline_seconds=5),
                original_headroom=dict(call_headroom=3,token_headroom=10000),
                original_main_ids=['original'],views={'a':dict(text='first'),'b':dict(text='second')},
                batches=[['a'],['b']],batch_bounds=[13,14],
                nominees=[dict(id='winner',source_view_ids=['a','b'],ce_rank=30)])
            inventory = root/'inventory.jsonl'
            inventory.write_text(json.dumps(job)+'\n')
            requests = [dict(search_id='s',batch=i,view_ids=batch,bound=job['batch_bounds'][i],
                body_sha256=body_digest(wire_body('qwen3-rerank','cohere','question',[job['views'][k]['text'] for k in batch])))
                for i,batch in enumerate(job['batches'])]
            settings = dict(FAKE=False,LLM_MODEL='gpt-4o-mini',CE_MODEL='qwen3-rerank',
                CE_API_FORMAT='cohere',CE_TIMEOUT_SECONDS=2,CE_BATCH_SIZE=24,
                CE_MAX_REQUEST_BYTES=90000,CE_MAX_DOCUMENT_BYTES=4000,
                RATE_LIMIT_RETRIES=0,PROVIDER_RPM=0,PROVIDER_SOFT_TPM=0,PROVIDER_CONCURRENCY=1)
            plan = dict(source_sha256={},input_sha256={},provider_settings=settings,
                provider_url_sha256=hashlib.sha256(b'https://example.invalid/rerank').hexdigest(),
                cohorts=dict(regression=dict(inventory=str(inventory),requests=requests,
                    search_ids=['s'],runtime_source=str(Path(__file__).resolve().parents[1]),runtime_sha256={})))
            (root/'plan.json').write_text(json.dumps(plan))
            bodies = []
            async def post(client,url,**kwargs):
                bodies.append(copy.deepcopy(kwargs['json']))
                if fail_second and len(bodies)==2:
                    raise httpx.ConnectError('offline fixture')
                return SimpleNamespace(raise_for_status=lambda:None,
                    json=lambda:dict(results=[dict(index=0,relevance_score=.8)],usage=dict(total_tokens=usage)))
            with patch.multiple(config,**{k:getattr(config,k) for k in settings}), \
                    patch.multiple(config,CE_API_URL='https://example.invalid/rerank',CE_API_KEY='test'), \
                    patch('httpx.AsyncClient.post',post),patch('socket.socket.connect',side_effect=AssertionError('Offline')), \
                    patch('os.getloadavg',return_value=(0,0,0)):
                await execute(SimpleNamespace(experiment=root,cohort='regression'))
            return (list(json_records(root/'regression-selections.jsonl'))[0],
                    list(json_records(root/'regression-requests.jsonl')),
                    list(json_records(root/'regression-ledger.jsonl')),bodies)

    async def test_real_adapter_complete_scores_preserve_main_prefix(self):
        row,events,ledger,bodies = await self.trial()
        self.assertEqual(row['proposed_main_ids'],['original','winner'])
        self.assertEqual(row['provider_calls'],2)
        self.assertEqual(row['reported_tokens'],20)
        self.assertEqual([r['status'] for r in events],['dispatch','ok','dispatch','ok'])
        self.assertEqual(len(ledger),2)
        self.assertEqual([b['documents'] for b in bodies],[['first'],['second']])

    async def test_second_batch_failure_discards_partial_scores_and_keeps_display(self):
        row,events,ledger,bodies = await self.trial(fail_second=True)
        self.assertEqual(row['scores'],{})
        self.assertEqual(row['proposed_main_ids'],['original'])
        self.assertTrue(row['original_display_preserved'])
        self.assertEqual(row['decision']['status'],'fallback')
        self.assertEqual(row['provider_calls'],2)
        self.assertEqual(events[-1]['status'],'error')
        self.assertEqual(ledger[-1]['fields']['status'],'error')

    async def test_excess_reported_usage_cannot_nominate(self):
        row,events,ledger,bodies = await self.trial(usage=10000)
        self.assertEqual(row['decision']['status'],'fallback')
        self.assertEqual(row['proposed_main_ids'],['original'])
        self.assertEqual(row['reported_tokens'],10000)
        self.assertEqual(row['provider_calls'],1)


if __name__ == '__main__':
    unittest.main()
