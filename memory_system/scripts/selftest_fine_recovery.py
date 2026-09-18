"""Bounded CE retry and facet-aware unscored recovery, with no providers."""
import os
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch
import httpx

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank as cascade, config, cross_encoder, llm, schemas
from selftest_graph_cascade import documents_in, permutation


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings = patch.multiple(config, CASCADE_COARSE_LIMIT=50, CASCADE_FINE_LIMIT=12,
            CASCADE_LLM_LIMIT=10, CE_BATCH_SIZE=8, CE_MAX_REQUEST_BYTES=90000,
            CE_MAX_DOCUMENT_BYTES=4000, CE_TIMEOUT_SECONDS=1., CE_DEADLINE_SECONDS=10.,
            CE_RETRY_BATCHES=2, CE_RETRY_BATCH_SIZE=4, RERANK_CONCURRENCY=2,
            RERANK_MAX_PROMPT_BYTES=24000, create=True)
        settings.start()
        self.addCleanup(settings.stop)

    def rows(self, count=20):
        return [dict(id=f'm{i}', content=f'Evidence {i}', _rank_text=f'Evidence {i}',
                     _fused=1-i/100) for i in range(count)]

    async def identity(self, prompt, **kwargs):
        limits = budget.current.get()
        if limits:
            limits.before_call()
        return permutation(len(documents_in(prompt)))

    async def test_timeout_retries_only_failed_documents_in_smaller_bounded_batches(self):
        calls, failed = [], set()
        async def ce(query, docs, **kwargs):
            budget.current.get().before_call()
            calls.append((kwargs['stage'], list(docs)))
            if len(calls) == 1:
                failed.update(docs)
                raise TimeoutError('single CE request')
            return [.7] * len(docs)
        plan = {}
        with budget.scope(seconds=30, calls=12, tokens=64000) as limits, \
                patch.object(cross_encoder, 'rerank', side_effect=ce), \
                patch.object(llm, 'complete_json', side_effect=self.identity):
            result = await cascade.rank(schemas.SearchRequest(user_id='u',query='Evidence'), plan, self.rows())
        retries = [docs for stage, docs in calls if '.retry_' in stage]
        self.assertTrue(retries)
        self.assertLessEqual(len(retries), 2)
        self.assertTrue(all(len(docs) <= 4 for docs in retries))
        self.assertEqual({s for docs in retries for s in docs}, failed)
        self.assertTrue(all(c.get('_ce_score') == .7 for c in result))
        self.assertEqual(plan['_cascade']['cross_encoder']['status'], 'ok')
        self.assertEqual(limits.reserved_calls, 0)
        self.assertEqual(limits.reserved_tokens, 0)

    async def test_retry_cannot_consume_last_reserved_listwise_call(self):
        calls = []
        async def ce(query, docs, **kwargs):
            budget.current.get().before_call()
            calls.append(kwargs['stage'])
            raise TimeoutError('single CE request')
        plan = {}
        with budget.scope(seconds=30,calls=2,tokens=64000) as limits, \
                patch.object(cross_encoder,'rerank',side_effect=ce), \
                patch.object(llm,'complete_json',side_effect=self.identity):
            await cascade.rank(schemas.SearchRequest(user_id='u',query='Evidence'),plan,self.rows(8))
        self.assertEqual(calls, ['search.cross_encoder.batch_1'])
        self.assertEqual(plan['_cascade']['listwise']['status'], 'ok')
        self.assertEqual(limits.calls, 2)
        self.assertEqual(limits.reserved_calls, 0)

    async def test_http_read_timeout_uses_same_bounded_retry_as_request_timeout(self):
        original_client = httpx.AsyncClient
        sent = []
        async def handler(request):
            self.assertEqual(str(request.url), 'https://fine-recovery.example.test/v1/rerank')
            documents = json.loads(request.content)['documents']
            sent.append(documents)
            if len(sent) == 1:
                raise httpx.ReadTimeout('controlled read timeout', request=request)
            return httpx.Response(200,json={'results':[
                {'index':i,'relevance_score':.7} for i in range(len(documents))],
                'usage':{'total_tokens':20}})
        def client(*args,**kwargs):
            return original_client(*args,**kwargs,transport=httpx.MockTransport(handler))
        plan={}
        with budget.scope(seconds=30,calls=12,tokens=64000) as limits, \
                patch.multiple(config,FAKE=False,CE_API_URL='https://fine-recovery.example.test/v1/rerank',
                    CE_API_KEY='offline-test-key',CE_MODEL='offline-model',CE_API_FORMAT='cohere',
                    PROVIDER_RPM=0,PROVIDER_SOFT_TPM=0,RATE_LIMIT_RETRIES=0), \
                patch.object(httpx,'AsyncClient',side_effect=client), \
                patch.object(llm,'complete_json',side_effect=self.identity):
            result=await cascade.rank(schemas.SearchRequest(user_id='u',query='Evidence'),plan,self.rows(8))
        self.assertEqual([len(rows) for rows in sent],[8,4,4])
        self.assertTrue(all(row.get('_ce_score')==.7 for row in result))
        self.assertEqual(plan['_cascade']['cross_encoder']['status'],'ok')
        self.assertEqual((limits.reserved_calls,limits.reserved_tokens),(0,0))

    async def test_unscored_facet_rescue_does_not_spend_both_slots_on_same_topic(self):
        rows = self.rows()
        texts = ['I follow basketball.', 'I like basketball.', 'Basketball history.',
                 'Basketball teams.', 'I enjoy spices in soup.', 'Spices have cultural history.',
                 'Basketball scores.', 'Spice trade history.']
        for row, text in zip(rows,texts):
            row.update(content=text,_rank_text=text,
                _packet_item=dict(content=text,sources=[dict(request_id=row['id'],message_index=0,role='user',content=text)]))
        plan = {'_query_specs': [dict(text='basketball',origin='option_premise'),
                               dict(text='spices',origin='option_premise')]}
        async def ce(query, docs, **kwargs):
            if kwargs['stage']=='search.cross_encoder.batch_1':
                raise ConnectionError('unavailable batch')
            return [.8]*len(docs)
        with patch.object(cross_encoder,'rerank',side_effect=ce), \
                patch.object(llm,'complete_json',side_effect=self.identity):
            await cascade.rank(schemas.SearchRequest(user_id='u',query='Cooking soup'),plan,rows)
        admitted=set(plan['_cascade']['listwise']['candidate_ids'])
        self.assertIn('m4',admitted)
        self.assertTrue(any(mid in admitted for mid in ('m0','m1','m2','m3','m6')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
