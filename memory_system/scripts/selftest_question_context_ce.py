"""Question-only CE: immutable prefix, provenance, budgets, failure and replay."""
import asyncio
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, evidence_packet, question_context_ce as context, schemas, search_pipeline
from selftest_fine_context_lexical import candidate


class QuestionContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.flags = dict(PACKET_QUESTION_CONTEXT_CE=True, PACKET_FINE_CONTEXT_RESERVE=True,
            PACKET_FINE_CONTEXT_5K=True, PACKET_FINE_CONTEXT_LEXICAL=True,
            PACKET_FINE_CONTEXT_FILL=False, PACKET_SOURCE_QUERY_CONTEXT=False, PACKET_ADJACENT_CONTEXT=False)
        self.config = patch.multiple(config, **self.flags)
        self.config.start()
        self.addCleanup(self.config.stop)
        self.base = schemas.SearchItem(id='base', content='Accepted evidence.', packet_hash_version=4).model_dump()
        self.fine = [candidate('one','Walking?'), candidate('two','Gardening?'),
            candidate('tea','Tea preferences?'), candidate('film','Why did older movies imply events?'),
            candidate('last','A different question.')]
        self.anchor = dict(id='base', _packet_item=self.base)
        self.req = schemas.SearchRequest(user_id='u',query='Tea?',options=['A fictional personal premise.'],
            top_k=10,evidence_token_budget=9000)
        self.plan = dict(_cascade=dict(fine=dict(selected_ids=['base']+[c['id'] for c in self.fine])),
            _selection=dict(selected_ids=['base']),_coverage_requirements=[])

    def prefix(self, plan=None, pool=None):
        return search_pipeline._pack_evidence(None,self.req,plan or self.plan,[self.anchor],None,pool or self.fine)

    async def run_extra(self, prefix, mock, plan=None, pool=None):
        with patch.object(context.cross_encoder,'rerank',mock):
            return await search_pipeline._append_question_context(self.req,plan or self.plan,
                pool or self.fine,*prefix)

    async def test_exact_question_whole_sources_prefix_and_hash_accounting(self):
        prefix=self.prefix();before=copy.deepcopy((prefix,self.fine))
        mock=AsyncMock(return_value=[.1,.9])
        with budget.scope(seconds=60,calls=1,tokens=9000):
            result,digest,manifest=await self.run_extra(prefix,mock)
        query,documents=mock.call_args.args
        self.assertEqual(query,self.req.query)
        self.assertTrue(all('Generated summary' not in d and 'fictional personal premise' not in d for d in documents))
        self.assertEqual(result[-1]['id'],'last')
        self.assertEqual(result[-1]['sources'],self.fine[-1]['_packet_item']['sources'])
        self.assertFalse(result[-1]['personal_evidence']);self.assertEqual(result[-1]['coverage_ids'],[])
        self.assertEqual(result[-1]['score'],.2)
        self.assertEqual([dict(p,packet_hash=None) for p in result[:-1]],
            [dict(p,packet_hash=None) for p in prefix[0]])
        self.assertEqual(evidence_packet.digest(result),digest)
        self.assertTrue(all(p['packet_hash']==digest for p in result))
        self.assertEqual(manifest['token_upper_bound'],sum(len(p['content'].encode()) for p in result)+len(result)-1)
        self.assertEqual(manifest['evidence_count'],len(result))
        self.assertEqual((prefix,self.fine),before)
        self.assertEqual(context.replay(*prefix,self.fine,manifest['conversation_context']['question_ce'],
            user_id='u',top_k=10,query=self.req.query),(result,digest,manifest))

    async def test_disabled_or_incompatible_features_make_no_calls(self):
        for key,value in [('PACKET_QUESTION_CONTEXT_CE',False),('PACKET_FINE_CONTEXT_RESERVE',False),
                ('PACKET_FINE_CONTEXT_5K',False),('PACKET_FINE_CONTEXT_LEXICAL',False),
                ('PACKET_FINE_CONTEXT_FILL',True),('PACKET_SOURCE_QUERY_CONTEXT',True)]:
            with self.subTest(key=key),patch.object(config,key,value):
                prefix=self.prefix();mock=AsyncMock()
                with budget.scope(seconds=60,calls=1,tokens=9000):
                    self.assertEqual(await self.run_extra(prefix,mock),prefix)
                mock.assert_not_called()

    async def test_call_token_reservations_and_deadline_skip_without_displacement(self):
        for kind in ('calls','reserved_calls','tokens','reserved_tokens','deadline','missing'):
            prefix=self.prefix();mock=AsyncMock()
            with self.subTest(kind=kind),budget.scope(seconds=60,calls=1,tokens=9000) as limits:
                if kind=='calls':limits.calls=1
                if kind=='reserved_calls':limits.reserved_calls=1
                if kind=='tokens':limits.tokens=9000
                if kind=='reserved_tokens':limits.reserved_tokens=9000
                if kind=='deadline':limits.deadline-=55
                token=budget.current.set(None) if kind=='missing' else None
                try:result,digest,manifest=await self.run_extra(prefix,mock)
                finally:
                    if token is not None:budget.current.reset(token)
            self.assertEqual((result,digest),prefix[:2]);mock.assert_not_called()
            self.assertEqual(manifest['conversation_context']['question_ce']['status'],
                'missing_budget' if kind=='missing' else 'budget_skipped')

    async def test_provider_failure_falls_back_and_cancellation_propagates(self):
        prefix=self.prefix()
        for exc in (TimeoutError(),RuntimeError('provider failure')):
            with budget.scope(seconds=60,calls=1,tokens=9000):
                result,digest,manifest=await self.run_extra(prefix,AsyncMock(side_effect=exc))
            self.assertEqual((result,digest),prefix[:2])
            self.assertEqual(manifest['conversation_context']['question_ce']['status'],'error')
        with budget.scope(seconds=60,calls=1,tokens=9000),self.assertRaises(asyncio.CancelledError):
            await self.run_extra(prefix,AsyncMock(side_effect=asyncio.CancelledError()))

    async def test_request_limits_and_no_capacity_never_clip_or_call(self):
        prefix=self.prefix()
        for key,value in [('CE_BATCH_SIZE',1),('CE_MAX_REQUEST_BYTES',1),('CE_MAX_DOCUMENT_BYTES',1)]:
            with patch.object(config,key,value),budget.scope(seconds=60,calls=1,tokens=9000):
                mock=AsyncMock();result,digest,_=await self.run_extra(prefix,mock)
            self.assertEqual((result,digest),prefix[:2]);mock.assert_not_called()
        for changes in (dict(top_k=len(prefix[0])),dict(evidence_token_budget=prefix[2]['token_upper_bound'])):
            req=self.req;self.req=self.req.model_copy(update=changes)
            try:
                with budget.scope(seconds=60,calls=1,tokens=9000):
                    mock=AsyncMock();new_prefix=self.prefix();result,digest,_=await self.run_extra(new_prefix,mock)
                self.assertEqual((result,digest),new_prefix[:2]);mock.assert_not_called()
            finally:self.req=req

    async def test_ineligible_sources_or_nonfine_candidates_never_reach_provider(self):
        prefix=self.prefix()
        for kind in ('foreign','invisible','stale','unscored','partial','assistant','constraint','duplicate','selected','nonfine','out_of_time'):
            bad=copy.deepcopy(self.fine[3]);plan=copy.deepcopy(self.plan)
            s=bad['_packet_item']['sources'][0]
            if kind=='foreign':s['user_id']='other'
            elif kind=='invisible':bad['_packet_item']['content']='Generated summary only'
            elif kind=='stale':bad['view_status']='stale'
            elif kind=='unscored':bad['_ce_score']=None
            elif kind=='partial':s['content_span']=dict(start=1,end=10,original_length=20)
            elif kind=='assistant':s['role']='assistant'
            elif kind=='constraint':bad['_packet_item']['is_constraint']=True
            elif kind=='duplicate':bad=copy.deepcopy(self.fine[0])
            elif kind=='selected':plan['_selection']['selected_ids'].append(bad['id'])
            elif kind=='nonfine':plan['_cascade']['fine']['selected_ids'].remove(bad['id'])
            elif kind=='out_of_time':
                bad.update(type='plan',temporal=dict(start='2020-01-01T00:00:00Z',end='2020-01-02T00:00:00Z'))
                plan['time_scope']={'from':'2025-01-01T00:00:00Z'}
            with self.subTest(kind=kind),budget.scope(seconds=60,calls=1,tokens=9000):
                mock=AsyncMock();result,digest,_=await self.run_extra(prefix,mock,plan=plan,pool=[bad])
            self.assertEqual((result,digest),prefix[:2]);mock.assert_not_called()

    async def test_replay_rejects_changed_requests_scores_and_budgets(self):
        prefix=self.prefix()
        with budget.scope(seconds=60,calls=1,tokens=9000):
            _,_,manifest=await self.run_extra(prefix,AsyncMock(return_value=[.1,.9]))
        original=manifest['conversation_context']['question_ce']
        for kind in ('candidate','document','query','score_count','nonfinite','budget','proposal'):
            trace=copy.deepcopy(original)
            if kind=='candidate':trace['candidate_ids'].reverse()
            elif kind=='document':trace['document_sha256'][0]='changed'
            elif kind=='query':trace['query_sha256']='changed'
            elif kind=='score_count':trace['scores'].pop()
            elif kind=='nonfinite':trace['scores'][0]=float('nan')
            elif kind=='budget':trace['budget_before']['max_calls']=0
            elif kind=='proposal':trace['proposal']['appended_ids']=[]
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                context.replay(*prefix,self.fine,trace,user_id='u',top_k=10,query=self.req.query)


if __name__=='__main__':unittest.main()
