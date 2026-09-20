"""Listwise recovery and source-aware selection; deterministic, no providers."""
import asyncio
import copy
import os
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank as cascade, config, cross_encoder, llm, schemas, search_pipeline as search
from selftest_graph_cascade import documents_in, permutation


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings=patch.multiple(config, CASCADE_COARSE_LIMIT=50, CASCADE_FINE_LIMIT=12,
            CASCADE_LLM_LIMIT=10, CE_BATCH_SIZE=8, RERANK_REPAIR_MAX_CALLS=1)
        settings.start()
        self.addCleanup(settings.stop)

    def req(self):
        return schemas.SearchRequest(user_id='u', query='Which activity fits?',
            options=['A. Films', 'B. Cycling', 'C. Cooking', 'D. Music'], top_k=50)

    def rows(self, count=20):
        return [dict(id=f'm{i}', content=f'Evidence {i}', _rank_text=f'Evidence {i}',
                     _fused=1-i/(count+1)) for i in range(count)]

    async def ce(self, query, documents, **kwargs):
        return [.2+int(re.search(r'Evidence (\d+)', text)[1])/100 for text in documents]

    async def identity(self, prompt, *args, **kwargs):
        return permutation(len(documents_in(prompt)))

    async def test_pipeline_filter_delegates_to_cascade_once(self):
        rows, plan = self.rows(2), {}
        with patch.object(cascade, 'rank', AsyncMock(return_value=rows)) as call:
            result=await search._filter_rerank(self.req(), plan, rows)
        call.assert_awaited_once()
        self.assertIs(result, rows)
        self.assertIs(call.call_args.args[1], plan)
        self.assertIs(call.call_args.args[2], rows)

    async def test_schema_requires_full_permutation_and_success_uses_one_listwise_call(self):
        for size in (1, 2, 9):
            with self.subTest(size=size):
                calls=[]
                async def rank(prompt, *args, **kwargs):
                    texts=documents_in(prompt)
                    calls.append(kwargs['stage'])
                    shape=kwargs['schema']['properties']['ranking']
                    self.assertEqual((shape['minItems'], shape['maxItems']), (size,size))
                    self.assertNotIn('scores', kwargs['schema']['properties'])
                    return permutation(len(texts))
                with patch.object(cross_encoder, 'rerank', side_effect=self.ce), \
                        patch.object(llm, 'complete_json', side_effect=rank):
                    plan={}
                    ranked=await cascade.rank(self.req(), plan, self.rows(size))
                self.assertEqual(len(search._select_evidence(self.req(),plan,ranked)),size)
                self.assertEqual(calls,['search.listwise'])

    async def test_format_repair_keeps_ce_results_and_applies_repaired_permutation(self):
        calls,expected=[],[]
        async def rank(prompt, *args, **kwargs):
            calls.append(kwargs['stage'])
            texts=documents_in(prompt)
            if kwargs['stage']=='search.listwise':
                return {'ranking':[0], 'irrelevant':[], 'groups':[]}
            expected.extend(re.search(r'\[id: ([^\]]+)\]', text)[1] for text in reversed(texts))
            return permutation(len(texts),reverse=True)
        plan={}
        with patch.object(cross_encoder,'rerank',side_effect=self.ce) as ce, \
                patch.object(llm,'complete_json',side_effect=rank):
            ranked=await cascade.rank(self.req(),plan,self.rows(4))
        result=search._select_evidence(self.req(),plan,ranked)
        self.assertEqual([c['id'] for c in result],expected)
        self.assertEqual(ce.call_count,1)
        self.assertEqual(calls,['search.listwise','search.listwise.repair'])
        self.assertEqual(plan['_rerank_status'],'recovered')

    async def test_raw_negative_and_zero_ce_logits_do_not_override_valid_listwise_order(self):
        values={'m0':-12.,'m1':0.,'m2':7.,'m3':-3.}
        expected=[]
        async def ce(query,documents,**kwargs):
            return [values['m'+re.search(r'Evidence (\d+)',text)[1]] for text in documents]
        async def rank(prompt,*args,**kwargs):
            texts=documents_in(prompt)
            expected.extend(re.search(r'\[id: ([^\]]+)\]',text)[1] for text in reversed(texts))
            return permutation(len(texts),reverse=True)
        plan={}
        with patch.object(cross_encoder,'rerank',side_effect=ce), \
                patch.object(llm,'complete_json',side_effect=rank):
            ranked=await cascade.rank(self.req(),plan,self.rows(4))
        result=search._select_evidence(self.req(),plan,list(reversed(ranked)))
        self.assertEqual([c['id'] for c in result],expected)
        self.assertEqual({c['id']:c['_final'] for c in result},values)

    async def test_all_explicitly_irrelevant_does_not_refill_from_coarse_tail(self):
        async def rank(prompt,*args,**kwargs):
            count=len(documents_in(prompt))
            return permutation(count,irrelevant=range(count))
        plan={}
        with patch.object(cross_encoder,'rerank',side_effect=self.ce), \
                patch.object(llm,'complete_json',side_effect=rank):
            ranked=await cascade.rank(self.req(),plan,self.rows(70))
        self.assertEqual(search._select_evidence(self.req(),plan,ranked),[])

    async def test_rules_survive_when_all_ordinary_evidence_is_irrelevant(self):
        rows=self.rows(4)
        rule=dict(id='rule',content='Do not recommend my forgotten hobby.',type='rule',_user_rule=True,_fused=-99)
        rows.append(rule)
        async def rank(prompt,*args,**kwargs):
            self.assertNotIn('forgotten hobby',prompt)
            count=len(documents_in(prompt))
            return permutation(count,irrelevant=range(count))
        plan={}
        with patch.object(cross_encoder,'rerank',side_effect=self.ce), \
                patch.object(llm,'complete_json',side_effect=rank):
            ranked=await cascade.rank(self.req(),plan,rows)
        result=search._select_evidence(self.req(),plan,ranked)
        self.assertEqual([c['id'] for c in result],['rule'])
        self.assertIsNone(result[0].get('_final'))

    async def test_rules_are_capped_by_topical_overlap_with_the_question(self):
        rows=self.rows(4)
        rules=[dict(id='r_cycling',content='The user asked the assistant to forget that they took up cycling.',type='rule',_user_rule=True,_fused=.1),
               dict(id='r_tax',content='The user asked the assistant to forget their tax bracket.',type='rule',_user_rule=True,_fused=.9),
               dict(id='r_cooking',content='The user asked the assistant to forget that they love cooking.',type='rule',_user_rule=True,_fused=.2),
               dict(id='r_pet',content='The user asked the assistant to forget their cat allergy.',type='rule',_user_rule=True,_fused=.8),
               dict(id='r_films',content='The user asked the assistant to forget their favourite films.',type='rule',_user_rule=True,_fused=.05)]
        plan={}
        with patch.object(config,'PACKET_RULE_LIMIT',3), \
                patch.object(cross_encoder,'rerank',side_effect=self.ce), \
                patch.object(llm,'complete_json',side_effect=self.identity):
            ranked=await cascade.rank(self.req(),plan,rows+rules)
        result=search._select_evidence(self.req(),plan,ranked)
        kept=[c['id'] for c in result if c.get('type')=='rule']
        self.assertEqual(sorted(kept),['r_cooking','r_cycling','r_films'])
        self.assertEqual(sorted(plan['_cascade']['rule_cap']['kept_ids']),sorted(kept))
        self.assertEqual({d['id'] for d in plan['_cascade']['rule_cap']['dropped']},{'r_tax','r_pet'})
        self.assertEqual({o['id'] for o in plan['_cascade']['omitted'] if o['stage']=='rules'},{'r_tax','r_pet'})
        self.assertFalse(any(c['_cascade_selected'] for c in ranked if c['id'] in ('r_tax','r_pet')))
        self.assertEqual(len([c for c in result if c.get('type')!='rule']),4)

    async def test_rule_cap_zero_keeps_every_rule(self):
        rules=[dict(id=f'r{i}',content=f'Forget rule {i}.',type='rule',_user_rule=True,_fused=0) for i in range(6)]
        plan={}
        with patch.object(config,'PACKET_RULE_LIMIT',0), \
                patch.object(cross_encoder,'rerank',side_effect=self.ce), \
                patch.object(llm,'complete_json',side_effect=self.identity):
            ranked=await cascade.rank(self.req(),plan,self.rows(2)+rules)
        result=search._select_evidence(self.req(),plan,ranked)
        self.assertEqual(len([c for c in result if c.get('type')=='rule']),6)
        self.assertEqual(plan['_cascade']['rule_cap']['dropped'],[])

    async def test_top_fused_candidates_keep_a_seat_when_ce_scores_them_low(self):
        rows=self.rows(20)
        async def ce(query,documents,**kwargs):
            # The best fused hit gets the worst CE score; everything else follows the fused order.
            indices=[int(re.search(r'Evidence (\d+)',text)[1]) for text in documents]
            return [-5. if i==0 else 1-i/100 for i in indices]
        plan={}
        with patch.object(config,'CASCADE_FINE_LIMIT',5), patch.object(config,'CASCADE_LLM_LIMIT',5), \
                patch.object(config,'CASCADE_FUSED_RESERVE',1), \
                patch.object(cross_encoder,'rerank',side_effect=ce), \
                patch.object(llm,'complete_json',side_effect=self.identity):
            ranked=await cascade.rank(self.req(),plan,rows)
        fine=plan['_cascade']['fine']
        self.assertIn('m0',fine['selected_ids'])
        self.assertEqual(fine['fused_reserved_ids'],['m0'])
        self.assertIn('m0',plan['_cascade']['listwise']['candidate_ids'])
        reasons={r['candidate_id']:r['reasons'] for r in fine['selection']['reservations']}
        self.assertEqual(reasons['m0'],['fused_head'])
        with patch.object(config,'CASCADE_FINE_LIMIT',5), patch.object(config,'CASCADE_LLM_LIMIT',5), \
                patch.object(config,'CASCADE_FUSED_RESERVE',0), \
                patch.object(cross_encoder,'rerank',side_effect=ce), \
                patch.object(llm,'complete_json',side_effect=self.identity):
            plan={}
            await cascade.rank(self.req(),plan,self.rows(20))
        self.assertNotIn('m0',plan['_cascade']['fine']['selected_ids'])

    async def test_ce_and_listwise_failure_have_a_bounded_unknown_fallback(self):
        plan={}
        with patch.object(config,'EVIDENCE_FALLBACK_ITEMS',3), \
                patch.object(cross_encoder,'rerank',AsyncMock(side_effect=ConnectionError('offline'))), \
                patch.object(llm,'complete_json',AsyncMock(side_effect=ConnectionError('offline'))):
            ranked=await cascade.rank(self.req(),plan,self.rows(30))
            result=search._select_evidence(self.req(),plan,ranked)
        self.assertTrue(result)
        self.assertLessEqual(len(result),3)
        self.assertTrue(all(c.get('_final') is None for c in result))
        self.assertTrue(all(c['_score_kind']=='graph_fallback' for c in result))

    async def test_successful_listwise_is_not_capped_as_unknown_when_ce_is_offline(self):
        plan={}
        with patch.object(config,'EVIDENCE_FALLBACK_ITEMS',3), \
                patch.object(cross_encoder,'rerank',AsyncMock(side_effect=ConnectionError('offline'))), \
                patch.object(llm,'complete_json',side_effect=self.identity):
            ranked=await cascade.rank(self.req(),plan,self.rows(30))
            result=search._select_evidence(self.req(),plan,ranked)
        self.assertEqual(len(result),10)
        self.assertTrue(all(c.get('_final') is None and c['_score_kind']=='listwise' for c in result))

    async def test_outer_cancellation_is_not_swallowed(self):
        started=asyncio.Event()
        async def ce(*args,**kwargs):
            started.set()
            await asyncio.Event().wait()
        with patch.object(cross_encoder,'rerank',side_effect=ce), \
                patch.object(llm,'complete_json',AsyncMock()) as rank:
            task=asyncio.create_task(cascade.rank(self.req(),{},self.rows()))
            await asyncio.wait_for(started.wait(),1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        rank.assert_not_awaited()

    async def test_format_repair_cannot_exceed_the_shared_call_budget(self):
        paid=[]
        async def ce(query,documents,**kwargs):
            budget.current.get().before_call()
            paid.append(kwargs['stage'])
            return await self.ce(query,documents)
        async def rank(prompt,*args,**kwargs):
            budget.current.get().before_call()
            paid.append(kwargs['stage'])
            return {'ranking':[], 'irrelevant':[], 'groups':[]}
        plan={}
        with budget.scope(seconds=30,calls=2,tokens=64000), \
                patch.object(cross_encoder,'rerank',side_effect=ce), \
                patch.object(llm,'complete_json',side_effect=rank):
            ranked=await cascade.rank(self.req(),plan,self.rows(4))
        self.assertEqual(paid,['search.cross_encoder.batch_1','search.listwise'])
        result=search._select_evidence(self.req(),plan,ranked)
        self.assertEqual([c['id'] for c in result],['m3','m2','m1','m0'])

    async def test_wrapped_format_failure_repairs_but_wrapped_connection_error_does_not(self):
        for cause,recoverable in ((ValueError('bad JSON'),True),(ConnectionError('offline'),False)):
            with self.subTest(cause=type(cause).__name__):
                calls=[]
                async def rank(prompt,*args,**kwargs):
                    calls.append(kwargs['stage'])
                    if len(calls)==1:
                        raise llm.LLMError('provider wrapper') from cause
                    return await self.identity(prompt)
                with patch.object(cross_encoder,'rerank',side_effect=self.ce), \
                        patch.object(llm,'complete_json',side_effect=rank):
                    plan={}
                    await cascade.rank(self.req(),plan,self.rows(2))
                self.assertEqual(len(calls),2 if recoverable else 1)
                self.assertEqual(plan['_cascade']['listwise']['status'],'recovered' if recoverable else 'fallback')

    async def test_model_stages_preserve_prepared_source_payloads(self):
        quote='Claire wrote this draft; it does not describe the user.'
        source=dict(request_id='r',message_index=0,role='assistant',
            content=quote, content_span=dict(start=0,end=len(quote),original_length=len(quote)))
        rows=self.rows(4)
        for row in rows:
            row['_rank_text']+=' [quoted assistant draft] '+source['content']
            row['_packet_item']=dict(id=row['id'],content=row['_rank_text'],sources=[copy.deepcopy(source)])
        originals={row['id']:copy.deepcopy(row['_packet_item']) for row in rows}
        async def ce(query,documents,**kwargs):
            self.assertTrue(all(source['content'] in text for text in documents))
            return await self.ce(query,documents)
        async def rank(prompt,*args,**kwargs):
            self.assertTrue(all(source['content'] in text for text in documents_in(prompt)))
            return await self.identity(prompt)
        with patch.object(cross_encoder,'rerank',side_effect=ce), \
                patch.object(llm,'complete_json',side_effect=rank):
            plan={}
            ranked=await cascade.rank(self.req(),plan,rows)
        for row in search._select_evidence(self.req(),plan,ranked):
            self.assertEqual(row['_packet_item'],originals[row['id']])


class SelectionTests(unittest.TestCase):
    def test_priority_is_derived_from_visible_sources_not_just_memory_type(self):
        class Snapshot:
            def get_amus_by_ids(self, ids, **kwargs):
                return [dict(id=mid, content='The user asked about movies.', type='preference') for mid in ids]

            def sources_for_amu(self, mid):
                source = dict(request_id=mid, message_index=0)
                if mid == 'user':
                    return [dict(source, role='user', content='Why do older movies use so much subtext?')]
                if mid == 'editorial':
                    return [dict(source, role='user', content='Please refine this note.')]
                return [dict(source, role='assistant', content='Movies use subtext to communicate indirectly.')]
        # Different source states prevent same-claim coalescing in this fixture.
        rows = [dict(id=mid, content=mid, type='fact') for mid in ('user', 'assistant', 'editorial')]
        with patch.object(search, '_coalesce_preferences', side_effect=lambda rows, plan: rows):
            prepared = search._prepare_candidates(Snapshot(),
                schemas.SearchRequest(user_id='u', query='movies'), {}, rows)
        by_id = {c['id']: c['_selection_evidence'] for c in prepared}
        self.assertTrue(by_id['user']['user_source'])
        self.assertFalse(by_id['assistant']['user_source'])
        self.assertFalse(by_id['editorial']['user_source'])

    def select(self, rows, limit=2):
        plan = {'_rerank': [dict(id=c['id'], reason='kept') for c in rows]}
        with patch.object(config, 'EVIDENCE_FALLBACK_ITEMS', limit):
            selected = search._select_evidence(schemas.SearchRequest(user_id='u', query='films'), plan, rows)
        return [c['id'] for c in selected], plan

    def row(self, mid, sources=(), user=False, score=.1, **kw):
        return dict(id=mid, content=mid, _final=score,
                    _selection_evidence=dict(user_source=user, source_ids=list(sources)), **kw)

    def test_user_source_wins_tie_against_assistant_explanations(self):
        rows = [self.row(f'assistant{i}', [f'a{i}']) for i in range(10)]
        rows.append(self.row('user_old_movies', ['u1'], True))
        ids, _ = self.select(rows)
        self.assertEqual(ids[0], 'user_old_movies')

    def test_distinct_sources_get_slots_before_repeats_at_same_score(self):
        rows = [self.row('fact', ['u1'], True), self.row('episode', ['u1'], True),
                self.row('independent', ['u2'], True)]
        ids, _ = self.select(rows)
        self.assertEqual(ids, ['fact', 'independent'])

    def test_same_source_distinct_claims_not_deleted(self):
        rows = [self.row('fact', ['u1'], True), self.row('episode', ['u1'], True)]
        ids, _ = self.select(rows)
        self.assertEqual(ids, ['fact', 'episode'])

    def test_source_ties_do_not_override_model_scores_or_zero_rejection(self):
        rows = [self.row('higher', ['a1'], score=.3), self.row('user', ['u1'], True),
                self.row('zero', ['u2'], True, score=0)]
        ids, _ = self.select(rows, 1)
        self.assertEqual(ids, ['higher'])

    def test_rules_survive_without_using_weak_evidence_capacity(self):
        rows = [self.row('assistant'), self.row('user', ['u1'], True),
                self.row('rule', score=0, _user_rule=True)]
        ids, _ = self.select(rows, 1)
        self.assertEqual(set(ids), {'user', 'rule'})


if __name__ == '__main__':
    unittest.main()
