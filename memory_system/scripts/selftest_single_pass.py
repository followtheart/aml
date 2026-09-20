"""Evidence-loss regressions for the shorter search/answer path, no providers."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from listwise_fixture import from_scores
from app import answer_context, cross_encoder, eval_scoring, schemas, search_pipeline as search, store
from app.embeddings import embed
import local_eval


class SinglePassTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(':memory:')

    def tearDown(self):
        self.st.conn.close()

    async def memory(self, content, sources=(), **kwargs):
        vec = (await embed([content]))[0]
        aid = self.st.insert_amu(user_id='u', session_id='s', content=content,
                                 embedding=vec, **kwargs)
        if sources:
            req = schemas.AddRequest(request_id=aid, user_id='u', session_id='s',
                                     messages=[schemas.Message(role=role, content=text)
                                               for role, text in sources])
            self.st.save_messages(req)
            self.st.link_sources(aid, aid, list(range(len(sources))))
        return aid

    async def rank(self, prompt, *args, **kwargs):
        self.assertEqual(kwargs.get('stage'), 'search.listwise')
        ids = re.findall(r'^\d+:', prompt, re.M)
        # Deliberately unhelpful ranking must not become another deletion gate.
        return from_scores([0.1 for _ in ids])

    async def search(self, query='morning routine', **kwargs):
        with patch.object(search, '_understand', AsyncMock(return_value={
                'intent': 'preference', 'entities': []})), patch.object(
                search.llm, 'complete_json', side_effect=self.rank):
            return await search.run_search(self.st, schemas.SearchRequest(
                user_id='u', query=query, **kwargs))

    async def test_named_facts_survive_wording_classifier_and_low_rerank(self):
        morning = await self.memory('Daniel describes his morning routine of stretches and breathing.',
                                    [('user', 'I start each morning with stretches and breathing.')])
        pollen = await self.memory('Daniel experiences discomfort from spring pollen.',
                                   [('user', 'I have difficulty staying outside when pollen is high.')])
        response = await self.search()
        self.assertEqual({x.id for x in response.data}, {morning, pollen})
        body = answer_context.build([x.model_dump() for x in response.data])
        self.assertIn('I start each morning', body)
        self.assertIn('I have difficulty staying outside', body)
        self.assertEqual(response.verification_status, 'not_run')
        self.assertEqual(response.evidence_status, 'retrieved')

    async def test_assistant_draft_span_survives_with_role_and_uncertainty(self):
        aid = await self.memory('A draft about a weekend at home.', [
            ('user', 'Please polish this personal draft.'),
            ('assistant', 'On Saturdays I set dough to rise and bake fresh bread at home.')],
            type='episode', epistemic_status='inferred')
        result = await self.search('weekend baking bread')
        item = next(x for x in result.data if x.id == aid)
        self.assertIn('set dough to rise', item.content)
        self.assertIn('role: assistant', item.content)
        self.assertIn('evidence: inferred', item.content)

    async def test_third_party_name_is_visible_to_answer(self):
        await self.memory('The user asked to polish a story.', [
            ('user', 'Claire struggled to combine remote teaching with parenting.')], type='episode')
        response = await self.search('work and family')
        qa = {'question': 'How can I manage my workload?', 'scoring': 'choice',
              'qa_type': 'single_choice', 'options': ['A. Since you taught remotely', 'B. Take breaks']}
        prompt = eval_scoring.answer_prompt(qa, [x.model_dump() for x in response.data])
        self.assertIn('Claire struggled', prompt)
        self.assertIn('Distinguish the speaker', prompt)

    async def test_forget_rule_survives_topk_even_when_ranked_last(self):
        rule = await self.memory('The user asked to forget their interest in electronic festivals.',
                                 [('user', 'Please forget that I enjoy electronic festivals.')], type='rule')
        fact = await self.memory('The user enjoys summer events.', type='preference')
        by_id = {row['id']: row for row in self.st.get_amus_by_ids([fact, rule])}
        rows = [by_id[fact], by_id[rule]]
        packet, _, manifest = search._pack_evidence(self.st, schemas.SearchRequest(
            user_id='u', query='event ideas', top_k=1), {}, rows, '2026-09-17T00:00:00Z')
        self.assertEqual([x['id'] for x in packet], [rule, fact])
        self.assertIn('do NOT use or recommend', packet[0]['content'])
        self.assertEqual(manifest['constraint_count'], 1)
        self.assertEqual(manifest['evidence_count'], 1)
        self.assertEqual(manifest['omitted'], [])

    async def test_core_does_not_reinject_unranked_assistant_advice(self):
        noise = await self.memory('The assistant suggested playing fetch with a dog.',
                                  [('assistant', 'Try playing fetch.')], type='rule')
        target = await self.memory('The user owns a vegetable garden.')
        packet, _, _ = search._pack_evidence(self.st, schemas.SearchRequest(
            user_id='u', query='garden', top_k=1), {'intent': 'preference'},
            self.st.get_amus_by_ids([target]), '2026-09-17T00:00:00Z')
        self.assertEqual([x['id'] for x in packet], [target])
        self.assertNotIn(noise, [x['id'] for x in packet])

    async def test_option_premises_are_queries_but_never_inserted_as_evidence(self):
        target = await self.memory('The user discussed record pressings.', type='preference')
        options = ['A. Since you collect rare vinyl, try a record marketplace',
                   'B. Since you own a spaceship, try an auction']
        embedding = AsyncMock(wraps=search.embed)
        plan = {'intent': 'fact', 'expanded_queries': ['generic ' + str(i) for i in range(20)]}
        with patch.object(search, 'embed', embedding), patch.object(
                search, '_understand', AsyncMock(return_value=plan)), patch.object(
                search.llm, 'complete_json', side_effect=self.rank):
            response = await search.run_search(self.st, schemas.SearchRequest(
                user_id='u', query='Where can I buy unique items?', options=options))
        queries = embedding.call_args.args[0]
        self.assertEqual(queries[1:3], ['Since you collect rare vinyl', 'Since you own a spaceship'])
        self.assertLessEqual(len(queries), 6)
        profile_route = next(r for r in plan['_routes'] if r['channel'] == 'profile_rule')
        self.assertNotIn(target, [c['id'] for c in profile_route['candidates']])
        self.assertIn(target, [x.id for x in response.data])
        self.assertNotIn('spaceship', answer_context.build([x.model_dump() for x in response.data]))

    async def test_default_choice_has_plan_ce_listwise_verified_answer_stages(self):
        await self.memory('The user lives in Kansas.')
        stages = []
        original_ce = cross_encoder.rerank
        async def ce(query, documents, **kwargs):
            stages.append(kwargs['stage'])
            return await original_ce(query, documents, **kwargs)
        async def json_call(prompt, *args, **kwargs):
            stages.append(kwargs['stage'])
            if kwargs['stage'] == 'search.understand':
                return {'intent': 'preference', 'entities': []}
            return await self.rank(prompt, *args, **kwargs)
        answer_packets = []
        async def answer(actual_qa, packet, diagnostics):
            stages.append('answer_choice.answer')
            self.assertIs(actual_qa, qa)
            answer_packets.append(packet)
            diagnostics.update(answer_policy=eval_scoring.answer_choice.VERSION)
            return 'A'
        qa = {'question': 'Weekend ideas?', 'options': ['A. A local event', 'B. Travel'],
              'scoring': 'choice', 'qa_type': 'single_choice', 'gold_labels': ['A']}
        with tempfile.TemporaryDirectory() as temp, patch.object(search.config, 'SEARCH_DEBUG_LOG',
                str(Path(temp) / 'trace.jsonl')), patch.object(search.llm, 'complete_json',
                side_effect=json_call), patch.object(eval_scoring.answer_choice, 'answer', side_effect=answer), \
                patch.object(cross_encoder, 'rerank', side_effect=ce):
            response = await search.run_search(self.st, schemas.SearchRequest(
                user_id='u', query=qa['question'], options=qa['options']))
            _, score, diagnostics = await eval_scoring.evaluate(qa, [x.model_dump() for x in response.data])
            trace = json.loads((Path(temp) / 'trace.jsonl').read_text())
        self.assertEqual(stages, ['search.understand', 'search.cross_encoder.batch_1',
                                  'search.listwise', 'answer_choice.answer'])
        self.assertEqual(score, 1.0)
        self.assertEqual(diagnostics['answer_policy'], eval_scoring.answer_choice.VERSION)
        self.assertEqual(answer_packets, [[x.model_dump() for x in response.data]])
        self.assertEqual(answer_context.build(answer_packets[0]),
                         answer_context.build(trace['returned']))
        self.assertEqual(response.coverage_manifest['rerank_status'], 'ok')
        recovery = response.coverage_manifest['cascade']['listwise']['recovery']
        self.assertEqual(recovery['status'], 'not_needed')
        self.assertEqual(recovery['reviewed_ids'], [])
        self.assertEqual(recovery['restored_ids'], [])
        self.assertEqual(len(trace['rounds']), 1)
        self.assertEqual(trace['pipeline'], search.run_metadata.SEARCH_POLICY)
        self.assertEqual(trace['versions']['search_policy'], search.run_metadata.SEARCH_POLICY)
        self.assertEqual(trace['versions']['settings']['SEARCH_DEADLINE_SECONDS'], search.config.SEARCH_DEADLINE_SECONDS)

    async def test_no_keyexp_ablation_uses_current_planner_signature(self):
        await self.memory('The user lives in Kyoto.')
        args = type('Args', (), dict(no_governance=False, no_graph=False,
                                    no_rerank=True, no_keyexp=True))()
        with patch.object(search, '_understand', search._understand), \
                patch.object(search, '_filter_rerank', search._filter_rerank):
            local_eval.apply_ablations(args)
            plan = await search._understand(self.st, schemas.SearchRequest(user_id='u', query='q'), None)
            response = await search.run_search(self.st, schemas.SearchRequest(user_id='u', query='Kyoto'))
        self.assertEqual(plan['expanded_queries'], ['q'])
        self.assertTrue(response.data)
        self.assertEqual(response.coverage_manifest['rerank_status'], 'not_run')


if __name__ == '__main__':
    unittest.main()
