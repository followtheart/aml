"""Contract, preservation and real cascade tests; no provider or quality claims."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import unittest
import tempfile
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, budget, cascade_rerank, config, cross_encoder, llm, schemas
from app import listwise_context as context, listwise_staged as staged


def row(i, text=None):
    source = dict(role='user', request_id='req', message_index=i,
                  content=text or f'I prefer item {i} for my current needs.')
    rendered = answer_context.with_evidence(f'Claim {i}', [source])
    return dict(id=f'm{i}', content=f'Claim {i}', _ce_score=1-i*.1, _fused=1-i*.1,
                _rank_text=rendered, _packet_item=dict(content=rendered, sources=[source]))


def decision(purpose='unrelated', refs=None, targets=None, owners=None):
    return dict(purpose=purpose, target_ids=targets or [], source_refs=refs or [], represented_by=(owners or [''])[0])


class Context(unittest.TestCase):
    def test_supplement_budget_uses_utf8_bytes_and_explicit_override(self):
        r = row(0, '证据。' * 1000)
        original = copy.deepcopy(r['_packet_item']['sources'])
        r['_packet_item']['sources'][0]['content'] = '证据。'
        r['_rank_text'] = ''
        with patch.object(config, 'LISTWISE_CONTEXT_MAX_BYTES', 12000):
            info, windows = context.prepare(r, original)
            self.assertEqual(info['review_context_bytes'], 9000)
            self.assertEqual(windows[0]['content'], original[0]['content'])
            _, windows = context.prepare(r, original, byte_limit=6000)
            self.assertEqual(windows, [])

    def test_exact_excerpt_and_full_context_are_distinguished(self):
        r = row(0, 'I used to run. I no longer run because of an injury. I now swim.')
        original = copy.deepcopy(r['_packet_item']['sources'])
        r['_packet_item']['sources'][0].update(content='I used to run.', content_span=dict(start=0,end=14,original_length=len(original[0]['content'])))
        r['_rank_text'] = 'I used to run.'
        info, windows = context.prepare(r, original)
        self.assertTrue(info['context_incomplete'])
        self.assertEqual(windows[0]['content'], original[0]['content'])
        r['_review_sources'] = windows
        c = staged.expanded_card(r, staged.cards_for([r])[0])
        self.assertFalse(c['context']['context_incomplete'])
        self.assertEqual(r['_rank_text'], 'I used to run.')

    def test_omitted_source_cannot_be_declared_complete(self):
        r = row(0)
        r['_packet_item']['sources'].append(dict(role='user', request_id='req', message_index=9,content_omitted='unit_budget'))
        self.assertIn('source_omitted', context.diagnose(r)['reasons'])

    def test_request_preserves_query_options_and_role_ownership(self):
        req = schemas.SearchRequest(user_id='u', query='What no longer applies after 2025?', options=['A. Not before 2024'])
        cards = staged.cards_for([row(0), row(1)])
        for shared in (False, True):
            request = staged.request_for(req, cards, shared=shared)
            self.assertIn(req.query, request['prompt']); self.assertIn(req.options[0], request['prompt'])
            self.assertEqual(request['schema']['properties']['c0']['properties']['source_refs']['items']['enum'], ['c0.s0'])
        self.assertNotIn('Source messages:', cards[0]['text'])

    def test_foreign_reference_and_missing_basis_are_invalid(self):
        card = staged.cards_for([row(0)])[0]
        self.assertEqual(staged.validate_decision(decision(refs=['c1.s0']),card,['c0'],['question']),'source_refs')
        self.assertEqual(staged.validate_decision(decision(),card,['c0'],['question']),'missing_source_basis')
        self.assertIsNone(staged.validate_decision(decision('unknown'),card,['c0'],['question']))

    def test_known_dependencies_are_not_split_by_batch_size(self):
        rows = [row(i) for i in range(5)]
        rows[0]['_relation_path_ids'] = ['m0','m4']
        batches = staged.batches_for(rows, 2)
        self.assertTrue(any(0 in b and 4 in b for b in batches))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(5)))

    def test_missing_dependency_blocks_negative_even_with_complete_sources(self):
        r=row(0);r['_relation_path_ids']=['m0','missing']
        card=staged.cards_for([r])[0]
        self.assertTrue(card['context']['context_incomplete'])
        self.assertIn('missing_dependency',card['context']['reasons'])

    def test_multiple_owned_references_are_not_rejected_by_arbitrary_two_ref_cap(self):
        card=staged.cards_for([row(0)])[0]
        card['sources'].update({'c0.s1':card['sources']['c0.s0'],'c0.s2':card['sources']['c0.s0']})
        self.assertIsNone(staged.validate_decision(decision('background',refs=list(card['sources']),targets=['question']),
                                                  card,['c0'],['question']))


class Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_larger_stage_requests_still_enforce_prompt_and_token_caps(self):
        from unittest.mock import AsyncMock
        request = dict(prompt='证' * 10000, reservation=31000)
        provider = AsyncMock(return_value={})
        with patch.object(config, 'LISTWISE_STAGED_MAX_PROMPT_BYTES', 48000), patch.object(llm, 'complete_json', provider):
            with budget.scope(seconds=30, tokens=128000) as limits:
                for stage in ('search.listwise.context', 'search.listwise.order'):
                    await staged.dispatch(request, time.monotonic()+25, stage)
                self.assertEqual(limits.reserved_tokens, 0)
                with self.assertRaisesRegex(budget.BudgetExceeded, 'prompt_budget'):
                    await staged.dispatch(dict(request, prompt='证'*16001), time.monotonic()+25, stage)
            with budget.scope(seconds=30, tokens=30000) as limits:
                with self.assertRaises(budget.BudgetExceeded):
                    await staged.dispatch(request, time.monotonic()+25, stage)
                self.assertEqual(limits.reserved_tokens, 0)
        self.assertEqual(provider.await_count, 2)

    async def run_case(self, rows, decisions, *, calls=8, fail_order=False, recovered=(), extra=None):
        self.requests = []
        async def provider(prompt, **kwargs):
            budget.current.get().before_call()
            self.requests.append(kwargs['stage'])
            payload = json.loads(prompt[prompt.index('\n{')+1:])
            if kwargs['stage'] == 'search.listwise.order':
                return dict(ranking=['bad']) if fail_order else dict(ranking=[c['key'] for c in reversed(payload['candidates'])])
            return {c['key']: copy.deepcopy((extra if kwargs['stage'].endswith('context') and extra else decisions)[c['key']])
                    for c in payload['candidates'] if c['key'] in decisions}
        async def recovery(req, submitted, selected, irrelevant, **kwargs):
            self.reviewed = irrelevant
            selected += [submitted[i] for i in irrelevant if i in recovered]
            return selected, dict(status='recovered' if recovered else 'ok', errors=[],represented=[],
                                 restored_ids=[submitted[i]['id'] for i in recovered])
        self.reviewed = set()
        with budget.scope(seconds=20,calls=calls,tokens=200000) as limits, \
             patch.object(llm,'complete_json',side_effect=provider), \
             patch.object(staged.listwise_recovery,'recover',side_effect=recovery), \
             patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
            result = await staged.select(schemas.SearchRequest(user_id='u',query='current needs'),{},rows,
                                         deadline=time.monotonic()+15,batch_size=2,shared=True)
            self.assertEqual((limits.reserved_tokens,limits.reserved_calls),(0,0))
        return result

    async def test_unrelated_requires_review_and_restored_item_is_ranked(self):
        values={'c0':decision(refs=['c0.s0']), 'c1':decision('background',refs=['c1.s0'],targets=['question'])}
        selected, irrelevant, trace = await self.run_case([row(0),row(1)],values,recovered=(0,))
        self.assertEqual(irrelevant,{0}); self.assertEqual(self.reviewed,{0})
        self.assertEqual([r['id'] for r in selected],['m1','m0'])
        self.assertEqual(trace['ranking_status'],'ok')

    async def test_missing_or_foreign_decision_is_retained(self):
        selected, irrelevant, trace = await self.run_case([row(0),row(1)],{'c0':decision(refs=['c1.s0'])})
        self.assertEqual(len(selected),2);self.assertFalse(irrelevant)
        self.assertEqual(trace['candidate_contract_status'],'invalid')

    async def test_incomplete_context_blocks_negative_and_duplicate_cycle(self):
        rows=[row(0),row(1),row(2)]
        rows[0]['_rank_excerpted']=True
        values={'c0':decision(refs=['c0.s0']),
                'c1':decision('duplicate',refs=['c1.s0'],owners=['c2']),
                'c2':decision('duplicate',refs=['c2.s0'],owners=['c1'])}
        selected, irrelevant, trace=await self.run_case(rows,values)
        self.assertEqual(len(selected),3);self.assertFalse(irrelevant)
        self.assertTrue(any(d['reason']=='context_incomplete' for d in trace['deletion_blocked']))

    async def test_budget_exhaustion_and_bad_ranking_never_delete(self):
        values={f'c{i}':decision('unknown') for i in range(3)}
        for calls, fail in [(1,False),(8,True)]:
            selected, irrelevant, trace=await self.run_case([row(i) for i in range(3)],values,calls=calls,fail_order=fail)
            self.assertEqual(len(selected),3); self.assertFalse(irrelevant)
            self.assertEqual(trace['ranking_status'],'fallback')

    async def test_supplement_retains_original_packet(self):
        r=row(0);r['_rank_excerpted']=True
        source=copy.deepcopy(r['_packet_item']['sources'][0]);text=source['content']
        source['content_span']=dict(start=0,end=len(text),original_length=len(text))
        r['_review_sources']=[source]
        before=copy.deepcopy(r)
        selected, irrelevant, trace=await self.run_case([r],{'c0':decision('unknown')},
            extra={'c0':decision('background',refs=['c0.s0'],targets=['question'])})
        self.assertEqual(selected,[before]);self.assertFalse(irrelevant)
        self.assertEqual(len(trace['context_supplements']),1)

    async def test_verified_duplicate_only_uses_retained_positive_owner(self):
        rows=[row(0),row(1)]
        rows[1]['_packet_item']=copy.deepcopy(rows[0]['_packet_item'])
        rows[1]['_rank_text']=rows[0]['_rank_text']
        values={'c0':decision('background',refs=['c0.s0'],targets=['question']),
                'c1':decision('duplicate',refs=['c1.s0'],owners=['c0'])}
        selected, irrelevant, trace=await self.run_case(rows,values)
        self.assertEqual(irrelevant,{1});self.assertEqual(self.reviewed,{1})
        self.assertEqual([r['id'] for r in selected],['m0'])

    async def test_real_cascade_and_fake_provider_staged_path(self):
        async def ce(query,documents,**kwargs):
            return [.8]*len(documents)
        rows=[row(i) for i in range(3)]
        with patch.multiple(config,LISTWISE_STAGED=True,FAKE=True,CASCADE_COARSE_LIMIT=3,
             CASCADE_FINE_LIMIT=3,CASCADE_LLM_LIMIT=3,FINE_EVIDENCE_ADAPTIVE=False), \
             patch.object(cross_encoder,'rerank',side_effect=ce), \
             patch('socket.socket.connect',side_effect=AssertionError('network forbidden')), \
             budget.scope(seconds=20,calls=10,tokens=200000):
            plan={}
            result=await cascade_rerank.rank(schemas.SearchRequest(user_id='u',query='current needs'),plan,rows)
        self.assertEqual(plan['_cascade']['listwise']['output_protocol'],'staged_usefulness_v1')
        self.assertEqual(plan['_cascade']['listwise']['ranking_status'],'ok')
        self.assertEqual(len([r for r in result if r['_cascade_selected']]),3)


class HumanMetrics(unittest.TestCase):
    def test_pending_labels_never_become_ground_truth(self):
        import evaluate_listwise_staged as evaluate
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            (root/'labels').write_text(json.dumps(dict(qa_id='0',candidate_id='m0',annotation=dict(status='pending')))+'\n')
            (root/'predictions').write_text(json.dumps(dict(qa_id='0',variant='v',status='complete',input_ids=['m0'],proposed_deleted=['m0']))+'\n')
            with redirect_stdout(io.StringIO()):
                evaluate.score(SimpleNamespace(labels=root/'labels',predictions=root/'predictions',output=root/'score'))
            report=json.loads((root/'score').read_text())
            self.assertEqual(report['status'],'awaiting_human_review')
            self.assertIsNone(report['variants']['v']['useful_false_deletion_rate'])

    def test_reviewed_useful_label_requires_real_source_quote(self):
        import evaluate_listwise_staged as evaluate
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            (root/'labels').write_text(json.dumps(dict(qa_id='0',candidate_id='m0',sources=[dict(content='Actual source.')],
                annotation=dict(status='reviewed',reviewer='reviewer',useful=True,roles=['unique_fact'],necessity='required',
                    rationale='Unique fact',source_quotes=['Invented quote.'])))+'\n')
            with self.assertRaisesRegex(ValueError,'quote not in source'):
                evaluate.score(SimpleNamespace(labels=root/'labels',predictions=root/'missing',output=root/'score'))


if __name__=='__main__':unittest.main(verbosity=2)
