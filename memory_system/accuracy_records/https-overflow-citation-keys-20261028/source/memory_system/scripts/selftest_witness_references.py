"""Offline source-reference transport and unchanged answer-gate regressions."""
import asyncio
import copy
import json
import random
import unittest
from unittest.mock import AsyncMock, patch

from selftest_answer_choice import assessment, entailment_response, memory, seal
from app import answer_choice, budget, choice_witness as cw, config, llm, run_metadata


OPTIONS = ['A. Since you practice yoga daily, try a different routine.', 'B. Rest.']
TEXT = 'I spend a quiet moment on the mat before breakfast.'


def payload(sid='s0', pid='s0.p0'):
    return dict(options=[dict(letter='A', witnesses=[dict(source_id=sid, span_id=pid)]),
                         dict(letter='B', witnesses=[])])


class SourcePartsTests(unittest.TestCase):
    def test_complete_sources_and_literal_offsets_across_formats(self):
        rng = random.Random(7)
        texts = [
            'I do not own a bike. My sister owns one.\nIf I had one, I might ride it.',
            '我以前喜歡騎車。現在不騎了！朋友問：「要一起嗎？」',
            '"role": {\n  "title": "Teacher",\n  "hours": 12\n}',
            '  I’m curious about astronomy.\n\nI do not own a telescope.  ',
            'word ' * 300, 'x' * 1001, '\n  \t', '',
            ''.join(rng.choice('abc 。！？\n\t') for _ in range(2000)),
        ]
        for text in texts:
            with self.subTest(text=text[:40]):
                sources = {'s0': dict(id='s0', role='user', text=text)}
                original = copy.deepcopy(sources)
                cards, registry = cw.reference_sources(sources)
                self.assertEqual(''.join(p['text'] for p in cards[0]['parts']), text)
                self.assertEqual(sources, original)
                for part in cards[0]['parts']:
                    self.assertLessEqual(len(part['text']), cw.MAX_QUOTE_CHARS)
                    if part['text'].strip():
                        span = registry[part['id']]
                        self.assertEqual(span['source_id'], 's0')
                        self.assertEqual(text[span['start']:span['end']], span['quote'])
                        self.assertEqual(part['text'].strip(), span['quote'])
                    else:
                        self.assertNotIn(part['id'], registry)

    def test_shared_words_do_not_merge_sources_or_authority(self):
        text = 'I enjoy playing chess.'
        sources = {
            's0': dict(id='s0', role='assistant', text=text),
            's1': dict(id='s1', role='user', declared='persona', text=text),
        }
        cards, registry = cw.reference_sources(sources)
        self.assertEqual([c['role'] for c in cards], ['assistant', 'persona'])
        self.assertEqual(registry['s0.p0']['source_id'], 's0')
        self.assertEqual(registry['s1.p0']['source_id'], 's1')


class ReferenceFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        patch.multiple(config, FAKE=False, CHOICE_SEMANTIC_WITNESSES=True,
                       CHOICE_WITNESS_SPAN_REFS=True).start()
        self.packet = seal(memory(TEXT))
        self.sources, _ = answer_choice.build_catalog(self.packet)
        self.diagnostics = {}

    async def select(self):
        return await cw.select('What routine should I try?', OPTIONS,
                               self.sources, self.diagnostics)

    async def test_no_keyword_prefilter_and_no_generated_quote(self):
        self.assertNotIn('A', cw.build(OPTIONS, self.sources))
        original = copy.deepcopy(self.sources)
        model = AsyncMock(return_value=payload())
        with patch.object(llm, 'complete_json', model):
            witnesses = await self.select()
        self.assertEqual(witnesses['A'][0]['quote'], TEXT)
        self.assertEqual(witnesses['A'][0]['match_method'], 'semantic')
        self.assertEqual(self.sources, original)
        cards = json.loads(model.call_args.args[0].split('<sources>\n')[1].split('\n</sources>')[0])
        self.assertEqual(''.join(p['text'] for p in cards[0]['parts']), TEXT)
        schema = model.call_args.kwargs['schema']
        self.assertEqual(set(schema['$defs']['SpanWitness']['properties']), {'source_id', 'span_id'})
        self.assertEqual(model.await_count, 1)
        call = self.diagnostics['answer_calls'][0]
        self.assertEqual(call['response'], payload())
        self.assertEqual(call['compiled_response']['options'][0]['witnesses'],
                         [dict(source_id='s0', quote=TEXT)])
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['citation_format'],
                         'source_span_refs_v1')

    async def test_invalid_reference_preserves_valid_sibling(self):
        data = payload()
        data['options'][1]['witnesses'] = [dict(source_id='s0', span_id='s0.missing')]
        with patch.object(llm, 'complete_json', AsyncMock(return_value=data)):
            witnesses = await self.select()
        self.assertEqual(witnesses['A'][0]['quote'], TEXT)
        self.assertNotIn('B', witnesses)
        trace = self.diagnostics['answer_witness_retrieval']
        self.assertEqual(trace['status'], 'partial')
        self.assertEqual(trace['rejected_witnesses'][0]['reason'], 'unknown_span')

    async def test_unknown_ids_cross_source_and_extra_fields_fall_back(self):
        self.sources['s1'] = dict(id='s1', role='assistant', text='I practice yoga daily.')
        bad_field = payload()
        bad_field['options'][0]['witnesses'][0]['quote'] = 'invented'
        duplicate = payload()
        duplicate['options'].append(copy.deepcopy(duplicate['options'][0]))
        for data in (payload(pid='s0.missing'), payload(sid='missing'),
                     payload(sid='s0', pid='s1.p0'), bad_field, duplicate):
            with self.subTest(data=data):
                self.diagnostics = {}
                model = AsyncMock(return_value=data)
                with patch.object(llm, 'complete_json', model):
                    await self.select()
                self.assertEqual(model.await_count, 1)
                self.assertEqual(self.diagnostics['answer_witness_retrieval']['status'], 'fallback')

    async def test_duplicate_source_does_not_duplicate_support(self):
        self.sources['s0']['text'] = TEXT + ' I like a quiet start.'
        data = payload()
        data['options'][0]['witnesses'].append(dict(source_id='s0', span_id='s0.p1'))
        with patch.object(llm, 'complete_json', AsyncMock(return_value=data)):
            witnesses = await self.select()
        self.assertEqual(len(witnesses['A']), 1)
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['rejected_witnesses'][0]['reason'],
                         'duplicate_source')

    async def test_complete_source_context_and_roles_survive_selection(self):
        text = 'Jordan wrote: I practice yoga daily. I never said this was my routine.'
        self.sources['s0']['text'] = text
        self.sources['s0']['role'] = 'assistant'
        with patch.object(llm, 'complete_json', AsyncMock(return_value=payload())):
            witnesses = await self.select()
        self.assertEqual(witnesses['A'][0]['role'], 'assistant')
        self.assertFalse(witnesses['A'][0]['first_party'])
        self.assertEqual(self.sources['s0']['text'], text)

    async def test_reference_is_not_proof_of_the_personal_premise(self):
        support = assessment('you practice yoga daily', 's0', TEXT)
        stages = []

        async def respond(prompt, **kwargs):
            stage = kwargs['stage']
            stages.append(stage)
            if stage == 'eval.choice_witness':
                return payload()
            if stage == 'eval.choice_support':
                self.assertIn(TEXT, prompt.split('<witnesses>')[1].split('</witnesses>')[0])
                return support
            self.assertEqual(stage, 'eval.choice_entailment')
            return entailment_response(prompt, {'A:0', 'A:option'})

        with patch.object(llm, 'complete_json', respond):
            selected = await answer_choice.answer(
                dict(question='What routine should I try?', options=OPTIONS),
                self.packet, self.diagnostics)
        self.assertEqual(selected, 'B')
        self.assertEqual(stages, ['eval.choice_witness', 'eval.choice_support', 'eval.choice_entailment'])

    async def test_budget_prompt_limit_and_cancellation_still_apply(self):
        with budget.scope(seconds=60, calls=9, tokens=128000) as limits:
            with patch.object(llm, 'complete_json', AsyncMock(side_effect=asyncio.CancelledError)):
                with self.assertRaises(asyncio.CancelledError):
                    await self.select()
            self.assertEqual((limits.reserved_calls, limits.reserved_tokens), (0, 0))
        with patch.object(cw, 'MAX_PROMPT_BYTES', 10), \
                patch.object(llm, 'complete_json', AsyncMock()) as model:
            await self.select()
        model.assert_not_awaited()
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['reason'], 'prompt_budget')
        with budget.scope(seconds=60, calls=4, tokens=128000), \
                patch.object(llm, 'complete_json', AsyncMock()) as model:
            await self.select()
        model.assert_not_awaited()
        self.assertEqual(self.diagnostics['answer_witness_retrieval']['reason'], 'answer_budget')

    def test_runtime_mode_is_recorded_in_experiment_metadata(self):
        run_metadata.versions.cache_clear()
        try:
            self.assertTrue(run_metadata.versions()['settings']['CHOICE_WITNESS_SPAN_REFS'])
        finally:
            run_metadata.versions.cache_clear()


if __name__ == '__main__':
    unittest.main()
