"""Local repair regressions with synthetic facts, no provider or gold inputs."""
import copy
import unittest
from unittest.mock import patch

from selftest_answer_choice import OfflineCase, memory, seal, assessment, staged_mock, entailment_response
from app import answer_context, choice_premises, evidence_units, eval_scoring, llm, metrics


class SupportRepairTests(OfflineCase, unittest.IsolatedAsyncioTestCase):
    async def test_global_failures_repair_all_options_with_scoped_span_ids(self):
        for failure in ['parse','wrapped_parse','missing','wrong_type','extra_key']:
            with self.subTest(failure=failure):
                qa,packet,valid=self.fixture();sources,_=self.gate.build_catalog(packet);calls=[]
                async def provider(prompt,**kwargs):
                    calls.append(kwargs)
                    if kwargs['stage']=='eval.choice_support':
                        if failure=='parse':raise ValueError('Malformed structured response')
                        if failure=='wrapped_parse':raise llm.LLMError('Structured response failed') from metrics.ResponseParseError({})
                        if failure=='missing':return {}
                        if failure=='wrong_type':return {'options':None}
                        return dict(options=copy.deepcopy(valid['options']),extra=True)
                    schema=kwargs['schema'];letter=schema['$defs']['Assessment']['properties']['letter']['enum'][0]
                    claim_schema=schema['$defs']['Claim'];self.assertNotIn('text',claim_schema['properties']);self.assertIn('span_id',claim_schema['required'])
                    ids=claim_schema['properties']['span_id']['enum'];self.assertTrue(all(i.startswith(letter+':') for i in ids))
                    row=copy.deepcopy(valid['options'][0 if letter=='A' else 1])
                    if row['claims']:
                        row['claims'][0].pop('text');row['claims'][0]['span_id']=ids[0]
                    return dict(options=[row])
                d={}
                with patch.object(llm,'complete_json',provider):result=await self.gate.assess_support('Options:\n'+'\n'.join(qa['options']),self.gate.Assessments.model_json_schema(),qa['options'],sources,d)
                self.assertEqual(len(calls),3);self.assertEqual(d['answer_support_validation']['accepted_options'],['A','B']);self.assertEqual(result[0]['status'],'supported');self.assertEqual(result[1]['status'],'generic')
    async def test_global_failure_does_not_accept_unknown_span_or_retry_forever(self):
        qa,packet,valid=self.fixture();sources,_=self.gate.build_catalog(packet);calls=[]
        async def provider(prompt,**kwargs):
            calls.append(kwargs)
            if kwargs['stage']=='eval.choice_support':return {}
            letter=kwargs['schema']['$defs']['Assessment']['properties']['letter']['enum'][0];row=copy.deepcopy(valid['options'][0 if letter=='A' else 1])
            if row['claims']:row['claims'][0].pop('text');row['claims'][0]['span_id']='A:999'
            return dict(options=[row])
        d={}
        with patch.object(llm,'complete_json',provider):result=await self.gate.assess_support('Options:\n'+'\n'.join(qa['options']),self.gate.Assessments.model_json_schema(),qa['options'],sources,d)
        self.assertEqual(len(calls),3);self.assertEqual(d['answer_support_validation']['unresolved_options'],['A']);self.assertEqual(result[1]['status'],'generic')

    async def test_repair_witnesses_only_cover_pending_option_and_preserve_multiline_quote(self):
        from app import choice_witness
        qa, packet, valid = self.fixture()
        sources, _ = self.gate.build_catalog(packet)
        witnesses = {
            'A': [dict(source_id='s0', role='user', quote='I own a telescope.\nIt is mine.')],
            'B': [dict(source_id='other', role='user', quote='Unrelated candidate hint.')],
        }
        before = copy.deepcopy(witnesses)
        prompt = ('Options:\n' + '\n'.join(qa['options']) + '\n<witnesses>\n'
                  + choice_witness.render(witnesses) + '\n</witnesses>\nCORE TASK\nAssess both.')
        mock = staged_mock({'options': [valid['options'][1]]}, {'options': [valid['options'][0]]})
        with patch.object(llm, 'complete_json', mock):
            result = await self.gate.assess_support(prompt, self.gate.Assessments.model_json_schema(),
                qa['options'], sources, dict(answer_witness_prefill=witnesses))
        initial, repair = [call.args[0] for call in mock.call_args_list]
        self.assertIn('B: [other user]', initial)
        block = repair.split('<witnesses>\n', 1)[1].split('\n</witnesses>', 1)[0]
        self.assertEqual(block, choice_witness.render({'A': witnesses['A']}))
        self.assertEqual(witnesses, before)
        self.assertEqual(result[1]['kind'], 'generic')

    async def test_pending_option_without_witness_does_not_inherit_sibling_hint(self):
        from app import choice_witness
        qa, packet, valid = self.fixture()
        sources, _ = self.gate.build_catalog(packet)
        witnesses = {'B': [dict(source_id='s0', role='user', quote='I own a telescope.')]}
        prompt = ('Options:\n' + '\n'.join(qa['options']) + '\n<witnesses>\n'
                  + choice_witness.render(witnesses) + '\n</witnesses>\nCORE TASK\nAssess both.')
        mock = staged_mock({'options': [valid['options'][1]]}, {'options': [valid['options'][0]]})
        with patch.object(llm, 'complete_json', mock):
            await self.gate.assess_support(prompt, self.gate.Assessments.model_json_schema(),
                qa['options'], sources, dict(answer_witness_prefill=witnesses))
        self.assertIn('<witnesses>\n(none found)\n</witnesses>', mock.call_args_list[1].args[0])

    def fixture(self):
        packet = seal(memory('I own a telescope.'))
        qa = dict(question='How can I observe the sky?', scoring='choice', qa_type='single_choice',
                  options=['A. Since you own a telescope, observe the moon.', 'B. Take a break.'],
                  gold_labels=['DO_NOT_SEND_GOLD'])
        support = assessment('you own a telescope', 's0', 'I own a telescope.')
        return qa, packet, support

    async def test_missing_option_repair_requires_original_spans(self):
        qa,packet,valid=self.fixture()
        first={'options':[valid['options'][1]]}
        mock=staged_mock(first,{'options':[valid['options'][0]]})
        sources,_=self.gate.build_catalog(packet)
        diagnostics={}
        with patch.object(llm,'complete_json',mock):
            result=await self.gate.assess_support('Options:\n'+'\n'.join(qa['options']),self.gate.Assessments.model_json_schema(),qa['options'],sources,diagnostics)
        self.assertEqual(mock.await_count,2)
        schema=mock.call_args_list[1].kwargs['schema']['$defs']['Claim']
        self.assertIn('span_id',schema['required'])
        self.assertNotIn('text',schema['properties'])
        self.assertTrue(all(x.startswith('A:') for x in schema['properties']['span_id']['enum']))
        self.assertEqual(result[1]['kind'],'generic')

    async def test_repair_source_cards_exclude_assistant_rewrites(self):
        import json
        qa, packet, valid = self.fixture()
        sources, _ = self.gate.build_catalog(packet)
        sources['rewrite'] = dict(id='rewrite', role='assistant', text='Rewrite-only marker.')
        before = copy.deepcopy(sources)
        prompt = 'Options:\n' + '\n'.join(qa['options']) + '\n' + json.dumps(
            self.gate._cards(sources), ensure_ascii=False)
        mock = staged_mock({'options': [valid['options'][1]]},
                           {'options': [valid['options'][0]]})
        with patch.object(llm, 'complete_json', mock):
            await self.gate.assess_support(prompt, self.gate.Assessments.model_json_schema(),
                                          qa['options'], sources, {})
        self.assertIn('Rewrite-only marker.', mock.call_args_list[0].args[0])
        self.assertNotIn('Rewrite-only marker.', mock.call_args_list[1].args[0])
        self.assertIn('I own a telescope.', mock.call_args_list[1].args[0])
        self.assertEqual(sources, before)

    async def test_bad_sibling_never_erases_valid_personal_evidence(self):
        qa, packet, first = self.fixture()
        first['options'][1]['claims'] = [dict(text='invented paraphrase', status='supported', citations=[])]
        second = copy.deepcopy(first)
        # A changed/invalid sibling response must not overwrite accepted A.
        second['options'][0]['claims'][0]['text'] = 'forged'
        diagnostics = {}
        mock = staged_mock(first, second, entailment_response)
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'A')
        self.assertEqual(diagnostics['answer_support_validation']['unresolved_options'], ['B'])
        self.assertEqual(diagnostics['choice_alignment'][0]['claims'][0]['text'], 'you own a telescope')
        self.assertEqual(diagnostics['choice_alignment'][1]['validation_status'], 'unresolved')
        self.assertIn('I own a telescope.', mock.call_args_list[2].args[0])
        repair = mock.call_args_list[1].args[0]
        self.assertIn('claim_not_in_option', repair)
        self.assertIn('invalid_options', repair)
        self.assertEqual(diagnostics['answer_calls'][0]['response'], first)
        self.assertEqual(diagnostics['answer_calls'][0]['failure_phase'], 'post_json_validator')
        self.assertNotIn('DO_NOT_SEND_GOLD', ''.join(c.args[0] for c in mock.call_args_list))

    async def test_schema_error_reports_path_and_only_retries_missing_option(self):
        qa, packet, first = self.fixture()
        first['options'][1]['kind'] = 'invented'
        second = {'options': [dict(letter='B', kind='generic', claims=[])]}
        mock = staged_mock(first, second, entailment_response, {'answer': 'A'})
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'A')
        self.assertIn('literal_error', mock.call_args_list[1].args[0])
        self.assertIn('kind', mock.call_args_list[1].args[0])
        initial_schema = mock.call_args_list[0].kwargs['schema']
        repair_schema = mock.call_args_list[1].kwargs['schema']
        self.assertEqual(initial_schema['properties']['options']['minItems'], 2)
        self.assertEqual(repair_schema['properties']['options']['minItems'], 1)
        self.assertEqual(repair_schema['properties']['options']['maxItems'], 1)
        self.assertEqual(repair_schema['$defs']['Assessment']['properties']['letter']['enum'], ['B'])
        self.assertEqual(diagnostics['answer_support_validation']['status'], 'valid')

    async def test_unlabelled_options_keep_original_letter_on_local_repair(self):
        qa, packet, first = self.fixture()
        qa['options'] = [s[3:] for s in qa['options']]
        mock = staged_mock(first, entailment_response, {'answer': 'A'})
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, {}), 'A')
        self.assertEqual(mock.await_count, 3)

    async def test_invalid_claim_quote_is_repaired_even_when_option_shape_is_valid(self):
        qa, packet, valid = self.fixture()
        invalid = copy.deepcopy(valid)
        invalid['options'][0]['claims'][0]['citations'][0]['quote'] = 'I own a microscope.'
        repair = {'options': [valid['options'][0]]}
        mock = staged_mock(invalid, repair, entailment_response, {'answer': 'A'})
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'A')
        self.assertEqual(diagnostics['answer_support_validation']['unresolved_options'], [])
        self.assertIn('quote_not_in_source', mock.call_args_list[1].args[0])
        self.assertIn('no_valid_user_support', mock.call_args_list[1].args[0])
        self.assertEqual(diagnostics['choice_alignment'][0]['claims'][0]['citations'][0]['quote'],
                         'I own a telescope.')

    async def test_failed_claim_quote_repair_remains_unresolved(self):
        qa, packet, invalid = self.fixture()
        invalid['options'][0]['claims'][0]['citations'][0]['quote'] = 'I own a microscope.'
        mock = staged_mock(invalid, {'options': [invalid['options'][0]]})
        sources, _ = self.gate.build_catalog(packet)
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            entries = await self.gate.assess_support('Options:\n' + '\n'.join(qa['options']),
                self.gate.Assessments.model_json_schema(), qa['options'], sources, diagnostics)
        self.assertEqual(diagnostics['answer_support_validation']['unresolved_options'], ['A'])
        self.assertEqual(entries[0]['validation_status'], 'unresolved')
        self.assertEqual(entries[1]['kind'], 'generic')

    async def test_single_option_responses_get_all_missing_siblings_repaired(self):
        qa, packet, first = self.fixture()
        qa['options'].extend(['C. Look up at night.', 'D. Ask a friend.'])
        first['options'] = first['options'][:1]
        responses = [{'options': [dict(letter=k, kind='generic', claims=[])]} for k in 'BCD']
        mock = staged_mock(first, *responses)
        sources, _ = self.gate.build_catalog(packet)
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            entries = await self.gate.assess_support('Options:\n' + '\n'.join(qa['options']),
                self.gate.Assessments.model_json_schema(), qa['options'], sources, diagnostics)
        self.assertEqual([e['letter'] for e in entries], list('ABCD'))
        self.assertEqual(diagnostics['answer_support_validation']['unresolved_options'], [])
        self.assertEqual(mock.await_count, 4)
        for letter, call in zip('BCD', mock.call_args_list[1:]):
            self.assertEqual(call.kwargs['schema']['$defs']['Assessment']['properties']['letter']['enum'], [letter])
            self.assertEqual(call.kwargs['schema']['properties']['options']['maxItems'], 1)

    async def test_one_failed_option_does_not_stop_later_option_repairs(self):
        qa, packet, first = self.fixture()
        qa['options'].extend(['C. Look up at night.', 'D. Ask a friend.'])
        first['options'] = first['options'][:1]
        responses = [{'options': [dict(letter=k, kind='generic', claims=[])]} for k in 'CD']
        mock = staged_mock(first, {}, *responses)
        sources, _ = self.gate.build_catalog(packet)
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            entries = await self.gate.assess_support('Options:\n' + '\n'.join(qa['options']),
                self.gate.Assessments.model_json_schema(), qa['options'], sources, diagnostics)
        self.assertEqual(diagnostics['answer_support_validation']['unresolved_options'], ['B'])
        self.assertEqual(entries[0]['claims'][0]['text'], 'you own a telescope')
        self.assertEqual(entries[1]['validation_status'], 'unresolved')
        self.assertEqual([e['kind'] for e in entries[2:]], ['generic', 'generic'])

    async def test_provider_failure_for_one_option_does_not_abort_later_repairs(self):
        qa, packet, first = self.fixture()
        qa['options'].extend(['C. Look up at night.', 'D. Ask a friend.'])
        first['options'] = first['options'][:1]
        later = [{'options': [dict(letter=k, kind='generic', claims=[])]} for k in 'CD']
        mock = staged_mock(first, llm.LLMError('temporary provider failure'), *later)
        sources, _ = self.gate.build_catalog(packet)
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            entries = await self.gate.assess_support('Options:\n' + '\n'.join(qa['options']),
                self.gate.Assessments.model_json_schema(), qa['options'], sources, diagnostics)
        self.assertEqual(diagnostics['answer_support_validation']['unresolved_options'], ['B'])
        self.assertEqual([entry['kind'] for entry in entries[2:]], ['generic', 'generic'])
        self.assertEqual(mock.await_count, 4)

    async def test_abstention_is_distinct_from_provider_failure_and_scores_zero(self):
        qa, packet, _ = self.fixture()
        mock = staged_mock({}, {}, {})
        with patch.object(llm, 'complete_json', mock):
            pred, score, diagnostics = await eval_scoring.evaluate(qa, packet)
        self.assertEqual((pred, score, diagnostics['answer_status']), ('ABSTAIN', 0, 'abstained'))
        self.assertNotIn('error_type', diagnostics)
        with patch.object(llm, 'complete_json', side_effect=llm.LLMError('Transport failed')):
            pred, score, diagnostics = await eval_scoring.evaluate(qa, packet)
        self.assertEqual((pred, score, diagnostics['answer_status']), ('', 0, 'error'))
        self.assertEqual(diagnostics['error_detail'], 'Transport failed')


class SemanticRecoveryTests(OfflineCase):
    def test_recovery_needs_anchor_strength_and_independent_premise(self):
        sources, _ = self.catalog('Which telescopes are useful for stargazing?')
        options = ['A. Given your interest in stargazing, visit an observatory.', 'B. Rest.']
        original = assessment('your interest in stargazing', 's0', sources['s0']['text'], basis='topic_interest')
        original['options'][0]['claims'][0].update(status='unsupported', reason='no_source', premise_type='interest')
        for rejected in (set(), {'A:0'}, {'A:option'}):
            entries = self.gate.validate_assessments(original, options, sources)
            checks = self.gate.entailment_checks(entries, sources, options)
            verdicts = dict(checks=[dict(claim_id=c['claim_id'], entailed=c['claim_id'] not in rejected) for c in checks])
            self.gate.validate_entailments(verdicts, entries, checks)
            expected = 'unsupported' if 'A:0' in rejected else 'partial' if rejected else 'supported'
            self.assertEqual(entries[0]['status'], expected)
        for reason, refs in [('contradicted', original['options'][0]['claims'][0]['citations']), ('no_source', [])]:
            data = copy.deepcopy(original)
            data['options'][0]['claims'][0].update(reason=reason, citations=refs)
            entries = self.gate.validate_assessments(data, options, sources)
            self.assertNotIn('A:0', [c['claim_id'] for c in self.gate.entailment_checks(entries, sources, options)])

    def test_interest_does_not_recover_ownership_even_with_true_semantic_vote(self):
        sources, _ = self.catalog('Which telescopes are useful?')
        options = ['A. Since you own a telescope, observe the moon.', 'B. Rest.']
        data = assessment('you own a telescope', 's0', sources['s0']['text'], basis='topic_interest')
        data['options'][0]['claims'][0].update(status='unsupported', reason='source_too_weak')
        entries = self.gate.validate_assessments(data, options, sources)
        checks = self.gate.entailment_checks(entries, sources, options)
        self.gate.validate_entailments(dict(checks=[dict(claim_id=c['claim_id'], entailed=True) for c in checks]), entries, checks)
        self.assertEqual(entries[0]['status'], 'unsupported')

    def test_claim_span_id_resolves_original_text_and_rejects_other_option(self):
        sources, _ = self.catalog('I own a telescope.')
        options = ['A. Since you own a telescope, observe the moon.', 'B. Rest.']
        data = assessment('you own a telescope', 's0', sources['s0']['text'])
        claim = data['options'][0]['claims'][0]
        claim.pop('text')
        claim['span_id'] = 'A:1'
        result = self.gate.validate_assessments(data, options, sources)
        c = result[0]['claims'][0]
        self.assertEqual(c['text'], 'Since you own a telescope,')
        self.assertEqual(options[0][c['option_span']['start']:c['option_span']['end']], c['text'])
        claim['span_id'] = 'B:0'
        with self.assertRaises(ValueError):
            self.gate.validate_assessments(data, options, sources)

    def test_explicit_contradiction_is_never_selected_as_partial(self):
        entries = [dict(letter='A', status='partial', claims=[dict(reason='contradicted')], validation_errors=[]),
                   dict(letter='B', status='generic', claims=[], validation_errors=[])]
        self.assertEqual(self.gate.eligible_choices(entries, set()), ['B'])

    def test_forgetting_creation_does_not_ban_display_or_reading(self):
        for rule, unrelated, prohibited in [
            ('Please forget that I take photos during holidays.', 'Display framed holiday photos.', 'Take photos during holidays.'),
            ('Please forget that I write detective stories.', 'Read detective stories.', 'Write new detective stories.'),
            ('Please forget that I bake fruit pies.', 'Buy fruit pies.', 'Bake fruit pies.')]:
            self.assertFalse(self.gate._scope_overlap(rule, unrelated))
            self.assertTrue(self.gate._scope_overlap(rule, prohibited))
        for rule, option in [
            ('Please forget that I take photos during holidays.', 'Since you took photos during holidays, print them.'),
            ('Please forget that I write detective stories.', 'Since you wrote detective stories, publish them.'),
            ('Please forget that I bake fruit pies.', 'Since you baked fruit pies, share the recipe.')]:
            self.assertTrue(self.gate._scope_overlap(rule, option))

    def test_future_advice_is_not_mistaken_for_past_ability(self):
        for claim in ('You could take a walk before bed', 'You might stretch when needed'):
            self.assertTrue(choice_premises.is_suggestion(claim, 'A. ' + claim + '.'))
        for claim in ('you could run marathons before the accident', 'you own a telescope', 'you have never walked'):
            self.assertFalse(choice_premises.is_suggestion(claim, 'A. Since ' + claim + ', rest.'))


class UserBudgetTests(OfflineCase):
    def source(self, index, text, role='user'):
        return dict(request_id='r', message_index=index, role=role, content=text)

    def test_long_assistant_cannot_displace_short_user_messages(self):
        sources = [self.source(0, 'Stars telescope guide. ' * 150, 'assistant'),
                   self.source(1, 'I enjoy stars.'), self.source(2, 'I use a telescope.')]
        before = copy.deepcopy(sources)
        unit = evidence_units.build({}, sources, [], ['stars telescope'], 'Summary', 1100)
        self.assertIsNotNone(unit)
        self.assertLessEqual(len(unit['content'].encode()), 1100)
        for source in sources[1:]:
            self.assertIn(source['content'], unit['content'])
        self.assertEqual(unit['content'], answer_context.with_evidence('Summary', unit['sources']))
        self.assertEqual(sources, before)

    def test_mandatory_assistant_context_remains_atomic_with_user_anchor(self):
        sources = [self.source(0, 'Do you own this telescope?', 'assistant'), self.source(1, 'Yes, I own it.')]
        evidence = [dict(request_id='r', message_index=i, quote=s['content']) for i, s in enumerate(sources)]
        unit = evidence_units.build({}, sources, evidence, ['telescope'], 'Summary', 1000)
        self.assertIsNotNone(unit)
        for source in sources:
            self.assertIn(source['content'], unit['content'])
        self.assertIsNone(evidence_units.build({}, sources, evidence, ['telescope'], 'Summary', 180))

    def test_optional_user_reservations_cannot_evict_a_mandatory_quote(self):
        quote = 'I observe stars with this telescope ' + 'very carefully ' * 45 + '.'
        sources = [self.source(0, quote)] + [self.source(i, f'I enjoy stars {i}.') for i in range(1, 9)]
        evidence = [dict(request_id='r', message_index=0, quote=quote)]
        unit = evidence_units.build({}, sources, evidence, ['stars'] * 8, 'Summary', 2000)
        self.assertIsNotNone(unit)
        self.assertIn(quote, unit['content'])


if __name__ == '__main__':
    unittest.main()
