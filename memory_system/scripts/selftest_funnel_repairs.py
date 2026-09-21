"""Gold-funnel regressions with synthetic controls and no provider calls."""
import copy
import unittest
from unittest.mock import AsyncMock, patch

from selftest_answer_choice import OfflineCase, memory, seal, assessment, staged_mock, entailment_response
from selftest_fine_selection import candidate, source
from app import budget, choice_premises, config, evidence_units, evidence_selection, llm


class CitationRepairs(OfflineCase):
    def test_type_label_cannot_turn_traits_into_interest(self):
        sources, _ = self.catalog('Which cameras take good pictures?')
        for text in ('you have a camera', 'you are recovering from cancer', 'you are an engineer', 'you exercise weekly'):
            data = assessment(text, 's0', sources['s0']['text'], basis='topic_interest')
            data['options'][0]['claims'][0]['premise_type'] = 'interest'
            entry = self.gate.validate_assessments(data, ['A. Since ' + text + ', rest.', 'B. Rest.'], sources)[0]
            self.assertEqual(entry['status'], 'unsupported')

    def test_existing_ability_and_negative_condition_are_not_suggestions(self):
        for text in ('you can play the piano', 'you can no longer run', 'you should never run because of your injury'):
            self.assertFalse(choice_premises.is_suggestion(text, 'A. Since ' + text + ', take a break.'))
        self.assertIsNone(choice_premises.canonical_span('you enjoy praying', 'you enjoy playing')[0])
        for claim, option in (
            ('you try hard at work', 'You try hard at work, so take a break.'),
            ('you consider yourself an introvert', 'You consider yourself an introvert, so rest.'),
            ('you start work at 6am', 'You start work at 6am, so rest.'),
            ('you volunteered last summer', 'You could visit the museum where you volunteered last summer.'),
            ('you passed the exam', 'Consider a concert to celebrate that you passed the exam.'),
            ('you could run marathons before the accident', 'You could run marathons before the accident, so rest.')):
            self.assertFalse(choice_premises.is_suggestion(claim, option))

    def test_interest_idioms_remain_interests(self):
        for claim in ('you are a fan of anime', 'you have an interest in anime', 'you have a passion for anime'):
            self.assertEqual(self.eligible('Which anime studios produce great animation?', claim, basis='topic_interest'), ['A'])

    def test_explicit_self_report_after_relation_is_not_third_party(self):
        self.assertEqual(self.eligible('My wife reads magazines and I collect vinyl records.',
            'you collect vinyl records', quote='I collect vinyl records'), ['A'])

    def test_fuzzy_repair_never_changes_negation_subject_or_time(self):
        self.assertEqual(choice_premises.canonical_span('you enjoy photgraphy', 'A. Since you enjoy photography, try this.'),
                         ('you enjoy photography', True))
        for claim, option in [('you enjoy photography', 'you never enjoy photography'),
                              ('your friend enjoys photography', 'your friends enjoy photography'),
                              ('you visited New Zealand', 'you visit New Zealand'),
                              ('you are now in London', 'you are not in London')]:
            self.assertIsNone(choice_premises.canonical_span(claim, option)[0])

    def test_invalid_citation_majority_and_duplicates_cannot_launder_claim(self):
        sources, _ = self.catalog('I injured my leg.')
        payload = assessment('your leg injury', 's0', 'I injured my leg.')
        refs = payload['options'][0]['claims'][0]['citations']
        refs.extend(dict(source_id=sid, quote='bad', basis='self_report', subject='current_user') for sid in ('bad1', 'bad2'))
        entry = self.gate.validate_assessments(payload, ['A. Given your leg injury, rest.', 'B. Rest.'], sources)[0]
        self.assertEqual(entry['status'], 'unsupported')
        refs[2] = copy.deepcopy(refs[0])
        entry = self.gate.validate_assessments(payload, ['A. Given your leg injury, rest.', 'B. Rest.'], sources)[0]
        self.assertEqual(len(entry['claims'][0]['citations']), 1)

    def test_inferred_is_opt_in_and_does_not_infer_strong_traits(self):
        sources, _ = self.catalog('I like talking about gardens.')
        data = assessment('you enjoy gardens', 's0', 'I like talking about gardens.')
        claim = data['options'][0]['claims'][0]
        claim.update(status='inferred', premise_type='interest')
        options = ['A. Since you enjoy gardens, read this.', 'B. Read this.']
        with patch.object(config, 'CHOICE_ALLOW_INFERRED', False):
            self.assertEqual(self.gate.eligible_choices(self.gate.validate_assessments(data, options, sources), set()), ['B'])
        with patch.object(config, 'CHOICE_ALLOW_INFERRED', True):
            entries = self.gate.validate_assessments(data, options, sources)
            self.assertEqual(entries[0]['status'], 'inferred')
            self.assertEqual(self.gate.eligible_choices(entries, set()), ['A'])
            claim['text'] = 'you own gardens'
            entries = self.gate.validate_assessments(data, ['A. Since you own gardens, read this.', options[1]], sources)
            self.assertEqual(self.gate.eligible_choices(entries, set()), ['B'])

    def test_soft_tier_requires_whole_option_core_and_secondary_anchors(self):
        sources, _ = self.catalog('I own a camera and I photograph birds.')
        options = ['A. Since you own a camera and you photograph birds, take a photo.', 'B. Rest.']
        data = assessment('you own a camera', 's0', sources['s0']['text'])
        data['options'][0]['claims'].append(dict(text='you photograph birds', status='unsupported', citations=[
            dict(source_id='s0', quote='I photograph birds', basis='self_report', subject='current_user')]))
        entries = self.gate.validate_assessments(data, options, sources)
        checks = self.gate.entailment_checks(entries, sources, options)
        verdicts = dict(checks=[dict(claim_id=c['claim_id'], entailed=True) for c in checks])
        self.gate.validate_entailments(verdicts, entries, checks)
        self.assertEqual(entries[0]['status'], 'partial')
        self.assertEqual(entries[0]['selection_tier'], 'supported')
        data['options'][0]['claims'][1]['citations'] = []
        entries = self.gate.validate_assessments(data, options, sources)
        self.gate.validate_entailments(verdicts, entries, checks)
        self.assertEqual(entries[0]['selection_tier'], 'partial')

    def test_valid_anchor_survives_one_invalid_citation(self):
        sources, _ = self.gate.build_catalog(seal(memory('I injured my leg.'),
            memory('You enjoy cycling.', role='assistant', mid='advice')))
        payload = assessment('your leg injury', 's0', 'I injured my leg.')
        payload['options'][0]['claims'][0]['citations'].append(dict(
            source_id='s1', quote='You enjoy cycling.', basis='self_report', subject='current_user'))
        entry = self.gate.validate_assessments(payload,
            ['A. Given your leg injury, rest.', 'B. Rest.'], sources)[0]
        self.assertEqual(entry['status'], 'supported')
        self.assertIn('citation_dropped', entry['warnings'])
        self.assertEqual(len(entry['claims'][0]['citations']), 1)

    def test_sentence_scoping_preserves_subject_and_denial(self):
        self.assertEqual(self.eligible('I love going through my travel photos.',
            'you love travel photos', quote='travel photos'), ['A'])
        for text in ('My friend loves travel photos.', 'I never liked travel photos.',
                     'If I liked travel photos, I would frame them.',
                     'Jordan wrote: I love travel photos.'):
            self.assertEqual(self.eligible(text, 'you love travel photos', quote='travel photos'), ['B'])

    def test_interest_does_not_prove_ownership_or_frequency(self):
        text = 'Which anime studios produce beautiful animation?'
        self.assertEqual(self.eligible(text, "you’re into anime", basis='topic_interest'), ['A'])
        for claim in ('you collect anime DVDs', 'you watch anime daily'):
            self.assertEqual(self.eligible(text, claim, basis='topic_interest'), ['B'])

    def test_persona_survives_headerless_excerpt_and_is_hash_protected(self):
        original = dict(request_id='persona', message_index=0, role='user',
            content='[system] User persona: {"padding": "' + 'x' * 500 + '.",\n"hobby": "vinyl records"}')
        # Identity is stamped before passage selection, including direct passages callers.
        excerpts = evidence_units.passages(original, [], ['vinyl records'], 140)
        self.assertTrue(excerpts)
        self.assertEqual(excerpts[0].get('source_role'), 'persona')
        row = memory('placeholder')
        row['sources'] = excerpts
        from app import answer_context
        row['content'] = answer_context.with_evidence('Profile', excerpts)
        packet = seal(row)
        cards, _ = self.gate.build_catalog(packet)
        self.assertEqual(next(iter(cards.values())).get('declared'), 'persona')
        self.assertEqual(self.gate._cards(cards)[0]['role'], 'persona (first-party profile)')
        packet[0]['sources'][0]['source_role'] = 'user'
        with self.assertRaises(ValueError):
            self.gate.build_catalog(packet)

    def test_proposed_advice_is_removed_but_existing_ownership_remains(self):
        options = ['A. You could try taking a walk.', 'B. Rest.']
        payload = assessment('taking a walk', 'missing', 'walk')
        entries = self.gate.validate_assessments(payload, options, {})
        self.assertEqual(entries[0]['status'], 'generic')
        self.assertEqual(entries[0]['claims'], [])
        payload = assessment('your own camera', 'missing', 'camera')
        entries = self.gate.validate_assessments(payload,
            ['A. You could try using your own camera.', 'B. Rest.'], {})
        self.assertEqual(entries[0]['status'], 'unsupported')
        self.assertTrue(choice_premises.is_suggestion('connect them with the school counselor',
            'Acknowledge their courage, and, when needed, connect them with the school counselor.'))
        self.assertTrue(choice_premises.is_suggestion('taking a short walk around campus',
            'Afterward, give yourself time to decompress—perhaps by taking a short walk around campus.'))

    def test_neighbors_are_context_only_and_same_request(self):
        cards = {'s0': dict(id='s0', role='user', text='I keep an eye on this project.', request_id='r', message_index=1),
                 's1': dict(id='s1', role='assistant', text='This is a crypto project.', request_id='r', message_index=0),
                 's2': dict(id='s2', role='user', text='I own crypto.', request_id='other', message_index=0)}
        options = ['A. Since you follow this project, read this.', 'B. Read this.']
        entries = self.gate.validate_assessments(assessment('you follow this project', 's0', cards['s0']['text']), options, cards)
        check = self.gate.entailment_checks(entries, cards, options)[0]
        neighbors = check.get('context_neighbors', [])
        self.assertEqual([c['id'] for c in neighbors], ['s1'])
        self.assertFalse(neighbors[0]['anchor'])


class SelectionRepairs(unittest.TestCase):
    def test_topic_alias_keeps_user_question_not_just_assistant_keywords(self):
        user = source('user', 'Why do European LPs sound different?')
        assistant = dict(source('assistant', 'Rare vintage vinyl collecting from international artists uses record marketplaces.'), role='assistant')
        rows = [candidate('essay', .99, [assistant]), candidate('irrelevant', .9, [source('noise', 'I read newspapers.')]),
                candidate('original', .005, [user])]
        requirements = [dict(id='option:1', origin='option_premise', text='rare vintage vinyl collecting from international artists, record marketplaces')]
        trace = {}
        result = evidence_selection.select(rows, 2, requirements, lambda c: c['_fused'], trace=trace)
        self.assertIn('original', [c['id'] for c in result])
        self.assertTrue(any(r['candidate_id'] == 'original' and 'option_topic_coverage' in r['reasons'] for r in trace['reservations']))
    def test_reserved_duplicates_do_not_consume_unique_budget(self):
        shared = source('same', 'I use Facebook for family updates.')
        rows = [candidate('a', .9, [shared]), candidate('b', .8, [shared]),
                candidate('c', .01, [source('new', 'I collect vinyl records.')])]
        result = evidence_selection.select(rows, 2, [], lambda c: c['_fused'], carry_ids=['a', 'b'])
        self.assertEqual([c['id'] for c in result], ['a', 'c'])

    def test_option_champion_uses_visible_evidence_below_ce_cut(self):
        from app import search_coverage
        rows = [candidate('a', .9, [source('a', 'I read newspapers.')]),
                candidate('b', .8, [source('b', 'I read magazines.')]),
                candidate('vinyl', .01, [source('v', 'I collect vinyl records and rare pressings.')])]
        requirements = [dict(id='option:0', origin='option_premise', text='vinyl records rare pressings')]
        for row in rows:
            search_coverage.annotate(row, requirements, row['_packet_item']['sources'])
        result = evidence_selection.select(rows, 2, requirements, lambda c: c['_fused'])
        self.assertIn('vinyl', [c['id'] for c in result])


class AnswerFlowRepairs(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        from app import answer_choice
        self.gate = answer_choice
        self.qa = dict(question='What would you recommend?', options=['A. Since you own a camera, take a photo.', 'B. Take a break.'])

    async def test_review_can_restore_only_original_rejected_supported_checks(self):
        packet = seal(memory('I own a camera.'))
        support = assessment('you own a camera', 's0', 'I own a camera.')
        mock = staged_mock(support, lambda p: entailment_response(p, {'A:0', 'A:option'}), entailment_response)
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa, packet, diagnostics), 'A')
        self.assertEqual(mock.await_count, 3)
        self.assertEqual(diagnostics['answer_entailment_review']['restored_check_ids'], ['A:0', 'A:option'])

    async def test_review_failure_keeps_first_verdict(self):
        packet = seal(memory('I own a camera.'))
        support = assessment('you own a camera', 's0', 'I own a camera.')
        mock = staged_mock(support, lambda p: entailment_response(p, {'A:0'}), {'checks': []})
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa, packet, diagnostics), 'B')
        self.assertEqual(mock.await_count, 3)
        self.assertEqual(diagnostics['answer_entailment_review']['status'], 'kept_first_verdict')

    async def test_review_budget_exhaustion_does_not_spend_the_final_call(self):
        packet = seal(memory('I own a camera.'))
        support = assessment('you own a camera', 's0', 'I own a camera.')
        mock = staged_mock(support, lambda p: entailment_response(p, {'A:0'}))
        diagnostics = {}
        with budget.scope(seconds=240, calls=2, tokens=128000), patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa, packet, diagnostics), 'B')
        self.assertEqual(mock.await_count, 2)
        self.assertEqual(diagnostics['answer_entailment_review']['status'], 'skipped_budget')

    async def test_format_fallback_verifies_whole_options_before_selecting_generic(self):
        mock = staged_mock({'options': []}, {'options': []}, lambda p: entailment_response(p, {'A:option'}))
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa, [], diagnostics), 'B')
        self.assertEqual(mock.await_count, 3)
        self.assertEqual(diagnostics['answer_support_fallback']['mode'], 'generic_only')
        self.assertIn(self.qa['options'][0], mock.call_args_list[2].args[0])

    async def test_provider_parse_failure_repairs_then_uses_verified_generic_fallback(self):
        from app import metrics
        async def respond(prompt, **kwargs):
            if kwargs['stage'].startswith('eval.choice_support'):
                try:
                    raise metrics.ResponseParseError({})
                except metrics.ResponseParseError as exc:
                    raise llm.LLMError('Malformed tool output') from exc
            return entailment_response(prompt, {'A:option'})
        mock = AsyncMock(side_effect=respond)
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa, [], {}), 'B')
        self.assertEqual(mock.await_count, 3)

    async def test_transport_failure_is_not_treated_as_malformed_support(self):
        mock = AsyncMock(side_effect=llm.LLMError('Provider unavailable'))
        with patch.object(llm, 'complete_json', mock), self.assertRaises(llm.LLMError):
            await self.gate.answer(self.qa, [], {})
        self.assertEqual(mock.await_count, 1)

    async def test_format_fallback_still_enforces_constraints(self):
        import json, re
        packet = seal(memory('Please forget that I own a camera.', constraint=True))
        self.qa['options'][1] = 'B. Buy a camera.'
        def constraints(prompt):
            pairs = json.loads(re.search(r'<pairs>\s*(.*?)\s*</pairs>', prompt, re.S)[1])
            return dict(decisions=[dict(pair_id=p['pair_id'], violates=True) for p in pairs])
        mock = staged_mock({'options': []}, {'options': []},
                           lambda p: entailment_response(p, {'A:option'}), constraints)
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock), self.assertRaises(ValueError):
            await self.gate.answer(self.qa, packet, diagnostics)
        self.assertIn('B', diagnostics['answer_blocked_options'])


if __name__ == '__main__':
    unittest.main()
