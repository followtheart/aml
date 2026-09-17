"""Regression cases for the PersonaMem evidence-loss and invented-citation bugs."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, choice_alignment as align, config, evidence_packet
from app import personal_evidence as personal, schemas, search_pipeline as search, store
from validate_choice_alignment import pair_traces


def source(text, index=0, role='user'):
    return dict(request_id='r', message_index=index, content=text, role=role, timestamp=None)


def entry(letter='A', claim='you enjoy anime', evidence='enjoys anime', **changes):
    result = dict(letter=letter, option_claim=claim, kind='persona', match='strong',
                  evidence_id='a', evidence=evidence, unsupported_claims=[], forbidden=False,
                  constraint_id='', constraint_span='')
    result.update(changes)
    return result


class ViewAndPacketTests(unittest.TestCase):
    def test_user_facts_and_plans_survive_while_assistant_knowledge_does_not(self):
        rows = [dict(id='p', type='preference', content='The user likes anime'),
                dict(id='f', type='fact', content='Daniel promised a convention trip', sources=[source('I promised a convention trip')]),
                dict(id='l', type='plan', content='Daniel adjusts his schedule for pollen', sources=[source('Pollen affects my comfort')]),
                dict(id='w', type='fact', content='Water stores heat', sources=[source('Water stores heat', role='assistant')])]
        plan = {'intent': 'preference'}
        self.assertEqual([r['id'] for r in search._select_memory_view([rows], plan)[0]], ['p', 'f', 'l'])
        self.assertEqual(plan['_view_excluded'][0]['id'], 'w')

    def test_personal_view_also_applies_to_options_without_profile_rows(self):
        rows = [dict(id='f', type='fact', content='The user owns a small garden')]
        result = search._select_memory_view([rows], {'intent': 'fact', '_personalization': True})
        self.assertEqual(result, [rows])

    def test_mixed_dialog_world_fact_is_not_personal_evidence(self):
        item = dict(type='fact', content='Regular exercise improves insulin sensitivity.',
                    sources=[source('How does exercise help?'),
                             source('Regular exercise improves insulin sensitivity.', 1, 'assistant')])
        self.assertFalse(personal.personal(item))

    def test_recorded_named_personal_facts_and_episodes_survive(self):
        texts = [
            'Daniel Robert Whitaker, a high school social studies teacher in Douglas County, Kansas, is adjusting his American Government class schedule to accommodate outdoor learning on days with manageable pollen levels due to increased pollen affecting his comfort and focus.',
            'Daniel Robert Whitaker, a high school social studies teacher in Lawrence, Kansas, is considering joining a community cycling fundraiser along the river trail.',
            "Daniel Robert Whitaker's routine has become more home-and-work centered due to his limited mobility.",
            'Daniel has a small garden', 'Daniel lives in Kansas', 'Daniel does yoga daily',
        ]
        for text in texts:
            self.assertTrue(personal.personal(dict(type='fact', content=text), [source('I have changed my plans. My health is affecting my life.')]), text)

    def test_capitalized_world_subject_with_user_question_is_not_personal(self):
        for text in ['Water is an effective heat sink.',
                     'Japan has a legal framework for cryptocurrencies.',
                     'Exercise has cardiovascular benefits.']:
            ss = [source('I am curious how this works. Could you explain it?'), source(text, 1, 'assistant')]
            self.assertFalse(personal.personal(dict(type='fact', content=text), ss), text)

    def test_general_choice_fact_view_keeps_assistant_recommendations(self):
        item = dict(id='a', type='fact', content='The assistant recommended Cafe Roma.',
                    sources=[source('Try Cafe Roma.', role='assistant')])
        self.assertEqual(search._select_memory_view([[item]], {'intent': 'fact'}), [[item]])

    def test_validated_assistant_context_and_late_user_support_are_retained(self):
        ss = [source('Were you living in Paris in 2020?', 0, 'assistant'),
              source('Yes, that was my previous home.', 1), source('Hello', 2),
              source('I keep a small garden.', 3)]
        candidate = dict(content='The user lived in Paris and has a garden', evidence=[
            dict(request_id='r', message_index=i, quote=ss[i]['content']) for i in [0, 1, 3]])
        result = personal.compact_sources(candidate, ss, 'home', 128, 1, persona=True)
        self.assertEqual([s['message_index'] for s in result if s.get('content')], [0, 1, 3])
        self.assertIn('Paris in 2020', result[0]['content'])

    def test_unknown_prefix_is_empty_but_relative_and_known_dates_survive(self):
        self.assertEqual(search._time_prefix(dict(temporal={'start': None, 'end': None, 'precision': 'unknown', 'raw': None})), '')
        prefix = search._time_prefix(dict(temporal={'start': None, 'end': None, 'precision': 'unknown', 'raw': 'last winter'}))
        self.assertIn('last winter', prefix)
        self.assertIn('event: unknown', prefix)
        self.assertIn('2025', search._time_prefix(dict(event_time='2025')))
        self.assertIn('valid:', search._time_prefix(dict(valid_to='2025-01-01')))

    def test_unrelated_core_cannot_displace_ranked_evidence(self):
        st = store.Store(':memory:')
        try:
            ids = [st.insert_amu(user_id='u', session_id='s', type='profile',
                                content='The user follows basketball ' + 'x' * 1000) for _ in range(12)]
            target = st.insert_amu(user_id='u', session_id='s', content='The user promised a convention trip', type='fact')
            ranked = st.get_amus_by_ids([target])
            ranked[0]['_final'] = .8
            packet, _, manifest = search._pack_evidence(st, schemas.SearchRequest(
                user_id='u', query='convention trip', evidence_token_budget=512),
                {'intent': 'preference'}, ranked, '2026-09-17T00:00:00+00:00')
            self.assertEqual([m['id'] for m in packet], [target])
            self.assertEqual(manifest['core_bytes'], 0)
        finally:
            st.conn.close()

    def test_supplemental_core_budget_reserves_space_for_query_evidence(self):
        items = [dict(id='core', content='relevant but long ' * 20, _core_injected=True),
                 dict(id='target', content='The user has a garden')]
        packet, _, manifest = evidence_packet.pack(items, 5, 512, core_budget=100)
        self.assertEqual([m['id'] for m in packet], ['target'])
        self.assertEqual(manifest['omitted'][0]['reason'], 'core_budget')

    def test_excerpt_preserves_support_and_negation_before_hash(self):
        text = 'Background filler. ' * 80 + 'I do not have asthma. I love cycling. More background.'
        candidate = dict(content='The user does not have asthma', evidence=[
            dict(request_id='r', message_index=0, quote='I do not have asthma')])
        ss = personal.compact_sources(candidate, [source(text)], 'asthma', 128, 3, persona=True)
        shown = ss[0]['content']
        self.assertIn('I do not have asthma', shown)
        self.assertLess(len(shown), len(text))
        span = ss[0]['content_span']
        self.assertEqual(text[span['start']:span['end']], shown)
        packet, digest, _ = evidence_packet.pack([dict(id='a', content=candidate['content'], sources=ss)], 5, 2000)
        self.assertEqual(answer_context.build(packet), packet[0]['content'])
        self.assertEqual(evidence_packet.digest(packet), digest)
        self.assertEqual(text, source(text)['content'])

    def test_different_excerpts_of_same_source_are_not_deduplicated(self):
        one = dict(source('I like anime'), content_span=dict(start=0, end=12))
        two = dict(source('I have a garden'), content_span=dict(start=50, end=65))
        packet, _, _ = evidence_packet.pack([
            dict(id='a', content='anime interest', sources=[one]),
            dict(id='b', content='garden', sources=[two])], 5, 2000)
        self.assertIn('I have a garden', packet[1]['content'])

    def test_metadata_only_reference_does_not_hide_later_visible_quote(self):
        reference = source('ignored')
        reference.pop('content')
        packet, _, _ = evidence_packet.pack([
            dict(id='a', content='a', sources=[reference]),
            dict(id='b', content='b', sources=[source('real evidence')])], 5, 2000)
        self.assertIn('real evidence', packet[1]['content'])

    def test_assistant_evidence_remains_for_nonpersonal_task(self):
        ss = personal.compact_sources(dict(content='Water stores heat'),
                                      [source('Water stores heat', role='assistant')], 'heat', 128, 3)
        self.assertEqual(ss[0]['content'], 'Water stores heat')


class AlignmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.options = ['A. Since you enjoy anime, try this series', 'B. Try a popular series']
        self.pool = {'a': dict(text='The user enjoys anime')}
        self.generic = entry('B', claim='', evidence='', evidence_id='', kind='generic', match='none')

    def test_fabricated_source_quote_and_unknown_id_are_rejected(self):
        for bad in [entry(evidence='user taught during remote schooling'), entry(evidence_id='not-in-packet')]:
            out = align.validate_entries([bad, self.generic], self.options, self.pool, {})
            self.assertEqual(out[0]['match'], 'none')
            self.assertFalse(out[0]['citation_valid'])
            self.assertEqual(out[0]['evidence'], '')
            self.assertIsNone(align.unique_supported_choice(out))

    def test_generic_mislabel_does_not_erase_cited_persona_premise(self):
        out = align.validate_entries([entry(kind='generic'), self.generic], self.options, self.pool, {})
        self.assertEqual((out[0]['kind'], out[0]['match']), ('persona', 'strong'))

    def test_unsupported_premise_prevents_strong_autopick(self):
        out = align.validate_entries([entry(unsupported_claims=['you enjoy anime']), self.generic], self.options, self.pool, {})
        self.assertEqual(out[0]['match'], 'weak')
        self.assertIsNone(align.unique_supported_choice(out))

    def test_fabricated_option_claim_does_not_gain_support(self):
        out = align.validate_entries([entry(claim='you own 300 anime DVDs'), self.generic], self.options, self.pool, {})
        self.assertEqual(out[0]['match'], 'none')

    def test_missing_and_duplicate_options_reject_entire_judgement(self):
        for rows in [[entry()], [entry(), entry()]]:
            with self.assertRaises(ValueError):
                align.validate_entries(rows, self.options, self.pool, {})

    def test_forbidden_needs_real_scoped_constraint_reference(self):
        out = align.validate_entries([entry(forbidden=True, constraint_id='invented', constraint_span='anime'), self.generic], self.options, self.pool, {})
        self.assertFalse(out[0]['forbidden'])
        constraint = {'r': dict(text='Please forget that I enjoy anime')}
        out = align.validate_entries([entry(forbidden=True, constraint_id='r', constraint_span='forget that I enjoy anime'), self.generic], self.options, self.pool, constraint)
        self.assertTrue(out[0]['forbidden'])

    def test_ties_cannot_autopick_even_with_opt_in(self):
        a = dict(entry(), citation_valid=True)
        b = dict(entry(letter='B'), citation_valid=True)
        self.assertIsNone(align.unique_supported_choice([a, b]))
        self.assertIsNone(align.unique_supported_choice([dict(a, match='none'), self.generic]))

    async def test_all_user_memory_types_and_source_text_reach_alignment(self):
        memories = [dict(id=f'm{i}', memory_type=kind, content='The user asked about yoga',
                         sources=[source('I have practiced yoga for years', i)])
                    for i, kind in enumerate(['fact', 'plan', 'episode', 'event'])]
        packet, _, _ = evidence_packet.pack(memories, 10, 5000)
        qa = dict(question='Ideas?', options=self.options)
        rows = [entry(match='none', evidence='', evidence_id=''), self.generic]
        call = AsyncMock(return_value={'options': rows})
        diagnostics = {}
        with patch.object(align.llm, 'complete_json', call):
            text, _ = await align.align(qa, packet, diagnostics)
        prompt = call.call_args.args[0]
        self.assertIn('practiced yoga for years', prompt)
        self.assertEqual(diagnostics['alignment_input_ids'], ['m0', 'm1', 'm2', 'm3'])
        self.assertIn('A = B', text)
        self.assertIn('do not break ties by letter', text)

    def test_modified_packet_cannot_be_aligned(self):
        packet, _, _ = evidence_packet.pack([dict(id='a', memory_type='fact', content='The user likes anime')], 5, 1000)
        packet[0]['content'] += ' changed'
        with self.assertRaises(ValueError):
            align.evidence_pool(packet)

    def test_alignment_metadata_is_part_of_new_packet_hash(self):
        packet, _, _ = evidence_packet.pack([dict(id='a', memory_type='fact',
            content='The user likes anime', sources=[source('I like anime')])], 5, 1000)
        for field in ['role', 'memory_type', 'personal_evidence']:
            changed = copy.deepcopy(packet)
            if field == 'role':
                changed[0]['sources'][0]['role'] = 'assistant'
            elif field == 'memory_type':
                changed[0]['memory_type'] = 'rule'
            else:
                changed[0]['personal_evidence'] = False
            with self.assertRaises(ValueError):
                align.evidence_pool(changed)

    def test_source_dedup_does_not_change_verified_personal_eligibility(self):
        sources = [source('I live in Kansas. I enjoy gardening.')]
        packet, _, _ = evidence_packet.pack([
            dict(id='a', content='Alice lives in Kansas', memory_type='fact', sources=sources, personal_evidence=True),
            dict(id='b', content='Alice enjoys gardening', memory_type='fact', sources=sources, personal_evidence=True)], 5, 2000)
        self.assertNotIn('content', packet[1]['sources'][0])
        pool, _ = align.evidence_pool(packet)
        self.assertEqual(set(pool), {'a', 'b'})


class ReplayPairingTests(unittest.TestCase):
    def test_matching_ignores_fake_tests_and_does_not_zip_by_order(self):
        conv = dict(dataset='d', conversation_id='c', qa=[
            dict(id='1', question='one', options=['A. a']),
            dict(id='2', question='two', options=['B. b'])])
        traces = [dict(user_id='local:d:c', query='two', options=['B. b'], fake=False, returned=[]),
                  dict(user_id='u', query='one', options=['A. a'], fake=True, returned=[]),
                  dict(user_id='local:d:c', query='one', options=['A. a'], fake=False, returned=[])]
        paired = pair_traces(traces, [conv])
        self.assertEqual([(q['id'], t['query']) for _, q, t in paired], [('1', 'one'), ('2', 'two')])
        with self.assertRaises(ValueError):
            pair_traces(traces + [traces[0]], [conv])

    def test_same_question_with_different_options_is_not_silently_matched(self):
        conv = dict(dataset='d', conversation_id='c', qa=[dict(id='1', question='one', options=['A. yes'])])
        trace = dict(user_id='local:d:c', query='one', options=['A. no'], fake=False, returned=[])
        with self.assertRaises(ValueError):
            pair_traces([trace], [conv])


if __name__ == '__main__':
    unittest.main()
