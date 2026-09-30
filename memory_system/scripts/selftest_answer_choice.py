"""Offline answer-gate regressions; --legacy-red probes the old direct evaluator.

The explicit legacy mode deliberately fails when a mocked model's inadmissible
choice is accepted. It is not part of normal self-test discovery and has no xfail.
"""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import re
import subprocess
import types
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['AML_FAKE'] = '1'
_read_text = Path.read_text


def _without_dotenv(path, *args, **kwargs):
    return '' if path.name == '.env' else _read_text(path, *args, **kwargs)


with patch.object(Path, 'read_text', _without_dotenv):
    from app import answer_context, budget, config, eval_scoring, evidence_packet, llm


def memory(text, *, role='user', mid='m0', summary='A retrieved memory.',
           constraint=False, personal=False):
    source = dict(request_id='answer-test:' + mid, message_index=0, role=role, timestamp=1700000000000,
                  source_event_id='source:' + mid, content=text)
    return dict(id=mid, content=answer_context.with_evidence(summary, [source]),
                memory_type='rule' if constraint else 'fact', sources=[source],
                personal_evidence=personal, is_constraint=constraint,
                packet_hash_version=4)


def seal(*items):
    packet = copy.deepcopy(list(items))
    digest = evidence_packet.digest(packet)
    for item in packet:
        item['packet_hash'] = digest
    return packet


def assessment(claim, sid, quote, *, basis='self_report', subject='current_user'):
    return {'options': [
        dict(letter='A', kind='personal', claims=[dict(text=claim, status='supported',
            citations=[dict(source_id=sid, quote=quote, basis=basis, subject=subject)])]),
        dict(letter='B', kind='generic', claims=[])]}


def entailed(*claim_ids):
    return {'checks': [dict(claim_id=cid, entailed=True) for cid in claim_ids]}


def entailment_response(prompt, rejected=()):
    match = re.search(r'<checks>\s*(.*?)\s*</checks>', prompt, flags=re.S)
    if match is None:
        raise AssertionError('Entailment prompt must expose its complete checks')
    checks = json.loads(match[1])
    return {'checks': [dict(claim_id=check['claim_id'], verdict=(
        'unsupported_personal' if check['claim_id'] in rejected else 'personal_supported'))
                       for check in checks]}


def staged_mock(*responses):
    remaining = iter(responses)

    async def respond(prompt, **kwargs):
        response = next(remaining)
        response = response(prompt) if callable(response) else copy.deepcopy(response)
        # Existing fixtures describe semantic intent with bools. Translate
        # only the fake wire response when the production schema is typed.
        if (kwargs.get('schema', {}).get('title') == 'TypedEntailments'
                and isinstance(response, dict) and isinstance(response.get('checks'), list)):
            for check in response['checks']:
                if 'entailed' in check and 'verdict' not in check:
                    check['verdict'] = ('personal_supported' if check.pop('entailed') else
                                        'unsupported_personal')
        return response

    return AsyncMock(side_effect=respond)


class OfflineCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')).start()
        patch.object(llm, 'complete', AsyncMock(side_effect=AssertionError('Unexpected model call'))).start()
        patch.object(llm, 'complete_json', AsyncMock(side_effect=AssertionError('Unexpected model call'))).start()
        self.gate = importlib.import_module('app.answer_choice')

    def catalog(self, text, **kwargs):
        return self.gate.build_catalog(seal(memory(text, **kwargs)))

    def eligible(self, source_text, claim, *, quote=None, role='user',
                 basis='self_report', subject='current_user'):
        sources, _ = self.catalog(source_text, role=role)
        sid = next(iter(sources))
        options = ['A. Since ' + claim + ', take a break.', 'B. Take a break.']
        payload = assessment(claim, sid, source_text if quote is None else quote,
                             basis=basis, subject=subject)
        entries = self.gate.validate_assessments(payload, options, sources)
        return self.gate.eligible_choices(entries, set())


class CatalogTests(OfflineCase):
    def test_visible_sources_survive_false_personal_flag_and_keep_real_role(self):
        packet = seal(memory('I have twin boys.', mid='user'),
                      memory('Try taking a break.', mid='assistant', role='assistant'))
        before = copy.deepcopy(packet)
        sources, constraints = self.gate.build_catalog(packet)
        self.assertEqual({s['text']: s['role'] for s in sources.values()},
                         {'I have twin boys.': 'user', 'Try taking a break.': 'assistant'})
        self.assertEqual({mid for s in sources.values() for mid in s['item_ids']}, {'user', 'assistant'})
        self.assertEqual(constraints, {})
        self.assertEqual(packet, before)

    def test_packet_role_mutation_is_rejected_before_catalog(self):
        packet = seal(memory('I have twin boys.', role='assistant'))
        packet[0]['sources'][0]['role'] = 'user'
        with self.assertRaises(ValueError):
            self.gate.build_catalog(packet)

    def test_unsourced_legacy_summary_never_becomes_original_evidence(self):
        sources, constraints = self.gate.build_catalog([
            dict(id='legacy', content='The user owns a camera.', memory_type='profile')])
        self.assertEqual(sources, {})
        self.assertEqual(constraints, {})

    def test_hidden_source_body_is_not_an_evidence_card(self):
        row = memory('I have twin boys.')
        row['content'] = 'Only this summary was visible.'
        row['sources'].append(dict(role='user', content_omitted='source_limit', request_id='hidden'))
        sources, _ = self.gate.build_catalog(seal(row))
        self.assertEqual(sources, {})

    def test_same_source_in_two_items_has_one_card_with_both_item_ids(self):
        first = memory('I have twin boys.', mid='one')
        second = copy.deepcopy(first)
        second['id'] = 'two'
        sources, _ = self.gate.build_catalog(seal(first, second))
        self.assertEqual(len(sources), 1)
        self.assertEqual(set(next(iter(sources.values()))['item_ids']), {'one', 'two'})

    def test_only_actual_user_forget_source_creates_constraint(self):
        cases = [memory('Please forget that I enjoy photography.', constraint=True),
                 memory('Please forget that I enjoy photography.', constraint=True, role='assistant'),
                 memory('I enjoy photography.', constraint=True, summary='Please forget photography.'),
                 memory('Claire wrote: Please forget that I enjoy photography.', constraint=True)]
        for index, item in enumerate(cases):
            with self.subTest(index=index):
                _, constraints = self.gate.build_catalog(seal(item))
                self.assertEqual(bool(constraints), index == 0)


class AssessmentTests(OfflineCase):
    def test_direct_user_statement_and_generic_remain_eligible(self):
        self.assertEqual(self.eligible('I have twin boys.', 'you have twin boys'), ['A', 'B'])

    def test_assistant_first_person_cannot_prove_current_user_history(self):
        self.assertEqual(self.eligible('I have twin boys.', 'you have twin boys', role='assistant'), ['B'])

    def test_inner_first_person_quote_keeps_third_party_attribution(self):
        self.assertEqual(self.eligible('Claire wrote: I have twin boys.', 'you have twin boys',
                                       quote='I have twin boys.'), ['B'])

    def test_my_friend_subject_does_not_prove_current_user_ownership(self):
        self.assertEqual(self.eligible('My friend owns a camera.', 'you own a camera'), ['B'])

    def test_quote_cannot_remove_its_denial_or_hypothetical_frame(self):
        for text in ('I never said that I own a camera.',
                     'If I owned a camera, I would say: I own a camera.'):
            with self.subTest(text=text):
                self.assertEqual(self.eligible(text, 'you own a camera', quote='I own a camera'), ['B'])

    def test_unrelated_previous_sentence_does_not_erase_a_real_self_report(self):
        for prefix in ('My friend owns a camera.', 'I never owned a camera.',
                       'If I owned a camera, I would take pictures.'):
            with self.subTest(prefix=prefix):
                self.assertEqual(self.eligible(prefix + ' I have twin boys.', 'you have twin boys',
                                               quote='I have twin boys.'), ['A', 'B'])

    def test_user_hypothetical_does_not_prove_ownership(self):
        self.assertEqual(self.eligible('Imagine this fictional scenario: I own a camera.',
                                       'you own a camera', quote='I own a camera.'), ['B'])

    def test_negated_user_fact_does_not_support_positive_option(self):
        for text in ('I do not own a camera.', 'I have never owned a camera.'):
            with self.subTest(text=text):
                self.assertEqual(self.eligible(text, 'you own a camera'), ['B'])

    def test_negative_self_report_can_support_the_same_negative_personal_premise(self):
        self.assertEqual(self.eligible('I do not own a camera.', 'you do not own a camera'), ['A', 'B'])

    def test_topic_question_can_support_interest_but_not_ownership(self):
        text = 'What camera should I buy for photography?'
        self.assertEqual(self.eligible(text, 'you are interested in photography', basis='topic_interest'), ['A', 'B'])
        self.assertEqual(self.eligible(text, 'you own a camera', basis='topic_interest'), ['B'])

    def test_self_report_citation_of_a_topic_question_falls_back_to_interest(self):
        text = 'Which anime studios produce the best sakuga this season?'
        self.assertEqual(self.eligible(text, 'you enjoy anime'), ['A', 'B'])
        self.assertEqual(self.eligible(text, 'you are interested in anime'), ['A', 'B'])
        self.assertEqual(self.eligible(text, 'you work professionally in anime'), ['B'])
        self.assertEqual(self.eligible(text, 'you own an anime studio'), ['B'])
        sources, _ = self.catalog(text)
        payload = assessment('you enjoy anime', next(iter(sources)), text)
        entries = self.gate.validate_assessments(payload, ['A. Since you enjoy anime, watch a film.', 'B. Rest.'], sources)
        self.assertEqual(entries[0]['claims'][0]['citations'][0]['basis'], 'topic_interest')

    def test_context_basis_uses_actual_source_scope_and_role(self):
        self.assertEqual(self.eligible('I have twin boys.', 'you have twin boys', basis='context'), ['A', 'B'])
        self.assertEqual(self.eligible('I have twin boys.', 'you have twin boys',
                                       basis='context', role='assistant'), ['B'])
        self.assertEqual(self.eligible('I have twin boys.', 'you have twin boys', subject='unknown'), ['B'])
        self.assertEqual(self.eligible('I have twin boys.', 'you have twin boys',
                                       basis='context', subject='unknown'), ['B'])

    def test_context_label_cannot_launder_third_party_hypothetical_or_denial(self):
        cases = [('Claire wrote: I own a camera.', 'I own a camera.'),
                 ('My friend owns a camera.', 'My friend owns a camera.'),
                 ('Imagine this fictional scenario: I own a camera.', 'I own a camera.'),
                 ('I never said that I own a camera.', 'I own a camera'),
                 ('If I owned a camera, I would say: I own a camera.', 'I own a camera')]
        for text, quote in cases:
            with self.subTest(text=text):
                self.assertEqual(self.eligible(text, 'you own a camera', quote=quote, basis='context'), ['B'])

    def test_claim_case_variation_preserves_actual_option_substring(self):
        sources, _ = self.catalog('I have twin boys.')
        payload = assessment('YOU HAVE TWIN BOYS', next(iter(sources)), 'I have twin boys.')
        options = ['A. Since You Have Twin Boys, rest.', 'B. Take a break.']
        entries = self.gate.validate_assessments(payload, options, sources)
        self.assertEqual(self.gate.eligible_choices(entries, set()), ['A', 'B'])
        self.assertEqual(entries[0]['claims'][0]['text'], 'You Have Twin Boys')

    def test_claim_case_tolerance_does_not_allow_paraphrase(self):
        sources, _ = self.catalog('I have twin boys.')
        payload = assessment('you are the parent of twin boys', next(iter(sources)), 'I have twin boys.')
        entries = self.gate.validate_assessments(payload,
            ['A. Since you have twin boys, rest.', 'B. Take a break.'], sources)
        self.assertEqual(self.gate.eligible_choices(entries, set()), ['B'])
        self.assertIn('claim_not_in_option', entries[0]['validation_errors'])

    def test_citation_case_stays_verbatim_even_when_claim_case_varies(self):
        sources, _ = self.catalog('I have twin boys.')
        payload = assessment('YOU HAVE TWIN BOYS', next(iter(sources)), 'i have twin boys.')
        entries = self.gate.validate_assessments(payload,
            ['A. Since you have twin boys, rest.', 'B. Take a break.'], sources)
        self.assertEqual(self.gate.eligible_choices(entries, set()), ['B'])

    def test_invalid_citation_downgrades_support_without_inventing_fallback(self):
        sources, _ = self.catalog('I have twin boys.', summary='The user has triplets.')
        sid = next(iter(sources))
        options = ['A. Since you have twin boys, rest.', 'B. Take a break.']
        for fake_sid, quote in [(sid, 'The user has triplets.'), (sid, 'I have three boys.'),
                                ('missing', 'I have twin boys.')]:
            with self.subTest(sid=fake_sid, quote=quote):
                entries = self.gate.validate_assessments(
                    assessment('you have twin boys', fake_sid, quote), options, sources)
                self.assertEqual(self.gate.eligible_choices(entries, set()), ['B'])

    def test_missing_secondary_claim_only_downgrades_to_partial(self):
        sources, _ = self.catalog('I have twin boys.')
        sid = next(iter(sources))
        options = ['A. Since you have twin boys and you own a camera, rest.', 'B. Take a break.']
        data = assessment('you have twin boys', sid, 'I have twin boys.')
        data['options'][0]['claims'].append(dict(text='you own a camera', status='unsupported', citations=[]))
        entries = self.gate.validate_assessments(data, options, sources)
        self.assertEqual(entries[0]['status'], 'partial')
        self.assertEqual(entries[0]['primary_claim'], 0)
        self.assertEqual(self.gate.eligible_choices(entries, set()), ['A', 'B'])

    def test_compound_interest_splits_verified_topic_from_unverified_preference(self):
        source = 'I’ve noticed some anime seem to deal with really weighty political or historical themes—what’s behind that storytelling approach?'
        sources, _ = self.catalog(source)
        sid = next(iter(sources))
        options = ['A. Since you’re into anime and enjoy layered stories with complex characters, try a series.',
                   'B. Try a popular series.']
        data = {'options': [
            dict(letter='A', kind='personal', claims=[dict(
                text='Since you’re into anime and enjoy layered stories with complex characters,',
                status='supported', premise_type='interest', citations=[dict(source_id=sid,
                    quote=source, basis='self_report', subject='current_user')])]),
            dict(letter='B', kind='generic', claims=[])]}
        entries = self.gate.validate_assessments(data, options, sources)
        diagnostics = {}
        entries = self.gate.split_compound_interest_claims(entries, sources, diagnostics)
        self.assertEqual([claim['text'] for claim in entries[0]['claims']],
                         ['Since you’re into anime', 'enjoy layered stories with complex characters'])
        self.assertEqual(entries[0]['claims'][0]['status'], 'supported')
        self.assertEqual(entries[0]['claims'][1]['status'], 'unsupported')
        self.assertEqual(entries[0]['primary_claim'], 0)
        self.assertEqual(entries[0]['status'], 'partial')
        self.assertEqual(self.gate.eligible_choices(entries, set()), ['A', 'B'])

    def test_missing_primary_claim_is_still_unsupported(self):
        sources, _ = self.catalog('I own a camera.')
        sid = next(iter(sources))
        options = ['A. Since you have twin boys and you own a camera, rest.', 'B. Take a break.']
        data = {'options': [
            dict(letter='A', kind='personal', claims=[
                dict(text='you have twin boys', status='unsupported', citations=[]),
                dict(text='you own a camera', status='supported',
                     citations=[dict(source_id=sid, quote='I own a camera.', basis='self_report', subject='current_user')])]),
            dict(letter='B', kind='generic', claims=[])]}
        entries = self.gate.validate_assessments(data, options, sources)
        self.assertEqual(entries[0]['status'], 'unsupported')
        self.assertEqual(self.gate.eligible_choices(entries, set()), ['B'])

    def test_verified_personal_cores_compete_across_evidence_tiers(self):
        packet = seal(memory('I have twin boys.', mid='boys'), memory('I own a camera.', mid='camera'))
        sources, _ = self.gate.build_catalog(packet)
        sid = {source['text']: key for key, source in sources.items()}
        options = ['A. Since you have twin boys and you live in Oslo, rest.',
                   'B. Since you own a camera, take photos.', 'C. Take a break.']
        data = {'options': [
            dict(letter='A', kind='personal', claims=[
                dict(text='you have twin boys', status='supported', citations=[dict(
                    source_id=sid['I have twin boys.'], quote='I have twin boys.', basis='self_report', subject='current_user')]),
                dict(text='you live in Oslo', status='unsupported', citations=[])]),
            dict(letter='B', kind='personal', claims=[dict(text='you own a camera', status='supported', citations=[dict(
                source_id=sid['I own a camera.'], quote='I own a camera.', basis='self_report', subject='current_user')])]),
            dict(letter='C', kind='generic', claims=[])]}
        entries = self.gate.validate_assessments(data, options, sources)
        self.assertEqual([e['status'] for e in entries], ['partial', 'supported', 'generic'])
        self.assertEqual(self.gate.eligible_choices(entries, set()), ['A', 'B', 'C'])

    def test_missing_duplicate_unknown_option_and_extra_fields_are_rejected(self):
        sources, _ = self.catalog('I have twin boys.')
        data = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        options = ['A. Since you have twin boys, rest.', 'B. Take a break.']
        variants = []
        missing = copy.deepcopy(data); missing['options'].pop(); variants.append(missing)
        duplicate = copy.deepcopy(data); duplicate['options'][1]['letter'] = 'A'; variants.append(duplicate)
        unknown = copy.deepcopy(data); unknown['options'][1]['letter'] = 'Z'; variants.append(unknown)
        extra = copy.deepcopy(data); extra['answer'] = 'A'; variants.append(extra)
        role = copy.deepcopy(data); role['options'][0]['claims'][0]['citations'][0]['role'] = 'user'; variants.append(role)
        empty_quote = copy.deepcopy(data); empty_quote['options'][0]['claims'][0]['citations'][0]['quote'] = ''; variants.append(empty_quote)
        bad_enum = copy.deepcopy(data); bad_enum['options'][0]['claims'][0]['status'] = True; variants.append(bad_enum)
        for index, payload in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.gate.validate_assessments(payload, options, sources)

    def test_personal_claim_cannot_be_laundered_into_generic_kind(self):
        sources, _ = self.catalog('I have twin boys.')
        payload = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        payload['options'][0]['kind'] = 'generic'
        entries = self.gate.validate_assessments(payload,
            ['A. Since you have twin boys, rest.', 'B. Take a break.'], sources)
        self.assertEqual(self.gate.eligible_choices(entries, set()), ['B'])

    def test_unsupported_personal_option_is_never_fallback_when_generic_blocked(self):
        sources, _ = self.catalog('I have twin boys.', role='assistant')
        entries = self.gate.validate_assessments(
            assessment('you have twin boys', next(iter(sources)), 'I have twin boys.'),
            ['A. Since you have twin boys, rest.', 'B. Take a break.'], sources)
        self.assertEqual(self.gate.eligible_choices(entries, {'B'}), [])


class EntailmentTests(OfflineCase):
    def fixture(self):
        packet = seal(memory('I have twin boys.', mid='boys'), memory('I own a camera.', mid='camera'))
        sources, _ = self.gate.build_catalog(packet)
        sid = {source['text']: key for key, source in sources.items()}
        options = ['A. Since you have twin boys and you own a camera, take photos.',
                   'B. Since you live in Oslo, go outside.', 'C. Take a break.']
        payload = assessment('you have twin boys', sid['I have twin boys.'], 'I have twin boys.')
        payload['options'][0]['claims'].append(assessment(
            'you own a camera', sid['I own a camera.'], 'I own a camera.')['options'][0]['claims'][0])
        payload['options'][1] = dict(letter='B', kind='personal', claims=[
            dict(text='you live in Oslo', status='unsupported', citations=[])])
        payload['options'].append(dict(letter='C', kind='generic', claims=[]))
        entries = self.gate.validate_assessments(payload, options, sources)
        checks = self.gate.entailment_checks(entries, sources, options)
        return sources, entries, checks

    def test_first_party_repeated_lab_question_supports_only_narrow_monitoring(self):
        quote = ('Why would cholesterol numbers change noticeably between two routine checkups a few months '
                 'apart, even if my diet and exercise stayed mostly the same?')
        check = dict(claim_id='A:0', check_type='premise',
                     claim='you’re keeping an eye on your cholesterol', premise_type='condition',
                     sources=[dict(id='s1', role='user', quote=quote, context=quote)])
        self.assertTrue(self.gate.direct_lab_monitoring_support(check))
        too_broad = dict(check, claim='you have high cholesterol')
        self.assertFalse(self.gate.direct_lab_monitoring_support(too_broad))
        wrong_topic = dict(check, claim='you’re trying to cut back on caffeine')
        self.assertFalse(self.gate.direct_lab_monitoring_support(wrong_topic))
        not_personal = dict(check, sources=[dict(id='s1', role='assistant', quote=quote, context=quote)])
        self.assertFalse(self.gate.direct_lab_monitoring_support(not_personal))
        unrelated = dict(check, sources=[dict(id='s1', role='user', quote=quote,
            context='Cholesterol trends can shift over time. Ask a doctor about results.')])
        self.assertFalse(self.gate.direct_lab_monitoring_support(unrelated))

        diagnostics = {}
        verdicts = self.gate.apply_direct_source_entailment_rules(
            {'checks': [dict(claim_id='A:0', entailed=False)]}, [check], diagnostics)
        self.assertTrue(verdicts['checks'][0]['entailed'])
        self.assertEqual(diagnostics['answer_entailment_rules']['claim_ids'], ['A:0'])

    def test_explanatory_topic_question_supports_only_narrow_interest(self):
        quote = ('I’ve noticed some anime seem to deal with really weighty political or historical themes—'
                 'what’s behind that storytelling approach?')
        check = dict(claim_id='D:0', check_type='premise', claim='Since you’re into anime',
                     premise_type='interest', sources=[dict(id='s9', role='user', quote=quote, context=quote)])
        self.assertTrue(self.gate.direct_topic_question_interest(check))
        sibling = dict(check, claim='enjoy layered stories with complex characters')
        self.assertFalse(self.gate.direct_topic_question_interest(sibling))
        stronger = dict(check, claim='you are an expert anime critic')
        self.assertFalse(self.gate.direct_topic_question_interest(stronger))
        advice_context = dict(check, sources=[dict(id='s9', role='user', quote=quote,
            context='Anime storytelling can cover many genres and themes.')])
        self.assertFalse(self.gate.direct_topic_question_interest(advice_context))

    def test_first_party_board_and_wave_joy_supports_only_surfing_passion(self):
        quote = ('I found myself waxing my board more than once just to enjoy the ritual, though the real joy '
                 'came in catching those early, still-glass waves before the crowds arrived.')
        check = dict(claim_id='C:0', check_type='premise',
            claim='Since you’re passionate about surfing when you’re near the coast',
            premise_type='interest', sources=[dict(id='s2', role='user',
                quote='the real joy came in catching those early, still-glass waves before the crowds arrived.',
                context=quote)])
        self.assertTrue(self.gate.direct_board_wave_joy_support(check))
        broader = dict(check, claim='you are an experienced surfer')
        self.assertFalse(self.gate.direct_board_wave_joy_support(broader))
        no_joy = dict(check, sources=[dict(id='s2', role='user', quote=quote,
            context='I used my board, and other people were catching waves.')])
        self.assertFalse(self.gate.direct_board_wave_joy_support(no_joy))
        no_board_context = dict(check, sources=[dict(id='s2', role='user',
            quote='the real joy came in catching those early, still-glass waves before the crowds arrived.',
            context='The real joy came in catching those early waves.')])
        self.assertFalse(self.gate.direct_board_wave_joy_support(no_board_context))
        assistant = dict(check, sources=[dict(id='s2', role='assistant', quote=quote, context=quote)])
        self.assertFalse(self.gate.direct_board_wave_joy_support(assistant))

    def test_direct_surfing_recovery_cites_exact_first_party_sentence_from_context(self):
        quote = ('There was something meditative in the simple act of waxing my board, though the true '
                 'exhilaration came in sliding across those glassy, first-light waves before the shoreline stirred with life.')
        sources, _ = self.catalog(quote)
        sid = next(iter(sources))
        claim = dict(text='Since you’re passionate about surfing when you’re near the coast',
            status='unsupported', reason='no_source', premise_type='interest', citations=[],
            validation_errors=[])
        entries = [dict(letter='C', kind='personal', status='unsupported', option='C. Since you’re passionate about surfing.',
            validation_status='valid', validation_errors=[], claims=[claim])]
        diagnostics = {}
        result = self.gate.recover_direct_board_wave_citations(entries, sources, diagnostics)
        attached = result[0]['claims'][0]['citations']
        self.assertEqual(len(attached), 1)
        self.assertEqual(attached[0]['source_id'], sid)
        self.assertEqual(attached[0]['quote'], quote)
        self.assertTrue(attached[0]['valid'])
        self.assertEqual(result[0]['claims'][0]['citation_recovery'], 'awaiting_entailment')
        check = dict(claim_id='C:0', check_type='premise', claim=claim['text'], sources=[dict(
            id=sid, role='user', quote=quote, context=sources[sid]['text'])])
        self.assertTrue(self.gate.direct_board_wave_joy_support(check))

        assistant_sources, _ = self.catalog(quote, role='assistant')
        assistant_claim = copy.deepcopy(claim)
        assistant_claim.update(status='unsupported', reason='no_source', citations=[], validation_errors=[])
        assistant_entries = [dict(letter='C', kind='personal', status='unsupported',
            option='C. Since you’re passionate about surfing.', validation_status='valid',
            validation_errors=[], claims=[assistant_claim])]
        self.gate.recover_direct_board_wave_citations(assistant_entries, assistant_sources, {})
        self.assertEqual(assistant_entries[0]['claims'][0]['citations'], [])

    def test_duplicate_citation_recovery_rows_merge_evidence_for_one_requested_claim(self):
        quote_a = 'I asked about anime storytelling.'
        quote_b = 'I watch a series I really enjoy.'
        citation = lambda sid, quote: dict(source_id=sid, quote=quote,
            basis='self_report', subject='current_user')
        payload = {'matches': [
            dict(claim_id='D:0', premise_type='interest', supported=True,
                 citations=[citation('s14', quote_a)]),
            dict(claim_id='D:0', premise_type='experience', supported=True,
                 citations=[citation('s18', quote_b), citation('s14', quote_a)]),
        ]}
        normalized = self.gate.normalize_citation_recovery_matches(payload, ['D:0'])
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].claim_id, 'D:0')
        self.assertEqual([c.source_id for c in normalized[0].citations], ['s14', 's18'])
        self.assertEqual(normalized[0].premise_type, 'interest')

    def test_conflicting_duplicate_citation_recovery_decisions_fail_closed(self):
        payload = {'matches': [
            dict(claim_id='D:0', premise_type='interest', supported=True, citations=[]),
            dict(claim_id='D:0', premise_type='interest', supported=False, citations=[]),
        ]}
        normalized = self.gate.normalize_citation_recovery_matches(payload, ['D:0'])
        self.assertFalse(normalized[0].supported)
        self.assertEqual(normalized[0].citations, [])

    def test_directly_verified_core_survives_rejected_full_option_without_upgrading_secondary(self):
        quote = ('I’ve noticed some anime seem to deal with really weighty political or historical themes—'
                 'what’s behind that storytelling approach?')
        check = dict(claim_id='D:0', check_type='premise', claim='Since you’re into anime',
                     premise_type='interest', sources=[dict(id='s9', role='user', quote=quote, context=quote)])
        anchor = dict(source_id='s9', quote=quote, valid=True, anchor=True, strength_gap=0)
        entries = [dict(letter='D', kind='personal', option='D. Since you’re into anime and enjoy layered stories.',
            status='unsupported', validation_status='valid', validation_errors=[], primary_claim=0, warnings=[],
            claims=[dict(text='Since you’re into anime', status='unsupported', reason='no_source',
                         validation_errors=[], citations=[anchor]),
                    dict(text='enjoy layered stories with complex characters', status='unsupported',
                         reason='no_source', validation_errors=[], citations=[])])]
        checks = [dict(claim_id='D:option', check_type='option', option=entries[0]['option']), check]
        diagnostics = {}
        verdicts = self.gate.apply_direct_source_entailment_rules(
            {'checks': [dict(claim_id='D:option', entailed=False), dict(claim_id='D:0', entailed=False)]},
            checks, diagnostics)
        result = self.gate.validate_entailments(verdicts, entries, checks,
            direct_source_claim_ids=diagnostics['answer_entailment_rules']['claim_ids'])
        self.assertEqual(result[0]['claims'][0]['status'], 'supported')
        self.assertEqual(result[0]['claims'][0]['recovery'], 'direct_source_entailment_rule')
        self.assertEqual(result[0]['claims'][1]['status'], 'unsupported')
        self.assertEqual(result[0]['status'], 'partial')
        self.assertEqual(self.gate.eligible_choices(result, set()), ['D'])

    def test_checks_contain_every_full_option_with_verified_claims_and_sources(self):
        sources, entries, checks = self.fixture()
        self.assertEqual({c['claim_id'] for c in checks}, {'A:option', 'A:0', 'A:1', 'B:option', 'C:option'})
        option_checks = [c for c in checks if c['check_type'] == 'option']
        self.assertEqual([c['option'] for c in option_checks], [
            'A. Since you have twin boys and you own a camera, take photos.',
            'B. Since you live in Oslo, go outside.', 'C. Take a break.'])
        self.assertEqual([c['claimed_kind'] for c in option_checks], ['personal', 'personal', 'generic'])
        premise_checks = [c for c in checks if c['check_type'] == 'premise']
        self.assertEqual([c['claim'] for c in premise_checks], ['you have twin boys', 'you own a camera'])
        for check in checks:
            self.assertNotIn('status', check)
            self.assertNotIn('validation_errors', check)
        self.assertEqual(len(option_checks[0]['sources']), 2)
        for source, text in zip(option_checks[0]['sources'], ['I have twin boys.', 'I own a camera.']):
            self.assertEqual(source['role'], 'user')
            self.assertEqual(source['context'], text)
            self.assertEqual(source['timestamp'], 1700000000000)
            self.assertEqual(source['context'], sources[source['id']]['text'])

    def test_unentailed_full_option_with_verified_core_only_downgrades(self):
        _, entries, checks = self.fixture()
        original_citations = copy.deepcopy([c['citations'] for e in entries for c in e['claims']])
        payload = entailed(*(c['claim_id'] for c in checks))
        for row in payload['checks']:
            row['entailed'] = row['claim_id'] != 'A:option'
        result = self.gate.validate_entailments(payload, entries, checks)
        self.assertEqual(result[0]['status'], 'partial')
        self.assertIn('option_not_entailed', result[0]['warnings'])
        self.assertEqual(self.gate.eligible_choices(result, set()), ['A', 'C'])
        self.assertEqual([c['citations'] for e in result for c in e['claims']], original_citations)

    def test_unentailed_generic_labelled_option_is_disqualified(self):
        _, entries, checks = self.fixture()
        payload = entailed(*(c['claim_id'] for c in checks))
        for row in payload['checks']:
            row['entailed'] = row['claim_id'] not in ('C:option', 'A:option', 'A:0')
        result = self.gate.validate_entailments(payload, entries, checks)
        self.assertEqual([e['status'] for e in result], ['unsupported', 'unsupported', 'unsupported'])
        self.assertEqual(self.gate.eligible_choices(result, set()), [])

    def test_true_entailment_never_promotes_an_unsupported_option(self):
        _, entries, checks = self.fixture()
        result = self.gate.validate_entailments(entailed(*(c['claim_id'] for c in checks)), entries, checks)
        self.assertEqual(result[1]['status'], 'unsupported')
        self.assertEqual(result[1]['claims'][0]['citations'], [])
        self.assertEqual(self.gate.eligible_choices(result, set()), ['A', 'C'])

    def test_missing_generic_duplicate_and_unknown_option_checks_are_rejected(self):
        _, entries, checks = self.fixture()
        complete = [c['claim_id'] for c in checks]
        payloads = [entailed(), entailed('A:option', 'B:option'),
                    entailed('A:option', 'A:option', 'C:option'),
                    entailed(*(cid for cid in complete if cid != 'C:option')),
                    entailed(*(cid for cid in complete if cid != 'A:0')),
                    entailed(*complete, 'Z:option')]
        before = copy.deepcopy(entries)
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.gate.validate_entailments(payload, entries, checks)
        self.assertEqual(entries, before)

    def test_entailment_boolean_and_schema_do_not_allow_new_citations(self):
        _, entries, checks = self.fixture()
        for value in (0, 1, 'true', None):
            payload = entailed(*(c['claim_id'] for c in checks))
            payload['checks'][0]['entailed'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.gate.validate_entailments(payload, entries, checks)
        payload = entailed(*(c['claim_id'] for c in checks))
        payload['checks'][0]['citations'] = [{'source_id': 'forged'}]
        with self.assertRaises(ValueError):
            self.gate.validate_entailments(payload, entries, checks)

    def test_rejected_primary_premise_demotes_option_even_when_full_option_check_says_true(self):
        _, entries, checks = self.fixture()
        payload = entailed(*(c['claim_id'] for c in checks))
        for row in payload['checks']:
            row['entailed'] = row['claim_id'] != 'A:0'
        result = self.gate.validate_entailments(payload, entries, checks)
        self.assertEqual(self.gate.eligible_choices(result, set()), ['C'])
        self.assertEqual(result[0]['status'], 'unsupported')

    def test_rejected_secondary_premise_keeps_option_partial(self):
        _, entries, checks = self.fixture()
        payload = entailed(*(c['claim_id'] for c in checks))
        for row in payload['checks']:
            row['entailed'] = row['claim_id'] != 'A:1'
        result = self.gate.validate_entailments(payload, entries, checks)
        self.assertEqual(result[0]['status'], 'partial')
        self.assertEqual(result[0]['claims'][1]['status'], 'unsupported')
        self.assertEqual(self.gate.eligible_choices(result, set()), ['A', 'C'])


class ConstraintTests(OfflineCase):
    def test_compound_forget_source_preserves_each_request_and_its_exact_span(self):
        text = 'Please forget that I enjoy photography. Please forget that I have asthma.'
        _, constraints = self.catalog(text, constraint=True)
        self.assertEqual({c['text'].strip() for c in constraints.values()}, {
            'Please forget that I enjoy photography.', 'Please forget that I have asthma.'})
        for constraint in constraints.values():
            span = constraint['constraint_span']
            self.assertEqual(constraint['source_text'], text)
            self.assertEqual(text[span['start']:span['end']], constraint['text'])
        pairs = self.gate.constraint_pairs(['A. Take a photography class.', 'B. Take a break.'], constraints)
        self.assertEqual([p['letter'] for p in pairs], ['A'])
        self.assertIn('photography', pairs[0]['constraint_quote'])
        self.assertNotIn('asthma', pairs[0]['constraint_quote'])

    def test_unrelated_preamble_cannot_dilute_a_real_forget_request(self):
        text = 'I live in Kansas and teach civics at a local college. Please forget that I enjoy photography.'
        _, constraints = self.catalog(text, constraint=True)
        self.assertEqual(len(constraints), 1)
        constraint = next(iter(constraints.values()))
        self.assertEqual(constraint['text'].strip(), 'Please forget that I enjoy photography.')
        span = constraint['constraint_span']
        self.assertEqual(constraint['source_text'][span['start']:span['end']], constraint['text'])
        pairs = self.gate.constraint_pairs(['A. Take a photography class.', 'B. Take a break.'], constraints)
        self.assertEqual([p['letter'] for p in pairs], ['A'])

    def test_dont_forget_reminder_does_not_cancel_a_later_real_forget_request(self):
        text = "Don't forget to call me tomorrow. Please forget that I enjoy photography."
        _, constraints = self.catalog(text, constraint=True)
        self.assertEqual(len(constraints), 1)
        constraint = next(iter(constraints.values()))
        self.assertEqual(constraint['text'].strip(), 'Please forget that I enjoy photography.')
        span = constraint['constraint_span']
        self.assertEqual(text[span['start']:span['end']], constraint['text'])
        pairs = self.gate.constraint_pairs(['A. Take a photography class.', 'B. Take a break.'], constraints)
        self.assertEqual([p['letter'] for p in pairs], ['A'])

    def decision_fixture(self):
        _, constraints = self.catalog('Please forget that I enjoy photography.', constraint=True)
        options = ['A. Since you enjoy photography, take photos.',
                   'B. Since you enjoy photography, visit a camera exhibit.',
                   'C. Since you enjoy surfing, visit a beach.']
        pairs = self.gate.constraint_pairs(options, constraints)
        expected = {'A:' + next(iter(constraints)), 'B:' + next(iter(constraints))}
        self.assertEqual({p['pair_id'] for p in pairs}, expected)
        return options, constraints, pairs

    def test_complete_pair_decisions_preserve_scope_and_verbatim_evidence(self):
        options, constraints, pairs = self.decision_fixture()
        payload = {'decisions': [dict(pair_id=p['pair_id'], violates=p['letter'] == 'A') for p in pairs]}
        blocked, judgments = self.gate.validate_decisions(payload, pairs, options, constraints)
        self.assertEqual(blocked, {'A'})
        self.assertEqual({j['letter'] for j in judgments}, {'A', 'B'})
        for judgment in judgments:
            self.assertIn(judgment['constraint_quote'], constraints[judgment['constraint_id']]['text'])
            self.assertIn(judgment['option_quote'], options[ord(judgment['letter']) - 65])

    def test_pair_decisions_reject_missing_duplicate_and_unknown_pairs(self):
        options, constraints, pairs = self.decision_fixture()
        complete = [dict(pair_id=p['pair_id'], violates=False) for p in pairs]
        variants = [[], complete[:1], [complete[0], complete[0]],
                    [complete[0], dict(pair_id='Z:missing', violates=False)]]
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.gate.validate_decisions({'decisions': rows}, pairs, options, constraints)

    def test_constraint_decision_boolean_is_strict_and_old_payload_is_not_complete(self):
        options, constraints, pairs = self.decision_fixture()
        for value in (0, 1, 'true', 'false', None):
            payload = {'decisions': [dict(pair_id=p['pair_id'], violates=value) for p in pairs]}
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.gate.validate_decisions(payload, pairs, options, constraints)
        with self.assertRaises(ValueError):
            self.gate.validate_decisions({'matches': []}, pairs, options, constraints)

    def constraint(self, rule, option, *, claimed_quote=None, option_quote=None):
        _, constraints = self.catalog(rule, constraint=True)
        cid = next(iter(constraints))
        options = ['A. ' + option, 'B. Take a break.']
        payload = {'matches': [dict(letter='A', constraint_id=cid,
            constraint_quote=rule if claimed_quote is None else claimed_quote,
            option_quote=option if option_quote is None else option_quote)]}
        return self.gate.validate_constraints(payload, options, constraints)

    def test_matching_forget_rule_blocks_even_supported_personal_choice(self):
        blocked, _ = self.constraint('Please forget that I enjoy photography.',
                                    'Since you enjoy photography, visit a camera exhibit.')
        self.assertEqual(blocked, {'A'})
        sources, _ = self.catalog('I enjoy photography.')
        entries = self.gate.validate_assessments(
            assessment('you enjoy photography', next(iter(sources)), 'I enjoy photography.'),
            ['A. Since you enjoy photography, visit a camera exhibit.', 'B. Take a break.'], sources)
        self.assertEqual(self.gate.eligible_choices(entries, blocked), ['B'])

    def test_photography_forget_does_not_block_surfing(self):
        blocked, _ = self.constraint('Please forget that I enjoy photography.',
                                    'Since you enjoy surfing, visit a beach.')
        self.assertEqual(blocked, set())

    def test_specific_music_festival_forget_does_not_ban_general_live_music(self):
        blocked, _ = self.constraint('Please forget that I love modern electronic music festivals.',
                                    'Listen to some live music at a local cafe.')
        self.assertEqual(blocked, set())

    def test_forged_constraint_or_option_quote_cannot_block(self):
        rule = 'Please forget that I enjoy photography.'
        option = 'Since you enjoy photography, visit a camera exhibit.'
        for kwargs in [dict(claimed_quote='Please forget surfing.'), dict(option_quote='You enjoy surfing.')]:
            with self.subTest(kwargs=kwargs):
                blocked, _ = self.constraint(rule, option, **kwargs)
                self.assertEqual(blocked, set())


class AnswerFlowTests(OfflineCase, unittest.IsolatedAsyncioTestCase):
    def qa(self, options=None):
        return dict(question='What should I do?', scoring='choice', qa_type='single_choice',
                    options=options or ['A. Since you have twin boys, rest.', 'B. Take a break.'],
                    gold_labels=['SECRET_GOLD'], answer='SECRET_REFERENCE')

    async def test_disabled_review_keeps_rejected_history_out_of_candidates(self):
        text = 'I spent the evening grading assignments.'
        packet = seal(memory(text))
        sources, _ = self.gate.build_catalog(packet)
        claim = 'you balanced remote schooling with parenting'
        qa = self.qa(['A. Since ' + claim + ', schedule family time.', 'B. Take a short break.'])
        support = assessment(claim, next(iter(sources)), text)
        mock = staged_mock(support, lambda prompt: entailment_response(prompt, {'A:0', 'A:option'}))
        diagnostics = {}
        with patch.object(config, 'CHOICE_ENTAILMENT_REVIEW', False), patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'B')
        self.assertEqual(diagnostics['answer_eligible_options'], ['B'])
        self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                         ['eval.choice_support', 'eval.choice_entailment'])
        self.assertIn('not_entailed', diagnostics['choice_alignment'][0]['validation_errors'])

    async def test_verified_generic_can_win_against_supported_personal_answer(self):
        packet = seal(memory('I have twin boys.'))
        sources, _ = self.gate.build_catalog(packet)
        payload = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        diagnostics = {}
        mock = staged_mock(payload, entailment_response, {'answer': 'B'}, {'answer': 'B'}, {'answer': 'B'})
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa(), packet, diagnostics), 'B')
        self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                         ['eval.choice_support', 'eval.choice_entailment', 'eval.choice_select',
                          'eval.choice_select_review'])
        self.assertEqual(diagnostics['answer_eligible_options'], ['A', 'B'])
        self.assertEqual(diagnostics['answer_validation'], 'validated')
        for call in mock.call_args_list:
            self.assertNotIn('SECRET_GOLD', call.args[0])
            self.assertNotIn('SECRET_REFERENCE', call.args[0])

    async def test_verified_personal_core_gets_focused_review_after_generic_first_choice(self):
        packet = seal(memory('I enjoy anime.'))
        sources, _ = self.gate.build_catalog(packet)
        sid = next(iter(sources))
        qa = self.qa(['A. Try a widely praised series.',
                      'B. Since you enjoy anime, try Attack on Titan.'])
        support = {'options': [
            dict(letter='A', kind='generic', claims=[]),
            dict(letter='B', kind='personal', claims=[dict(text='you enjoy anime', status='supported',
                premise_type='interest', reason='none', citations=[dict(source_id=sid,
                    quote='I enjoy anime.', basis='self_report', subject='current_user')])]),
        ]}

        def typed_entailment(prompt):
            match = re.search(r'<checks>\s*(.*?)\s*</checks>', prompt, flags=re.S)
            checks = json.loads(match[1])
            return {'checks': [dict(claim_id=check['claim_id'], verdict=(
                'personal_supported' if check['claim_id'].startswith('B:') else 'no_personal_premise'))
                for check in checks]}

        mock = staged_mock(
            lambda prompt: dict(options=[dict(letter=letter, witnesses=[]) for letter in 'AB']),
            support, {'matches': [dict(claim_id='B:0', premise_type='interest',
                                     supported=False, citations=[])]},
            typed_entailment, {'answer': 'A'}, {'answer': 'B'})
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'B')
        self.assertEqual(diagnostics['answer_selection_review']['first'], 'A')
        self.assertEqual(diagnostics['answer_selection_review']['second'], 'B')
        self.assertEqual(diagnostics['answer_selection_review']['context_evidence'][0]['letter'], 'B')
        self.assertIn('eval.choice_select_review', [c.kwargs['stage'] for c in mock.call_args_list])

    async def test_first_party_context_can_reconsider_generic_options(self):
        source = ('A student shared troubling personal circumstances. I listened closely and connected '
                  'them with the school counselor.')
        packet = seal(memory(source))
        qa = self.qa(['A. Try a popular show.',
                      'B. When someone shares something heavy, listen fully and point them to support.',
                      'C. Take a calming walk afterwards.',
                      'D. When a student shares something heavy, listen, then connect them with the school counselor.'])
        support = {'options': [dict(letter=letter, kind='generic', claims=[])
                               for letter in 'ABCD']}

        def typed_entailment(prompt):
            match = re.search(r'<checks>\s*(.*?)\s*</checks>', prompt, flags=re.S)
            checks = json.loads(match[1])
            return {'checks': [dict(claim_id=check['claim_id'], verdict='no_personal_premise')
                               for check in checks]}

        mock = staged_mock(
            lambda prompt: dict(options=[dict(letter=letter, witnesses=[]) for letter in 'ABCD']),
            support, typed_entailment, {'answer': 'B'}, {'answer': 'D'})
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'D')
        self.assertEqual(diagnostics['answer_selection_review']['first'], 'B')
        self.assertEqual(diagnostics['answer_selection_review']['context_options'], ['D'])

    async def test_disqualified_generic_keeps_unique_personal_fast_path(self):
        packet = seal(memory('I have twin boys.'))
        sources, _ = self.gate.build_catalog(packet)
        support = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        mock = staged_mock(support, lambda prompt: entailment_response(prompt, {'B:option'}))
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa(), packet, diagnostics), 'A')
        self.assertEqual(diagnostics['answer_eligible_options'], ['A'])
        self.assertEqual(diagnostics['answer_selection'], 'unique_eligible')
        self.assertEqual(mock.await_count, 2)

    async def test_assistant_only_personal_support_cannot_override_generic_fallback(self):
        packet = seal(memory('I have twin boys.', role='assistant'))
        sources, _ = self.gate.build_catalog(packet)
        payload = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        mock = staged_mock(payload, payload, entailment_response)
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa(), packet, diagnostics), 'B')
        self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                         ['eval.choice_support', 'eval.choice_support.repair', 'eval.choice_entailment'])
        self.assertEqual(diagnostics['answer_support_validation']['unresolved_options'], ['A'])
        self.assertIn('not_user_source', mock.call_args_list[1].args[0])

    async def test_question_date_and_source_timestamp_reach_independent_check_unchanged(self):
        packet = seal(memory('I have twin boys.'))
        qa = self.qa()
        qa['question_date'] = '2023-11-15'
        sources, _ = self.gate.build_catalog(packet)
        support = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        before = copy.deepcopy((qa, packet))
        mock = staged_mock(support, entailment_response, {'answer': 'A'})
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, {}), 'A')
        check_prompt = mock.call_args_list[1].args[0]
        self.assertIn('2023-11-15', check_prompt)
        self.assertIn(qa['question'], check_prompt)
        checks = json.loads(re.search(r'<checks>\s*(.*?)\s*</checks>', check_prompt, flags=re.S)[1])
        cited_sources = [source for check in checks for source in check['sources']]
        self.assertTrue(cited_sources)
        self.assertTrue(all(source['timestamp'] == 1700000000000 for source in cited_sources))
        self.assertTrue(all(source['role'] == 'user' and source['context'] == 'I have twin boys.'
                            for source in cited_sources))
        self.assertEqual((qa, packet), before)

    async def test_invalid_assessment_repairs_once_using_original_sources(self):
        packet = seal(memory('I have twin boys.'))
        sources, _ = self.gate.build_catalog(packet)
        payload = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        mock = staged_mock({'options': []}, payload, entailment_response, {'answer': 'A'})
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa(), packet, {}), 'A')
        self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                         ['eval.choice_support', 'eval.choice_support.repair', 'eval.choice_entailment', 'eval.choice_select'])
        for call in mock.call_args_list:
            self.assertEqual(call.kwargs['attempts'], 1)
            self.assertIn('I have twin boys.', call.args[0])

    async def test_inconsistent_support_shapes_repair_instead_of_silent_generic_answer(self):
        packet = seal(memory('I have twin boys.'))
        sources, _ = self.gate.build_catalog(packet)
        valid = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        generic_claims = copy.deepcopy(valid)
        generic_claims['options'][0]['kind'] = 'generic'
        missing_claim = copy.deepcopy(valid)
        missing_claim['options'][0]['claims'] = []
        paraphrased_span = copy.deepcopy(valid)
        paraphrased_span['options'][0]['claims'][0]['text'] = 'you are the parent of twin boys'
        for invalid in (generic_claims, missing_claim, paraphrased_span):
            with self.subTest(invalid=invalid):
                mock = staged_mock(invalid, valid, entailment_response, {'answer': 'A'})
                with patch.object(llm, 'complete_json', mock):
                    self.assertEqual(await self.gate.answer(self.qa(), packet, {}), 'A')
                self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                                 ['eval.choice_support', 'eval.choice_support.repair', 'eval.choice_entailment', 'eval.choice_select'])

    async def test_repeated_inconsistent_support_cannot_bypass_independent_generic_verification(self):
        packet = seal(memory('I have twin boys.'))
        sources, _ = self.gate.build_catalog(packet)
        invalid = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        invalid['options'][0]['kind'] = 'generic'
        mock = AsyncMock(return_value=invalid)
        with patch.object(llm, 'complete_json', mock), self.assertRaises(ValueError):
            await self.gate.answer(self.qa(), packet, {})
        self.assertEqual(mock.await_count, 4)
        self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list][-2:],
                         ['eval.choice_entailment', 'eval.choice_entailment.repair'])

    async def test_malformed_support_abstains_without_inventing_generic_evidence(self):
        mock = AsyncMock(return_value={'options': []})
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(self.qa(), [], diagnostics), 'ABSTAIN')
        self.assertEqual(mock.await_count, 3)
        self.assertEqual(diagnostics['answer_abstention_reason'], 'support_unresolved')

    async def test_no_supported_or_generic_choice_fails_explicitly(self):
        qa = self.qa(['A. Since you own a camera, take photos.'])
        data = {'options': [dict(letter='A', kind='personal', claims=[
            dict(text='you own a camera', status='unsupported', citations=[])])]}
        diagnostics = {}
        scope_responses = ([{'claims': [{'claim_id': 'A:0', 'personal_fact': True}]}]
                           if hasattr(self.gate, 'reclassify_uncited') else [])
        mock = staged_mock(data, *scope_responses, entailment_response)
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, [], diagnostics), 'ABSTAIN')
        self.assertEqual(diagnostics['answer_abstention_reason'], 'no_admissible_option')
        self.assertEqual(diagnostics['answer_eligible_options'], [])
        self.assertNotEqual(diagnostics['answer_validation'], 'validated')
        self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                         ['eval.choice_support'] + (['eval.choice_premise_scope'] if scope_responses else [])
                         + ['eval.choice_entailment'])

    async def test_valid_source_that_does_not_entail_remote_schooling_or_hiking_is_rejected(self):
        cases = [('I spent the evening grading my students\' assignments.',
                  'you have experience with remote schooling'),
                 ('I took a long ride through the hills.', 'you enjoy hiking')]
        for text, claim in cases:
            with self.subTest(claim=claim):
                packet = seal(memory(text))
                sources, _ = self.gate.build_catalog(packet)
                qa = self.qa(['A. Since ' + claim + ', make a plan.', 'B. Take a break.'])
                support = assessment(claim, next(iter(sources)), text)
                mock = staged_mock(support, lambda prompt: entailment_response(prompt, {'A:0'}))
                diagnostics = {}
                with patch.object(config, 'CHOICE_ENTAILMENT_REVIEW', True), patch.object(llm, 'complete_json', mock):
                    self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'B')
                self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                                 ['eval.choice_support', 'eval.choice_entailment'])
                self.assertIn(text, mock.call_args_list[1].args[0])
                self.assertIn(claim, mock.call_args_list[1].args[0])
                self.assertEqual(diagnostics['answer_eligible_options'], ['B'])

    async def test_generic_label_cannot_skip_verifying_a_complete_personal_option(self):
        packet = seal(memory('I am single and have no children.'))
        qa = self.qa(['A. Since you have twin boys, arrange childcare.', 'B. Take a break.'])
        support = {'options': [dict(letter=letter, kind='generic', claims=[]) for letter in 'AB']}
        mock = staged_mock(support, lambda prompt: entailment_response(prompt, {'A:option'}))
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'B')
        self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                         ['eval.choice_support', 'eval.choice_entailment'])
        for option in qa['options']:
            self.assertIn(option, mock.call_args_list[1].args[0])
        self.assertEqual(diagnostics['answer_eligible_options'], ['B'])

    async def test_hidden_secondary_premise_downgrades_option_to_partial_but_keeps_it_eligible(self):
        packet = seal(memory('I own a camera.'))
        sources, _ = self.gate.build_catalog(packet)
        qa = self.qa(['A. Since you own a camera and photograph weddings professionally, advertise wedding shoots.',
                      'B. Take a break.'])
        support = assessment('you own a camera', next(iter(sources)), 'I own a camera.')
        mock = staged_mock(support, lambda prompt: entailment_response(prompt, {'A:option'}),
                           lambda prompt: entailment_response(prompt, {'A:option'}), {'answer': 'A'})
        diagnostics = {}
        with patch.object(config, 'CHOICE_ENTAILMENT_REVIEW', True), patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'A')
        self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                         ['eval.choice_support', 'eval.choice_entailment', 'eval.choice_entailment_review', 'eval.choice_select'])
        self.assertIn(qa['options'][0], mock.call_args_list[1].args[0])
        self.assertIn('I own a camera.', mock.call_args_list[1].args[0])
        self.assertEqual(diagnostics['answer_eligible_options'], ['A', 'B'])
        self.assertEqual(diagnostics['choice_alignment'][0]['status'], 'partial')
        self.assertIn('option_not_entailed', diagnostics['choice_alignment'][0]['warnings'])

    async def test_unsupported_nonliteral_claims_do_not_block_another_supported_answer(self):
        packet = seal(memory('I have twin boys.'))
        sources, _ = self.gate.build_catalog(packet)
        qa = self.qa(['A. Since you enjoy hiking, take a trail walk.',
                      'B. Since you have twin boys, rest.', 'C. Take a break.'])
        payload = {'options': [dict(letter='A', kind='personal', claims=[
            dict(text='you have a hiking habit', status='unsupported', citations=[]),
            dict(text='you have a hiking habit', status='unsupported', citations=[])]),
            dict(letter='B', kind='personal', claims=assessment('you have twin boys',
                 next(iter(sources)), 'I have twin boys.')['options'][0]['claims']),
            dict(letter='C', kind='generic', claims=[])]}
        mock = staged_mock(payload, payload, entailment_response, {'answer': 'B'})
        diagnostics = {}
        with patch.object(llm, 'complete_json', mock):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'B')
        self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list],
                         ['eval.choice_support', 'eval.choice_support.repair', 'eval.choice_entailment', 'eval.choice_select'])
        self.assertEqual(diagnostics['answer_support_validation']['unresolved_options'], ['A'])
        self.assertNotIn('A:option', mock.call_args_list[2].args[0])
        self.assertEqual(diagnostics['answer_eligible_options'], ['B', 'C'])

    async def test_missing_entailment_check_repairs_once_and_cannot_be_silently_skipped(self):
        packet = seal(memory('I have twin boys.'))
        sources, _ = self.gate.build_catalog(packet)
        support = assessment('you have twin boys', next(iter(sources)), 'I have twin boys.')
        for last, success in [(entailment_response, True), (entailed('A:option'), False)]:
            with self.subTest(success=success):
                mock = staged_mock(support, entailed(), last, {'answer': 'A'})
                with patch.object(llm, 'complete_json', mock):
                    if success:
                        self.assertEqual(await self.gate.answer(self.qa(), packet, {}), 'A')
                    else:
                        with self.assertRaises(ValueError):
                            await self.gate.answer(self.qa(), packet, {})
                self.assertEqual([c.kwargs['stage'] for c in mock.call_args_list], [
                    'eval.choice_support', 'eval.choice_entailment', 'eval.choice_entailment.repair']
                    + (['eval.choice_select'] if success else []))

    async def tie_fixture(self, selected, constraint_responses=None):
        packet = seal(memory('I enjoy photography.', mid='fact'),
                      memory('Please forget that I enjoy photography.', mid='rule', constraint=True))
        sources, constraints = self.gate.build_catalog(packet)
        qa = self.qa(['A. Since you enjoy photography, take photos.', 'B. Take a break.', 'C. Rest indoors.'])
        support = assessment('you enjoy photography', next(iter(sources)), 'I enjoy photography.')
        support['options'].append(dict(letter='C', kind='generic', claims=[]))
        pairs = self.gate.constraint_pairs(qa['options'], constraints)
        self.assertEqual({p['pair_id'] for p in pairs}, {'A:' + next(iter(constraints))})
        decisions = {'decisions': [dict(pair_id=p['pair_id'], violates=True) for p in pairs]}
        responses = iter(constraint_responses) if constraint_responses is not None else None
        calls = []

        async def respond(prompt, **kwargs):
            calls.append((prompt, kwargs))
            if kwargs['stage'] == 'eval.choice_support':
                return copy.deepcopy(support)
            if kwargs['stage'] == 'eval.choice_entailment':
                return entailment_response(prompt)
            if kwargs['stage'] in ('eval.choice_constraints', 'eval.choice_constraints.repair'):
                return copy.deepcopy(next(responses) if responses is not None else decisions)
            self.assertIn(kwargs['stage'], ('eval.choice_select', 'eval.choice_select.repair'))
            self.assertNotIn(qa['options'][0], prompt)
            self.assertEqual(set(kwargs['schema']['properties']['answer']['enum']), {'B', 'C'})
            return {'answer': selected}

        return qa, packet, calls, respond

    async def test_valid_tie_selection_is_limited_to_verified_eligible_choices(self):
        qa, packet, calls, respond = await self.tie_fixture('C')
        diagnostics = {}
        with patch.object(llm, 'complete_json', respond):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'C')
        self.assertEqual(diagnostics['answer_blocked_options'], ['A'])
        self.assertEqual(diagnostics['answer_eligible_options'], ['B', 'C'])
        self.assertEqual(len(calls), 4)

    async def test_model_cannot_reintroduce_blocked_answer_even_during_repair(self):
        qa, packet, calls, respond = await self.tie_fixture('A')
        diagnostics = {}
        with patch.object(llm, 'complete_json', respond), self.assertRaises(ValueError):
            await self.gate.answer(qa, packet, diagnostics)
        self.assertEqual([kwargs['stage'] for _, kwargs in calls], [
            'eval.choice_support', 'eval.choice_entailment', 'eval.choice_constraints',
            'eval.choice_select', 'eval.choice_select.repair'])
        self.assertNotEqual(diagnostics.get('answer_validation'), 'validated')
        self.assertNotIn('answer_selected', diagnostics)

    async def test_empty_pair_decisions_cannot_silently_skip_forget_rule(self):
        qa, packet, calls, respond = await self.tie_fixture('C', [{'decisions': []}, {'decisions': []}])
        diagnostics = {}
        with patch.object(llm, 'complete_json', respond), self.assertRaises(ValueError):
            await self.gate.answer(qa, packet, diagnostics)
        self.assertEqual([kwargs['stage'] for _, kwargs in calls], [
            'eval.choice_support', 'eval.choice_entailment', 'eval.choice_constraints', 'eval.choice_constraints.repair'])
        self.assertNotIn('answer_selected', diagnostics)

    async def test_missing_pair_can_be_repaired_once_before_selection(self):
        repaired = {'decisions': [dict(pair_id='A:r0', violates=True)]}
        qa, packet, calls, respond = await self.tie_fixture('C', [{'decisions': []}, repaired])
        diagnostics = {}
        with patch.object(llm, 'complete_json', respond):
            self.assertEqual(await self.gate.answer(qa, packet, diagnostics), 'C')
        self.assertEqual(diagnostics['answer_blocked_options'], ['A'])
        self.assertEqual([kwargs['stage'] for _, kwargs in calls], [
            'eval.choice_support', 'eval.choice_entailment', 'eval.choice_constraints',
            'eval.choice_constraints.repair', 'eval.choice_select'])

    async def test_standalone_budget_allows_each_option_and_downstream_stage_repairs(self):
        qa, packet, _, respond = await self.tie_fixture('C')
        calls, scopes = [], []

        async def repaired(prompt, **kwargs):
            scope = budget.current.get()
            self.assertIsNotNone(scope)
            scope.before_call()
            scopes.append(scope)
            stage = kwargs['stage']
            calls.append(stage)
            if not stage.endswith('.repair'):
                return {}
            result = await respond(prompt, **dict(kwargs, stage=stage.removesuffix('.repair')))
            if stage == 'eval.choice_support.repair':
                letters = kwargs['schema']['$defs']['Assessment']['properties']['letter']['enum']
                result = {'options': [r for r in result['options'] if r['letter'] in letters]}
            return result

        self.assertIsNone(budget.current.get())
        with patch.object(llm, 'complete_json', repaired):
            self.assertEqual(await self.gate.answer(qa, packet, {}), 'C')
        self.assertEqual(calls, ['eval.choice_support']
                         + ['eval.choice_support.repair'] * len(qa['options'])
                         + [stage + suffix for stage in (
                             'eval.choice_entailment', 'eval.choice_constraints', 'eval.choice_select')
                            for suffix in ('', '.repair')])
        self.assertTrue(all(scope is scopes[0] for scope in scopes))
        self.assertEqual(scopes[0].calls, 10)
        self.assertGreaterEqual(scopes[0].max_calls, 10)
        self.assertIsNone(budget.current.get())

    async def test_packet_tamper_fails_before_any_model_call(self):
        packet = seal(memory('I have twin boys.'))
        packet[0]['content'] += 'FORGED'
        mock = AsyncMock(side_effect=AssertionError('Model must not see an unverified packet'))
        with patch.object(llm, 'complete_json', mock), self.assertRaises(ValueError):
            await self.gate.answer(self.qa(), packet, {})
        mock.assert_not_awaited()


def legacy_red_suite():
    """Run pinned git evaluator source, never a handwritten legacy replica."""
    revision = '32f8ccddc6b924d744345080c4e5e220344c67cb'
    path = 'memory_system/app/eval_scoring.py'
    code = subprocess.check_output(['git', 'show', revision + ':' + path], cwd=ROOT.parent)
    baseline = types.ModuleType('app._answer_choice_legacy_probe')
    baseline.__package__ = 'app'
    exec(compile(code, 'git:' + revision + ':' + path, 'exec'), baseline.__dict__)
    class LegacyDirectRegression(unittest.IsolatedAsyncioTestCase):
        async def test_forbidden_option_must_not_be_accepted(self):
            await self.reject(memory('Please forget that I enjoy photography.', constraint=True),
                              'Since you enjoy photography, take photos.')

        async def test_third_party_history_must_not_be_accepted_as_personal(self):
            await self.reject(memory('Claire wrote: I have twin boys.'), 'Since you have twin boys, rest.')

        async def test_assistant_only_fact_must_not_be_accepted_as_personal(self):
            await self.reject(memory('I have twin boys.', role='assistant'), 'Since you have twin boys, rest.')

        async def reject(self, item, option):
            qa = dict(question='What should I do?', scoring='choice', qa_type='single_choice',
                      options=['A. ' + option, 'B. Take a break.'], gold_labels=['B'])
            with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
                    patch.object(llm, 'complete', AsyncMock(return_value='A')), \
                    patch.object(llm, 'complete_json', AsyncMock(side_effect=AssertionError('Unexpected call'))):
                prediction, _, diagnostics = await baseline.evaluate(qa, seal(item))
            self.assertNotEqual(prediction, 'A', f'Inadmissible answer accepted: {diagnostics}')

    print('REAL_EVALUATOR_SHA256=' + hashlib.sha256(code).hexdigest(), flush=True)
    print('REAL_EVALUATOR=git:' + revision + ':' + path, flush=True)
    return unittest.defaultTestLoader.loadTestsFromTestCase(LegacyDirectRegression)


if __name__ == '__main__':
    if '--legacy-red' in sys.argv:
        outcome = unittest.TextTestRunner(verbosity=2).run(legacy_red_suite())
        raise SystemExit(not outcome.wasSuccessful())
    unittest.main()
