"""Offline checks for replay integrity and evidence attribution validation."""
import copy
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import controlled_validation as cv
import audit_controlled_validation as audit
from app import answer_context, eval_scoring, evidence_packet


class ReplayChecks(unittest.TestCase):
    def setUp(self):
        quote = 'Claire wrote: I have twin boys.'
        rule = 'Please forget that I enjoy festivals.'
        self.packet = [dict(id='m1', content=answer_context.with_evidence('A quoted story.', [{'role': 'user', 'content': quote}]),
                            sources=[dict(role='user', content=quote, source_event_id='s1')]),
                       dict(id='m2', content=answer_context.with_evidence('A forget rule.', [{'role': 'user', 'content': rule}]),
                            sources=[dict(role='user', content=rule, source_event_id='s2')], is_constraint=True)]
        packet_hash = evidence_packet.digest(self.packet)
        for item in self.packet:
            item['packet_hash'] = packet_hash
        self.qa = dict(question='What helps?', options=['A. Since you have twin boys, rest.', 'B. Take a break.'],
                       scoring='choice', qa_type='single_choice')
        self.fixture = dict(qa=self.qa, packet=self.packet)
        self.response = dict(answer='B', options=[
            dict(letter='A', premise='you have twin boys', subject='third_party', support='unsupported',
                 citations=[dict(item_id='m1', quote=quote, source_role='user')],
                 constraint=dict(applies=False, item_id='', quote='')),
            dict(letter='B', premise='', subject='generic', support='generic', citations=[],
                 constraint=dict(applies=False, item_id='', quote=''))])

    def test_sealed_fixture_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            cv.write_new(path, cv.seal({'prompt': 'original'}))
            self.assertEqual(cv.read_sealed(path), {'prompt': 'original'})
            changed = json.loads(path.read_text())
            changed['payload']['prompt'] = 'changed'
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                cv.read_sealed(path)
            with self.assertRaises(FileExistsError):
                cv.write_new(path, {})

    def test_diagnostic_preserves_every_evidence_text_and_packet(self):
        before = copy.deepcopy(self.packet)
        prompt = cv.diagnostic_prompt(self.qa, self.packet)
        for item in self.packet:
            self.assertEqual(prompt.count(item['content']), 1)
        self.assertEqual(before, self.packet)
        self.assertIn('Return exactly one uppercase option letter', eval_scoring.answer_prompt(self.qa, self.packet))
        self.assertNotIn('Return exactly one uppercase option letter', prompt)

    def test_gold_is_not_a_prompt_input(self):
        prompt = cv.diagnostic_prompt(self.qa, self.packet)
        with_gold = dict(self.qa, answer='SECRET_GOLD_TEXT', gold_labels=['SECRET_LABEL'])
        self.assertEqual(prompt, cv.diagnostic_prompt(with_gold, self.packet))
        self.assertNotIn('gold_labels', cv.QA_PUBLIC)
        self.assertNotIn('answer', cv.QA_PUBLIC)

    def test_valid_literal_third_party_citation_not_auto_personal_support(self):
        result = cv.check_diagnostic(self.response, self.fixture)
        self.assertTrue(result['valid'])
        self.assertIn('model_claim_only', result['subject_validation'])

    def test_wrong_role_or_wrong_item_rejected(self):
        for field, value in [('source_role', 'assistant'), ('item_id', 'm2'), ('quote', 'I have a daughter.')]:
            response = copy.deepcopy(self.response)
            response['options'][0]['citations'][0][field] = value
            self.assertFalse(cv.check_diagnostic(response, self.fixture)['valid'])

    def test_summary_only_quote_is_not_source_evidence(self):
        self.response['options'][0]['citations'][0]['quote'] = 'A quoted story.'
        self.assertFalse(cv.check_diagnostic(self.response, self.fixture)['valid'])

    def test_missing_options_and_unbacked_support_rejected(self):
        self.response['options'][0]['support'] = 'supported'
        self.response['options'][0]['citations'] = []
        self.assertFalse(cv.check_diagnostic(self.response, self.fixture)['valid'])
        self.response['options'].pop()
        self.assertIn('missing_or_duplicate_options', cv.check_diagnostic(self.response, self.fixture)['errors'])

    def test_constraint_requires_actual_rule_quote(self):
        self.response['options'][0]['constraint'] = dict(applies=True, item_id='m2', quote='Please forget that I enjoy festivals.')
        self.assertTrue(cv.check_diagnostic(self.response, self.fixture)['valid'])
        self.response['options'][0]['constraint']['item_id'] = 'm1'
        self.assertFalse(cv.check_diagnostic(self.response, self.fixture)['valid'])

    def test_retention_uses_full_visible_source_and_source_identity(self):
        target = dict(chunk=6, message_index=0, quote='Mild asthma in childhood, no recent symptoms.')
        source = dict(request_id='conversation:chunk:6', message_index=0, content=target['quote'])
        row = dict(id='m1', _rank_text=target['quote'], _packet_item=dict(sources=[source]))
        self.assertEqual(cv.source_carriers([row], target), ['m1'])
        row['_rank_text'] = 'Mild asthma in childhood'
        self.assertEqual(cv.source_carriers([row], target), [])
        row['_rank_text'] = target['quote']
        source['request_id'] = 'conversation:chunk:16'
        self.assertEqual(cv.source_carriers([row], target), [])

    def test_resume_does_not_repeat_completed_or_uncertain_calls(self):
        from app import budget, config, llm
        request = dict(qid='x', arm='answer_baseline', kind='text', prompt='Frozen prompt', system=None,
                       max_tokens=10, settings=dict(model='openai/qwen3-14b', temperature=0.0),
                       prompt_sha256='p', request_sha256='r')
        fixture = dict(qa=self.qa, oracle=['B'])
        data = dict(requests=[request], fixtures={'x': fixture}, repetitions=3, code_hashes={})
        calls = []
        async def fake_complete(**kwargs):
            calls.append(kwargs)
            budget.current.get().before_call()
            return 'B'
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            path = Path(directory)
            cv.write_new(path / 'fixtures.json', cv.seal(data))
            # Interrupted call has no result: preserve uncertainty, do not rebill.
            (path / 'calls.jsonl').write_text(json.dumps(dict(call_id='Qx/answer_baseline/1', event='started')) + '\n')
            for key, value in dict(FAKE=False, LLM_DISABLE_THINKING=True, LLM_MODEL='openai/qwen3-14b',
                                   LLM_TEMPERATURE=0.0, RATE_LIMIT_RETRIES=config.RATE_LIMIT_RETRIES).items():
                stack.enter_context(patch.object(config, key, value))
            stack.enter_context(patch.object(llm, 'complete', fake_complete))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            asyncio.run(cv.run(path))
            asyncio.run(cv.run(path))
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(c['prompt'] == 'Frozen prompt' and c['attempts'] == 1 for c in calls))
            finished = [r for r in cv.journal_rows(path) if r['event'] == 'finished']
            self.assertEqual([r['repetition'] for r in finished], [2, 3])
            self.assertTrue(all(r['provider_calls'] == 1 for r in finished))

    def test_schema_audit_does_not_confuse_literal_validity_with_structure(self):
        response = dict(answer='B', options=[dict(letter='A', support='not_an_enum'),
                                            dict(letter='B', support='not_an_enum')])
        # The literal-only pass makes no unsupported promise of schema validity.
        self.assertTrue(cv.check_diagnostic(response, self.fixture)['valid'])
        errors = list(audit.Draft202012Validator(cv.diagnostic_schema(['A', 'B'])).iter_errors(response))
        self.assertGreater(len(errors), 0)

    def test_matched_cap_preserves_prompt_and_admitted_order(self):
        common = dict(qid='16', kind='json', prompt='same', system='system', schema={},
                      settings={}, prompt_sha256=cv.digest(b'same'))
        data = dict(fixtures={'16': {}}, repetitions=3, requests=[
            dict(common, arm='listwise_admitted10', max_tokens=256, candidate_ids=['a', 'b']),
            dict(common, arm='listwise_full12', max_tokens=288, candidate_ids=['a', 'b', 'c'])])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)
            cv.write_new(path / 'fixtures.json', cv.seal(data))
            audit.prepare_matched(path)
            matched = cv.read_sealed(path / 'matched-cap/fixtures.json')['requests'][0]
        self.assertEqual(matched['max_tokens'], 288)
        self.assertEqual(matched['prompt_sha256'], common['prompt_sha256'])
        self.assertEqual(matched['candidate_ids'], ['a', 'b'])

    def test_invalid_listwise_response_does_not_count_as_source_deletion(self):
        target = dict(chunk=6, message_index=0, quote='Qualified history.')
        fine = [dict(id='a', _rank_text=target['quote'], _packet_item=dict(sources=[
            dict(request_id='x:chunk:6', message_index=0, content=target['quote'])]))]
        request = dict(qid='16', arm='listwise_full12', kind='json', schema={'required': ['ranking']},
                       prompt='Frozen', candidate_ids=['a'], max_tokens=288,
                       prompt_sha256='p', request_sha256='r', exceeds_production_prompt_limit=True)
        data = dict(fixtures={'16': dict(fine=fine, targets=[target])}, requests=[request], repetitions=1)
        failed = dict(call_id='Q16/listwise_full12/1', qid='16', arm='listwise_full12',
                      fixture_sha256=cv.digest(data), prompt_sha256='p', request_sha256='r',
                      event='finished', status='error', response={}, provider_calls=1, tokens=10)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)
            cv.write_new(path / 'fixtures.json', cv.seal(data))
            (path / 'calls.jsonl').write_text(json.dumps(dict(call_id=failed['call_id'], event='started'))
                                            + '\n' + json.dumps(failed) + '\n')
            audit.audit(path)
            result = json.loads((path / 'audited-results.json').read_text())['arms'][0]
        self.assertEqual(result['source_retention']['S6:0']['input'], 1)
        self.assertEqual(result['source_retention']['S6:0']['output_evaluable'], 0)
        self.assertEqual(result['selected_counts'], [None])


if __name__ == '__main__':
    unittest.main()
