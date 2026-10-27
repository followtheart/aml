"""Guards and controller integration for bounded, request-local interest retries."""
import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac, budget, config


def entry(letter='A', status='unsupported', kind='interest'):
    return dict(letter=letter, kind='personal', status=status, primary_claim=0,
                validation_status='valid', claims=[dict(premise_type=kind, status=status,
                reason='none', validation_errors=['not_entailed'] if status == 'unsupported' else [],
                entailment_verified=status == 'supported',
                citations=[dict(valid=True, anchor=True, strength_gap=0)])])


def diag(entries=None, eligible=(), blocked=()):
    return dict(choice_alignment=entries if entries is not None else [entry()],
                answer_eligible_options=list(eligible), answer_blocked_options=list(blocked),
                answer_witness_prefill={'A': []}, answer_calls=[])


class Guards(unittest.TestCase):
    def test_anchored_interest_and_validated_primary(self):
        self.assertEqual(ac.interest_retry_letters(diag()), ['A'])
        self.assertTrue(ac.accept_interest_retry('A', diag([entry(status='supported')], ['A']), ['A']))
        for kind in ('ownership', 'condition', 'habit', 'experience', 'occupation', 'location', 'unknown'):
            self.assertEqual(ac.interest_retry_letters(diag([entry(kind=kind)])), [])

    def test_verified_strict_core_is_kept(self):
        self.assertEqual(ac.interest_retry_letters(diag([entry(), entry('B', 'supported')], ['B'])), [])

    def test_invalid_or_missing_anchors_and_contradictions(self):
        for updates in ({'valid': False}, {'anchor': False}, {'strength_gap': 1}, {'strength_gap': None}):
            e = entry(); e['claims'][0]['citations'][0].update(updates)
            self.assertEqual(ac.interest_retry_letters(diag([e])), [])
        for updates in ({'reason': 'contradicted'}, {'citations': []}, {'validation_errors': ['quote_not_in_source']}):
            e = entry(); e['claims'][0].update(updates)
            self.assertEqual(ac.interest_retry_letters(diag([e])), [])
        self.assertEqual(ac.interest_retry_letters(diag(blocked=['A'])), [])
        e = entry(); e['validation_status'] = 'unresolved'
        self.assertEqual(ac.interest_retry_letters(diag([e])), [])

    def test_retry_is_eligible_same_letter_verified_not_inferred(self):
        for status in ('generic', 'inferred', 'unsupported'):
            e = entry(status=status); e['claims'][0]['entailment_verified'] = True
            self.assertFalse(ac.accept_interest_retry('A', diag([e], ['A']), ['A']))
        d = diag([entry(status='supported')], ['A'])
        self.assertFalse(ac.accept_interest_retry('A', d, ['B']))
        d['answer_blocked_options'] = ['A']
        self.assertFalse(ac.accept_interest_retry('A', d, ['A']))
        d['answer_blocked_options'] = []; d['choice_alignment'][0]['claims'][0]['entailment_verified'] = False
        self.assertFalse(ac.accept_interest_retry('A', d, ['A']))


class Controller(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.flags = patch.multiple(config, CHOICE_ADAPTIVE_INTEREST_RETRY=True, FAKE=False)
        self.flags.start(); self.addCleanup(self.flags.stop)
        self.qa = dict(question='test', options=['A. interest', 'B. generic'])

    async def test_success_reuses_witness_shared_budget_and_separate_logs(self):
        seen = []
        async def fake(qa, memories, diagnostics, *, witness_prefill=None):
            seen.append((ac.inference_enabled(), budget.current.get(), copy.deepcopy(witness_prefill)))
            budget.current.get().before_call()
            diagnostics.update(diag([entry(status='supported' if ac.inference_enabled() else 'unsupported')],
                                    ['A'] if ac.inference_enabled() else ['B']))
            diagnostics['answer_calls'].append(dict(stage='mock', response=ac.inference_enabled()))
            return 'A' if ac.inference_enabled() else 'B'
        d = {}
        with patch.object(ac, '_answer', fake):
            self.assertEqual(await ac.answer(self.qa, [], d), 'A')
        self.assertEqual([s[0] for s in seen], [False, True])
        self.assertIs(seen[0][1], seen[1][1]); self.assertEqual(seen[1][2], {'A': []})
        self.assertEqual(d['answer_adaptive_retry']['selected_attempt'], 1)
        self.assertEqual(d['answer_attempts'][0]['diagnostics']['answer_calls'][0]['response'], False)
        self.assertEqual(d['answer_calls'][0]['response'], True)
        json.dumps(d)  # Nested attempt diagnostics must be serializable and acyclic.
        self.assertIsNone(ac._inference_override.get())

    async def test_retry_failure_preserves_first_answer(self):
        async def fake(qa, memories, diagnostics, **kwargs):
            if ac.inference_enabled():
                diagnostics['answer_calls'] = [dict(stage='failed')]
                raise budget.BudgetExceeded('retry exhausted')
            diagnostics.update(diag()); return 'B'
        d = {}
        with patch.object(ac, '_answer', fake):
            self.assertEqual(await ac.answer(self.qa, [], d), 'B')
        self.assertEqual(d['answer_adaptive_retry']['status'], 'retry_failed')
        self.assertEqual(len(d['answer_attempts']), 2)
        self.assertIsNone(ac._inference_override.get())

    async def test_caller_budget_skip_and_reservations(self):
        calls = []
        async def fake(qa, memories, diagnostics, **kwargs):
            calls.append(budget.current.get()); diagnostics.update(diag()); return 'B'
        for kwargs in [dict(calls=3), dict(tokens=15999), dict(seconds=10)]:
            d = {}
            with budget.scope(**kwargs) as limits, patch.object(ac, '_answer', fake):
                self.assertEqual(await ac.answer(self.qa, [], d), 'B')
                self.assertIs(calls[-1], limits)
                self.assertEqual(d['answer_adaptive_retry']['status'], 'skipped_budget')
        self.assertEqual(len(calls), 3)

    async def test_parallel_requests_do_not_share_inference_overrides(self):
        ready, release = asyncio.Event(), asyncio.Event()
        async def fake(qa, memories, diagnostics, **kwargs):
            diagnostics.update(diag())
            if qa['question'] == 'retry':
                if ac.inference_enabled():
                    ready.set(); await release.wait()
            else:
                await ready.wait(); self.assertFalse(ac.inference_enabled()); release.set()
                diagnostics.update(diag([entry(status='supported')], ['A']))
            return 'B'
        with patch.object(ac, '_answer', fake):
            await asyncio.gather(ac.answer(dict(self.qa, question='retry'), [], {}),
                                 ac.answer(dict(self.qa, question='other'), [], {}))
        self.assertIsNone(ac._inference_override.get())

    async def test_disabled_path_keeps_legacy_configuration(self):
        async def fake(*args, **kwargs):
            self.assertTrue(ac.inference_enabled()); return 'A'
        with patch.multiple(config, CHOICE_ADAPTIVE_INTEREST_RETRY=False, CHOICE_ALLOW_INFERRED=True), patch.object(ac, '_answer', fake):
            d = {}; self.assertEqual(await ac.answer(self.qa, [], d), 'A')
            self.assertNotIn('answer_attempts', d)

if __name__ == '__main__':
    unittest.main()
