"""Exercise live scope-stage plumbing with deterministic provider responses."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import budget
from selftest_answer_choice import entailment_response

path = os.environ.get('ANSWER_CANDIDATE', str(ROOT / 'runs/iteration-20260924-13/candidate/answer_choice.py'))
spec = importlib.util.spec_from_file_location('app.scope_answer_candidate', path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ScopeFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_scope_then_mandatory_verifier_with_shared_budget(self):
        for scenario in ['advice', 'whole_rejects', 'personal', 'malformed', 'timeout']:
            with self.subTest(scenario=scenario):
                stages, usage = [], []
                qa = dict(question='How can I relax?', options=['A. You could walk every evening.'],
                          gold_labels=['DO_NOT_INCLUDE_GOLD'])

                async def respond(prompt, **kwargs):
                    self.assertNotIn('DO_NOT_INCLUDE_GOLD', prompt)
                    limits = budget.current.get()
                    limits.before_call()
                    usage.append((limits.calls, limits.max_calls))
                    stage = kwargs['stage']
                    stages.append(stage)
                    if stage == 'eval.choice_support':
                        return {'options': [dict(letter='A', kind='personal', claims=[dict(
                            text='walk every evening', status='unsupported', reason='no_source',
                            premise_type='habit', citations=[])])]}
                    if stage == 'eval.choice_premise_scope':
                        if scenario == 'timeout':
                            raise TimeoutError('simulated')
                        if scenario == 'malformed':
                            return {'claims': []}
                        return {'claims': [dict(claim_id='A:0', personal_fact=scenario == 'personal')]}
                    if stage == 'eval.choice_entailment':
                        return entailment_response(prompt, {'A:option'} if scenario == 'whole_rejects' else set())
                    raise AssertionError(stage)

                diagnostics = {}
                with patch.object(gate.choice_witness, 'select', AsyncMock(return_value={})), \
                     patch.object(gate.llm, 'complete_json', respond):
                    answer = await gate.answer(qa, [], diagnostics)
                self.assertEqual(answer, 'A' if scenario == 'advice' else 'ABSTAIN')
                self.assertEqual(stages, ['eval.choice_support', 'eval.choice_premise_scope', 'eval.choice_entailment'])
                self.assertEqual([x[0] for x in usage], [1, 2, 3])
                self.assertTrue(all(x[1] >= 10 for x in usage))


if __name__ == '__main__':
    unittest.main()
