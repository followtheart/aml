"""Group isolation, identity validation and real budget fallback contracts."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as gate, budget, entailment_groups


class Grouped(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, *, calls=20, seconds=240, tokens=64000, failure=None):
        checks = [dict(claim_id=f'{letter}:{suffix}', check_type=kind,
                       sources=[dict(id='source', role='user', timestamp='2024-01-01',
                                     quote='not my experience', context='quoted draft')])
                  for letter in 'AB' for suffix, kind in [('0', 'claim'), ('option', 'option')]]
        before = copy.deepcopy(checks)
        prompt = 'guidance\n<checks>\n' + json.dumps(checks, ensure_ascii=False) + '\n</checks>\nwhole context'
        diagnostics, seen, scopes = {}, [], {}
        async def judge(p, schema, validator, stage, diag, **kwargs):
            budget.current.get().before_call()
            items = json.loads(p.split('<checks>\n')[1].split('\n</checks>')[0])
            seen.extend(items)
            self.assertTrue(p.startswith('guidance\n'))
            self.assertTrue(p.endswith('\nwhole context'))
            response = dict(checks=[dict(claim_id=c['claim_id'], verdict='personal_supported') for c in items])
            if stage.endswith('.A'):
                if failure == 'timeout':
                    raise TimeoutError('provider timeout')
                if failure == 'cross':
                    response['checks'][0]['claim_id'] = 'B:0'
                if failure == 'duplicate':
                    response['checks'][0]['claim_id'] = 'A:option'
            return validator(response)
        with budget.scope(seconds=seconds, calls=calls, tokens=tokens) as limits:
            output = await entailment_groups.verify(prompt, gate.TypedEntailments.model_json_schema(),
                checks, checks, judge, lambda value, subset: gate._typed_verdicts(value, subset, scope=scopes), diagnostics)
            self.assertEqual(limits.reserved_calls, 0)
        self.assertEqual(checks, before)
        self.assertEqual(seen, checks)
        self.assertEqual([c['claim_id'] for c in output['checks']], [c['claim_id'] for c in checks])
        return output, diagnostics, scopes

    async def test_lossless_groups(self):
        result, diag, scopes = await self.run_case()
        self.assertEqual(diag['answer_entailment_groups']['status'], 'grouped')
        self.assertTrue(all(c['entailed'] for c in result['checks']))
        self.assertEqual(len(scopes), 4)

    async def test_failed_group_retains_all_ids_and_other_group(self):
        for failure in ('timeout', 'cross', 'duplicate'):
            with self.subTest(failure=failure):
                result, diag, scopes = await self.run_case(failure=failure)
                self.assertEqual([c['entailed'] for c in result['checks']], [False, False, True, True])
                self.assertEqual(scopes['A:option'], 'unsupported_personal')
                self.assertEqual(len(diag['answer_entailment_groups']['failures']), 1)

    async def test_budget_fallback(self):
        for kwargs, reason in [(dict(calls=5), 'call_headroom'),
                               (dict(seconds=69), 'time_headroom'),
                               (dict(tokens=1000), 'token_headroom')]:
            result, diag, _ = await self.run_case(**kwargs)
            self.assertEqual(diag['answer_entailment_groups']['status'], 'fallback')
            self.assertEqual(diag['answer_entailment_groups']['reason'], reason)
            self.assertTrue(all(c['entailed'] for c in result['checks']))


if __name__ == '__main__':
    unittest.main()
