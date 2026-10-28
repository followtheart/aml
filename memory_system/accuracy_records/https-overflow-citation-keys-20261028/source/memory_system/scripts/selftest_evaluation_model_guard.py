"""The fixed-model experiment gate must reject mismatches before dispatch."""
import unittest
from unittest.mock import patch

import evaluate_frozen_https as evaluator
from evaluation_model_guard import check_runtime, check_versions, check_provider_log


class ModelGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_runtime_fails_before_touching_experiment(self):
        with patch.object(evaluator.config, 'LLM_MODEL', 'other-model'), \
                patch.object(evaluator.config, 'FAKE', False):
            with self.assertRaisesRegex(ValueError, 'runtime: evaluation requires'):
                await evaluator.evaluate(None)

    def test_missing_or_conflicting_trace_model_rejected(self):
        for versions in ({}, {'model': 'other-model'},
                         {'model': 'gpt-4o-mini', 'settings': {'LLM_MODEL': 'other-model'}}):
            with self.subTest(versions=versions), self.assertRaises(ValueError):
                check_versions(versions, 'trace')
        check_versions({'model': 'gpt-4o-mini'}, 'trace')

    def test_dispatch_audit_allows_embedding_but_rejects_mixed_llms(self):
        log = ('provider_call kind=embedding model=embedding-model\n'
               'provider_call kind=llm model=gpt-4o-mini status=ok\n')
        self.assertEqual(check_provider_log(log, 'server'), 1)
        for extra in ('provider_call kind=llm model=other-model\n',
                      'provider_call kind=llm status=ok\n'):
            with self.assertRaises(ValueError):
                check_provider_log(log + extra, 'server')
        with self.assertRaises(ValueError):
            check_provider_log('', 'server')

    def test_runtime_guard_does_not_rewrite_configuration(self):
        with patch.object(evaluator.config, 'LLM_MODEL', 'other-model'):
            with self.assertRaises(ValueError):
                check_runtime()
            self.assertEqual(evaluator.config.LLM_MODEL, 'other-model')


if __name__ == '__main__':
    unittest.main()
