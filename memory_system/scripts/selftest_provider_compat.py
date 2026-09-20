"""離線驗證 provider 的思考開關及結構化呼叫相容性。"""
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, llm


class ProviderCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings = patch.multiple(config, FAKE=False, LLM_API_BASE=None,
            LLM_API_KEY=None, LLM_MODEL='deepseek/deepseek-flash',
            LLM_DISABLE_THINKING=True, PROVIDER_RPM=0, PROVIDER_SOFT_TPM=0)
        settings.start()
        self.addCleanup(settings.stop)

    async def test_deepseek_structured_call_disables_thinking_before_forcing_tool(self):
        async def provider(**kwargs):
            if kwargs.get('extra_body', {}).get('thinking') != {'type': 'disabled'}:
                raise ValueError('Thinking mode does not support this tool_choice')
            self.assertEqual(kwargs['tool_choice']['function']['name'], 'emit_json_result')
            return {'choices': [{'message': {'tool_calls': [
                {'function': {'name': 'emit_json_result', 'arguments': '{"ok": true}'}}]}}]}

        for model in ('deepseek/deepseek-flash', 'deepseek/deepseek-v4-pro'):
            with self.subTest(model=model), patch.object(config, 'LLM_MODEL', model), \
                    patch('litellm.acompletion', AsyncMock(side_effect=provider)):
                result = await llm.complete_json('Return ok.', schema={
                    'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
                    'required': ['ok']}, attempts=1)
            self.assertEqual(result, {'ok': True})

    async def test_deepseek_plain_call_uses_same_thinking_setting(self):
        call = AsyncMock(return_value={'choices': [{'message': {'content': 'OK'}}]})
        with patch('litellm.acompletion', call):
            self.assertEqual(await llm.complete('Return OK.', attempts=1), 'OK')
        self.assertEqual(call.await_args.kwargs['extra_body'], {'thinking': {'type': 'disabled'}})

    def test_switch_off_does_not_override_provider_thinking(self):
        with patch.object(config, 'LLM_DISABLE_THINKING', False):
            self.assertNotIn('extra_body', llm._provider_kwargs())

    def test_other_providers_keep_their_own_parameter_contract(self):
        with patch.object(config, 'LLM_MODEL', 'openai/qwen3-14b'):
            self.assertEqual(llm._provider_kwargs(), {'extra_body': {'enable_thinking': False}})
        with patch.object(config, 'LLM_MODEL', 'gpt-4o-mini'):
            self.assertEqual(llm._provider_kwargs(), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
