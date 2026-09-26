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

    async def test_structured_result_requests_one_call_for_all_options(self):
        result = {'options': [{'letter': 'A'}, {'letter': 'B'}]}
        response = {'choices': [{'message': {'tool_calls': [
            {'function': {'name': 'emit_json_result', 'arguments': result}}]}}]}
        call = AsyncMock(return_value=response)
        with patch.object(config, 'LLM_MODEL', 'gpt-4o-mini'), patch('litellm.acompletion', call):
            actual = await llm.complete_json('Assess all options.', schema={'type': 'object'}, attempts=1)
        self.assertEqual(actual, result)
        self.assertIs(call.await_args.kwargs['parallel_tool_calls'], False)

    async def test_multiple_tool_calls_are_not_silently_truncated(self):
        calls = [{'function': {'name': 'emit_json_result',
                               'arguments': {'options': [{'letter': letter}]}}}
                 for letter in ('A', 'B', 'C', 'D')]
        with patch('litellm.acompletion', AsyncMock(return_value={
                'choices': [{'message': {'tool_calls': calls}}]})):
            with self.assertRaises(llm.LLMError):
                await llm.complete_json('Assess all options.', schema={'type': 'object'}, attempts=1)

    async def test_wrong_function_name_is_not_accepted_as_result(self):
        response = {'choices': [{'message': {'tool_calls': [
            {'function': {'name': 'other_function', 'arguments': {'ok': True}}}]}}]}
        with patch('litellm.acompletion', AsyncMock(return_value=response)):
            with self.assertRaises(llm.LLMError):
                await llm.complete_json('Return ok.', schema={'type': 'object'}, attempts=1)

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
