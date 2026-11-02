"""Readiness and bounded query planning with provider cold imports excluded."""
import asyncio
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import AsyncMock, patch

os.environ['AML_FAKE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, llm, provider_startup, schemas, search_pipeline


class ProviderStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_waits_for_import_without_blocking_event_loop(self):
        entered, release = threading.Event(), threading.Event()
        ready = asyncio.Event()
        main_thread = threading.get_ident()
        workers = []

        def initialize():
            workers.append(threading.get_ident())
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test failed to release initializer')

        async def serve():
            async with provider_startup.lifespan(None):
                ready.set()

        with patch.object(config, 'FAKE', False), patch.object(llm, 'initialize_provider', initialize):
            task = asyncio.create_task(serve())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                await asyncio.sleep(0)
                self.assertFalse(ready.is_set())
                self.assertNotEqual(workers, [main_thread])
            finally:
                release.set()
                await task
        self.assertTrue(ready.is_set())

    async def test_import_failure_prevents_readiness(self):
        with patch.object(config, 'FAKE', False), patch.object(
                llm, 'initialize_provider', side_effect=ImportError('SDK missing')):
            with self.assertRaisesRegex(ImportError, 'SDK missing'):
                async with provider_startup.lifespan(None):
                    self.fail('must not become ready')

    async def test_fake_mode_never_loads_provider(self):
        with patch.object(config, 'FAKE', True), patch.object(llm.importlib, 'import_module') as load:
            llm.initialize_provider()
            async with provider_startup.lifespan(None):
                pass
            load.assert_not_called()

    def test_sdk_preparation_resolves_entry_points_without_calling_them(self):
        class Provider:
            acompletion = AsyncMock()
            aembedding = AsyncMock()
        provider = Provider()
        with patch.object(config, 'FAKE', False), patch.object(
                llm.importlib, 'import_module', return_value=provider):
            llm.initialize_provider()
        provider.acompletion.assert_not_called()
        provider.aembedding.assert_not_called()


class PlanningDeadlineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.req = schemas.SearchRequest(user_id='u', query='OrchidLedger', options=['A', 'B'])

    async def understand(self):
        return await search_pipeline._understand(None, self.req, '2026-01-01T00:00:00Z')

    async def test_configured_limit_preserves_successful_plan(self):
        plan = {'intent': 'fact', 'entities': ['OrchidLedger']}
        with patch.object(config, 'QUERY_UNDERSTAND_TIMEOUT_SECONDS', .5), patch.object(
                llm, 'complete_json', AsyncMock(return_value=plan)) as call:
            self.assertEqual(await self.understand(), plan)
        self.assertEqual(call.call_args.kwargs['timeout'], .5)
        self.assertEqual(call.call_args.kwargs['attempts'], 1)

    async def test_stage_timeout_cancels_call_and_falls_back(self):
        cancelled = asyncio.Event()

        async def slow(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch.object(config, 'QUERY_UNDERSTAND_TIMEOUT_SECONDS', .02), patch.object(llm, 'complete_json', slow):
            plan = await self.understand()
        self.assertTrue(cancelled.is_set())
        self.assertEqual(plan['_planning_error'], 'TimeoutError')
        self.assertEqual(plan['sub_queries'], [self.req.query])

    async def test_remaining_search_budget_caps_provider_timeout(self):
        with patch.object(config, 'QUERY_UNDERSTAND_TIMEOUT_SECONDS', 30), patch.object(
                llm, 'complete_json', AsyncMock(return_value={'intent': 'fact'})) as call:
            with budget.scope(seconds=.2, calls=1, tokens=1024):
                await self.understand()
        self.assertGreater(call.call_args.kwargs['timeout'], 0)
        self.assertLessEqual(call.call_args.kwargs['timeout'], .2)

    async def test_exhausted_total_budget_is_not_swallowed_as_fallback(self):
        async def slow(*args, **kwargs):
            await asyncio.Event().wait()

        with patch.object(llm, 'complete_json', slow):
            with budget.scope(seconds=.02, calls=1, tokens=1024):
                with self.assertRaises(TimeoutError):
                    await self.understand()

    async def test_external_cancellation_propagates(self):
        with patch.object(llm, 'complete_json', AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await self.understand()


if __name__ == '__main__':
    unittest.main()
