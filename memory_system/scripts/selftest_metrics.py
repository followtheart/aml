"""Offline retry regression tests; no provider calls or real waiting."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import metrics


class RateLimitError(Exception):
    def __init__(self, headers=None):
        self.response = SimpleNamespace(status_code=429, headers=headers or {})


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_answer_paces_judge_before_provider_call(self):
        now = [100.0]
        async def advance(seconds):
            now[0] += seconds
        judge_times = []
        async def judge(attempt):
            judge_times.append(now[0])
            return {}
        with patch.object(metrics.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(metrics.asyncio, "sleep", side_effect=advance) as sleep, \
                patch.object(metrics.config, "PROVIDER_SOFT_TPM", 60000):
            await self.run_call(AsyncMock(return_value={"usage": {"total_tokens": 97184}}))
            await metrics.measured_call(kind="llm", stage="eval.judge_refined",
                                        model="offline", call=judge)
        self.assertEqual(judge_times, [160.0])
        sleep.assert_awaited_once_with(60.0)

    async def test_small_calls_are_not_delayed(self):
        with patch.object(metrics.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            for _ in range(3):
                await self.run_call(AsyncMock(return_value={"usage": {"total_tokens": 100}}))
        sleep.assert_not_awaited()

    async def test_request_window_includes_failed_attempts(self):
        now = [100.0]
        async def advance(seconds):
            now[0] += seconds
        with patch.object(metrics.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(metrics.asyncio, "sleep", side_effect=advance) as sleep, \
                patch.object(metrics.config, "PROVIDER_RPM", 1):
            await self.run_call(AsyncMock(side_effect=[ValueError(), {}]))
        sleep.assert_awaited_once_with(60.0)

    async def test_token_window_accumulates_across_stages(self):
        now = [100.0]
        async def advance(seconds):
            now[0] += seconds
        with patch.object(metrics.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(metrics.asyncio, "sleep", side_effect=advance) as sleep, \
                patch.object(metrics.config, "PROVIDER_SOFT_TPM", 60000):
            for _ in range(2):
                await self.run_call(AsyncMock(return_value={"usage": {"total_tokens": 30000}}))
            await self.run_call(AsyncMock(return_value={}))
        sleep.assert_awaited_once_with(60.0)

    async def test_concurrent_calls_serialize(self):
        active = 0
        peak = 0
        async def provider(attempt):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await metrics.asyncio.sleep(0)
            active -= 1
            return {}
        await metrics.asyncio.gather(*(self.run_call(provider) for _ in range(3)))
        self.assertEqual(peak, 1)

    async def test_other_models_do_not_share_token_budget(self):
        with patch.object(metrics.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            await self.run_call(AsyncMock(return_value={"usage": {"total_tokens": 97184}}))
            await metrics.measured_call(kind="llm", stage="test", model="other",
                                        call=AsyncMock(return_value={}))
        sleep.assert_not_awaited()

    async def run_call(self, call, **kwargs):
        return await metrics.measured_call(
            kind="llm", stage="test", model="offline", call=call, **kwargs)

    async def test_recovers_after_more_than_two_attempts(self):
        call = AsyncMock(side_effect=[RateLimitError(), RateLimitError(), {}])
        with patch.object(metrics.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            await self.run_call(call)
        self.assertEqual(call.await_count, 3)
        self.assertEqual(sleep.await_count, 2)
        self.assertGreater(sleep.await_args_list[1].args[0],
                           sleep.await_args_list[0].args[0])

    async def test_retry_after_not_capped(self):
        call = AsyncMock(side_effect=[RateLimitError({"Retry-After": "600"}), {}])
        with patch.object(metrics.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            await self.run_call(call)
        sleep.assert_awaited_once_with(600.0)

    async def test_exhaustion_raises_original(self):
        error = RateLimitError()
        call = AsyncMock(side_effect=error)
        with patch.object(metrics.config, "RATE_LIMIT_RETRIES", 2), \
                patch.object(metrics.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaises(RateLimitError) as raised:
                await self.run_call(call)
        self.assertIs(raised.exception, error)
        self.assertEqual(call.await_count, 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_other_errors_keep_original_budget(self):
        call = AsyncMock(side_effect=ValueError("invalid tool JSON"))
        with patch.object(metrics.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaises(ValueError):
                await self.run_call(call)
        self.assertEqual(call.await_count, 2)
        sleep.assert_not_awaited()

    async def test_mixed_errors_have_separate_budgets(self):
        call = AsyncMock(side_effect=[ValueError(), RateLimitError(), {}])
        with patch.object(metrics.asyncio, "sleep", new_callable=AsyncMock):
            await self.run_call(call)
        self.assertEqual(call.await_count, 3)

    async def test_cancel_during_cooldown_propagates(self):
        call = AsyncMock(side_effect=RateLimitError())
        with patch.object(metrics.asyncio, "sleep",
                          side_effect=metrics.asyncio.CancelledError):
            with self.assertRaises(metrics.asyncio.CancelledError):
                await self.run_call(call)
        self.assertEqual(call.await_count, 1)

    def test_http_date_and_invalid_headers(self):
        with patch.object(metrics.time, "time", return_value=0):
            self.assertEqual(metrics._retry_after(RateLimitError(
                {"retry-after": "Thu, 01 Jan 1970 00:01:00 GMT"})), 60)
        for value in ("bad", "NaN", "inf", "-1"):
            self.assertEqual(metrics._retry_after(
                RateLimitError({"retry-after": value})), 0)

    async def test_http_status_without_litellm_class(self):
        error = RuntimeError()
        error.status_code = 429
        call = AsyncMock(side_effect=[error, {}])
        with patch.object(metrics.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            await self.run_call(call)
        sleep.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
