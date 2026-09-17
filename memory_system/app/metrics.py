"""Privacy-safe timing and token metrics for external model calls."""
import asyncio
import logging
import math
import random
import time
import weakref
from collections import deque
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, Optional

from . import budget, config

log = logging.getLogger("aml.metrics")


class _Pacer:
    """Bound a model's in-flight calls and pace them using a rolling local window."""

    def __init__(self):
        self.lock = asyncio.Semaphore(config.PROVIDER_CONCURRENCY)
        self.requests = deque()
        self.usage = deque()

    async def wait(self, kind, stage, model):
        while True:
            now = time.monotonic()
            while self.requests and self.requests[0] <= now - 60:
                self.requests.popleft()
            while self.usage and self.usage[0][0] <= now - 60:
                self.usage.popleft()
            delay = 0.0
            if config.PROVIDER_RPM and len(self.requests) >= config.PROVIDER_RPM:
                delay = max(delay, self.requests[0] + 60 - now)
            if (config.PROVIDER_SOFT_TPM and self.usage
                    and sum(tokens for _, tokens in self.usage) >= config.PROVIDER_SOFT_TPM):
                delay = max(delay, self.usage[0][0] + 60 - now)
            if delay <= 0:
                self.requests.append(now)
                return
            log.info("provider_throttle kind=%s stage=%s model=%s wait_s=%.3f",
                     kind, stage, model, delay)
            await asyncio.sleep(delay)

    def record(self, tokens):
        if tokens:
            # Completion time is conservative: do not age out long requests early.
            self.usage.append((time.monotonic(), tokens))


_pacers = weakref.WeakKeyDictionary()


async def measured_call(*, kind, stage, model, call, attempts=2,
                        input_count=None, response_getter=None):
    # Per-loop state also supports repeated asyncio.run() in local tools/tests.
    states = _pacers.setdefault(asyncio.get_running_loop(), {})
    pacer = states.setdefault((kind, model), _Pacer())
    async with pacer.lock:
        return await _measured_call(
            kind=kind, stage=stage, model=model, call=call, attempts=attempts,
            input_count=input_count, response_getter=response_getter, pacer=pacer)


def _rate_limited(exc):
    return (getattr(exc, "status_code", None) == 429
            or getattr(getattr(exc, "response", None), "status_code", None) == 429
            or any(cls.__name__ == "RateLimitError" for cls in type(exc).__mro__))


def _retry_after(exc):
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    headers = {str(k).lower(): v for k, v in (headers or {}).items()}
    value = headers.get("retry-after")
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


def _rate_limit_delay(exc, retry):
    backoff = min(config.RATE_LIMIT_MAX_BACKOFF_SECONDS,
                  config.RATE_LIMIT_BACKOFF_SECONDS * 2 ** min(retry, 30))
    # Server cooldown is a minimum, even when it exceeds our backoff cap.
    return max(_retry_after(exc), backoff * random.uniform(0.8, 1.0))


def _field(obj, name, default=0):
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _usage(response):
    usage = _field(response, "usage", None)
    prompt = int(_field(usage, "prompt_tokens", 0) or 0)
    completion = int(_field(usage, "completion_tokens", 0) or 0)
    total = int(_field(usage, "total_tokens", prompt + completion) or 0)
    return prompt, completion, total


async def _measured_call(
        *, kind: str, stage: str, model: str,
        call: Callable[[int], Awaitable], attempts: int = 2,
        input_count: Optional[int] = None,
        response_getter: Optional[Callable] = None, pacer=None):
    """Run an async provider call with explicit, observable retries.

    Prompt/input contents and credentials are deliberately never logged.
    ``call`` receives a one-based attempt number. ``response_getter`` lets a
    structured call return parsed data while exposing its provider response
    solely for usage accounting.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    attempt = 0
    failures = 0
    rate_retries = 0
    while True:
        await pacer.wait(kind, stage, model)
        limits = budget.current.get()
        if limits:
            limits.before_call()
        attempt += 1
        started = time.perf_counter()
        try:
            result = await call(attempt)
            duration = time.perf_counter() - started
            response = response_getter(result) if response_getter else result
            prompt_tokens, completion_tokens, total_tokens = _usage(response)
            if limits:
                limits.tokens += total_tokens
            pacer.record(total_tokens)
            parts = [
                "provider_call",
                f"kind={kind}",
                f"stage={stage}",
                f"model={model}",
                "status=ok",
                f"attempt={attempt}",
                f"retries={attempt - 1}",
                f"duration_s={duration:.3f}",
                f"prompt_tokens={prompt_tokens}",
                f"completion_tokens={completion_tokens}",
                f"total_tokens={total_tokens}",
            ]
            if input_count is not None:
                parts.append(f"input_count={input_count}")
            log.info(" ".join(parts))
            return result
        except Exception as exc:
            duration = time.perf_counter() - started
            delay = 0.0
            if _rate_limited(exc):
                will_retry = rate_retries < config.RATE_LIMIT_RETRIES
                if will_retry:
                    delay = _rate_limit_delay(exc, rate_retries)
                    rate_retries += 1
            else:
                failures += 1
                will_retry = failures < attempts
            parts = [
                "provider_call",
                f"kind={kind}",
                f"stage={stage}",
                f"model={model}",
                "status=error",
                f"attempt={attempt}",
                f"retries={attempt - 1}",
                f"duration_s={duration:.3f}",
                f"will_retry={str(will_retry).lower()}",
                f"retry_delay_s={delay:.3f}",
                f"error_type={type(exc).__name__}",
            ]
            if input_count is not None:
                parts.append(f"input_count={input_count}")
            log.warning(" ".join(parts))
            if not will_retry:
                raise
            if delay:
                await asyncio.sleep(delay)


def log_fake(*, kind: str, stage: str, model: str,
             input_count: Optional[int] = None):
    parts = [
        "provider_call",
        f"kind={kind}",
        f"stage={stage}",
        f"model={model}",
        "status=fake",
        "attempt=1",
        "retries=0",
        "duration_s=0.000",
        "prompt_tokens=0",
        "completion_tokens=0",
        "total_tokens=0",
    ]
    if input_count is not None:
        parts.append(f"input_count={input_count}")
    log.info(" ".join(parts))
