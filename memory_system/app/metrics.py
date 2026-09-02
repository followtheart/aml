"""Privacy-safe timing and token metrics for external model calls."""
import logging
import time
from typing import Awaitable, Callable, Optional


log = logging.getLogger("aml.metrics")


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


async def measured_call(
        *, kind: str, stage: str, model: str,
        call: Callable[[int], Awaitable], attempts: int = 2,
        input_count: Optional[int] = None,
        response_getter: Optional[Callable] = None):
    """Run an async provider call with explicit, observable retries.

    Prompt/input contents and credentials are deliberately never logged.
    ``call`` receives a one-based attempt number. ``response_getter`` lets a
    structured call return parsed data while exposing its provider response
    solely for usage accounting.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            result = await call(attempt)
            duration = time.perf_counter() - started
            response = response_getter(result) if response_getter else result
            prompt_tokens, completion_tokens, total_tokens = _usage(response)
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
            last_error = exc
            will_retry = attempt < attempts
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
                f"error_type={type(exc).__name__}",
            ]
            if input_count is not None:
                parts.append(f"input_count={input_count}")
            log.warning(" ".join(parts))
    raise last_error


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
