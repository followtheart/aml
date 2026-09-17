"""Request-scoped provider limits; retries share the same accounting."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import time


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    deadline: float
    max_calls: int
    max_tokens: int
    calls: int = 0
    tokens: int = 0

    def before_call(self):
        if time.monotonic() >= self.deadline or self.calls >= self.max_calls or self.tokens >= self.max_tokens:
            raise BudgetExceeded('request provider budget exhausted')
        self.calls += 1


current = ContextVar('request_budget', default=None)


@contextmanager
def scope(seconds=45, calls=12, tokens=64000):
    value = Budget(time.monotonic() + seconds, calls, tokens)
    token = current.set(value)
    try:
        yield value
    finally:
        current.reset(token)
