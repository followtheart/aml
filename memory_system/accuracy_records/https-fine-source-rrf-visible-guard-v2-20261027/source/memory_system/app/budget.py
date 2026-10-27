"""Request-scoped provider limits; retries share the same accounting."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import time
import threading
from dataclasses import field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    deadline: float
    max_calls: int
    max_tokens: int
    calls: int = 0
    tokens: int = 0
    reserved_tokens: int = 0
    reserved_calls: int = 0
    cancelled: threading.Event = field(default_factory=threading.Event)

    def check(self):
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise TimeoutError('Search local work deadline exhausted')

    def before_call(self):
        self.check()
        if self.calls + self.reserved_calls >= self.max_calls or self.tokens >= self.max_tokens:
            raise BudgetExceeded('request provider budget exhausted')
        self.calls += 1

    def reserve_calls(self, amount):
        """Keep later-stage calls unavailable to current calls and retries."""
        self.check()
        if self.calls + self.reserved_calls + amount > self.max_calls:
            raise BudgetExceeded('request has insufficient unreserved calls')
        self.reserved_calls += amount

    def release_calls(self, amount):
        self.reserved_calls -= amount

    def reserve_tokens(self, amount):
        """Reserve an input/output bound before concurrent provider dispatch."""
        self.check()
        if self.tokens + self.reserved_tokens + amount > self.max_tokens:
            raise BudgetExceeded('request has insufficient unreserved tokens')
        self.reserved_tokens += amount

    def release_tokens(self, amount):
        self.reserved_tokens -= amount


current = ContextVar('request_budget', default=None)


def check():
    value = current.get()
    if value:
        value.check()


@contextmanager
def scope(seconds=45, calls=12, tokens=64000):
    value = Budget(time.monotonic() + seconds, calls, tokens)
    token = current.set(value)
    try:
        yield value
    finally:
        current.reset(token)
