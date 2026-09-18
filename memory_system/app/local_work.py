"""Bounded synchronous search work with cooperative cancellation and safe cleanup."""
import asyncio
import contextvars
import functools
from concurrent.futures import ThreadPoolExecutor
from . import budget, config

_executor = ThreadPoolExecutor(max_workers=config.SEARCH_LOCAL_CONCURRENCY,
                               thread_name_prefix='aml-search')


async def _drain(future):
    """Repeated cancellation must not detach a worker from its resources."""
    while not future.done():
        try:
            await asyncio.wait({future})
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if future.done() and not future.cancelled():
        future.exception()  # Retrieve a worker failure even when its caller was cancelled.


async def run(call, *args, **kwargs):
    limits = budget.current.get()
    context = contextvars.copy_context()
    def execute():
        budget.check()
        return call(*args, **kwargs)
    future = asyncio.get_running_loop().run_in_executor(_executor, context.run, execute)
    try:
        await asyncio.wait({future})
        return future.result()
    except asyncio.CancelledError:
        if limits:
            limits.cancelled.set()
        # Workers inspect this token and SQLite progress handlers interrupt SQL.
        # Never close a snapshot while a worker can still access its connection.
        await _drain(future)
        raise
    except Exception:
        if limits:
            limits.check()
        raise


async def close_snapshot(cm):
    # Cleanup must run even after the request deadline/cancellation token is set.
    future = asyncio.get_running_loop().run_in_executor(_executor, functools.partial(cm.__exit__, None, None, None))
    try:
        await asyncio.wait({future})
        future.result()
    except asyncio.CancelledError:
        await _drain(future)
        raise
