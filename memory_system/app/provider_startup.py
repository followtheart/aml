"""Prepare provider code before the HTTP server advertises readiness."""
import asyncio
from contextlib import asynccontextmanager
import logging
import time

from . import config, llm

log = logging.getLogger('aml.startup')


@asynccontextmanager
async def lifespan(app):
    if not config.FAKE:
        started = time.monotonic()
        # Keep blocking imports off the event loop. Fail startup if dependencies
        # cannot load instead of accepting requests that all fail initialization.
        await asyncio.to_thread(llm.initialize_provider)
        log.info('provider_sdk_ready duration_s=%.3f', time.monotonic() - started)
    yield
