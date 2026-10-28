"""Rotating JSONL HTTP metadata log; excludes credentials and request bodies."""
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from datetime import datetime, timezone
import uuid


class AccessLogMiddleware:
    def __init__(self, app, path):
        self.app = app
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f'aml.access.{target.resolve()}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = RotatingFileHandler(target, maxBytes=10 * 1024 * 1024,
                                          backupCount=5, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        status = 500
        error = None

        async def logged_send(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
                message = dict(message)
                message['headers'] = list(message.get('headers', [])) + [
                    (b'x-request-id', request_id.encode('ascii'))]
            await send(message)

        try:
            await self.app(scope, receive, logged_send)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            route = scope.get('route')
            event = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                         request_id=request_id, method=scope['method'],
                         route=getattr(route, 'path', '<unmatched>'),
                         status=status, duration_ms=round((time.monotonic()-started)*1000, 2))
            if error:
                event['error_type'] = error
            self.logger.info(json.dumps(event, ensure_ascii=False))
