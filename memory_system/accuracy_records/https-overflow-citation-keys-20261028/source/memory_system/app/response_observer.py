"""Task-local, opt-in capture of public provider responses before parsing."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import hashlib

_sink = ContextVar('source_response_observer', default=None)


@contextmanager
def observe(sink):
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def record(kind, response, request=None):
    sink = _sink.get()
    if sink is None:
        return
    data = response if isinstance(response, dict) else response.model_dump(mode='json')
    fields = ('id', 'model', 'usage', 'meta', 'results', 'output') if kind == 'rerank' else (
        'id', 'model', 'object', 'created', 'choices', 'usage', 'system_fingerprint')
    public = {k: data[k] for k in fields if k in data}
    event = dict(kind=kind, status='provider_response', response=json.loads(json.dumps(public, ensure_ascii=False)))
    if request is not None:
        event['request_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True,
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    sink(event)
