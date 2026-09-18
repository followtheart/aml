"""Provider query-document reranking with strict identity and budget checks."""
import asyncio
import math
from urllib.parse import urlsplit

import httpx
from . import budget, config, metrics, personal_evidence


class RerankUnavailable(RuntimeError):
    pass


def request_size(query, documents):
    # Conservative input-token bound, including repeated query-document pairs.
    return sum(len(d.encode('utf-8')) + len(query.encode('utf-8')) for d in documents)


def _scores(payload, count):
    if not isinstance(payload, dict):
        raise ValueError('Rerank response must be an object')
    result = payload.get('results')
    if result is None and isinstance(payload.get('output'), dict):
        result = payload['output'].get('results')
    if not isinstance(result, list) or len(result) != count:
        raise ValueError('Rerank response must score every input exactly once')
    scores = [None] * count
    for row in result:
        if not isinstance(row, dict):
            raise ValueError('Invalid rerank result')
        index, value = row.get('index'), row.get('relevance_score')
        if type(index) is not int or not 0 <= index < count or scores[index] is not None:
            raise ValueError('Unknown or repeated rerank index')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Rerank score must be finite')
        # Scores are provider-specific ordering signals, never probabilities.
        scores[index] = float(value)
    return scores


def _usage(payload, estimate):
    usage = payload.get('usage') or (payload.get('meta') or {}).get('tokens') or {}
    if not isinstance(usage, dict):
        usage = {}
    total = usage.get('total_tokens')
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        total = usage.get('input_tokens', usage.get('prompt_tokens'))
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            total = estimate
    return {'usage': {'prompt_tokens': total, 'completion_tokens': 0, 'total_tokens': total}}


async def rerank(query, documents, stage='search.cross_encoder'):
    if not documents:
        return []
    if not isinstance(query, str) or not query.strip() or any(not isinstance(d, str) or not d for d in documents):
        raise ValueError('Rerank requires a query and nonempty document strings')
    size = request_size(query, documents)
    if (len(documents) > config.CE_BATCH_SIZE or size > config.CE_MAX_REQUEST_BYTES
            or len(query.encode('utf-8')) > config.CE_MAX_DOCUMENT_BYTES
            or any(len(d.encode('utf-8')) > config.CE_MAX_DOCUMENT_BYTES for d in documents)):
        raise ValueError('Rerank input exceeds configured bounds; evidence is not truncated')
    budget.check()
    if config.FAKE:
        metrics.log_fake(kind='rerank', stage=stage, model='fake-cross-encoder', input_count=len(documents))
        words = personal_evidence.terms(query)
        return [len(words & personal_evidence.terms(d)) / max(1, len(words)) for d in documents]
    if not config.CE_API_URL or not config.CE_MODEL or not config.CE_API_KEY:
        raise RerankUnavailable('Cross-Encoder endpoint, model or credential is not configured')
    url = urlsplit(config.CE_API_URL)
    if (url.username or url.password or url.query or url.fragment or
            (url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost', '::1')))):
        raise RerankUnavailable('Cross-Encoder requires HTTPS or a loopback endpoint')
    if config.CE_API_FORMAT == 'cohere':
        body = dict(model=config.CE_MODEL, query=query, documents=documents, top_n=len(documents))
    elif config.CE_API_FORMAT == 'dashscope':
        body = dict(model=config.CE_MODEL, input=dict(query=query, documents=documents),
                    parameters=dict(top_n=len(documents), return_documents=False))
    else:
        raise RerankUnavailable('Unsupported Cross-Encoder API format')

    async def call(_attempt):
        async with httpx.AsyncClient(timeout=config.CE_TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = await client.post(config.CE_API_URL, json=body,
                headers={'Authorization': 'Bearer ' + config.CE_API_KEY})
            response.raise_for_status()
            payload = response.json()
            return payload

    limits = budget.current.get()
    if limits:
        limits.reserve_tokens(size)
    try:
        payload = await asyncio.wait_for(metrics.measured_call(kind='rerank', stage=stage,
            model=config.CE_MODEL, call=call, attempts=1, input_count=len(documents),
            response_getter=lambda data: _usage(data, size)), config.CE_TIMEOUT_SECONDS)
    finally:
        if limits:
            limits.release_tokens(size)
    return _scores(payload, len(documents))
