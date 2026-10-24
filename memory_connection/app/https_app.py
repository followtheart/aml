"""AML-compatible HTTPS entrypoint backed exclusively by memory_connection."""
import hmac
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from collections import Counter
from fastapi import Depends, FastAPI, Header, HTTPException
from . import https_schemas as contract
from .access_log import AccessLogMiddleware
from .config import Settings
from .https_store import HTTPSStore, digest
from .providers import HTTPProvider, OfflineProvider
from .schemas import AddRequest, Message, SearchRequest
from .service import MemoryService
from .store import Conflict

logger = logging.getLogger(__name__)


def create_app(settings=None, provider=None):
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        if not config.api_key:
            raise ValueError('MC_API_KEY is required')
        active = provider or (OfflineProvider() if config.fake else HTTPProvider(config))
        store = HTTPSStore(config.db_path, active.space_id)
        app.state.service = MemoryService(store, active, config)
        try:
            yield
        finally:
            try:
                if hasattr(active, 'aclose'):
                    await active.aclose()
            finally:
                store.close()

    app = FastAPI(title='memory_connection HTTPS', version='1.0.0', lifespan=lifespan)
    app.add_middleware(AccessLogMiddleware, path=os.environ.get('MC_ACCESS_LOG',
        str(Path(__file__).resolve().parents[1] / 'logs' / 'https-access.jsonl')))

    async def auth(authorization: str = Header(default='')):
        if not hmac.compare_digest(authorization.encode(), f'Bearer {config.api_key}'.encode()):
            raise HTTPException(401, 'invalid token')

    @app.middleware('http')
    async def identify(request, call_next):
        response = await call_next(request)
        response.headers['X-Memory-System'] = 'memory_connection'
        return response

    @app.get('/health')
    async def health():
        return {'status': 'ok'}

    @app.post('/add', response_model=contract.AddResponse, dependencies=[Depends(auth)])
    async def add(req: contract.HTTPAddRequest):
        try:
            # The contract model preserves original provenance and accepts large histories.
            return await app.state.service.add(req)
        except ValueError as exc:
            raise HTTPException(422, detail={'reason': str(exc)}) from exc
        except Exception as exc:
            logger.error('add failed: %s', type(exc).__name__)
            raise HTTPException(500, detail={'reason': 'add failed'}) from exc

    async def run_search(req):
        service = app.state.service
        # A per-request settings copy prevents one client's budget changing another's.
        if req.evidence_token_budget is not None:
            service = MemoryService(service.store, service.provider,
                config.model_copy(update={'evidence_bytes': req.evidence_token_budget}))
        request = req.model_copy(update={'min_revision': req.min_revision or 0,
            'include_history': bool(req.include_history), 'options': req.options or [], 'debug': False})
        result = await service.search(request)
        # Keep the established response schema; expose content-free diagnostics
        # in its existing coverage_manifest instead of silently dropping them.
        reasons = Counter(e['reason'] for e in result.get('verification_errors', []))
        metrics = result.get('metrics', {})
        result['coverage_manifest'] = {
            'system': 'memory_connection', 'search_degraded': bool(result.get('degradation')),
            'degradation': result.get('degradation', []),
            'verification_errors': dict(reasons),
            'omitted': dict(Counter(e['reason'] for e in result.get('omitted', []))),
            'metrics': {k: v for k, v in metrics.items() if k != 'bridge_reserved_events'},
            'empty_reason': None if result['data'] else
                'snapshot_invalidated' if result.get('verification_status') == 'invalidated' else
                'no_candidates' if not metrics.get('candidate_count') else
                'verification_incomplete' if reasons else
                'evidence_budget' if result.get('omitted') else 'no_relevant_evidence',
        }
        for item in result['data']:
            item['created_at'] = item['valid_from']
            item['is_constraint'] = item['relation_type'] == 'constraint'
            item['memory_type'] = item.get('memory_type', 'fact')
            item['source_count'] = len(item['sources'])
            item['coverage_ids'] = [s['source_id'] for s in item['sources']]
            item['personal_evidence'] = any(s['role'] == 'user' and 'personal' in s.get('trust_scope', ['personal']) for s in item['sources'])
            item['temporal'] = {k: item[k] for k in ('valid_from', 'valid_to')}
        return result

    @app.post('/search', response_model=contract.SearchResponse, dependencies=[Depends(auth)])
    async def search(req: contract.HTTPSearchRequest):
        try:
            return await run_search(req)
        except Exception as exc:
            logger.error('search failed: %s', type(exc).__name__)
            raise HTTPException(500, detail={'reason': 'search failed'}) from exc

    @app.delete('/memory/{user_id}', response_model=contract.PurgeResponse, dependencies=[Depends(auth)])
    async def purge(user_id: str):
        return app.state.service.store.purge(user_id)

    @app.delete('/users/{user_id}', dependencies=[Depends(auth)], include_in_schema=False)
    async def native_purge(user_id: str):
        return app.state.service.store.purge(user_id)

    @app.post('/answer', dependencies=[Depends(auth)])
    async def answer(req: SearchRequest):
        try:
            return await app.state.service.answer(req)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/feedback', response_model=contract.FeedbackResponse, dependencies=[Depends(auth)])
    async def feedback(req: contract.FeedbackRequest):
        service = app.state.service
        payload_hash = digest(req.model_dump(exclude={'feedback_event_id'}))
        event_id = req.feedback_event_id or payload_hash
        try:
            existing = service.store.feedback(req.user_id, event_id)
            if existing:
                if existing['payload_hash'] != payload_hash:
                    raise ValueError('Feedback event was reused with different content')
                return dict(success=True, feedback_event_id=event_id, memory_ids=existing['memory_ids'])
            epoch = service.store.state(req.user_id)[1]
            events = []
            if req.distill:
                # Preserve the supplied execution as an unvalidated hypothesis. No
                # generated strategy is promoted to an original memory fact.
                request = AddRequest(request_id='feedback:' + event_id, user_id=req.user_id,
                    session_id=req.session_id, messages=[Message(role='tool', source_kind='tool',
                        content=f'Task: {req.task}\nOutcome: {req.outcome}\nExecution kind: {req.execution_kind}\nTrace:\n{req.trace}',
                        timestamp=int(datetime.now(timezone.utc).timestamp() * 1000))])
                events, _ = await service.prepare_add(request)
            ids = service.store.save_feedback(req, event_id, payload_hash, events, epoch)
            return dict(success=True, feedback_event_id=event_id, memory_ids=ids)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.error('feedback failed: %s', type(exc).__name__)
            raise HTTPException(500, detail={'reason': 'feedback failed'}) from exc

    @app.get('/feedback/{user_id}/{event_id}', dependencies=[Depends(auth)])
    async def observation(user_id: str, event_id: str):
        event = app.state.service.store.feedback(user_id, event_id)
        if event is None:
            raise HTTPException(404, 'Feedback event not found')
        kind = event['payload'].get('execution_kind', 'observed')
        return dict(kind='observation' if kind == 'observed' else kind, **event)

    @app.get('/experience/{user_id}/{memory_id}', dependencies=[Depends(auth)])
    async def experience(user_id: str, memory_id: str):
        try:
            return app.state.service.store.experience(user_id, memory_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post('/experience/review', dependencies=[Depends(auth)])
    async def review(req: contract.ExperienceReviewRequest):
        try:
            return app.state.service.store.review(req)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
