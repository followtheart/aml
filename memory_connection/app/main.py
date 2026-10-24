import hmac
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Header, HTTPException
from .config import Settings
from .providers import HTTPProvider, OfflineProvider
from .schemas import AddRequest, SearchRequest
from .service import MemoryService
from .store import Store, Conflict


def create_app(settings=None, provider=None):
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        if not config.api_key:
            raise ValueError("MC_API_KEY is required; this key grants access to all user namespaces")
        active_provider = provider or (OfflineProvider() if config.fake else HTTPProvider(config))
        store = Store(config.db_path, active_provider.space_id)
        app.state.service = MemoryService(store, active_provider, config)
        try:
            yield
        finally:
            if hasattr(active_provider, "aclose"):
                await active_provider.aclose()
            store.close()

    app = FastAPI(title="memory_connection", lifespan=lifespan)

    async def auth(authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization, f"Bearer {config.api_key}"):
            raise HTTPException(401, "invalid service API key")

    @app.get("/health")
    async def health():
        return {"status": "ok", "system": "memory_connection", "fake": config.fake}

    @app.post("/add", dependencies=[Depends(auth)])
    async def add(req: AddRequest):
        try:
            return await app.state.service.add(req)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, "invalid event or provider result") from exc
        except Exception as exc:
            raise HTTPException(503, "memory write failed") from exc

    @app.post("/search", dependencies=[Depends(auth)])
    async def search(req: SearchRequest):
        try:
            return await app.state.service.search(req)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/users/{user_id}", dependencies=[Depends(auth)])
    async def purge(user_id: str):
        return app.state.service.store.purge(user_id)

    @app.post('/answer', dependencies=[Depends(auth)])
    async def answer(req: SearchRequest):
        try:
            return await app.state.service.answer(req)
        except Conflict as exc:
            raise HTTPException(409,str(exc)) from exc

    return app


app = create_app()
