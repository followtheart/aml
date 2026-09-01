"""FastAPI entrypoint: /add, /search, /health (AML synchronous contract)."""
import logging

from fastapi import Depends, FastAPI, Header, HTTPException

from . import add_pipeline, config, schemas, search_pipeline, store

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AML Memory System", version="0.2.0")
_store = store.Store()


def auth(authorization: str = Header(default="")):
    if not config.API_KEY:
        return  # open mode (local smoke only)
    if authorization != f"Bearer {config.API_KEY}":
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/add", response_model=schemas.AddResponse,
          dependencies=[Depends(auth)])
async def add(req: schemas.AddRequest):
    try:
        await add_pipeline.run_add(_store, req)
    except Exception as e:  # never return 202; retryable 5xx only
        raise HTTPException(status_code=500,
                            detail={"reason": f"add failed: {e}"})
    # contract: echo ids byte-for-byte
    return schemas.AddResponse(success=True, request_id=req.request_id,
                               user_id=req.user_id, session_id=req.session_id)


@app.post("/search", response_model=schemas.SearchResponse,
          dependencies=[Depends(auth)])
async def search(req: schemas.SearchRequest):
    try:
        return await search_pipeline.run_search(_store, req)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail={"reason": f"search failed: {e}"})
