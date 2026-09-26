"""FastAPI entrypoint: /add, /search, /health (AML synchronous contract)."""
import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from .errors import InvalidRequest
from .access_log import AccessLogMiddleware

from . import (add_pipeline, config, experience, memory_debug, schemas,
               search_debug, search_pipeline, store)

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AML Memory System", version="0.4.0")
app.add_middleware(AccessLogMiddleware, path=os.environ.get(
    "AML_ACCESS_LOG", str(Path(__file__).resolve().parents[1] / "logs" / "access.jsonl")))
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
async def add(req: schemas.HTTPAddRequest):
    try:
        result = await add_pipeline.run_add(_store, req)
    except InvalidRequest as e:
        raise HTTPException(status_code=422, detail={"reason": str(e)}) from e
    except Exception as e:  # provider and internal failures remain retryable
        raise HTTPException(status_code=500,
                            detail={"reason": f"add failed: {e}"})
    # contract: echo ids byte-for-byte; ULM §2.5: return the write revision
    return schemas.AddResponse(success=True, request_id=req.request_id,
                               user_id=req.user_id, session_id=req.session_id,
                               write_revision=result["write_revision"],
                               scope_epoch=result["scope_epoch"])


@app.post("/search", response_model=schemas.SearchResponse,
          dependencies=[Depends(auth)])
async def search(req: schemas.HTTPSearchRequest):
    try:
        return await search_pipeline.run_search(_store, req)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail={"reason": f"search failed: {e}"})


@app.delete("/memory/{user_id}", response_model=schemas.PurgeResponse,
            dependencies=[Depends(auth)])
async def purge_memory(user_id: str):
    try:
        receipt = _store.purge_user(user_id)
        memory_debug.purge_user(user_id)
        search_debug.purge_user(user_id)
        return schemas.PurgeResponse(**receipt)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail={"reason": f"purge failed: {e}"})


@app.post("/feedback", response_model=schemas.FeedbackResponse,
          dependencies=[Depends(auth)])
async def feedback(req: schemas.FeedbackRequest):
    try:
        ids = await experience.run_feedback(_store, req)
        return schemas.FeedbackResponse(memory_ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail={"reason": f"feedback failed: {e}"})
