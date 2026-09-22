"""Embedding abstraction over LiteLLM with a deterministic offline fallback."""
import asyncio
import hashlib
import re
from typing import List

import numpy as np

from . import config, metrics


async def embed(texts: List[str], stage: str = "embedding") -> np.ndarray:
    if config.FAKE:
        metrics.log_fake(kind="embedding", stage=stage, model="fake",
                         input_count=len(texts))
        return np.stack([_hash_embed(t) for t in texts])
    import litellm
    kwargs = {}
    if config.EMBED_API_BASE:
        kwargs["api_base"] = config.EMBED_API_BASE
    if config.EMBED_API_KEY:
        kwargs["api_key"] = config.EMBED_API_KEY
    # Matryoshka-capable models shorten natively; slicing other models' vectors
    # would distort the similarity space (REVIEW P0-1). LiteLLM accepts the
    # top-level parameter only for OpenAI's own text-embedding-3 models and
    # rejects it for any other `openai/` model, so compatible providers
    # (Qwen3-Embedding, Qwen3.7 text embedding, DashScope v3/v4) get it as a raw
    # request-body field instead.
    model_lower = config.EMBED_MODEL.lower()
    if "text-embedding-3" in model_lower:
        kwargs["dimensions"] = config.EMBED_DIM
    elif ("qwen3-embedding" in model_lower
          or model_lower.rsplit("/", 1)[-1] in {
              "qwen3.7-text-embedding", "qwen3.7-text-embedding-flash"}
          or re.search(r"text-embedding-v[34]", model_lower)):
        kwargs["extra_body"] = {"dimensions": config.EMBED_DIM}

    async def _embed_batch(batch: List[str]) -> np.ndarray:
        async def _call(_attempt):
            return await litellm.aembedding(
                model=config.EMBED_MODEL, input=batch,
                timeout=config.EMBED_TIMEOUT_SECONDS, num_retries=0, **kwargs)

        # A real provider failure must fail Add/Search. Hash vectors are only
        # used in explicit AML_FAKE mode and never contaminate the real space.
        resp = await metrics.measured_call(
            kind="embedding", stage=stage, model=config.EMBED_MODEL,
            call=_call, attempts=2, input_count=len(batch))
        vecs = np.array([d["embedding"] for d in resp["data"]], dtype=np.float32)
        if len(vecs) != len(batch):
            raise ValueError("Embedding provider returned the wrong number of vectors")
        return vecs

    # Providers cap inputs per request (DashScope: 10); split into ordered
    # batches and let the pacer bound how many are in flight.
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    size = config.EMBED_BATCH_SIZE
    batches = [texts[i:i + size] for i in range(0, len(texts), size)]
    parts = await asyncio.gather(*(_embed_batch(b) for b in batches))
    return _fit_dim(np.concatenate(parts, axis=0))


def _fit_dim(vecs: np.ndarray) -> np.ndarray:
    d = vecs.shape[1]
    if d == config.EMBED_DIM:
        out = vecs
    else:
        raise ValueError(
            f"Embedding provider returned {d} dimensions; configured space "
            f"requires {config.EMBED_DIM}. Configure the native model dimension "
            "or use a provider-supported dimensions parameter.")
    return _normalize(out)


def _hash_embed(text: str) -> np.ndarray:
    """Word-level hashing embedding; deterministic, offline, decent overlap
    semantics for plumbing tests."""
    vec = np.zeros(config.EMBED_DIM, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % config.EMBED_DIM] += 1.0
    return _normalize(vec[None, :])[0]


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n
