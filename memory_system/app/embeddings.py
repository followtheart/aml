"""Embedding abstraction over LiteLLM with a deterministic offline fallback."""
import hashlib
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
    # would distort the similarity space (REVIEW P0-1). LiteLLM recognizes the
    # OpenAI parameter directly, while Qwen3 needs it forwarded to compatible
    # providers as an extra request-body field.
    if "text-embedding-3" in config.EMBED_MODEL:
        kwargs["dimensions"] = config.EMBED_DIM
    elif "qwen3-embedding" in config.EMBED_MODEL.lower():
        kwargs["extra_body"] = {"dimensions": config.EMBED_DIM}
    async def _call(_attempt):
        return await litellm.aembedding(
            model=config.EMBED_MODEL, input=texts,
            timeout=60, num_retries=0, **kwargs)

    # A real provider failure must fail Add/Search. Hash vectors are only used
    # in explicit AML_FAKE mode and can never contaminate the real vector space.
    resp = await metrics.measured_call(
        kind="embedding", stage=stage, model=config.EMBED_MODEL,
        call=_call, attempts=2, input_count=len(texts))
    vecs = np.array([d["embedding"] for d in resp["data"]],
                    dtype=np.float32)
    if len(vecs) != len(texts):
        raise ValueError("Embedding provider returned the wrong number of vectors")
    return _fit_dim(vecs)


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
