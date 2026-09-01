"""Embedding abstraction over LiteLLM with a deterministic offline fallback."""
import hashlib
from typing import List

import numpy as np

from . import config


async def embed(texts: List[str]) -> np.ndarray:
    if config.FAKE:
        return np.stack([_hash_embed(t) for t in texts])
    import litellm
    kwargs = {}
    if config.EMBED_API_BASE:
        kwargs["api_base"] = config.EMBED_API_BASE
    if config.EMBED_API_KEY:
        kwargs["api_key"] = config.EMBED_API_KEY
    try:
        resp = await litellm.aembedding(model=config.EMBED_MODEL, input=texts,
                                        timeout=60, num_retries=1, **kwargs)
        vecs = np.array([d["embedding"] for d in resp["data"]],
                        dtype=np.float32)
        return _fit_dim(vecs)
    except Exception as e:
        # Credential/network failure: degrade to hash embeddings so writes
        # never hard-fail. Vector space is then hash-based until restart —
        # acceptable for dev; fix credentials and rebuild for real runs.
        import logging
        logging.getLogger("aml.embed").warning(
            "embedding call failed (%s); falling back to hash embedding", e)
        return np.stack([_hash_embed(t) for t in texts])


def _fit_dim(vecs: np.ndarray) -> np.ndarray:
    d = vecs.shape[1]
    if d == config.EMBED_DIM:
        out = vecs
    elif d > config.EMBED_DIM:
        out = vecs[:, : config.EMBED_DIM]
    else:
        out = np.pad(vecs, ((0, 0), (0, config.EMBED_DIM - d)))
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
