"""Provider contract and explicit offline plumbing provider (never a fallback)."""
import hashlib
import json
import math
import re
import time
import asyncio
from pathlib import Path
from typing import Protocol
import httpx
from .schemas import Expansion, Verdicts
from .references import ReferenceVerdicts, reference_packet, decode_references


def terms(text):
    words = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.casefold())
    result = []
    for word in words:
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            result.extend(word[i:i + 2] for i in range(max(1, len(word) - 1)))
        else:
            result.append(word)
    return result


def normalize_vectors(vectors, count, dimension):
    if len(vectors) != count:
        raise ValueError("embedding count mismatch")
    result = []
    for vector in vectors:
        if len(vector) != dimension or not all(math.isfinite(x) for x in vector):
            raise ValueError("invalid embedding dimension or non-finite value")
        norm = math.sqrt(sum(x*x for x in vector))
        if not norm:
            raise ValueError("zero embedding")
        result.append([x/norm for x in vector])
    return result


class Provider(Protocol):
    space_id: str
    dimension: int
    async def expand_memory(self, event: dict) -> dict: ...
    async def expand_query(self, query: dict) -> dict: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def verify(self, query: dict, originals: list[dict]) -> dict: ...


class HTTPProvider:
    def __init__(self, settings, usage_sink=None):
        if not settings.model or not settings.embed_model or not settings.provider_api_key:
            raise ValueError("MC_MODEL, MC_EMBED_MODEL and MC_PROVIDER_API_KEY are required")
        self.settings = settings
        self.usage_sink = usage_sink
        self.client = None
        self.dimension = settings.embed_dimension
        origin = hashlib.sha256((settings.embed_api_base or settings.api_base).encode()).hexdigest()[:12]
        self.space_id = f"{origin}:{settings.embed_model}:{self.dimension}:normalize-v1"

    async def post(self, path, payload):
        for attempt in range(1, 4):
            try:
                return await self._post_once(path, payload, attempt)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 3 or (status is not None and status != 429 and status < 500):
                    raise
                await asyncio.sleep(attempt)

    async def aclose(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _post_once(self, path, payload, attempt):
        embedding = path == "/embeddings"
        base = (self.settings.embed_api_base if embedding else "") or self.settings.api_base
        key = (self.settings.embed_api_key if embedding else "") or self.settings.provider_api_key
        started = time.perf_counter()
        record = {"model": payload["model"], "kind": "embedding" if embedding else "llm",
                  "status": "error", "usage_reported": False, "attempt": attempt}
        try:
            if self.client is None:
                self.client = httpx.AsyncClient(timeout=self.settings.add_seconds)
            response = await self.client.post(base.rstrip("/") + path,
                                         headers={"Authorization": f"Bearer {key}"}, json=payload)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            record.update(status="ok", usage_reported=bool(usage),
                          prompt_tokens=usage.get("prompt_tokens", usage.get("input_tokens")),
                          completion_tokens=usage.get("completion_tokens", usage.get("output_tokens", 0)),
                          total_tokens=usage.get("total_tokens"))
            return data
        finally:
            record["duration_s"] = time.perf_counter() - started
            if self.usage_sink:
                self.usage_sink(record)

    async def complete(self, task, payload, schema):
        prompt = (Path(__file__).resolve().parents[1] / "prompts" / f"{task}.txt").read_text(encoding="utf-8")
        data = await self.post("/chat/completions", {
            "model": self.settings.model, "temperature": 0, "max_tokens": 5000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": prompt + "\nJSON schema:\n" + json.dumps(schema.model_json_schema())},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]})
        return json.loads(data["choices"][0]["message"]["content"])

    async def expand_memory(self, event):
        return await self.complete("expand_memory_v1", event, Expansion)

    async def expand_query(self, query):
        return await self.complete("expand_query_v1", query, Expansion)

    async def verify(self, query, originals):
        packet, aliases, refs = reference_packet(originals)
        raw = await self.complete("verify_original_pair_v6", {"query": query, **packet}, ReferenceVerdicts)
        return decode_references(raw, aliases, refs, originals)

    def verification_size(self, query, originals):
        packet, _, _ = reference_packet(originals)
        return len(json.dumps({"query": query, **packet}, ensure_ascii=False).encode())

    async def rerank(self, query, documents):
        from urllib.parse import urlsplit
        cfg = self.settings
        url = urlsplit(cfg.rerank_api_url)
        if not cfg.rerank_model or not cfg.rerank_api_key or url.username or url.password or url.query or url.fragment:
            raise ValueError("invalid reranker configuration")
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost", "::1"}):
            raise ValueError("reranker requires HTTPS or loopback")
        if len(documents) > cfg.rerank_batch_size or any(len(d.encode()) > cfg.rerank_document_bytes for d in documents):
            raise ValueError("rerank document budget exceeded")
        if sum(len(d.encode())+len(query.encode()) for d in documents) > cfg.rerank_request_bytes:
            raise ValueError("rerank request budget exceeded")
        if cfg.rerank_api_format == "dashscope":
            payload = {"model": cfg.rerank_model, "input": {"query": query, "documents": documents},
                       "parameters": {"top_n": len(documents), "return_documents": False}}
        elif cfg.rerank_api_format == "cohere":
            payload = {"model": cfg.rerank_model, "query": query, "documents": documents, "top_n": len(documents)}
        else:
            raise ValueError("unsupported reranker format")
        started = time.perf_counter()
        record = {"model": cfg.rerank_model, "kind": "rerank", "status": "error", "usage_reported": False}
        try:
            if self.client is None:
                self.client = httpx.AsyncClient(timeout=cfg.add_seconds)
            response = await self.client.post(cfg.rerank_api_url,
                headers={"Authorization": "Bearer " + cfg.rerank_api_key}, json=payload)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            tokens = usage.get("input_tokens", usage.get("prompt_tokens", usage.get("total_tokens")))
            reported = type(tokens) is int and tokens >= 0
            record.update(status="ok", usage_reported=reported, prompt_tokens=tokens if reported else None,
                          completion_tokens=0 if reported else None, usage_source="provider" if reported else "missing")
            rows = data.get("results", data.get("output", {}).get("results", []))
            scores = [None]*len(documents)
            for row in rows:
                i, score = row.get("index"), row.get("relevance_score")
                if type(i) is not int or not 0 <= i < len(scores) or scores[i] is not None:
                    raise ValueError("rerank identity mismatch")
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError("invalid rerank score")
                scores[i] = float(score)
            if any(s is None for s in scores):
                raise ValueError("missing rerank scores")
            return scores
        finally:
            record["duration_s"] = time.perf_counter()-started
            if self.usage_sink:
                self.usage_sink(record)

    async def embed(self, texts):
        vectors = []
        size = self.settings.embed_batch_size
        for start in range(0, len(texts), size):
            batch = texts[start:start + size]
            payload = {"model": self.settings.embed_model, "input": batch}
            if self.settings.embed_send_dimensions:
                payload["dimensions"] = self.dimension
            result = await self.post("/embeddings", payload)
            data = sorted(result["data"], key=lambda item: item["index"])
            if [item["index"] for item in data] != list(range(len(batch))):
                raise ValueError("invalid embedding indices")
            vectors.extend(item["embedding"] for item in data)
        return normalize_vectors(vectors, len(texts), self.dimension)


class OfflineProvider:
    """Lexical smoke mode. No semantic quality claims or invented expansions."""
    dimension = 128
    space_id = "offline-hash-128-v1"

    async def expand_memory(self, event):
        return {"views": []}

    async def expand_query(self, query):
        return {"views": []}

    async def embed(self, texts):
        result = []
        for text in texts:
            vector = [0.0] * self.dimension
            for term in terms(text) or [text]:
                vector[int(hashlib.sha256(term.encode()).hexdigest(), 16) % self.dimension] += 1
            result.append(vector)
        return normalize_vectors(result, len(texts), self.dimension)

    async def verify(self, query, originals):
        # Conservative: offline mode cannot verify semantics.
        return {"decisions": [{"event_id": event["event_id"], "verdict": "uncertain",
                               "relation_type": "other", "subject_match": "unknown",
                               "time_match": "unknown", "polarity": "unknown",
                               "evidence_spans": [], "reason": "offline provider cannot verify semantics"}
                              for event in originals]}
