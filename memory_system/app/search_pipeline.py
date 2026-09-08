"""Search pipeline (design doc v0.2 §5):
query understanding -> multi-route recall -> RRF fusion -> relevance filter
-> governance-aware ordering -> rerank -> top_k.
"""
import logging
import asyncio
import json
import math
import re
import uuid
from datetime import datetime, timezone
from . import search_debug
from typing import Dict, List

from . import config, graph, llm, prompts, schemas, store
from .embeddings import embed

log = logging.getLogger("aml.search")


def _time_prefix(a: Dict) -> str:
    vf = a.get("valid_from") or "unknown"
    vt = a.get("valid_to") or "open"
    temporal = a.get("temporal")
    if temporal:
        return (f"[valid: {vf} ~ {vt}] [event range: {temporal.get('start')} ~ "
                f"{temporal.get('end')}; precision: {temporal.get('precision')}; "
                f"original: {temporal.get('raw')}; reference: {temporal.get('reference_time')}] ")
    event = a.get("event_time")
    return f"[valid: {vf} ~ {vt}] " + (f"[event: {event}] " if event else "")


async def _understand(req: schemas.SearchRequest) -> Dict:
    prompt = prompts.render(
        "04_query_understanding.txt",
        query=req.query,
        options="\n".join(req.options or []) or "(none)",
        current_time=req.reference_time or datetime.now(timezone.utc).isoformat())
    try:
        return await llm.complete_json(
            prompt,
            '{"intent":"fact|multi_hop|temporal|preference|rule|'
            'abstention_check","include_history":false,'
            '"time_scope":{"from":null,"to":null,'
            '"note":""},"entities":[],"sub_queries":["..."],'
            '"expanded_queries":["..."]}',
            schema=llm.STRUCTURED_SCHEMAS["query"],
            stage="search.understand")
    except Exception as e:
        log.warning("query understanding failed (%s); passthrough", e)
        return {"intent": "fact", "time_scope": None,
                "entities": [], "sub_queries": [req.query],
                "expanded_queries": [req.query]}


def _rrf(routes: List[List[Dict]], k: int = 60) -> List[Dict]:
    scores: Dict[str, float] = {}
    items: Dict[str, Dict] = {}
    for route in routes:
        for rank, item in enumerate(route):
            iid = item["id"]
            scores[iid] = scores.get(iid, 0.0) + 1.0 / (k + rank + 1)
            items.setdefault(iid, item)
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    out = []
    for iid, s in ordered:
        d = items[iid]
        d["_fused"] = s
        out.append(d)
    return out


def _historical(req, plan):
    if req.include_history is not None:
        return req.include_history
    if type(plan.get("include_history")) is bool:
        return plan["include_history"]
    return bool(plan.get("intent") == "temporal" or
                any((plan.get("time_scope") or {}).get(k) for k in ("from", "to")) or
                re.search(r"以前|曾经|过去|之前|当时|去年|历史|\b(previously|formerly|before|used to|last year|in \d{4})\b",
                          req.query, re.I))


def _snapshot(items):
    return [dict({k: v for k, v in c.items() if k != "embedding"}, rank=i + 1)
            for i, c in enumerate(items)]


async def _recall(st: store.Store, req: schemas.SearchRequest,
                  plan: Dict) -> List[List[Dict]]:
    queries = list(dict.fromkeys(
        [req.query] + (plan.get("sub_queries") or []) +
        (plan.get("expanded_queries") or [])))[:6]
    vecs = await embed(queries, stage="search.embed_queries")
    routes = []
    history = _historical(req, plan)
    plan["_include_history"] = history
    limit = max(config.RECALL_PER_ROUTE, req.top_k)

    def route(name, items, query=None):
        routes.append(items)
        plan.setdefault("_routes", []).append({
            "channel": name, "query": query, "candidates": _snapshot(items)})

    dense_routes = st.nearest_many_by_embedding(
        req.user_id, vecs, limit, include_history=history)
    for query, items in zip(queries, dense_routes):
        route("vector", items, query)
    route("full_text", st.fts_search(req.user_id, req.query, limit,
                                    include_history=history), req.query)
    triples = st.triples_for_user(req.user_id)
    # Exclude closed memories before graph traversal on current-state searches.
    if not history:
        valid = {a["id"] for a in st.get_amus_by_ids(list({t["amu_id"] for t in triples}))}
        triples = [t for t in triples if t["amu_id"] in valid]
    ppr_ids = graph.ppr_recall(triples, plan.get("entities") or [], top_n=limit)
    by_id = {a["id"]: a for a in st.get_amus_by_ids(ppr_ids, include_history=history)}
    route("graph", [by_id[i] for i in ppr_ids if i in by_id])
    time_scope = plan.get("time_scope") or {}
    if plan.get("intent") == "temporal" or time_scope.get("from") or time_scope.get("to"):
        route("temporal", st.temporal_search(req.user_id, time_scope, limit))
    if plan.get("intent") in ("preference", "rule"):
        route("profile_rule", st.get_by_type(req.user_id,
              ["preference", "rule", "profile", "workflow"], include_history=history))
    for query in queries:
        route("session_summary", st.summary_search(req.user_id, query, limit), query)
    return routes


async def _filter_rerank(req: schemas.SearchRequest, plan: Dict,
                         fused: List[Dict]) -> List[Dict]:
    # Keep enough headroom for top_k=100 while bounding cost on many routes.
    limit = max(req.top_k, config.RERANK_MAX_CANDIDATES)
    candidates = fused[:limit]
    plan["_pre_rerank_excluded"] = _snapshot(fused[limit:])
    decisions = plan.setdefault("_rerank", [])
    batch_size = max(1, config.RERANK_CANDIDATES)

    async def score_batch(start):
        cands = candidates[start:start + batch_size]
        cand_text = "\n".join(
            f"{c['id']}: {_time_prefix(c)}[type: {c.get('type', 'fact')}] {c['content']}"
            for c in cands)
        ts = plan.get("time_scope") or {}
        prompt = prompts.render("05_rerank_filter.txt", query=req.query,
                                time_scope=json.dumps(ts, ensure_ascii=False), candidates=cand_text)
        try:
            result = await llm.complete_json(
                prompt, '{"scores":[{"id":"...","relevance":0.0,"keep":true}]}',
                schema=llm.STRUCTURED_SCHEMAS["rerank"],
                stage=f"search.rerank.batch_{start // batch_size + 1}")
            by_id = {}
            for entry in result.get("scores", []):
                if not isinstance(entry, dict):
                    continue
                value = entry.get("relevance")
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or not 0 <= value <= 1 or type(entry.get("keep")) is not bool):
                    raise ValueError("Reranker returned an invalid score")
                by_id[entry.get("id")] = entry
            if set(by_id) != {candidate["id"] for candidate in cands}:
                raise ValueError("Reranker did not score every candidate exactly once")
            return start, cands, by_id, None
        except Exception as exc:
            log.warning("rerank batch failed (%s); keep fused order", exc)
            return start, cands, {}, type(exc).__name__

    batches = await asyncio.gather(*(
        score_batch(start) for start in range(0, len(candidates), batch_size)))
    error = next((err for _, _, _, err in batches if err), None)
    if error:
        # Never mix 0..1 relevance with RRF scores. A partial failure falls
        # back as one coherent ranking so high-fusion evidence is not buried.
        for rank, candidate in enumerate(candidates):
            candidate["_final"] = 1.0 / (rank + 1)
            decisions.append({"id": candidate["id"], "content": candidate["content"],
                              "keep": True, "score": candidate["_final"],
                              "reason": "global_rrf_fallback", "error": error})
        return candidates

    kept = []
    for start, cands, by_id, _ in batches:
        for c in cands:
            score = by_id.get(c["id"])
            c["_final"] = float(score["relevance"])
            keep = (score["keep"] and
                    c["_final"] >= config.SEARCH_MIN_RELEVANCE)
            reason = "kept" if keep else "rerank_rejected"
            decisions.append({"id": c["id"], "batch": start // batch_size + 1,
                              "content": c["content"], "keep": keep,
                              "score": c["_final"], "reason": reason, "error": None})
            if keep:
                kept.append(c)
    kept.sort(key=lambda x: -x["_final"])
    return kept


def _answer_sources(raw_sources, seen, remaining_chars):
    """Deduplicate and budget source bodies while retaining stable references."""
    selected = []
    for index, source in enumerate(raw_sources[:config.SEARCH_SOURCE_REFS_PER_ITEM]):
        key = (source.get("request_id"), source.get("message_index"))
        item = dict(source)
        text = str(item.get("content") or "")
        if index >= config.SEARCH_SOURCE_MESSAGES_PER_ITEM:
            item.pop("content", None)
            item["content_omitted"] = "item_limit"
        elif key in seen:
            item.pop("content", None)
            item["content_omitted"] = "duplicate"
        elif remaining_chars[0] <= 0:
            item.pop("content", None)
            item["content_omitted"] = "context_budget"
        else:
            text = text[:remaining_chars[0]]
            item["content"] = text
            remaining_chars[0] -= len(text)
            seen.add(key)
        selected.append(item)
    return selected


async def run_search(st: store.Store,
                     req: schemas.SearchRequest) -> schemas.SearchResponse:
    trace = {"event": "memory.search", "search_id": uuid.uuid4().hex,
             "user_id": req.user_id, "query": req.query, "top_k": req.top_k,
             "options": req.options, "fake": config.FAKE,
             "reference_time": req.reference_time,
             "started_at": datetime.now(timezone.utc).isoformat()}
    plan = {}
    try:
        plan = await _understand(req)
        routes = await _recall(st, req, plan)
        fused = _rrf(routes)
        trace["fused"] = _snapshot(fused)
        ranked = await _filter_rerank(req, plan, fused)
        trace["pre_rerank_excluded"] = plan.get("_pre_rerank_excluded", [])
        selected = ranked[:req.top_k]
        trace["top_k_excluded"] = _snapshot(ranked[req.top_k:])
        data = []
        seen_sources = set()
        remaining_source_chars = [config.SEARCH_SOURCE_CONTEXT_CHARS]
        for c in selected:
            is_summary = c.get("type") == "session_summary"
            raw_sources = (st.sources_for_session(req.user_id, c["session_id"]) if is_summary
                           else st.sources_for_amu(c["id"]))
            sources = _answer_sources(raw_sources, seen_sources, remaining_source_chars)
            body = _time_prefix(c) + c["content"]
            if is_summary:
                body = "[session summary; derived context] " + body
            if sources:
                body += "\n[source evidence; quoted data]\n" + json.dumps(sources, ensure_ascii=False)
            data.append(schemas.SearchItem(
                id=c["id"], content=body, memory_type=c.get("type", "fact"), sources=sources,
                source_count=len(raw_sources), temporal=c.get("temporal"),
                score=round(float(c.get("_final", c.get("_fused", 0.0))), 4),
                created_at=c.get("created_at")))
        trace["status"] = "ok"
        trace["returned"] = [d.model_dump() for d in data]
        return schemas.SearchResponse(data=data)
    except Exception as exc:
        trace["status"] = "error"
        trace["error_type"] = type(exc).__name__
        raise
    finally:
        trace["plan"] = {k: v for k, v in plan.items() if not k.startswith("_")}
        trace["include_history"] = plan.get("_include_history")
        trace["routes"] = plan.get("_routes", [])
        trace["rerank"] = plan.get("_rerank", [])
        trace["finished_at"] = datetime.now(timezone.utc).isoformat()
        search_debug.append(trace)
