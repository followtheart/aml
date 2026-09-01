"""Search pipeline (design doc v0.2 §5):
query understanding -> multi-route recall -> RRF fusion -> relevance filter
-> governance-aware ordering -> rerank -> top_k.
"""
import logging
from typing import Dict, List

from . import config, graph, llm, prompts, schemas, store
from .embeddings import embed

log = logging.getLogger("aml.search")


def _time_prefix(a: Dict) -> str:
    vf = (a.get("valid_from") or "")[:10]
    vt = (a.get("valid_to") or "")[:10] or "now"
    return f"[valid: {vf} ~ {vt}] " if vf else ""


async def _understand(req: schemas.SearchRequest) -> Dict:
    prompt = prompts.render(
        "04_query_understanding.txt",
        query=req.query,
        options="\n".join(req.options or []) or "(none)",
        current_time="2026-09-01T00:00:00Z")
    try:
        return await llm.complete_json(
            prompt,
            '{"intent":"fact|multi_hop|temporal|preference|rule|'
            'abstention_check","time_scope":{"from":null,"to":null,'
            '"note":""},"entities":[],"sub_queries":["..."],'
            '"expanded_queries":["..."]}',
            schema=llm.STRUCTURED_SCHEMAS["query"])
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


async def _recall(st: store.Store, req: schemas.SearchRequest,
                  plan: Dict) -> List[List[Dict]]:
    queries = list(dict.fromkeys(
        [req.query] + (plan.get("sub_queries") or []) +
        (plan.get("expanded_queries") or [])))[:6]
    vecs = await embed(queries)
    routes: List[List[Dict]] = []

    # route 1+3: dense vector over (sub)queries — primary route
    for v in vecs:
        routes.append(st.nearest_by_embedding(
            req.user_id, v, config.RECALL_PER_ROUTE))

    # route 2: BM25/FTS
    routes.append(st.fts_search(req.user_id, req.query,
                                config.RECALL_PER_ROUTE))

    # route 4: graph PPR expansion
    ppr_ids = graph.ppr_recall(st.triples_for_user(req.user_id),
                               plan.get("entities") or [])
    if ppr_ids:
        routes.append(st.get_amus_by_ids(ppr_ids))

    # route 6: profile/rule direct feed
    if plan.get("intent") in ("preference", "rule"):
        routes.append(st.get_by_type(req.user_id,
                                     ["preference", "rule", "profile",
                                      "workflow"]))
    return routes


async def _filter_rerank(req: schemas.SearchRequest, plan: Dict,
                         fused: List[Dict]) -> List[Dict]:
    cands = fused[: config.RERANK_CANDIDATES]
    if not cands:
        return []
    cand_text = "\n".join(f"{c['id']}: {c['content']}" for c in cands)
    ts = plan.get("time_scope") or {}
    prompt = prompts.render(
        "05_rerank_filter.txt",
        query=req.query,
        time_scope=f"{ts.get('from')} ~ {ts.get('to')}" if ts else "(any)",
        candidates=cand_text)
    try:
        result = await llm.complete_json(
            prompt,
            '{"scores":[{"id":"...","relevance":0.0,"keep":true}]}',
            schema=llm.STRUCTURED_SCHEMAS["rerank"])
        scores = result.get("scores", [])
        by_id = {s["id"]: s for s in scores if isinstance(s, dict)}
    except Exception as e:
        log.warning("rerank failed (%s); keep fused order", e)
        return cands
    kept = []
    for c in cands:
        s = by_id.get(c["id"])
        if s is None:
            c["_final"] = c["_fused"] * 0.1  # unscored tail
            kept.append(c)
        elif s.get("keep", True):
            c["_final"] = float(s.get("relevance", 0.0))
            kept.append(c)
    kept.sort(key=lambda x: -x["_final"])
    return kept


async def run_search(st: store.Store,
                     req: schemas.SearchRequest) -> schemas.SearchResponse:
    plan = await _understand(req)
    routes = await _recall(st, req, plan)
    fused = _rrf(routes)
    ranked = await _filter_rerank(req, plan, fused)

    # abstention: nothing survived filtering with any relevance
    if ranked and all(c.get("_final", 0) <= 0 for c in ranked[:5]):
        ranked = []

    data = [
        schemas.SearchItem(
            id=c["id"],
            content=_time_prefix(c) + c["content"],
            score=round(float(c.get("_final", c.get("_fused", 0.0))), 4),
            created_at=c.get("created_at"))
        for c in ranked[: req.top_k]
    ]
    return schemas.SearchResponse(data=data)
