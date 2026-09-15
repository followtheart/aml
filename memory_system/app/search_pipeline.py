"""Search pipeline (ULM design §5 "reconstructive recall"):
query understanding (anchored to the user's latest memory time) -> multi-route
recall (dense / sparse / graph PPR / temporal / scene->cell / profile) -> RRF
-> small-R rerank -> sufficiency verification with bounded iterative retrieval
-> foresight validity filter -> abstention -> top_k. Never generates answers.
"""
import logging
import asyncio
import json
import math
import re
import uuid
from datetime import datetime, timezone
from . import search_debug
from typing import Dict, List, Optional

from . import answer_context, config, graph, integrity, llm, prompts, scenes, schemas, store
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


def _anchor_time(st: store.Store, req: schemas.SearchRequest) -> str:
    """§5.1: relative dates resolve against the user's latest known moment,
    never the service wall clock (REVIEW P0-3)."""
    return (req.reference_time or st.latest_time(req.user_id)
            or datetime.now(timezone.utc).isoformat())


async def _understand(req: schemas.SearchRequest, anchor: Optional[str] = None) -> Dict:
    prompt = prompts.render(
        "04_query_understanding.txt",
        query=req.query,
        options="\n".join(req.options or []) or "(none)",
        current_time=req.reference_time or anchor or datetime.now(timezone.utc).isoformat())
    try:
        return await llm.complete_json(
            prompt,
            '{"intent":"fact|multi_hop|temporal|preference|rule|profile|'
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
    return [dict({k: v for k, v in c.items() if k not in ("embedding", "centroid")}, rank=i + 1)
            for i, c in enumerate(items)]


def _query_terms(queries: List[str]) -> List[str]:
    terms = set()
    for q in queries:
        for tok in re.findall(r"[\w\u3400-\u9fff]+", q):
            key = graph.normalize_entity(tok)
            if key and len(key) > 1:
                terms.add(key)
    return sorted(terms)


async def _recall(st: store.Store, req: schemas.SearchRequest,
                  plan: Dict) -> List[List[Dict]]:
    # A follow-up round (§5.5) supplies its own queries via plan["_queries"].
    queries = list(dict.fromkeys(
        plan.get("_queries") or
        ([req.query] + (plan.get("sub_queries") or []) +
         (plan.get("expanded_queries") or []))))[:6]
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
    # §5.2 scene->cell two-stage route: pick top-m MemScenes, then rank cells inside.
    top_scenes = scenes.rank_scenes(st.list_scenes(req.user_id), vecs,
                                    _query_terms(queries), config.SCENE_TOP_M)
    plan["_scenes"] = [{"id": s["id"], "score": round(s["_score"], 4),
                        "summary": (s.get("summary") or "")[:200]} for s in top_scenes]
    if top_scenes:
        per_scene = st.nearest_many_by_embedding(
            req.user_id, vecs[:1], config.SCENE_CELLS_PER_SCENE * len(top_scenes),
            include_history=history, scene_ids=[s["id"] for s in top_scenes])[0]
        route("scene", per_scene, queries[0])
    # §2.3 rules are always injected; other profile memories on matching intents.
    profile_types = ["rule"]
    if plan.get("intent") in ("preference", "rule", "profile"):
        profile_types += ["preference", "profile", "workflow"]
    route("profile_rule", st.get_by_type(req.user_id, profile_types,
                                         include_history=history))
    if config.SUMMARY_ROUTE:
        for query in queries:
            route("session_summary", st.summary_search(req.user_id, query, limit), query)
    return routes


def _one_line(text: str) -> str:
    return " / ".join(part.strip() for part in str(text).splitlines() if part.strip())


async def _filter_rerank(req: schemas.SearchRequest, plan: Dict,
                         fused: List[Dict], scored: Optional[Dict] = None) -> List[Dict]:
    """§5.4 small-R rerank: LLM-score only the fused head; unscored candidates
    keep their fusion order below the kept items instead of being penalised."""
    scored = scored if scored is not None else {}
    head_limit = config.RERANK_MAX_CANDIDATES
    head = list(fused[:head_limit])
    tail = list(fused[head_limit:])
    candidates = [c for c in head if c["id"] not in scored]
    plan["_pre_rerank_excluded"] = _snapshot(tail)
    decisions = plan.setdefault("_rerank", [])
    batch_size = max(1, config.RERANK_CANDIDATES)

    async def score_batch(start):
        cands = candidates[start:start + batch_size]
        cand_text = "\n".join(
            f"{c['id']}: {_time_prefix(c)}[type: {c.get('type', 'fact')}] {_one_line(c['content'])}"
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
        for rank, candidate in enumerate(fused):
            candidate["_final"] = 1.0 / (rank + 1)
            decisions.append({"id": candidate["id"], "content": candidate["content"],
                              "keep": True, "score": candidate["_final"],
                              "reason": "global_rrf_fallback", "error": error})
        return list(fused)

    for start, cands, by_id, _ in batches:
        for c in cands:
            scored[c["id"]] = (float(by_id[c["id"]]["relevance"]),
                               bool(by_id[c["id"]]["keep"]))
    kept, unscored = [], []
    for c in head:
        relevance, keep_flag = scored[c["id"]]
        c["_final"] = relevance
        keep = keep_flag and relevance >= config.SEARCH_MIN_RELEVANCE
        decisions.append({"id": c["id"], "batch": 1, "content": c["content"],
                          "keep": keep, "score": relevance,
                          "reason": "kept" if keep else "rerank_rejected", "error": None})
        if keep:
            kept.append(c)
    kept.sort(key=lambda x: -x["_final"])
    for c in tail:
        c.pop("_final", None)
        decisions.append({"id": c["id"], "content": c["content"], "keep": True,
                          "score": c.get("_fused"), "reason": "unscored_fused", "error": None})
        unscored.append(c)
    return kept + unscored


def _evidence_lines(ranked: List[Dict]) -> str:
    return "\n".join(
        f"{i + 1}. {_time_prefix(c)}[type: {c.get('type', 'fact')}] {_one_line(c['content'])[:400]}"
        for i, c in enumerate(ranked[:config.VERIFY_EVIDENCE_ITEMS]))


async def _verify(req: schemas.SearchRequest, plan: Dict, ranked: List[Dict]) -> Dict:
    """§5.5 sufficiency verifier. Fails open: an invalid verdict never
    suppresses evidence."""
    fallback = {"sufficient": True, "confidence": 1.0, "missing": "",
                "follow_up_queries": [], "_fallback": True}
    if not ranked:
        return {"sufficient": False, "confidence": 0.0, "missing": "no evidence",
                "follow_up_queries": list(plan.get("sub_queries") or []), "_fallback": False}
    prompt = prompts.render("08_sufficiency_verify.txt", query=req.query,
                            time_scope=json.dumps(plan.get("time_scope") or {}, ensure_ascii=False),
                            evidence=_evidence_lines(ranked))
    try:
        verdict = await llm.complete_json(
            prompt, '{"sufficient":true,"confidence":0.0,"missing":"","follow_up_queries":[]}',
            schema=llm.STRUCTURED_SCHEMAS["verify"], stage="search.verify")
        if (not isinstance(verdict, dict) or type(verdict.get("sufficient")) is not bool
                or type(verdict.get("confidence")) not in (int, float)):
            raise ValueError("verifier returned an invalid verdict")
        follow = [q for q in verdict.get("follow_up_queries") or []
                  if isinstance(q, str) and q.strip()]
        return {"sufficient": verdict["sufficient"],
                "confidence": max(0.0, min(1.0, float(verdict["confidence"]))),
                "missing": str(verdict.get("missing") or ""),
                "follow_up_queries": follow[:3], "_fallback": False}
    except Exception as exc:
        log.warning("sufficiency verification failed (%s); accepting evidence", exc)
        return fallback


def _foresight_status(item: Dict, anchor: str) -> Optional[str]:
    """§5.3: plans carry a validity window; report whether it is still open."""
    if item.get("type") != "plan":
        return None
    end = (item.get("temporal") or {}).get("end")
    if not end:
        return "pending"
    try:
        return "expired" if integrity.instant(end) < integrity.instant(anchor) else "pending"
    except ValueError:
        return "pending"


def _intersects_scope(item: Dict, scope: Dict) -> bool:
    temporal = item.get("temporal") or {}
    start, end = temporal.get("start"), temporal.get("end")
    if not (start and end):
        return True
    lo, hi = scope.get("from"), scope.get("to")
    try:
        if lo and integrity.instant(end) < integrity.instant(lo):
            return False
        if hi and integrity.instant(start) > integrity.instant(hi):
            return False
    except ValueError:
        return True
    return True


def _foresight_filter(ranked: List[Dict], plan: Dict) -> List[Dict]:
    scope = plan.get("time_scope") or {}
    if not (scope.get("from") or scope.get("to")):
        return ranked
    kept = []
    for item in ranked:
        if item.get("type") == "plan" and not _intersects_scope(item, scope):
            plan.setdefault("_foresight_dropped", []).append(item["id"])
            continue
        kept.append(item)
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


def _memory_prefix(c: Dict, full: Optional[Dict], anchor: str) -> str:
    item = full or c
    prefix = _time_prefix(c)
    status = _foresight_status(item, anchor)
    if status:
        prefix += f"[plan; status: {status}] "
    if item.get("type") in ("preference", "profile", "rule"):
        prefix += f"[profile: {scenes.profile_stability(item)}] "
    if c.get("type") == "session_summary":
        prefix = "[session summary; derived context] " + prefix
    return prefix


async def run_search(st: store.Store,
                     req: schemas.SearchRequest) -> schemas.SearchResponse:
    trace = {"event": "memory.search", "search_id": uuid.uuid4().hex,
             "user_id": req.user_id, "query": req.query, "top_k": req.top_k,
             "options": req.options, "fake": config.FAKE,
             "reference_time": req.reference_time,
             "started_at": datetime.now(timezone.utc).isoformat()}
    plan = {}
    try:
        anchor = _anchor_time(st, req)
        trace["anchor_time"] = anchor
        plan = await _understand(req, anchor)
        routes_all, scored, rounds = [], {}, []
        ranked, verdict = [], {"sufficient": True, "confidence": 1.0, "_fallback": True}
        for round_index in range(1, config.SEARCH_MAX_ROUNDS + 1):
            routes_all.extend(await _recall(st, req, plan))
            fused = _rrf(routes_all)
            ranked = await _filter_rerank(req, plan, fused, scored)
            verdict = await _verify(req, plan, ranked)
            rounds.append({"round": round_index, "queries": plan.get("_queries"),
                           "fused": len(fused), "ranked": len(ranked),
                           "verdict": {k: v for k, v in verdict.items() if not k.startswith("_")}})
            if verdict["sufficient"] or not verdict.get("follow_up_queries"):
                break
            plan["_queries"] = verdict["follow_up_queries"]
        trace["fused"] = _snapshot(_rrf(routes_all))
        trace["rounds"] = rounds
        ranked = _foresight_filter(ranked, plan)
        trace["pre_rerank_excluded"] = plan.get("_pre_rerank_excluded", [])
        abstain = (not verdict["sufficient"] and not verdict.get("_fallback")
                   and verdict["confidence"] < config.ABSTAIN_CONFIDENCE)
        trace["abstained"] = abstain
        selected = [] if abstain else ranked[:req.top_k]
        trace["top_k_excluded"] = _snapshot(ranked if abstain else ranked[req.top_k:])
        full_rows = {a["id"]: a for a in st.get_amus_by_ids(
            [c["id"] for c in selected if not str(c["id"]).startswith("summary_")],
            include_history=True)}
        data = []
        seen_sources = set()
        remaining_source_chars = [config.SEARCH_SOURCE_CONTEXT_CHARS]
        for c in selected:
            is_summary = c.get("type") == "session_summary"
            raw_sources = (st.sources_for_session(req.user_id, c["session_id"]) if is_summary
                           else st.sources_for_amu(c["id"]))
            sources = _answer_sources(raw_sources, seen_sources, remaining_source_chars)
            body = _memory_prefix(c, full_rows.get(c["id"]), anchor) + c["content"]
            body = answer_context.with_evidence(body, sources)
            data.append(schemas.SearchItem(
                id=c["id"], content=body, memory_type=c.get("type", "fact"), sources=sources,
                source_count=len(raw_sources), temporal=c.get("temporal"),
                score=round(float(c.get("_final", c.get("_fused", 0.0))), 4),
                created_at=c.get("created_at")))
        # §4.4/§4.5 usage strengthens memories and heats their scenes.
        st.record_recall(
            [c["id"] for c in selected if c["id"] in full_rows],
            sorted({full_rows[c["id"]]["scene_id"] for c in selected
                    if c["id"] in full_rows and full_rows[c["id"]].get("scene_id")}))
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
        trace["scenes"] = plan.get("_scenes", [])
        trace["foresight_dropped"] = plan.get("_foresight_dropped", [])
        trace["rerank"] = plan.get("_rerank", [])
        trace["finished_at"] = datetime.now(timezone.utc).isoformat()
        search_debug.append(trace)
