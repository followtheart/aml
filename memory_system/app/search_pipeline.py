"""Search pipeline (ULM design §5 "reconstructive recall"):
query understanding (explicit reference_time anchor) -> multi-route
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

from . import answer_context, budget, config, evidence_packet, graph, integrity, llm, personal_evidence, profile, prompts, run_metadata, scenes, schemas, store
from .embeddings import embed

log = logging.getLogger("aml.search")


def _parse_iso(value) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _readable_range(temporal: Dict) -> str:
    """Spell the event range as 'D Month YYYY' so the answer model never has
    to parse ISO 'YYYY-MM-DD' (qwen-14b read 2023-05-07 as 5 July)."""
    start = _parse_iso(temporal.get("start"))
    end = _parse_iso(temporal.get("end"))
    if not start:
        return "unknown"
    precision = (temporal.get("precision") or "").lower()
    if precision == "year":
        return start.strftime("%Y")
    if precision == "month":
        return start.strftime("%B %Y")
    if end and end.date() != start.date():
        if start.year != end.year:
            return f"{start.day} {start.strftime('%B %Y')} - {end.day} {end.strftime('%B %Y')}"
        if start.month != end.month:
            return f"{start.day} {start.strftime('%B')} - {end.day} {end.strftime('%B %Y')}"
        return f"{start.day}-{end.day} {start.strftime('%B %Y')}"
    return f"{start.day} {start.strftime('%B %Y')}"


def _time_prefix(a: Dict) -> str:
    prefix = ''
    if a.get('valid_from') or a.get('valid_to'):
        prefix = f"[valid: {a.get('valid_from') or 'unknown'} ~ {a.get('valid_to') or 'open'}] "
    temporal = a.get("temporal")
    if temporal and any(temporal.get(k) for k in ('start', 'end', 'raw', 'reference_time')):
        fields = [f"event: {_readable_range(temporal)}", f"precision: {temporal.get('precision', 'unknown')}"]
        fields += [f'{label}: {temporal[key]}' for key, label in
                   [('raw', 'original'), ('reference_time', 'reference')] if temporal.get(key)]
        return prefix + '[' + '; '.join(fields) + '] '
    event = a.get("event_time")
    return prefix + (f"[event: {event}] " if event else "")


def _anchor_time(st: store.Store, req: schemas.SearchRequest) -> str:
    """§5.1: query_time comes from the request layer (reference_time). When it
    is missing the receive-time wall clock is the documented fallback — never
    the user's latest memory time, which would leak future messages into the
    anchor (ULM §5.1)."""
    return req.reference_time or datetime.now(timezone.utc).isoformat()


async def _understand(st: store.Store, req: schemas.SearchRequest,
                      anchor: Optional[str] = None) -> Dict:
    # P3: expansions bind generic questions to known user traits (the profile
    # digest is compact and never answers the question by itself).
    digest = "(none)"
    if config.QUERY_PROFILE_DIGEST:
        digest = profile.render_digest(
            st.core_profile(req.user_id, config.CORE_PROFILE_MAX_ITEMS),
            config.CORE_PROFILE_MAX_CHARS)
    prompt = prompts.render(
        "04_query_understanding.txt",
        query=req.query,
        options="\n".join(req.options or []) or "(none)",
        current_time=req.reference_time or anchor or datetime.now(timezone.utc).isoformat(),
        user_profile=digest)
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


def _overlaps_scope(item: Dict, scope: Dict) -> bool:
    start, end = scope.get("from"), scope.get("to")
    if not (start or end):
        return True
    valid_from = item.get("valid_from") or "0001-01-01T00:00:00+00:00"
    valid_to = item.get("valid_to") or "9999-12-31T23:59:59+00:00"
    try:
        return (not end or integrity.instant(valid_from) <= integrity.instant(end)) and (
            not start or integrity.instant(valid_to) >= integrity.instant(start))
    except ValueError:
        return True


def _fold_versions(items: List[Dict], plan: Dict) -> List[Dict]:
    """Restrict version chains to versions valid in an explicit time scope.

    Unbounded change-history questions retain all versions. Current-state
    searches are already restricted to open versions by the storage layer.
    """
    scope = plan.get("time_scope") or {}
    if not (scope.get("from") or scope.get("to")):
        return items
    return [item for item in items if _overlaps_scope(item, scope)]


def _select_memory_view(routes: List[List[Dict]], plan: Dict, st=None) -> List[List[Dict]]:
    document = plan.get("intent") in ("narrative", "document")
    flat = [item for route in routes for item in route]
    # P3 Structural Memory: advice/preference intents read the persona view —
    # first-person traits, rules and episodes; assistant world knowledge is
    # hidden once the user actually owns persona memories.
    if (config.PERSONA_VIEW_FILTER
            and (plan.get("intent") in ("preference", "profile") or plan.get('_personalization'))):
        plan["_memory_mode"] = "personal_evidence"
        cache = {}
        for item in flat:
            if item['id'] in cache:
                continue
            sources = (st.sources_for_amu(item['id']) if st and item.get('type') != 'session_summary'
                       else item.get('sources', []))
            cache[item['id']] = personal_evidence.personal(item, sources)
            if not cache[item['id']]:
                plan.setdefault('_view_excluded', []).append(
                    {'id': item['id'], 'round': plan.get('_round', 1), 'reason': 'not_user_evidence'})
        return [[item for item in route if cache[item['id']]] for route in routes]
    has_preferred = any((item.get("type") == "episode") == document for item in flat)
    plan["_memory_mode"] = "memory_doc" if document else "memory_only"
    if not has_preferred:
        return routes
    return [[item for item in route
             if item.get("type") == "rule"
             or (item.get("type") == "episode") == document] for route in routes]


def _profile_not_expired(item: Dict, anchor: str, include_history: bool) -> bool:
    expires = item.get("expires_at")
    if include_history or not expires:
        return True
    try:
        return integrity.instant(expires) > integrity.instant(anchor)
    except ValueError:
        return True


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


def _merge_query_results(routes: List[List[Dict]], limit: int) -> List[Dict]:
    """One retrieval channel gets one RRF vote, independent of rewrite count."""
    best = {}
    for route in routes:
        for item in route:
            current = best.get(item["id"])
            if current is None or item.get("_score", 0.0) > current.get("_score", 0.0):
                best[item["id"]] = item
    return sorted(best.values(), key=lambda item: -item.get("_score", 0.0))[:limit]


async def _recall(st: store.Store, req: schemas.SearchRequest,
                  plan: Dict) -> List[List[Dict]]:
    # A follow-up round (§5.5) supplies its own queries via plan["_queries"].
    queries = list(dict.fromkeys(
        plan.get("_queries") or
        ([req.query] + (plan.get("sub_queries") or []) +
         (plan.get("expanded_queries") or []))))[:6]
    plan["_used_queries"] = queries
    vecs = await embed(queries, stage="search.embed_queries")
    routes = []
    plan["_round_channels"] = []
    history = _historical(req, plan)
    plan["_include_history"] = history
    sensitive = bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive)
    plan["_include_sensitive"] = sensitive
    limit = max(config.RECALL_PER_ROUTE, req.top_k)

    def route(name, items, query=None):
        routes.append(items)
        plan["_round_channels"].append(name)
        plan.setdefault("_routes", []).append({
            "channel": name, "round": plan.get("_round", 1), "query": query,
            "candidates": _snapshot(items)})

    dense_routes = st.nearest_many_by_embedding(
        req.user_id, vecs, limit, include_history=history,
        include_sensitive=sensitive, include_cold=bool(plan.get("_cold")))
    route("vector", _merge_query_results(dense_routes, limit), " | ".join(queries))
    route("full_text", st.fts_search(req.user_id, req.query, limit,
                                    include_history=history,
                                    include_sensitive=sensitive, include_cold=bool(plan.get("_cold"))), req.query)
    if plan.get("intent") == "multi_hop" or plan.get("_expand"):
        triples = st.triples_for_user(req.user_id, include_sensitive=sensitive)
        # Exclude closed memories before graph traversal on current-state searches.
        if not history:
            valid = {a["id"] for a in st.get_amus_by_ids(
                list({t["amu_id"] for t in triples}), include_sensitive=sensitive)}
            triples = [t for t in triples if t["amu_id"] in valid]
        triples = graph.filter_triples(triples, req.query, plan.get("entities") or [])
        ppr_ids = graph.ppr_recall(triples, plan.get("entities") or [], top_n=limit)
        by_id = {a["id"]: a for a in st.get_amus_by_ids(
            ppr_ids, include_history=history, include_sensitive=sensitive)}
        route("graph", [by_id[i] for i in ppr_ids if i in by_id])
    time_scope = plan.get("time_scope") or {}
    if plan.get("intent") == "temporal" or time_scope.get("from") or time_scope.get("to"):
        route("temporal", st.temporal_search(
            req.user_id, time_scope, limit, include_sensitive=sensitive))
    # §5.2 scene->cell two-stage route: pick top-m MemScenes, then rank cells inside.
    top_scenes = (scenes.rank_scenes(
        [s for s in st.list_scenes(req.user_id) if s.get("view_status") != "stale"], vecs,
        _query_terms(queries), config.SCENE_TOP_M)
        if plan.get("intent") in ("multi_hop", "narrative", "document") or plan.get("_expand") else [])
    plan["_scenes"] = [{"id": s["id"], "score": round(s["_score"], 4),
                        "summary": (s.get("summary") or "")[:200]} for s in top_scenes]
    if top_scenes:
        per_scene = st.nearest_many_by_embedding(
            req.user_id, vecs[:1], config.SCENE_CELLS_PER_SCENE * len(top_scenes),
            include_history=history, scene_ids=[s["id"] for s in top_scenes],
            include_sensitive=sensitive)[0]
        route("scene", per_scene, queries[0])
    # §2.3 rules are always injected; other profile memories on matching intents.
    profile_types = ["rule"]
    if plan.get("intent") in ("preference", "rule", "profile"):
        profile_types += ["preference", "profile", "workflow"]
    route("profile_rule", st.get_by_type(req.user_id, profile_types,
                                         include_history=history,
                                         include_sensitive=sensitive))
    if plan.get("intent") == "procedural":
        route("experience", st.get_by_type(
            req.user_id, ["strategy", "workflow", "skill", "playbook"],
            include_history=history, include_sensitive=sensitive))
    if config.SUMMARY_ROUTE:
        for query in queries:
            route("session_summary", st.summary_search(req.user_id, query, limit), query)
    return routes


def _one_line(text: str) -> str:
    return " / ".join(part.strip() for part in str(text).splitlines() if part.strip())


def _options_text(req: schemas.SearchRequest) -> str:
    return "\n".join(req.options or []) or "(none)"


def _personalization_query(req: schemas.SearchRequest, plan: Dict) -> bool:
    """Advice/choice questions are answered from user traits, not from a fact
    that literally answers them, so the verifier's answer-presence test does
    not apply (PersonaMem-style tasks)."""
    return bool(req.options) or plan.get("intent") in ("preference", "profile")


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
    round_index = plan.get("_round", 1)
    batch_size = max(1, config.RERANK_CANDIDATES)

    async def score_batch(start):
        cands = candidates[start:start + batch_size]
        cand_text = "\n".join(
            f"{c['id']}: {_time_prefix(c)}[type: {c.get('type', 'fact')}] {_one_line(c['content'])}"
            for c in cands)
        ts = plan.get("time_scope") or {}
        prompt = prompts.render("05_rerank_filter.txt", query=req.query,
                                options=_options_text(req),
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
                if entry.get('id') in by_id:
                    raise ValueError('Reranker returned a duplicate id')
                by_id[entry.get("id")] = entry
            if set(by_id) != {candidate["id"] for candidate in cands}:
                raise ValueError("Reranker did not score every candidate exactly once")
            return start, cands, by_id, None
        except Exception as exc:
            plan.setdefault('_rerank_errors', []).append({
                'round': round_index, 'batch': start // batch_size + 1,
                'error': type(exc).__name__,
                'detail': str(exc) if isinstance(exc, ValueError) else 'provider call failed'})
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
            decisions.append({"id": candidate["id"], "round": round_index,
                              "content": candidate["content"],
                              "keep": True, "score": candidate["_final"],
                              "reason": "global_rrf_fallback", "error": error})
        return list(fused)

    newly = {c["id"] for c in candidates}
    for start, cands, by_id, _ in batches:
        for c in cands:
            scored[c["id"]] = (float(by_id[c["id"]]["relevance"]),
                               bool(by_id[c["id"]]["keep"]))
    # P2: personalization queries sort by relevance but never filter
    # (HippoRAG 2 reports ~26% recall loss from hard recognition filtering).
    personalization = _personalization_query(req, plan)
    min_relevance = 0.0 if personalization else config.SEARCH_MIN_RELEVANCE
    kept, unscored = [], []
    for c in head:
        relevance, keep_flag = scored[c["id"]]
        c["_final"] = relevance
        keep = personalization or (keep_flag and relevance >= min_relevance)
        # P1: governance memories (forget requests, user instructions) are
        # rerank-whitelisted: ranked, never dropped.
        if c.get("type") == "rule":
            keep = True
        # Cached scores from earlier rounds are not logged again.
        if c["id"] in newly:
            decisions.append({"id": c["id"], "round": round_index, "batch": 1,
                              "content": c["content"], "keep": keep, "score": relevance,
                              "reason": "kept" if keep else "rerank_rejected", "error": None})
        if keep:
            kept.append(c)
    kept.sort(key=lambda x: (-x["_final"], x.get('type') in ('episode', 'session_summary')))
    for c in tail:
        c.pop("_final", None)
        decisions.append({"id": c["id"], "round": round_index, "content": c["content"],
                          "keep": True, "score": c.get("_fused"),
                          "reason": "unscored_fused", "error": None})
        unscored.append(c)
    return kept + unscored


def _evidence_lines(ranked: List[Dict]) -> str:
    return "\n".join(c['content'] for c in ranked)


async def _verify(req: schemas.SearchRequest, plan: Dict, ranked: List[Dict]) -> Dict:
    """§5.5 sufficiency verifier. Fails open: an invalid verdict never
    suppresses evidence."""
    fallback = {"sufficient": False, "confidence": 0.0, "missing": "verification failed",
                "follow_up_queries": [], "_fallback": True, "verification_status": "error"}
    if not ranked:
        return {"sufficient": False, "confidence": 0.0, "missing": "no evidence",
                "follow_up_queries": list(plan.get("sub_queries") or []), "_fallback": False}
    prompt = prompts.render("08_sufficiency_verify.txt", query=req.query,
                            options=_options_text(req),
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
                "follow_up_queries": follow[:3], "_fallback": False, "verification_status": "verified"}
    except Exception as exc:
        log.warning("sufficiency verification failed (%s); returning partial evidence", exc)
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
    if profile.is_forget_rule(item):
        # Negative constraint: the forgotten trait must not be used or
        # recommended, even if other memories still mention it.
        prefix += "[constraint: forgotten by user request; do NOT use or recommend this trait] "
    if c.get("type") == "session_summary":
        prefix = "[session summary; derived context] " + prefix
    return prefix


def _pack_evidence(st, req, plan, ranked, anchor):
    candidates = list(_foresight_filter(ranked, plan))
    injected = set()
    if config.CORE_PROFILE_INJECT:
        core = [a for a in st.core_profile(req.user_id, config.CORE_PROFILE_MAX_ITEMS)
                if _profile_not_expired(a, anchor, False)]
        ids = {a['id'] for a in candidates}
        terms = personal_evidence.terms(req.query + ' ' + _options_text(req))
        core = [a for a in core if a['id'] not in ids and
                (a.get('type') == 'rule' or personal_evidence.terms(a['content']) & terms)]
        core.sort(key=lambda a: (a.get('type') != 'rule',
                                -len(personal_evidence.terms(a['content']) & terms)))
        injected = {a['id'] for a in core}
        candidates = core + candidates
    document = plan.get('intent') in ('document', 'narrative')
    persona = plan.get('intent') in ('preference', 'profile') or plan.get('_personalization', False)
    items = []
    for candidate in candidates:
        if candidate.get('view_status') == 'stale':
            continue
        is_summary = candidate.get('type') == 'session_summary'
        sources = (st.sources_for_session(req.user_id, candidate['session_id']) if is_summary
                   else st.sources_for_amu(candidate['id']))
        sources = [dict(source) for source in sources]
        body = _memory_prefix(candidate, candidate, anchor) + candidate['content']
        support_queue = list(st.dependencies_for(candidate['id']))
        support_evidence = []
        visited = {candidate['id']}
        support_missing = False
        while support_queue:
            dep = support_queue.pop()
            if dep['source_id'] in visited:
                continue
            visited.add(dep['source_id'])
            rows = st.get_amus_by_ids([dep['source_id']], include_history=True,
                                      include_sensitive=bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive))
            if not rows or rows[0].get('version') != dep['source_version']:
                support_missing = True
                break
            support = rows[0]
            support_evidence.extend(support.get('evidence') or [])
            body += '\n[support: ' + support['id'] + '] ' + _memory_prefix(support, support, anchor) + support['content']
            sources.extend(st.sources_for_amu(support['id']))
            support_queue.extend(st.dependencies_for(support['id']))
        if support_missing:
            continue
        sources = list({(s['request_id'], s['message_index']): s for s in sources}.values())
        is_personal = personal_evidence.personal(candidate, sources)
        if not document:
            # Fetch validated support quotes omitted from lightweight recall rows.
            full = st.get_amus_by_ids([candidate['id']], include_history=True,
                                     include_sensitive=bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive))
            grounded = dict(full[0] if full else candidate)
            grounded['evidence'] = list(grounded.get('evidence') or []) + support_evidence
            sources = personal_evidence.compact_sources(
                grounded, sources, req.query + ' ' + _options_text(req),
                config.SEARCH_SOURCE_EXCERPT_CHARS, config.SEARCH_SOURCE_MESSAGES_PER_ITEM,
                persona=persona)
        body = answer_context.with_evidence(body, sources)
        items.append(dict(id=candidate['id'], content=body,
                          _core_injected=candidate['id'] in injected,
                          personal_evidence=is_personal,
                          memory_type=candidate.get('type', 'fact'), sources=sources,
                          source_count=len(sources), temporal=candidate.get('temporal'),
                          created_at=candidate.get('created_at'),
                          score=round(float(candidate.get('_final', candidate.get('_fused', 0))), 4)))
    return evidence_packet.pack(items, req.top_k, req.evidence_token_budget,
                                min(config.CORE_PROFILE_TOKEN_BUDGET, req.evidence_token_budget // 5))


async def _run_search(st: store.Store,
                      req: schemas.SearchRequest) -> schemas.SearchResponse:
    epoch = st.user_state(req.user_id)["epoch"]
    trace = {"event": "memory.search", "search_id": uuid.uuid4().hex,
             "user_id": req.user_id, "query": req.query, "top_k": req.top_k,
             "options": req.options, "fake": config.FAKE,
             "reference_time": req.reference_time,
             "started_at": datetime.now(timezone.utc).isoformat(),
             'evidence_token_budget': req.evidence_token_budget, 'versions': run_metadata.versions()}
    plan = {}
    try:
        anchor = _anchor_time(st, req)
        trace["anchor_time"] = anchor
        plan = await _understand(st, req, anchor)
        plan['_personalization'] = plan.get('intent') in ('preference', 'profile')
        routes_all, scored, rounds = [], {}, []
        channels = {}
        ranked, verdict = [], {"sufficient": True, "confidence": 1.0, "_fallback": True}
        tried = set()
        for round_index in range(1, config.SEARCH_MAX_ROUNDS + 1):
            plan["_round"] = round_index
            round_routes = _select_memory_view(await _recall(st, req, plan), plan, st)
            history = bool(plan.get("_include_history"))
            round_routes = [[item for item in route
                             if _profile_not_expired(item, anchor, history)]
                            for route in round_routes]
            names = plan.get('_round_channels') or [str(i) for i in range(len(round_routes))]
            for name, items in zip(names, round_routes):
                channels[name] = _merge_query_results([channels.get(name, []), items], max(config.RECALL_PER_ROUTE, req.top_k))
            routes_all = list(channels.values())
            tried.update(q.strip().casefold() for q in plan.get("_used_queries", []))
            fused = _fold_versions(_rrf(routes_all), plan)
            ranked = await _filter_rerank(req, plan, fused, {})
            packed, packet_hash, manifest = _pack_evidence(st, req, plan, ranked, anchor)
            verdict = await _verify(req, plan, packed)
            rounds.append({"round": round_index, "queries": plan.get("_queries"),
                           "fused": len(fused), "ranked": len(ranked),
                           "verdict": {k: v for k, v in verdict.items() if not k.startswith("_")}})
            # Another round only pays off when it brings queries not yet tried.
            fresh = [q for q in verdict.get("follow_up_queries") or []
                     if q.strip().casefold() not in tried]
            if verdict['sufficient']:
                break
            if not plan.get('_cold'):
                plan['_cold'] = True
                plan['_expand'] = True
                plan['_queries'] = fresh or [req.query]
            elif fresh:
                plan['_queries'] = fresh
            else:
                break
        trace["fused"] = _snapshot(_fold_versions(_rrf(routes_all), plan))
        trace["rounds"] = rounds
        data = [schemas.SearchItem(**item) for item in packed]
        conflicts = [c['id'] for c in ranked if c.get('resolution_status') == 'disputed'
                     and c['id'] in manifest['included_ids']]
        status = ('not_found' if not data else 'conflicting' if conflicts else
                  'complete' if verdict['sufficient'] else 'partial')
        manifest['conflicting_ids'] = conflicts
        trace['top_k_excluded'] = [x for x in manifest['omitted'] if x['reason'] == 'top_k']
        trace['status'] = status
        trace['abstained'] = status == 'not_found'
        trace['abstain_exempt'] = False
        trace['returned'] = [d.model_dump() for d in data]
        trace['ranked'] = _snapshot(ranked)
        trace['coverage_manifest'] = manifest
        trace['packet_hash'] = packet_hash
        st.assert_epoch(req.user_id, epoch)
        return schemas.SearchResponse(
            search_id=trace['search_id'],
            data=data, evidence_status=status, packet_hash=packet_hash,
            verification_status=verdict.get('verification_status', 'verified'),
            read_revision=getattr(st, 'read_revision', st.user_state(req.user_id)['revision']),
            read_epoch=epoch, coverage_manifest=manifest,
            missing_evidence=[verdict['missing']] if verdict.get('missing') else [])
    except BaseException as exc:
        # Includes CancelledError from the hard deadline in run_search: a
        # timed-out search is still logged as an error trace, then re-raised.
        trace["status"] = "error"
        trace["error_type"] = type(exc).__name__
        raise
    finally:
        trace["plan"] = {k: v for k, v in plan.items() if not k.startswith("_")}
        trace["include_history"] = plan.get("_include_history")
        trace["routes"] = plan.get("_routes", [])
        trace["scenes"] = plan.get("_scenes", [])
        trace["foresight_dropped"] = list(dict.fromkeys(plan.get("_foresight_dropped", [])))
        trace["rerank"] = plan.get("_rerank", [])
        trace['view_excluded'] = plan.get('_view_excluded', [])
        trace['rerank_errors'] = plan.get('_rerank_errors', [])
        trace["finished_at"] = datetime.now(timezone.utc).isoformat()
        if st.user_state(req.user_id)["epoch"] == epoch:
            search_debug.append(trace)


async def run_search(st: store.Store, req: schemas.SearchRequest) -> schemas.SearchResponse:
    with budget.scope(seconds=config.SEARCH_DEADLINE_SECONDS,
                      calls=config.SEARCH_MAX_CALLS,
                      tokens=config.SEARCH_MAX_TOKENS) as limits:
        async with asyncio.timeout(config.SEARCH_DEADLINE_SECONDS):
            with st.snapshot(req.user_id, min_revision=req.min_revision, as_of=req.as_of) as snapshot:
                response = await _run_search(snapshot, req)
                st.assert_epoch(req.user_id, snapshot.read_epoch)
                # §4.4/§4.5: memories in the final packet count as one actual
                # use per query; candidate hits and extra rounds never reheat.
                st.record_recall([item.id for item in response.data
                                  if item.id.startswith('amu_')], [])
                response.read_revision = snapshot.read_revision
                response.read_epoch = snapshot.read_epoch
                response.coverage_manifest['provider_calls'] = limits.calls
                response.coverage_manifest['provider_tokens'] = limits.tokens
                return response
