"""Single-pass search: plan -> multi-route recall -> rank -> immutable packet.

Only the answering model decides whether evidence supports an answer. Search
keeps provenance and deterministic storage/time/epoch checks, without a second
semantic verifier or an iterative retrieval loop.
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
    # Options are hypotheses to retrieve evidence for, never evidence themselves.
    # Allocate one slot per option before generic rewrites can consume the budget.
    queries = list(dict.fromkeys(
        [req.query] + list(req.options or []) +
        (plan.get("sub_queries") or []) + (plan.get("expanded_queries") or [])))
    queries = queries[:6]
    plan["_used_queries"] = queries
    vecs = await embed(queries, stage="search.embed_queries")
    routes = []
    history = _historical(req, plan)
    plan["_include_history"] = history
    sensitive = bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive)
    plan["_include_sensitive"] = sensitive
    limit = max(config.RECALL_PER_ROUTE, req.top_k)

    def route(name, items, query=None):
        routes.append(items)
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
    route('source_text', _merge_query_results([
        st.source_search(req.user_id, q, limit, include_history=history,
                         include_sensitive=sensitive, include_cold=bool(plan.get('_cold')))
        for q in queries], limit), ' | '.join(queries))
    if plan.get("intent") == "multi_hop" or plan.get("_expand"):
        triples = st.triples_for_user(req.user_id, include_sensitive=sensitive)
        # Exclude closed memories before graph traversal on current-state searches.
        if not history:
            valid = {a["id"] for a in st.get_amus_by_ids(
                list({t["amu_id"] for t in triples}), include_sensitive=sensitive)}
            triples = [t for t in triples if t["amu_id"] in valid]
        # Seed with actual retrieved nodes when a planner's abstract concepts
        # do not match graph entity labels. No extra model or retrieval round.
        seed_ids = list(dict.fromkeys(c['id'] for r in (routes[0], routes[2]) for c in r[:3]))
        plan['_graph_seed_ids'] = seed_ids
        triples = graph.filter_triples(triples, req.query, plan.get('entities') or [],
                                       seed_amu_ids=seed_ids)
        ppr_ids = graph.ppr_recall(triples, plan.get("entities") or [], top_n=limit,
                                   seed_amu_ids=seed_ids)
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
    # Rules always join recall. Choice questions also recall profiles, even
    # when the planner classifies a generic recommendation as a fact query.
    profile_types = ["rule"]
    if req.options or plan.get("intent") in ("preference", "rule", "profile"):
        profile_types += ["preference", "profile", "workflow"]
    profiles = st.get_by_type(req.user_id, profile_types, include_history=history,
                              include_sensitive=sensitive)
    terms = personal_evidence.terms(' '.join(queries))
    for c in profiles:
        c['_score'] = len(personal_evidence.terms(c['content']) & terms)
    # The route's order must be relevance, not SQLite insertion order. Legacy
    # assistant advice remains available via normal recall, not a rule boost.
    profiles = [c for c in profiles if not (
        c.get('type') == 'rule' and c.get('epistemic_status') == 'inferred'
        and not any(s.get('role') == 'user' for s in st.sources_for_amu(c['id'])))]
    route('profile_rule', sorted(profiles, key=lambda c: (-c['_score'], c['id'])))
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


def _coalesce_preferences(items, plan):
    """Merge identical current claims, retaining all provenance for packing."""
    groups, out = {}, []
    for original in items:
        item = dict(original)
        if '_equivalent_ids' in item:
            item['_equivalent_ids'] = list(item['_equivalent_ids'])
        if item.get('type') != 'preference':
            out.append(item)
            continue
        temporal = item.get('temporal') or {}
        if not any(temporal.get(k) for k in ('raw', 'start', 'end')):
            temporal = None
        key = (item.get('user_id'), item['content'].strip().rstrip('.').casefold(),
               json.dumps(temporal, sort_keys=True),
               item.get('resolution_status') or 'accepted',
               *(json.dumps(item.get(k), sort_keys=True) for k in
                 ('valid_from', 'valid_to', 'state', 'sensitivity')))
        if key not in groups:
            groups[key] = item
            out.append(item)
            continue
        representative = groups[key]
        representative.setdefault('_equivalent_ids', []).extend(
            [item['id']] + item.get('_equivalent_ids', []))
        plan.setdefault('_deduplicated', []).append(
            {'id': item['id'], 'retained_id': representative['id'], 'reason': 'same_preference'})
    return out


def _candidate_sources(st, req, plan, candidate):
    ids = [candidate['id']] + candidate.get('_equivalent_ids', [])
    key = tuple(ids)
    cache = plan.setdefault('_source_cache', {})
    if key not in cache:
        full = st.get_amus_by_ids(ids, include_history=True,
                                  include_sensitive=bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive))
        full.sort(key=lambda m: ids.index(m['id']))
        sources = (st.sources_for_session(req.user_id, candidate['session_id'])
                   if candidate.get('type') == 'session_summary' else
                   [s for mid in ids for s in st.sources_for_amu(mid)])
        sources = list({(s['request_id'], s['message_index']): dict(s) for s in sources}.values())
        cache[key] = (full, sources)
    return cache[key]


def _prepare_candidates(st, req, plan, fused):
    candidates = _coalesce_preferences(fused, plan)
    query = req.query + ' ' + _options_text(req)
    for c in candidates:
        full, sources = _candidate_sources(st, req, plan, c)
        grounded = dict(full[0] if full else c)
        grounded['evidence'] = [e for m in full for e in m.get('evidence') or []]
        snippets = personal_evidence.compact_sources(grounded, sources, query, 320, 2)
        # Rank original excerpts as well as facts. Meta summaries such as
        # "asked to refine a note" must not hide the note from the ranker.
        body = ('Conversation excerpts; verify speaker attribution.'
                if c.get('type') == 'episode' and sources else c['content'][:500])
        c['_rank_text'] = answer_context.with_evidence(body, snippets)
    return candidates


async def _filter_rerank(req: schemas.SearchRequest, plan: Dict,
                         fused: List[Dict], scored: Optional[Dict] = None) -> List[Dict]:
    """One positional scoring call over a pool larger than the output capacity."""
    head_limit = max(config.RERANK_MAX_CANDIDATES, req.top_k + 20)
    head, tail = list(fused[:head_limit]), list(fused[head_limit:])
    decisions = plan.setdefault('_rerank', [])
    plan['_rerank_pool_size'] = len(head)
    if not head:
        return []
    cand_text = '\n'.join(
        f"{i}: {_time_prefix(c)}[type: {c.get('type', 'fact')}] {_one_line(c.get('_rank_text', c['content']))}"
        for i, c in enumerate(head))
    prompt = prompts.render('05_rerank_filter.txt', query=req.query,
                            options=_options_text(req),
                            time_scope=json.dumps(plan.get('time_scope') or {}, ensure_ascii=False),
                            candidates=cand_text, candidate_count=len(head))
    values = None
    try:
        result = await llm.complete_json(
            prompt, '{"scores":[0.0]}', schema=llm.STRUCTURED_SCHEMAS['rerank'],
            stage='search.rerank.batch_1')
        values = result.get('scores')
        if not isinstance(values, list) or len(values) != len(head):
            raise ValueError('Reranker score count does not match candidate count')
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Reranker returned an invalid score')
    except Exception as exc:
        plan.setdefault('_rerank_errors', []).append({
            'round': 1, 'batch': 1, 'error': type(exc).__name__,
            'expected_count': len(head),
            'received_count': len(values) if isinstance(values, list) else None,
            'detail': str(exc) if isinstance(exc, ValueError) else 'provider call failed'})
        log.warning('rerank failed (%s); keep fused order', exc)
        # Never combine partially returned relevance values with RRF scores.
        for rank, c in enumerate(fused):
            c['_final'] = 1.0 / (rank + 1)
            decisions.append(dict(id=c['id'], round=1, content=c['content'], keep=True,
                                  score=c['_final'], reason='global_rrf_fallback', error=type(exc).__name__))
        return list(fused)
    for c, value in zip(head, values):
        c['_final'] = float(value)
        decisions.append(dict(id=c['id'], round=1, content=c['content'], keep=True,
                              score=c['_final'], reason='kept', error=None))
    document = plan.get('intent') in ('narrative', 'document')
    head.sort(key=lambda c: (-c['_final'],
                            (c.get('type') in ('episode', 'session_summary')) != document))
    for c in tail:
        c.pop('_final', None)
        decisions.append(dict(id=c['id'], round=1, content=c['content'], keep=True,
                              score=c.get('_fused'), reason='unscored_fused', error=None))
    return head + tail


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


def _memory_prefix(c: Dict, full: Optional[Dict], anchor: str) -> str:
    item = full or c
    prefix = _time_prefix(c)
    status = _foresight_status(item, anchor)
    if status:
        prefix += f"[plan; status: {status}] "
    if item.get("type") in ("preference", "profile", "rule"):
        prefix += f"[profile: {scenes.profile_stability(item)}] "
    if item.get("epistemic_status") == "inferred":
        prefix += "[evidence: inferred; check source attribution] "
    if profile.is_forget_rule(item):
        # Negative constraint: the forgotten trait must not be used or
        # recommended, even if other memories still mention it.
        prefix += "[constraint: forgotten by user request; do NOT use or recommend this trait] "
    if c.get("type") == "session_summary":
        prefix = "[session summary; derived context] " + prefix
    return prefix


def _pack_evidence(st, req, plan, ranked, anchor):
    # Rules are already recalled through profile_rule. Avoid a second injection
    # path that bypasses ranking and can add assistant advice as user rules.
    candidates = sorted(_coalesce_preferences(_foresight_filter(ranked, plan), plan),
                        key=lambda c: not profile.is_forget_rule(c))
    document = plan.get('intent') in ('document', 'narrative')
    items = []
    for candidate in candidates:
        if candidate.get('view_status') == 'stale':
            continue
        full, raw_sources = _candidate_sources(st, req, plan, candidate)
        sources = [dict(source) for source in raw_sources]
        grounded = dict(full[0] if full else candidate)
        if any(m.get('epistemic_status') == 'inferred' for m in full):
            grounded['epistemic_status'] = 'inferred'
        body = ('[conversation excerpts; quoted context, verify speaker attribution]'
                if candidate.get('type') == 'episode' and sources
                else candidate['content'])
        body = _memory_prefix(candidate, grounded, anchor) + body
        support_queue = [d for mid in [candidate['id']] + candidate.get('_equivalent_ids', [])
                         for d in st.dependencies_for(mid)]
        support_evidence = [e for m in full for e in m.get('evidence') or []]
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
            grounded['evidence'] = support_evidence
            sources = personal_evidence.compact_sources(
                grounded, sources, req.query + ' ' + _options_text(req),
                config.SEARCH_SOURCE_EXCERPT_CHARS, config.SEARCH_SOURCE_MESSAGES_PER_ITEM)
        body = answer_context.with_evidence(body, sources)
        items.append(dict(id=candidate['id'], content=body,
                          is_constraint=profile.is_forget_rule(grounded),
                          equivalent_ids=candidate.get('_equivalent_ids', []),
                          personal_evidence=is_personal,
                          memory_type=candidate.get('type', 'fact'), sources=sources,
                          source_count=len(sources), temporal=candidate.get('temporal'),
                          created_at=candidate.get('created_at'),
                          score=round(float(candidate.get('_final', candidate.get('_fused', 0))), 4)))
    return evidence_packet.pack(items, req.top_k, req.evidence_token_budget)


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
        # Include cold memories and expansion routes in the same pass, instead of
        # waiting for an LLM to declare the first packet insufficient.
        plan.update(_round=1, _cold=True, _expand=True)
        routes = await _recall(st, req, plan)
        history = bool(plan.get("_include_history"))
        routes = [[item for item in route if _profile_not_expired(item, anchor, history)]
                  for route in routes]
        fused = _fold_versions(_rrf(routes), plan)
        candidates = _prepare_candidates(st, req, plan, fused)
        ranked = await _filter_rerank(req, plan, candidates)
        packed, packet_hash, manifest = _pack_evidence(st, req, plan, ranked, anchor)
        trace["pipeline"] = "single_pass_v2"
        trace["fused"] = _snapshot(fused)
        trace["rounds"] = [{"round": 1, "queries": plan.get("_used_queries", []),
                            "fused": len(fused), "ranked": len(ranked),
                            "verification_status": "not_run"}]
        trace["verification_status"] = "not_run"
        data = [schemas.SearchItem(**item) for item in packed]
        conflicts = [c['id'] for c in ranked if c.get('resolution_status') == 'disputed'
                     and c['id'] in manifest['included_ids']]
        status = ('not_found' if not data else 'conflicting' if conflicts else
                  'retrieved')
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
            verification_status='not_run',
            read_revision=getattr(st, 'read_revision', st.user_state(req.user_id)['revision']),
            read_epoch=epoch, coverage_manifest=manifest,
            missing_evidence=[])
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
        trace['deduplicated'] = plan.get('_deduplicated', [])
        trace['rerank_pool_size'] = plan.get('_rerank_pool_size', 0)
        trace['graph_seed_ids'] = plan.get('_graph_seed_ids', [])
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
