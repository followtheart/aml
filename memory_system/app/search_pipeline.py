"""Bounded search with coverage, one grounded follow-up and immutable evidence.

Only the answer model decides semantic sufficiency. Search reports observable
coverage, provenance, degradation and deterministic storage/time/epoch checks.
"""
import logging
import asyncio
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from . import search_debug
from typing import Dict, List, Optional

from . import answer_context, budget, cascade_rerank, config, graph_fusion, evidence_packet, evidence_units, graph, integrity, llm, local_work, personal_evidence, profile, prompts, retrieval_queries, run_metadata, scenes, schemas, search_coverage, store
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
    # Choice hypotheses come from the options, not an unverified profile digest.
    if config.QUERY_PROFILE_DIGEST and not req.options:
        scope = personal_evidence.terms(req.query)
        profiles = [p for p in await local_work.run(st.core_profile, req.user_id, config.CORE_PROFILE_MAX_ITEMS * 4)
                    if _profile_not_expired(p, anchor or _anchor_time(st, req), False)
                    and personal_evidence.terms(p['content']) & scope]
        profiles.sort(key=lambda p: -len(personal_evidence.terms(p['content']) & scope))
        digest = profile.render_digest(
            profiles[:config.CORE_PROFILE_MAX_ITEMS],
            config.CORE_PROFILE_MAX_CHARS)
    prompt = prompts.render(
        "04_query_understanding.txt",
        query=req.query,
        options="\n".join(req.options or []) or "(none)",
        current_time=req.reference_time or anchor or datetime.now(timezone.utc).isoformat(),
        user_profile=digest)
    try:
        return await asyncio.wait_for(llm.complete_json(
            prompt,
            json.dumps(llm.STRUCTURED_SCHEMAS['query']),
            schema=llm.STRUCTURED_SCHEMAS["query"],
            stage="search.understand", attempts=1, timeout=8.0), 8.0)
    except Exception as e:
        log.warning("query understanding failed (%s); passthrough", e)
        return {"intent": "fact", "time_scope": None, '_planning_error': type(e).__name__,
                "entities": [], "sub_queries": [req.query],
                "expanded_queries": [req.query]}


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
    return sorted(set().union(*(personal_evidence.terms(q) for q in queries)))


def _merge_query_results(routes: List[List[Dict]], limit: int) -> List[Dict]:
    """Balance query variants by rank; BM25 scales differ between queries.

    Every variant's first hit competes before any variant's second hit. The
    combined channel retains a single candidate entry per memory.
    """
    best, positions, query_ranks = {}, {}, {}
    for query_index, route in enumerate(routes):
        seen = set()
        for item in route:
            iid = item['id']
            if iid in seen:
                continue
            seen.add(iid)
            position = (len(seen), query_index)
            query_ranks.setdefault(iid, {})[str(query_index)] = len(seen)
            if iid not in positions or position < positions[iid]:
                best[iid], positions[iid] = item, position
    return [dict(best[iid], _query_ranks=query_ranks[iid])
            for iid in sorted(best, key=lambda iid: positions[iid])[:limit]]


def _route_weight(name, intent):
    # Graph/scene are expansions of direct hits, not independent corroboration.
    if name == 'graph':
        return 0.85 if intent == 'multi_hop' else 0.5
    if name == 'scene':
        return 0.85 if intent in ('multi_hop', 'narrative', 'document') else 0.5
    return 1.0


def _route(plan, routes, name, items, query=None):
    routes.append(items)
    family = ('lexical' if name in ('full_text', 'source_text', 'session_summary') else
              'expansion' if name in ('graph', 'scene') else name)
    plan.setdefault('_routes', []).append(dict(channel=name, family=family,
        round=plan.get('_round', 1), query=query, weight=_route_weight(name, plan.get('intent')),
        candidates=_snapshot(items)))


def _lexical_recall(st, req, plan, specs):
    history, sensitive = plan['_include_history'], plan['_include_sensitive']
    fts, sources = [], []
    for spec in specs:
        budget.check()
        fts.append(st.fts_search(req.user_id, spec['text'], config.RECALL_FTS_LIMIT,
            include_history=history, include_sensitive=sensitive, include_cold=True))
        sources.append(st.source_search(req.user_id, spec['text'], config.RECALL_SOURCE_LIMIT,
            include_history=history, include_sensitive=sensitive, include_cold=True))
    merged = [_merge_query_results(fts, config.RECALL_FTS_LIMIT),
              _merge_query_results(sources, config.RECALL_SOURCE_LIMIT)]
    # Count independent original observations per query, never AMUs from one source.
    coverage = []
    for index, spec in enumerate(specs):
        independent = set()
        for candidate in fts[index] + sources[index]:
            budget.check()
            _, raw = _candidate_sources(st, req, plan, candidate)
            text = candidate['content'] + '\n' + '\n'.join(x.get('content', '') for x in raw)
            if search_coverage.matches(text, spec['text']):
                independent.update(search_coverage.source_keys(
                    [source for source in raw if search_coverage.matches(source.get('content', ''), spec['text'])]))
        coverage.append(dict(query=spec['text'], coverage_ids=spec.get('coverage_ids', []),
                             independent_sources=len(independent)))
    plan['_direct_coverage'] = coverage
    return merged


async def _recall(st: store.Store, req: schemas.SearchRequest, plan: Dict) -> List[List[Dict]]:
    specs = plan.pop('_supplemental_specs', None) or retrieval_queries.build(req.query, req.options or [], plan)
    supplemental = plan.get('_round', 1) > 1
    if not supplemental:
        plan['_query_specs'] = specs
        plan['_used_queries'] = [x['text'] for x in specs]
    queries = [entry['text'] for entry in specs]
    history = _historical(req, plan)
    sensitive = bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive)
    plan['_include_history'], plan['_include_sensitive'] = history, sensitive
    lexical = await local_work.run(_lexical_recall, st, req, plan, specs)
    vecs = None
    try:
        limits = budget.current.get()
        seconds = min(8.0, max(.01, (limits.deadline - time.monotonic()) * .3)) if limits else 8.0
        vecs = await asyncio.wait_for(embed(queries, stage='search.embed_queries'), seconds)
    except Exception as exc:
        budget.check()
        plan.setdefault('_recall_errors', []).append(dict(channel='vector', error=type(exc).__name__))
    routes = []
    dense = []
    if vecs is not None:
        dense = await local_work.run(st.nearest_many_by_embedding,
            req.user_id, vecs, config.RECALL_VECTOR_LIMIT, include_history=history,
            include_sensitive=sensitive, include_cold=True)
        plan.setdefault('_vector_diagnostics', []).append(getattr(st, 'vector_diagnostics', {}))
    _route(plan, routes, 'vector', _merge_query_results(dense, config.RECALL_VECTOR_LIMIT), ' | '.join(queries))
    _route(plan, routes, 'full_text', lexical[0], ' | '.join(queries))
    _route(plan, routes, 'source_text', lexical[1], ' | '.join(queries))
    if supplemental:
        # Grounded follow-up results retain their target only when the candidate
        # actually matches the follow-up text (a full dense list is not coverage).
        for route in routes:
            for item in route:
                for spec in specs:
                    if search_coverage.matches(item['content'], spec['text']):
                        item['_coverage_ids'] = sorted(set(item.get('_coverage_ids', [])) |
                                                       set(spec.get('coverage_ids', [])))
                        item['_bridge_ids'] = sorted(set(item.get('_bridge_ids', [])) | set(spec.get('seed_ids', [])))
                        item.setdefault('_coverage_proofs', []).append(dict(spec))
        return routes
    return await local_work.run(_expand_recall, st, req, plan, routes, specs, vecs)


def _expand_recall(st, req, plan, routes, specs, vecs):
    history, sensitive = plan['_include_history'], plan['_include_sensitive']
    queries = [x['text'] for x in specs]
    contextual = plan.get('intent') in ('multi_hop', 'narrative', 'document')
    sparse = any(c['independent_sources'] < config.RECALL_EXPANSION_MIN_DIRECT
                 for c in plan['_direct_coverage'])
    expansion_limit = config.RECALL_EXPANSION_LIMIT if contextual or sparse else 0
    plan['_expansion'] = dict(enabled=bool(expansion_limit),
        reason='intent' if contextual else 'sparse_direct' if sparse else 'direct_coverage',
        direct_count=len({c['id'] for r in routes[1:3] for c in r}),
        coverage=plan['_direct_coverage'], limit=expansion_limit)
    graph_count = 0
    if expansion_limit // 2:
        budget.check()
        triples = st.triples_for_user(req.user_id, include_sensitive=sensitive)
        if not history:
            valid = {a['id'] for a in st.get_amus_by_ids(
                list({t['amu_id'] for t in triples}), include_sensitive=sensitive)}
            triples = [t for t in triples if t['amu_id'] in valid]
        seed_ids = list(dict.fromkeys(c['id'] for r in routes[:3] for c in r[:3]))
        plan['_graph_seed_ids'] = seed_ids
        triples = graph.filter_triples(triples, req.query, plan.get('entities') or [], seed_amu_ids=seed_ids)
        ppr_ids = graph.ppr_recall(triples, plan.get('entities') or [], top_n=expansion_limit // 2,
                                   seed_amu_ids=seed_ids, query=req.query)
        by_id = {a['id']: a for a in st.get_amus_by_ids(ppr_ids, include_history=history, include_sensitive=sensitive)}
        for triple in triples:
            if triple['amu_id'] in by_id:
                by_id[triple['amu_id']].setdefault('_bridge_ids', triple.get('_path_ids', []))
        _route(plan, routes, 'graph', [by_id[i] for i in ppr_ids if i in by_id])
        graph_count = len(routes[-1])
    scope = plan.get('time_scope') or {}
    if plan.get('intent') == 'temporal' or scope.get('from') or scope.get('to'):
        _route(plan, routes, 'temporal', st.temporal_search(req.user_id, scope,
            config.RECALL_PER_ROUTE, include_sensitive=sensitive))
    top_scenes = (scenes.rank_scenes([s for s in st.list_scenes(req.user_id, include_cold=True)
                  if s.get('view_status') != 'stale'], vecs, _query_terms(queries), config.SCENE_TOP_M)
                  if vecs is not None and expansion_limit > graph_count else [])
    plan['_scenes'] = [dict(id=s['id'], score=round(s['_score'], 4), summary=s.get('summary', '')[:200]) for s in top_scenes]
    if top_scenes:
        # One lane per (query, scene), round-robin merged before the shared cap.
        # No scene or original-question embedding can consume every slot first.
        per_query = [[] for _ in queries]
        for scene in top_scenes:
            budget.check()
            found = st.nearest_many_by_embedding(req.user_id, vecs,
                config.SCENE_CELLS_PER_SCENE, include_history=history,
                scene_ids=[scene['id']], include_cold=True, include_sensitive=sensitive)
            for i, hits in enumerate(found):
                per_query[i].append(hits)
        lanes = [_merge_query_results(group, expansion_limit - graph_count) for group in per_query]
        _route(plan, routes, 'scene', _merge_query_results(lanes, expansion_limit - graph_count), ' | '.join(queries))
    types = ['rule']
    if req.options or plan.get('intent') in ('preference', 'rule', 'profile'):
        types += ['preference', 'profile', 'workflow']
    profiles = st.get_by_type(req.user_id, types, include_history=history, include_sensitive=sensitive)
    terms = personal_evidence.terms(' '.join(queries))
    for c in profiles:
        c['_score'] = len(personal_evidence.terms(c['content']) & terms)
    profiles = [c for c in profiles if not (c.get('type') == 'rule' and c.get('epistemic_status') == 'inferred'
                and not any(s.get('role') == 'user' for s in st.sources_for_amu(c['id'])))]
    profiles = [c for c in profiles if c['_score'] > 0 or c.get('type') == 'rule' or plan.get('intent') == 'profile']
    ordered = sorted(profiles, key=lambda c: (-c['_score'], c['id']))
    _route(plan, routes, 'profile_rule', [c for c in ordered if c.get('type') != 'rule'][:config.RECALL_PROFILE_LIMIT]
           + [c for c in ordered if c.get('type') == 'rule'])
    if plan.get('intent') == 'procedural':
        _route(plan, routes, 'experience', st.get_by_type(req.user_id, ['strategy', 'workflow', 'skill', 'playbook'],
               include_history=history, include_sensitive=sensitive))
    if config.SUMMARY_ROUTE:
        _route(plan, routes, 'session_summary', _merge_query_results([
            st.summary_search(req.user_id, q, config.RECALL_PER_ROUTE) for q in queries], config.RECALL_PER_ROUTE))
    return routes


def _one_line(text: str) -> str:
    return " / ".join(part.strip() for part in str(text).splitlines() if part.strip())


async def _followup(st, req, plan, candidates):
    if plan.get('intent') != 'multi_hop' or not candidates or not config.SEARCH_FOLLOWUP_QUERIES:
        return []
    limits = budget.current.get()
    # Reserve provider capacity and time for the scoring/packing stages.
    if limits and (limits.max_calls - limits.calls < 5 or limits.deadline - time.monotonic() < 12):
        plan['_followup'] = dict(status='budget_skipped', queries=[])
        return []
    seeds = candidates[:6]
    requirement_ids = {x['id'] for x in plan.get('_coverage_requirements', [])}
    prompt = prompts.render('04b_query_followup.txt', query=req.query,
        requirements=json.dumps(plan.get('_coverage_requirements', []), ensure_ascii=False),
        queries=json.dumps(plan.get('_used_queries', []), ensure_ascii=False),
        evidence='\n'.join(c['id'] + ': ' + c['_rank_text'][:1500] for c in seeds),
        limit=config.SEARCH_FOLLOWUP_QUERIES)
    schema = dict(type='object', properties={'queries': dict(type='array', maxItems=config.SEARCH_FOLLOWUP_QUERIES,
        items=dict(type='object', properties={'text': dict(type='string', maxLength=240),
            'seed_ids': dict(type='array', items=dict(type='string')),
            'coverage_ids': dict(type='array', items=dict(type='string'))},
            required=['text', 'seed_ids', 'coverage_ids']))}, required=['queries'])
    try:
        result = await asyncio.wait_for(llm.complete_json(prompt, schema=schema,
            stage='search.followup', attempts=1, max_tokens=512,
            timeout=config.SEARCH_FOLLOWUP_SECONDS), config.SEARCH_FOLLOWUP_SECONDS)
    except Exception as exc:
        budget.check()
        plan['_followup'] = dict(status='error', error=type(exc).__name__, queries=[])
        return []
    by_id = {c['id']: c for c in seeds}
    specs, seen = [], {q.casefold() for q in plan.get('_used_queries', [])}
    original = personal_evidence.terms(req.query)
    proposals = result.get('queries') if isinstance(result, dict) else None
    if not isinstance(proposals, list):
        plan['_followup'] = dict(status='error', error='invalid_queries', queries=[])
        return []
    for proposed in proposals:
        if not isinstance(proposed, dict) or len(specs) >= config.SEARCH_FOLLOWUP_QUERIES:
            continue
        query = proposed.get('text')
        ids, targets = proposed.get('seed_ids'), proposed.get('coverage_ids')
        if not isinstance(query, str) or not query.strip() or len(query) > 240 or query.casefold() in seen:
            continue
        if not isinstance(ids, list) or not ids or not all(isinstance(mid, str) and mid in by_id for mid in ids):
            continue
        if not isinstance(targets, list) or not targets or not all(isinstance(rid, str) and rid in requirement_ids for rid in targets):
            continue
        words = personal_evidence.terms(query)
        observed = personal_evidence.terms(' '.join(by_id[mid]['_rank_text'][:1500] for mid in ids))
        if not words - original or not (words - original) <= observed:
            continue
        seen.add(query.casefold())
        specs.append(dict(text=query, origin='grounded_followup', seed_ids=ids,
                          coverage_ids=targets, option_index=None))
    plan['_followup'] = dict(status='queried' if specs else 'no_grounded_bridge', queries=specs)
    if not specs:
        return []
    plan['_round'] = 2
    plan['_supplemental_specs'] = specs
    plan['_query_specs'].extend(specs)
    plan['_used_queries'].extend(s['text'] for s in specs)
    return await _recall(st, req, plan)


def _options_text(req: schemas.SearchRequest) -> str:
    return "\n".join(req.options or []) or "(none)"


def _source_query(req, plan):
    # Full advice prose rewards verbose generic explanations. Preview and pack
    # must use the same short premise queries, including fallback queries.
    queries = plan.get('_used_queries') or [s['text'] for s in
        retrieval_queries.build(req.query, req.options or [], plan)]
    return ' '.join(queries)


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
                 ('valid_from', 'valid_to', 'state', 'sensitivity', 'epistemic_status')))
        if key not in groups:
            groups[key] = item
            out.append(item)
            continue
        representative = groups[key]
        representative.setdefault('_equivalent_ids', []).extend(
            [item['id']] + item.get('_equivalent_ids', []))
        for field in ('_coverage_ids', '_bridge_ids'):
            representative[field] = sorted(set(representative.get(field, [])) | set(item.get(field, [])))
        proofs = representative.get('_coverage_proofs', []) + item.get('_coverage_proofs', [])
        representative['_coverage_proofs'] = list({json.dumps(p, sort_keys=True): p for p in proofs}.values())
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
    hydrated = []
    for c in fused:
        budget.check()
        full, sources = _candidate_sources(st, req, plan, c)
        item = dict(full[0]) if full else dict(c)
        item.pop('embedding', None)
        item.update({k: v for k, v in c.items() if k.startswith('_')})
        item['_user_rule'] = item.get('type') == 'rule' and (
            any(s.get('role') == 'user' for s in sources) or
            (not sources and item.get('epistemic_status') != 'inferred'))
        hydrated.append(item)
    candidates = _coalesce_preferences(hydrated, plan)
    anchor = plan.get('_anchor') or _anchor_time(st, req)
    queries = plan.get('_used_queries') or [req.query]
    cap = min(config.SEARCH_ITEM_MAX_BYTES, config.CE_MAX_DOCUMENT_BYTES, req.evidence_token_budget,
              max(256, config.RERANK_MAX_PROMPT_BYTES - len(_rerank_prompt(req, plan, []).encode('utf-8')) - 256))
    prepared = []
    for c in candidates:
        budget.check()
        full, sources = _candidate_sources(st, req, plan, c)
        sources = [dict(x) for x in sources]
        primary_source_keys = {(s['request_id'], s['message_index']) for s in sources}
        support_evidence = [e for m in full for e in m.get('evidence') or []]
        body = ('[conversation excerpts; verify speaker attribution]'
                if c.get('type') == 'episode' and sources else c['content'])
        body = _memory_prefix(c, full[0] if full else c, anchor) + body
        primary_body = body
        queue = [d for mid in [c['id']] + c.get('_equivalent_ids', [])
                 for d in st.dependencies_for(mid)] if hasattr(st, 'dependencies_for') else []
        if c.get('_bridge_ids'):
            bridges = st.get_amus_by_ids(c['_bridge_ids'], include_history=bool(plan.get('_include_history')),
                include_sensitive=bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive))
            queue.extend(dict(source_id=m['id'], source_version=m.get('version')) for m in bridges)
        visited, failure = {c['id']}, None
        while queue:
            budget.check()
            dep = queue.pop()
            if dep['source_id'] in visited:
                continue
            visited.add(dep['source_id'])
            if len(visited) > 64:
                failure = 'dependency_limit'
                break
            rows = st.get_amus_by_ids([dep['source_id']], include_history=True,
                include_sensitive=bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive))
            if not rows or rows[0].get('version') != dep['source_version']:
                failure = 'missing_or_changed_dependency'
                break
            support = rows[0]
            body += '\n[support: ' + support['id'] + '] ' + _memory_prefix(support, support, anchor) + support['content']
            support_evidence.extend(support.get('evidence') or [])
            sources.extend(st.sources_for_amu(support['id']))
            queue.extend(st.dependencies_for(support['id']))
        if failure:
            plan.setdefault('_unit_omitted', []).append(dict(id=c['id'], reason=failure))
            continue
        # Very long unsourced narrative text also uses whole verbatim passages.
        if len(body.encode('utf-8')) > cap // 2 and not queue and len(visited) == 1:
            if c.get('type') in ('episode', 'session_summary') and sources:
                body = _memory_prefix(c, c, anchor) + '[source excerpts; full source retained by reference]'
            elif not _protected_rule(c):
                chunks = evidence_units.passages(dict(content=c['content']), queries, [], max(128, cap // 2 - 160))
                if chunks:
                    body = _memory_prefix(c, c, anchor) + '[verbatim memory excerpts] ' + '\n'.join(x['content'] for x in chunks)
            primary_body = body
        unit = evidence_units.build(c, sources, support_evidence, queries, body, cap)
        if unit is None:
            plan.setdefault('_unit_omitted', []).append(dict(id=c['id'], reason='unit_budget', budget_bytes=cap))
            continue
        c['_rank_text'] = unit['content']
        visible = [x for x in unit['sources'] if x.get('content')]
        informative = [x for x in visible if not personal_evidence.editorial_request(x['content'])]
        c['_selection_evidence'] = dict(user_source=any(x.get('role') == 'user' for x in informative)
            and personal_evidence.personal(c, informative), source_ids=sorted(search_coverage.source_keys(informative)))
        # Recompute direct coverage from the prepared unit, never from discarded
        # document text. A follow-up label also needs its target text and every
        # grounding bridge still present in this same unit.
        c['_coverage_ids'] = []
        primary_text = primary_body + '\n' + '\n'.join(s['content'] for s in visible
            if (s['request_id'], s['message_index']) in primary_source_keys)
        for proof in c.get('_coverage_proofs', []):
            if (proof.get('seed_ids') and set(proof['seed_ids']) <= visited
                    and search_coverage.matches(primary_text, proof['text'])):
                c['_coverage_ids'].extend(proof.get('coverage_ids', []))
        search_coverage.annotate(c, plan.get('_coverage_requirements', []))
        c['_packet_item'] = dict(id=c['id'], **unit, _rank_aligned=True, is_constraint=profile.is_forget_rule(c),
            equivalent_ids=c.get('_equivalent_ids', []), memory_type=c.get('type', 'fact'),
            personal_evidence=personal_evidence.personal(c, visible), source_count=len(unit['sources']),
            coverage_ids=c.get('_coverage_ids', []), temporal=c.get('temporal'), created_at=c.get('created_at'))
        prepared.append(c)
    return _coalesce_episodes(prepared, plan)


def _coalesce_episodes(candidates, plan):
    """Only merge identical episode previews with the exact same provenance."""
    groups, out = {}, []
    for c in candidates:
        _, sources = plan.get('_source_cache', {}).get(tuple(
            [c['id']] + c.get('_equivalent_ids', [])), ([], []))
        if c.get('type') != 'episode' or not sources:
            out.append(c)
            continue
        key = (c.get('_rank_text'), tuple(sorted((s['request_id'], s['message_index']) for s in sources)),
               *(json.dumps(c.get(k), sort_keys=True) for k in (
                   'user_id', 'sensitivity', 'epistemic_status', 'resolution_status',
                   'valid_from', 'valid_to', 'temporal', 'view_status', 'state',
                   'knowledge_status', 'polarity', 'trust_scope', 'event_time', 'session_id')))
        if key not in groups:
            groups[key] = c
            out.append(c)
            continue
        retained = groups[key]
        retained.setdefault('_equivalent_ids', []).extend([c['id']] + c.get('_equivalent_ids', []))
        for field in ('_coverage_ids', '_bridge_ids'):
            retained[field] = sorted(set(retained.get(field, [])) | set(c.get(field, [])))
        if '_packet_item' in retained:
            retained['_packet_item']['coverage_ids'] = retained.get('_coverage_ids', [])
        plan.setdefault('_deduplicated', []).append(dict(
            id=c['id'], retained_id=retained['id'], reason='same_episode_sources_and_preview'))
    return out


def _rerank_head(fused, plan, limit):
    """Historical v5/v6 replay only: reserve route/query champions.

    Most of the head remains RRF-ranked. Only already-retrieved candidates are
    used, with no second retrieval or additional model scoring batch.
    """
    representative = {mid: c['id'] for c in fused
                      for mid in [c['id']] + c.get('_equivalent_ids', [])}
    anchors = []
    for requirement in plan.get('_coverage_requirements', []):
        champion = next((c for c in fused if requirement['id'] in c.get('_coverage_ids', [])), None)
        if champion and champion['id'] not in anchors:
            anchors.append(champion['id'])
    for route in plan.get('_routes', []):
        name, rows = route['channel'], route['candidates']
        champions = []
        if name in ('vector', 'source_text', 'full_text'):
            per_query = {}
            for row in rows:
                if row['id'] not in representative:
                    continue
                for query, rank in row.get('_query_ranks', {'0': row.get('rank', 1)}).items():
                    if query not in per_query or rank < per_query[query][0]:
                        per_query[query] = (rank, row['id'])
            champions = [mid for _, mid in per_query.values()]
        elif name == 'graph' and plan.get('intent') == 'multi_hop':
            champions = [row['id'] for row in rows[:2]]
        elif name == 'scene' and plan.get('intent') in ('narrative', 'document'):
            champions = [row['id'] for row in rows[:1]]
        for mid in champions:
            mid = representative.get(mid)
            if mid and mid not in anchors:
                anchors.append(mid)
    anchors = anchors[:max(1, min(limit // 2, max(limit // 4, len(plan.get('_coverage_requirements', [])))))]
    head_ids = set(anchors)
    for c in fused:
        if len(head_ids) >= limit:
            break
        head_ids.add(c['id'])
    plan['_rerank_reserved_ids'] = anchors
    return ([c for c in fused if c['id'] in head_ids],
            [c for c in fused if c['id'] not in head_ids])


def _rerank_prompt(req, plan, rows):
    return cascade_rerank.listwise_prompt(req, plan, rows)


async def _filter_rerank(req, plan, fused, scored=None):
    return await cascade_rerank.rank(req, plan, fused)


def _protected_rule(candidate):
    return profile.is_forget_rule(candidate) or bool(candidate.get('_user_rule'))


def _selection_order(ranked):
    """Coverage and independent observations break near-score ties, not just exact ties."""
    pending, ordered, seen, covered = list(ranked), [], set(), set()
    while pending:
        budget.check()
        def priority(c):
            evidence = c.get('_selection_evidence', {})
            score = c.get('_final')
            return (int((score + 1e-9) / .1) if score is not None else -1,
                    len(set(c.get('_coverage_ids', [])) - covered),
                    bool(set(evidence.get('source_ids', [])) - seen),
                    bool(evidence.get('user_source')), score if score is not None else -1)
        candidate = max(pending, key=priority)
        pending.remove(candidate)
        ordered.append(candidate)
        if not _protected_rule(candidate):
            seen.update(candidate.get('_selection_evidence', {}).get('source_ids', []))
            covered.update(candidate.get('_coverage_ids', []))
    return ordered


def _select_evidence(req, plan, ranked, defer_limits=False):
    """Preserve v7 ordinal decisions; retain pointwise rules for historical replay."""
    if '_cascade' in plan:
        selected = [c for c in ranked if c.get('_cascade_selected')]
        selected.sort(key=lambda c: c.get('_listwise_rank', 0))
        for c in selected:
            c['_selection_bucket'] = None
        plan['_selection'] = dict(mode='graph_cascade', input_count=len(ranked),
            selected_count=len(selected), selected_ids=[c['id'] for c in selected],
            tie_policy='listwise_order_then_atomic_group_budget',
            omitted=plan['_cascade'].get('omitted', []))
        return selected
    decisions = {d['id']: d for d in plan.get('_rerank', [])}
    fallback = not decisions or any(d['reason'] == 'global_rrf_fallback' for d in decisions.values())
    threshold = config.EVIDENCE_MIN_RELEVANCE
    strong = any(c.get('_final', 0) >= threshold for c in ranked if c.get('_final') is not None
                 and decisions.get(c['id'], {}).get('reason') == 'kept')
    mode = 'rrf_fallback' if fallback else 'relevance' if strong else 'weak_fallback'
    partial = plan.get('_rerank_status') == 'partial'
    limit = min(req.top_k, config.EVIDENCE_FALLBACK_ITEMS)
    selected, omitted, counts = [], [], dict(fallback=0, weak=0)
    ordered = ranked if fallback else _selection_order(ranked)
    for c in ordered:
        reason, bucket = None, None
        score = c.get('_final')
        decision = decisions.get(c['id'], {}).get('reason')
        if not _protected_rule(c):
            if fallback or decision == 'batch_failed':
                bucket = 'fallback'
            elif decision != 'kept':
                reason = 'unscored_tail'
            elif score is None or score <= 0:
                reason = 'irrelevant'
            elif score < threshold:
                if strong:
                    reason = 'low_relevance'
                else:
                    bucket = 'weak'
            if bucket:
                if not defer_limits and counts[bucket] >= limit:
                    reason = 'fallback_limit'
                counts[bucket] += 1
        if reason:
            omitted.append(dict(id=c['id'], reason=reason, score=score))
        else:
            c['_selection_bucket'] = bucket
            selected.append(c)
    plan['_selection'] = dict(mode=('partial_' + mode if partial else mode),
        input_count=len(ranked), selected_count=len(selected),
        tie_policy='score_band_then_query_coverage_and_source_diversity',
        selected_ids=[c['id'] for c in selected], min_relevance=threshold,
        fallback_limit=limit, omitted=omitted)
    return selected


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
    if item.get('resolution_status') == 'disputed':
        prefix += '[evidence: disputed; unresolved conflicting claims] '
    if profile.is_forget_rule(item):
        # Negative constraint: the forgotten trait must not be used or
        # recommended, even if other memories still mention it.
        prefix += "[constraint: forgotten by user request; do NOT use or recommend this trait] "
    if c.get("type") == "session_summary":
        prefix = "[session summary; derived context] " + prefix
    return prefix


def _pack_evidence(st, req, plan, ranked, anchor):
    plan['_anchor'] = anchor
    # Legacy callers may supply raw rows; production prepares units before rerank.
    if any('_packet_item' not in c for c in ranked):
        ranked = _prepare_candidates(st, req, plan, ranked)
    ranked = [c for c in _foresight_filter(ranked, plan) if c.get('view_status') != 'stale']
    eligible = {c['id'] for c in ranked}
    groups = plan.get('_evidence_groups', [])
    broken = {mid for group in groups if not set(group) <= eligible for mid in group}
    if broken:
        ranked = [c for c in ranked if c['id'] not in broken]
        plan['_evidence_groups'] = [g for g in groups if not set(g) & broken]
        plan.setdefault('_cascade', {}).setdefault('omitted', []).extend(
            dict(id=mid, stage='packet', reason='atomic_group_scope') for mid in sorted(broken))
    items = []
    for candidate in ranked:
        budget.check()
        if candidate.get('view_status') == 'stale':
            continue
        item = dict(candidate['_packet_item'])
        item['equivalent_ids'] = candidate.get('_equivalent_ids', [])
        item['score'] = candidate.get('_final')
        item['score_kind'] = candidate.get('_score_kind', 'unknown')
        item['_fused'] = candidate.get('_fused', 0)
        item['_selection_bucket'] = candidate.get('_selection_bucket')
        items.append(item)
    requirements = [r['id'] for r in plan.get('_coverage_requirements', [])]
    if '_cascade' in plan:
        packed, packet_hash, manifest = evidence_packet.pack_ranked(items, req.top_k, req.evidence_token_budget,
            required_coverage=requirements, groups=plan.get('_evidence_groups', []))
    else:
        packed, packet_hash, manifest = evidence_packet.pack(items, req.top_k, req.evidence_token_budget,
            required_coverage=requirements, fallback_limit=config.EVIDENCE_FALLBACK_ITEMS)
    manifest['unit_omitted'] = plan.get('_unit_omitted', [])
    return packed, packet_hash, manifest


def _fuse_candidates(st, req, plan, routes):
    # Version/privacy filters run before any graph association is admitted.
    routes = [_fold_versions(route, plan) for route in routes]
    # Bound the metadata reads themselves, not just the hydrated graph.
    admission = dict(plan)
    admitted = {c['id'] for c in graph_fusion.fuse(routes, admission)}
    routes = [[c for c in route if c['id'] in admitted] for route in routes]
    pool = {c['id']: c for route in routes for c in route}
    ids = list(pool)
    triples = st.graph_rows_for_candidates(req.user_id, ids,
        include_history=bool(plan.get('_include_history')),
        include_sensitive=bool(config.SENSITIVE_RECALL_ENABLED and req.include_sensitive))
    sources, dependencies = {}, {}
    for mid, c in pool.items():
        budget.check()
        _, raw = _candidate_sources(st, req, plan, c)
        sources[mid] = [{key: s.get(key) for key in ('request_id', 'message_index', 'source_event_id')} for s in raw]
        dependencies[mid] = st.dependencies_for(mid)
    result = graph_fusion.fuse(routes, plan, triples=triples, sources=sources, dependencies=dependencies)
    plan['_graph_fusion']['truncated_ids'] = admission['_graph_fusion']['truncated_ids']
    return result


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
        plan['_anchor'] = anchor
        # Cold evidence remains eligible; contextual/sparse expansion is gated
        # locally inside recall; multi-hop may add one grounded follow-up.
        plan.update(_round=1, _cold=True)
        routes = await _recall(st, req, plan)
        history = bool(plan.get("_include_history"))
        routes = [[item for item in route if _profile_not_expired(item, anchor, history)]
                  for route in routes]
        plan['query'] = req.query
        fused = await local_work.run(_fuse_candidates, st, req, plan, routes)
        candidates = await local_work.run(_prepare_candidates, st, req, plan, fused)
        candidates = _foresight_filter(candidates, plan)
        extra = await _followup(st, req, plan, candidates)
        if extra:
            routes += [[item for item in route if _profile_not_expired(item, anchor, history)] for route in extra]
            fused = await local_work.run(_fuse_candidates, st, req, plan, routes)
            candidates = await local_work.run(_prepare_candidates, st, req, plan, fused)
            candidates = _foresight_filter(candidates, plan)
        ranked = await _filter_rerank(req, plan, candidates)
        selected = _select_evidence(req, plan, ranked, defer_limits=True)
        packed, packet_hash, manifest = await local_work.run(_pack_evidence, st, req, plan, selected, anchor)
        manifest['selection_mode'] = plan['_selection']['mode']
        manifest['selection_omitted'] = plan['_selection']['omitted']
        manifest['rerank_status'] = plan.get('_rerank_status', 'not_run')
        manifest['rerank_errors'] = plan.get('_rerank_errors', [])
        manifest['recall_errors'] = plan.get('_recall_errors', [])
        manifest['planning_error'] = plan.get('_planning_error')
        manifest['calibration'] = plan.get('_calibration', {})
        manifest['fusion'] = plan.get('_graph_fusion', {})
        manifest['cascade'] = plan.get('_cascade', {})
        manifest.setdefault('evidence_groups', [])
        manifest['followup'] = plan.get('_followup', {'status': 'not_needed'})
        manifest['vector_search'] = plan.get('_vector_diagnostics', [])
        manifest['coverage'] = search_coverage.report(plan.get('_coverage_requirements', []), candidates, packed)
        manifest['search_degraded'] = bool(manifest['planning_error'] or manifest['recall_errors'] or manifest['rerank_errors'] or plan.get('_unit_omitted')
            or any(q.get('candidate_budget_exhausted') for v in manifest['vector_search'] for q in v.get('queries', []))
            or manifest['followup'].get('status') in ('error', 'budget_skipped'))
        trace["pipeline"] = run_metadata.SEARCH_POLICY
        trace["fused"] = _snapshot(fused)
        trace["rounds"] = [{"round": r, "queries": [q['text'] for q in plan.get('_query_specs', [])
                            if (q['origin'] == 'grounded_followup') == (r == 2)],
                            "fused": len(fused), "ranked": len(ranked),
                            "verification_status": "not_run"} for r in range(1, plan.get('_round', 1) + 1)]
        trace["verification_status"] = "not_run"
        data = [schemas.SearchItem(**item) for item in packed]
        conflicts = [c['id'] for c in ranked if c.get('resolution_status') == 'disputed'
                     and c['id'] in manifest['included_ids']]
        status = ('incomplete' if not data and manifest['search_degraded'] else
                  'not_found' if not data else 'conflicting' if conflicts else
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
            missing_evidence=list(dict.fromkeys(r['text'] for r in manifest['coverage']['requirements']
                                                if r['status'] != 'packed')))
    except BaseException as exc:
        # Includes CancelledError from the hard deadline in run_search: a
        # timed-out search is still logged as an error trace, then re-raised.
        trace["status"] = "error"
        trace["error_type"] = type(exc).__name__
        raise
    finally:
        trace["plan"] = {k: v for k, v in plan.items() if not k.startswith("_")}
        trace['query_specs'] = plan.get('_query_specs', [])
        trace['expansion'] = plan.get('_expansion', {})
        trace["include_history"] = plan.get("_include_history")
        trace["routes"] = plan.get("_routes", [])
        trace["scenes"] = plan.get("_scenes", [])
        trace["foresight_dropped"] = list(dict.fromkeys(plan.get("_foresight_dropped", [])))
        trace["rerank"] = plan.get("_rerank", [])
        trace['view_excluded'] = plan.get('_view_excluded', [])
        trace['rerank_errors'] = plan.get('_rerank_errors', [])
        trace['rerank_status'] = plan.get('_rerank_status', 'not_run')
        trace['deduplicated'] = plan.get('_deduplicated', [])
        trace['rerank_pool_size'] = plan.get('_rerank_pool_size', 0)
        trace['rerank_batches'] = sorted(plan.get('_rerank_batches', []), key=lambda b: b['batch'])
        trace['rerank_reserved_ids'] = plan.get('_rerank_reserved_ids', [])
        trace['graph_seed_ids'] = plan.get('_graph_seed_ids', [])
        trace['selection'] = plan.get('_selection', {})
        trace['fusion'] = plan.get('_graph_fusion', {})
        trace['fusion_edges'] = plan.get('_fusion_edges', [])
        trace['fusion_triples'] = plan.get('_fusion_triples', [])
        trace['cascade'] = plan.get('_cascade', {})
        trace['evidence_groups'] = plan.get('_evidence_groups', [])
        trace["finished_at"] = datetime.now(timezone.utc).isoformat()
        if getattr(st, '_origin', st).user_state(req.user_id)["epoch"] == epoch:
            search_debug.append(trace)


async def run_search(st: store.Store, req: schemas.SearchRequest) -> schemas.SearchResponse:
    with budget.scope(seconds=config.SEARCH_DEADLINE_SECONDS,
                      calls=config.SEARCH_MAX_CALLS, tokens=config.SEARCH_MAX_TOKENS) as limits:
        cm = st.snapshot(req.user_id, min_revision=req.min_revision, as_of=req.as_of)
        opened = []
        def enter():
            snapshot = cm.__enter__()
            opened.append(snapshot)
            return snapshot
        try:
            async with asyncio.timeout(config.SEARCH_DEADLINE_SECONDS):
                snapshot = await local_work.run(enter)
                response = await _run_search(snapshot, req)
                await local_work.run(st.assert_epoch, req.user_id, snapshot.read_epoch)
                await local_work.run(st.record_recall, [item.id for item in response.data
                    if item.id.startswith('amu_')], [])
                response.read_revision, response.read_epoch = snapshot.read_revision, snapshot.read_epoch
                response.coverage_manifest['provider_calls'] = limits.calls
                response.coverage_manifest['provider_tokens'] = limits.tokens
        finally:
            if opened:
                await local_work.close_snapshot(cm)
        # No await after the final live deletion guard: cleanup is part of Search.
        st.assert_epoch(req.user_id, response.read_epoch)
        return response
