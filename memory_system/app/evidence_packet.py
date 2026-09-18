"""Immutable, whole-item evidence packing shared with answer consumption.

UTF-8 bytes provide a conservative token upper bound for byte-based tokenizers;
no text or supporting quote is silently clipped after packing.
"""
import hashlib
import json
import math
from . import budget


def digest(items):
    payload = [{'id': x['id'], 'content': x['content']} for x in items]
    versions = {x.get('packet_hash_version') for x in items}
    if versions in ({2}, {3}, {4}):
        payload = [dict(p, memory_type=x.get('memory_type', 'fact'), sources=x.get('sources', []),
                        personal_evidence=x.get('personal_evidence'))
                   for p, x in zip(payload, items)]
        if versions in ({3}, {4}):
            payload = [dict(p, is_constraint=x.get('is_constraint', False),
                            equivalent_ids=x.get('equivalent_ids', [])) for p, x in zip(payload, items)]
        if versions == {4}:
            payload = [dict(p, coverage_ids=x.get('coverage_ids', []), score_kind=x.get('score_kind', 'unknown'))
                       for p, x in zip(payload, items)]
    elif versions - {None, 1}:
        raise ValueError('Mixed or unsupported evidence packet hash versions')
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def pack_ranked(items, top_k, token_budget, required_coverage=(), groups=()):
    """Preserve a listwise permutation and admit joint evidence atomically.

    Trial packing uses the same exact deduplication/cost rules as final packing;
    it never splits a model-validated group to squeeze under a budget.
    """
    by_id = {item['id']: item for item in items}
    membership = {}
    for group in groups:
        if len(group) < 2 or len(set(group)) != len(group) or any(mid not in by_id or mid in membership for mid in group):
            raise ValueError('Evidence groups must be disjoint known candidate IDs')
        for mid in group:
            membership[mid] = set(group)
    positions = {item['id']: index for index, item in enumerate(items)}
    accepted, visited, omitted = [], set(), []
    for item in items:
        budget.check()
        if item['id'] in visited:
            continue
        members = membership.get(item['id'], {item['id']})
        chunk = [candidate for candidate in items if candidate['id'] in members]
        visited.update(members)
        proposed = sorted(accepted + chunk, key=lambda c: positions[c['id']])
        trial, _, details = pack(proposed, top_k, token_budget)
        included = {x['id'] for x in trial}
        if members | {c['id'] for c in accepted} <= included:
            accepted = proposed
        else:
            reasons = {x['reason'] for x in details['omitted'] if x['id'] in members}
            reason = ('atomic_group_' if len(members) > 1 else '') + (
                'top_k' if 'top_k' in reasons else 'token_budget' if 'token_budget' in reasons else 'redundant')
            omitted.extend(dict(id=mid, reason=reason) for mid in sorted(members))
    packed, packet_hash, manifest = pack(accepted, top_k, token_budget)
    included = {x['id'] for x in packed}
    manifest['omitted'].extend(omitted)
    manifest['missing_requirements'] = sorted(set(required_coverage) - set(manifest['covered_requirements']))
    manifest['evidence_groups'] = [list(group) for group in groups if set(group) <= included]
    manifest['requested_evidence_groups'] = [list(group) for group in groups]
    manifest['ordering'] = 'listwise_or_stage_fallback; atomic_groups'
    return packed, packet_hash, manifest


def pack(items, top_k, token_budget, core_budget=None, required_coverage=None, fallback_limit=8):
    selected, omitted, used, core_used = [], [], 0, 0
    evidence_count, constraint_bytes = 0, 0
    seen_sources = {}
    pending = list(enumerate(items))
    requirements = set(required_coverage or [])
    costs = {index: len(item['content'].encode('utf-8')) + 1 for index, item in pending}
    facets = {index: set(item.get('coverage_ids', [])) & requirements for index, item in pending}
    covered, bucket_counts = set(), dict(weak=0, fallback=0)
    def priority(entry):
        index, item = entry
        cost = costs[index]
        gain = len(facets[index] - covered)
        novel = any((s['request_id'], s['message_index']) not in seen_sources
                    for s in item.get('sources', []) if s.get('content'))
        score = item.get('score')
        missing = requirements - covered - facets[index]
        # Avoid consuming the budget/last slot needed by another attainable facet.
        attainable = [remaining_costs[rid] for rid in missing if rid in remaining_costs]
        completion_possible = (not attainable or (evidence_count + 1 < top_k and
            max(attainable) <= token_budget - used - cost))
        return (bool(item.get('is_constraint')), cost <= token_budget - used,
                completion_possible, bool(gain), int((score or 0) / .1),
                gain / math.sqrt(max(1, cost)), novel, (score or 0) / math.sqrt(max(1, cost)), -index)
    while pending:
        budget.check()
        remaining_costs = {}
        for index, _ in pending:
            for rid in facets[index] - covered:
                remaining_costs[rid] = min(remaining_costs.get(rid, costs[index]), costs[index])
        entry = max(pending, key=priority) if required_coverage is not None else pending[0]
        pending.remove(entry)
        _, item = entry
        from . import answer_context
        item = dict(item)
        original_sources = item.get('sources', [])
        sources = [dict(s) for s in item.get('sources', [])]
        for source in sources:
            key = (source['request_id'], source['message_index'])
            span = source.get('content_span')
            interval = (span['start'], span['end']) if span else (0, float('inf'))
            if source.get('content') and any(a <= interval[0] and b >= interval[1]
                                           for a, b in seen_sources.get(key, [])):
                source.pop('content', None)
                source['content_omitted'] = 'duplicate'
        if sources:
            # Only remove the exact suffix we rendered. Source-marker text may
            # also appear verbatim inside the user's original memory or quotes.
            suffix = answer_context.with_evidence('', original_sources)
            body = item.get('_evidence_body', item['content'].removesuffix(suffix) if suffix else item['content'])
            item['content'] = answer_context.with_evidence(body, sources)
            item['sources'] = sources
        cost = len(item['content'].encode('utf-8')) + bool(selected)
        is_core = item.pop('_core_injected', False)
        is_constraint = bool(item.get('is_constraint'))
        bucket = item.get('_selection_bucket')
        if bucket and bucket_counts.get(bucket, 0) >= min(top_k, fallback_limit) and not is_constraint:
            omitted.append(dict(id=item['id'], reason='fallback_limit', cost=cost))
            continue
        if (item.get('memory_type') == 'episode' and not is_constraint and sources
                and any(s.get('content_omitted') == 'duplicate' for s in sources)
                and all(s.get('content_omitted') in ('duplicate', 'source_limit', 'duplicate_source', 'unit_budget') for s in sources)):
            # A quoted-context wrapper with no new source span adds no evidence.
            # A full document or wider excerpt survives unless fully covered.
            omitted.append({'id': item['id'], 'reason': 'redundant_episode', 'cost': cost})
            continue
        if is_core and core_budget is not None and core_used + cost > core_budget:
            omitted.append({'id': item['id'], 'reason': 'core_budget', 'cost': cost})
            continue
        full = not is_constraint and evidence_count >= top_k
        if full or used + cost > token_budget:
            omitted.append({'id': item['id'], 'reason': 'top_k' if full else 'token_budget', 'cost': cost})
            continue
        # Only already-visible duplicate spans may be removed after ranking.
        item = {key: value for key, value in item.items() if not key.startswith('_')}
        selected.append(dict(item, packet_hash_version=4))
        covered.update(item.get('coverage_ids', []))
        if bucket:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        evidence_count += not is_constraint
        constraint_bytes += cost if is_constraint else 0
        for s in sources:
            if not s.get('content'):
                continue
            span = s.get('content_span')
            seen_sources.setdefault((s['request_id'], s['message_index']), []).append(
                (span['start'], span['end']) if span else (0, float('inf')))
        used += cost
        if is_core:
            core_used += cost
    packet_hash = digest(selected)
    for item in selected:
        item['packet_hash'] = packet_hash
    return selected, packet_hash, {'included_ids': [x['id'] for x in selected],
                                    'omitted': omitted, 'token_upper_bound': used,
                                    'token_budget': token_budget, 'estimator': 'utf8_bytes',
                                    'budget_unit': 'utf8_bytes_conservative_token_bound',
                                    'covered_requirements': sorted(covered),
                                    'missing_requirements': sorted(set(required_coverage or []) - covered),
                                    'evidence_count': evidence_count,
                                    'constraint_count': len(selected) - evidence_count,
                                    'constraint_bytes': constraint_bytes,
                                    'core_bytes': core_used, 'core_budget': core_budget}
