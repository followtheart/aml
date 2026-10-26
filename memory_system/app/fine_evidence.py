"""Bounded, source-backed additions to a CE core; relevance is not entailment.

No gold labels, provider calls, inferred personal facts or blended CE/graph scores.
The caller must admit additions against the actual whole-request budget.
"""
from collections import Counter
import math

from . import evidence_selection as selection


def key(row):
    return (row['_ce_score'] if row.get('_ce_score') is not None else -1e30,
            row.get('_fused', 0))


def relevance(row, requirements, visible=None):
    """Return retrieval facets witnessed by visible text, never support labels."""
    pieces = selection.fragments(row) if visible is None else visible
    terms_by_source = [(role, selection._topic_concepts(text)) for (_, role), text in pieces]
    matches = {}
    for requirement in requirements:
        wanted = selection._topic_concepts(requirement['text'])
        if not wanted:
            continue
        for role, terms in terms_by_source:
            hits = wanted & terms
            family = any(t.startswith('topic_family:') for t in hits)
            ratio = len(hits) / len(wanted)
            if family or (len(hits) >= min(2, len(wanted)) and ratio >= .3):
                value = (role == 'user', family, ratio)
                matches[requirement['id']] = max(matches.get(requirement['id'], (False, False, 0)), value)
    return matches


def sparse_matches(rows, requirements, visible):
    """Rare, requirement-specific source terms can earn inspection, not support.

    Count independent source identities rather than repeated memory projections.
    A single shared generic term must not unlock every spare seat.
    """
    corpus = {}
    for pieces in visible.values():
        for source, text in pieces:
            corpus.setdefault(source, set()).update(selection._topic_concepts(text))
    frequency = Counter(term for terms in corpus.values() for term in terms)
    wanted = {r['id']: selection._topic_concepts(r['text']) for r in requirements}
    requirement_frequency = Counter(term for terms in wanted.values() for term in terms)
    rare_limit = max(1, len(corpus) // 10)
    out = {}
    for row in rows:
        matches = {}
        for rid, terms in wanted.items():
            for (_, role), text in visible[row['id']]:
                hits = terms & selection._topic_concepts(text)
                distinctive = {t for t in hits if len(t) >= 4 and not t.startswith('topic_family:')
                               and requirement_frequency[t] == 1 and frequency[t] <= rare_limit}
                if distinctive and len(hits) / len(terms) >= .3:
                    value = (role == 'user', False, len(hits) / len(terms))
                    matches[rid] = max(matches.get(rid, (False, False, 0)), value)
        out[row['id']] = matches
    return out


def dependencies(row):
    # _bridge_ids can be unordered query seeds. Only ordered recorded paths
    # justify joint inspection, and even these do not establish entailment.
    return set(row.get('_relation_path_ids', [])) - {row['id']}


def closure(row, by_id):
    pending, found, missing = [row['id']], set(), set()
    while pending:
        mid = pending.pop()
        if mid in found:
            continue
        if mid not in by_id:
            missing.add(mid)
            continue
        found.add(mid)
        pending.extend(dependencies(by_id[mid]) - found)
    return found, missing


def propose(rows, count, requirements, *, rescue_ids=(), fused_ids=(), extra_limit=8,
            context_queries=(), core_pool_ids=None, soft_promotion_limit=None,
            semantic_reserve=False, new_coarse_insurance=False, new_coarse_missing_only=False):
    """Select evidence, optionally insuring only candidates missing from baseline.

    The missing-only policy computes the original fine selection on these exact
    inputs before excluding its members from the extra semantic nomination.
    """
    options = dict(rescue_ids=rescue_ids, fused_ids=fused_ids, extra_limit=extra_limit,
                   context_queries=context_queries, core_pool_ids=core_pool_ids,
                   soft_promotion_limit=soft_promotion_limit, semantic_reserve=semantic_reserve)
    baseline_ids = ()
    if new_coarse_missing_only:
        baseline, _ = _propose(rows, count, requirements, **options)
        baseline_ids = tuple(c['id'] for c in baseline)
    chosen, trace = _propose(rows, count, requirements, **options,
        new_coarse_insurance=new_coarse_insurance or new_coarse_missing_only,
        excluded_insurance_ids=baseline_ids)
    if new_coarse_missing_only:
        trace['adaptive']['new_coarse_insurance'].update(missing_only=True, baseline_fine_ids=list(baseline_ids))
    return chosen, trace


def _propose(rows, count, requirements, *, rescue_ids=(), fused_ids=(), extra_limit=8,
             context_queries=(), core_pool_ids=None, soft_promotion_limit=None,
             semantic_reserve=False, new_coarse_insurance=False, excluded_insurance_ids=()):
    """Keep the existing bounded selection core; add novel relevant groups.

    Extra capacity is earned by visible evidence gain, not requirement count.
    This is a proposal: no downstream capacity or token budget is bypassed.
    """
    ordered = sorted(rows, key=key, reverse=True)
    by_id = {c['id']: c for c in ordered}
    visible = {c['id']: selection.fragments(c) for c in ordered}
    # Supplemental query topics are useful retrieval facets, not assertions or
    # obligations. Only the optional extra seats see them; keep the original
    # requirements for core selection and its supported-coverage reservations.
    context = []
    seen = {r['text'].casefold() for r in requirements}
    for spec in context_queries:
        text = spec.get('text')
        if (spec.get('origin') != 'option_topic' or not isinstance(text, str)
                or not text.strip() or len(text) > 240 or text.casefold() in seen):
            continue
        seen.add(text.casefold())
        context.append(dict(id=f'context:{len(context)}', text=text, origin='option_topic'))
        if len(context) == 4:
            break
    facets = list(requirements) + context
    matches = {c['id']: relevance(c, facets, visible[c['id']]) for c in ordered}
    sparse = sparse_matches(ordered, facets, visible)
    sparse_ids = {mid for mid in matches if not matches[mid] and sparse[mid]}
    for mid in matches:
        for rid, value in sparse[mid].items():
            matches[mid][rid] = max(matches[mid].get(rid, (False, False, 0)), value)
    trace = {}
    core_pool = ordered
    if core_pool_ids is not None:
        allowed = set(core_pool_ids)
        if not allowed <= set(by_id):
            raise ValueError('Fine core pool must be drawn from the current coarse candidates')
        core_pool = [c for c in ordered if c['id'] in allowed]
    core = selection.select(core_pool, count, requirements, key, reserve_ids=rescue_ids,
                            fused_ids=fused_ids, bound_soft_reservations=True, trace=trace,
                            soft_promotion_limit=soft_promotion_limit)
    if core_pool_ids is not None:
        # Keep every candidate visible in the decision trace. Core ranks remain
        # relative to the restricted pool; extra candidates have no core rank.
        for c in ordered:
            if c['id'] not in allowed:
                trace['decisions'].append(dict(id=c['id'], rank=None, selected=False,
                    reason='outside_core_pool', score=c.get('_ce_score'), represented_by=[],
                    visible_sources=sorted({s[0] for s, _ in visible[c['id']]})))
    chosen = list(core)
    additions, rejected = [], []
    capacity = min(len(rows), len(core) + max(0, extra_limit))
    # A bounded semantic second chance for a score-core candidate displaced by
    # reservations. Lexical overlap is not required for cross-language recall.
    # Spend one existing extra seat, never alter the core or split dependencies.
    semantic_ids = {c['id'] for c in core_pool[:max(0, count)]
                    if type(c.get('_ce_score')) in (int, float)
                    and math.isfinite(c['_ce_score']) and visible[c['id']]}
    # The core quota may exclude a strong new-coarse candidate altogether.
    # Give only full-pool CE heads a chance in the existing extra capacity.
    # Both insurance policies share one seat; neither changes the core pool.
    new_coarse_ids = ({c['id'] for c in ordered[:max(0, count)]
                      if c['id'] not in allowed and c['id'] not in excluded_insurance_ids
                      and type(c.get('_ce_score')) in (int, float)
                      and math.isfinite(c['_ce_score']) and visible[c['id']]}
                     if new_coarse_insurance and core_pool_ids is not None else set())
    semantic_used = False
    new_coarse_used = False
    # Complete recorded dependencies of the core before optional topical gains.
    pending = [c for c in ordered if c not in core or dependencies(c)]
    while pending and len(chosen) < capacity:
        chosen_ids = {c['id'] for c in chosen}
        covered = {rid for c in chosen for rid in matches[c['id']]}
        proposals = []
        for c in pending:
            group, missing = closure(c, by_id)
            new_ids = group - chosen_ids
            if missing:
                rejected.append(dict(id=c['id'], reason='missing_dependency', missing_ids=sorted(missing)))
                continue
            if not new_ids:
                continue
            required = c['id'] in chosen_ids and bool(dependencies(c))
            semantic = (not semantic_used and not required and len(new_ids) == 1
                        and ((semantic_reserve and c['id'] in semantic_ids)
                             or c['id'] in new_coarse_ids))
            if not required and ((not semantic and not matches[c['id']]) or
                                 selection.represented_by(c, chosen, visible)):
                continue
            if len(chosen) + len(new_ids) > capacity:
                rejected.append(dict(id=c['id'], reason='group_capacity', group_ids=sorted(group)))
                continue
            new_facets = set(matches[c['id']]) - covered
            # CE decides ties after evidence gain. Source count is not a score.
            # A retrieval-head disagreement gets one bounded second chance only
            # after visible relevance and novelty checks; never displace the core.
            utility = (required, semantic, key(c) if semantic else (-1e30, 0),
                       c['id'] in fused_ids, len(new_facets),
                       max(matches[c['id']].values(), default=(False, False, 0)),
                       key(c))
            proposals.append((utility, c, group, new_ids, semantic))
        if not proposals:
            break
        _, c, group, new_ids, semantic = max(proposals, key=lambda p: p[0])
        semantic_used = semantic_used or semantic
        new_coarse_used = new_coarse_used or (semantic and c['id'] in new_coarse_ids)
        chunk = [r for r in ordered if r['id'] in new_ids]
        additions.append(dict(candidate_id=c['id'], ids=[r['id'] for r in chunk],
            group_ids=sorted(group), requirement_ids=sorted(matches[c['id']]),
            reason=('path_completion' if c['id'] in chosen_ids else
                    'new_coarse_core_insurance' if semantic and c['id'] in new_coarse_ids else
                    'ce_core_insurance' if semantic else
                    'sparse_topic_evidence' if c['id'] in sparse_ids else 'novel_visible_evidence'),
            unscored=c.get('_ce_score') is None))
        chosen.extend(chunk)
        pending = [r for r in pending if r['id'] != c['id']]
    # A lexical gate must not strand spare seats when CE found a nearby,
    # source-backed semantic match. Bound this insurance to the effective CE
    # cutoff, never unscored items or arbitrary low-ranked unique sources.
    for c in ordered[:capacity]:
        if len(chosen) >= capacity:
            break
        chosen_ids = {r['id'] for r in chosen}
        if c['id'] in chosen_ids or c.get('_ce_score') is None or not visible[c['id']]:
            continue
        if selection.represented_by(c, chosen, visible):
            continue
        group, missing = closure(c, by_id)
        new_ids = group - chosen_ids
        if missing or len(chosen) + len(new_ids) > capacity:
            continue
        chunk = [r for r in ordered if r['id'] in new_ids]
        chosen.extend(chunk)
        additions.append(dict(candidate_id=c['id'], ids=[r['id'] for r in chunk],
            group_ids=sorted(group), requirement_ids=sorted(matches[c['id']]),
            reason='ce_boundary_backfill', unscored=False))
    chosen_ids = {c['id'] for c in chosen}
    path_groups = set()
    for c in chosen:
        group, missing = closure(c, by_id)
        if len(group) > 1 and not missing and group <= chosen_ids:
            path_groups.add(tuple(sorted(group)))
    trace['adaptive'] = dict(core_ids=[c['id'] for c in core], extra_limit=extra_limit,
                             context_facets=context,
                             additions=additions, rejected=rejected,
                             path_groups=[list(g) for g in sorted(path_groups)],
                             proposed_ids=[c['id'] for c in chosen], capacity=capacity, unused_capacity=capacity-len(chosen),
                             support_labels_unchanged=True)
    if semantic_reserve:
        trace['adaptive']['semantic_reserve'] = dict(limit=1, max_ce_rank=count,
            rank_scope='core_pool', used=semantic_used and not new_coarse_used,
            eligible_ids=[c['id'] for c in core_pool if c['id'] in semantic_ids])
    if new_coarse_insurance:
        trace['adaptive']['new_coarse_insurance'] = dict(limit=1, max_ce_rank=count,
            rank_scope='full_coarse_outside_core_pool', used=new_coarse_used,
            eligible_ids=[c['id'] for c in ordered if c['id'] in new_coarse_ids])
    if core_pool_ids is not None:
        trace['adaptive'].update(core_policy='restricted_core_pool_v1',
                                 core_pool_ids=[c['id'] for c in core_pool])
    extra_ids = {c['id'] for c in chosen} - {c['id'] for c in core}
    for decision in trace['decisions']:
        if decision['id'] in extra_ids:
            reason = next(a['reason'] for a in additions if decision['id'] in a['ids'])
            decision.update(selected=True, reason=reason, represented_by=[])
        elif not decision['selected'] and not decision['represented_by']:
            decision['reason'] = ('missing_dependency' if closure(by_id[decision['id']], by_id)[1] else
                                  'capacity' if matches[decision['id']] else 'no_visible_relevance')
    return chosen, trace


def admit_additions(submitted, candidates, additions, *, count_limit, byte_limit,
                    token_headroom, request_for):
    """Append whole groups only; never evict a previously admitted core unit."""
    admitted = list(submitted)
    decisions = []
    by_id = {c['id']: c for c in candidates}
    for addition in additions:
        ids = {c['id'] for c in admitted}
        group = set(addition['group_ids'])
        chunk = [by_id[mid] for mid in addition['group_ids'] if mid not in ids and mid in by_id]
        reason = None
        if not group <= set(by_id):
            reason = 'missing_dependency'
        elif len(admitted) + len(chunk) > count_limit:
            reason = 'adaptive_count_limit'
        else:
            request = request_for(admitted + chunk)
            if len(request['prompt'].encode()) > byte_limit:
                reason = 'prompt_budget'
            elif token_headroom is not None and request['reservation'] > token_headroom:
                reason = 'token_budget'
        decisions.append(dict(candidate_id=addition['candidate_id'], group_ids=addition['group_ids'],
                              admitted=reason is None, reason=reason or addition.get('reason', 'novel_visible_evidence')))
        if reason is None:
            admitted.extend(chunk)
    return admitted, decisions


def provisional_candidates(submitted, selected, recovery, requirements, limit):
    """Only unresolved reviews may fall back; a valid negative is not an error."""
    unresolved = set(recovery.get('unreviewed_ids', []))
    reviewed = set(recovery.get('reviewed_ids', []))
    accepted = {c['id'] for c in selected}
    candidates = []
    for c in sorted(submitted, key=key, reverse=True):
        if c['id'] not in unresolved or c['id'] in reviewed or c['id'] in accepted:
            continue
        if not relevance(c, requirements) or selection.represented_by(c, selected + candidates):
            continue
        candidates.append(c)
        if len(candidates) >= limit:
            break
    return candidates if limit > 0 else []
