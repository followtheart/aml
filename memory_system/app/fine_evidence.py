"""Bounded, source-backed additions to a CE core; relevance is not entailment.

No gold labels, provider calls, inferred personal facts or blended CE/graph scores.
The caller must admit additions against the actual whole-request budget.
"""
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


def propose(rows, count, requirements, *, rescue_ids=(), fused_ids=(), extra_limit=8):
    """Keep the existing bounded selection core; add novel relevant groups.

    Extra capacity is earned by visible evidence gain, not requirement count.
    This is a proposal: no downstream capacity or token budget is bypassed.
    """
    ordered = sorted(rows, key=key, reverse=True)
    by_id = {c['id']: c for c in ordered}
    visible = {c['id']: selection.fragments(c) for c in ordered}
    matches = {c['id']: relevance(c, requirements, visible[c['id']]) for c in ordered}
    trace = {}
    core = selection.select(ordered, count, requirements, key, reserve_ids=rescue_ids,
                            fused_ids=fused_ids, bound_soft_reservations=True, trace=trace)
    chosen = list(core)
    additions, rejected = [], []
    capacity = min(len(rows), len(core) + max(0, extra_limit))
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
            if not required and (not matches[c['id']] or
                                 selection.represented_by(c, chosen, visible)):
                continue
            if len(chosen) + len(new_ids) > capacity:
                rejected.append(dict(id=c['id'], reason='group_capacity', group_ids=sorted(group)))
                continue
            new_facets = set(matches[c['id']]) - covered
            # CE decides ties after evidence gain. Source count is not a score.
            # A retrieval-head disagreement gets one bounded second chance only
            # after visible relevance and novelty checks; never displace the core.
            utility = (required, c['id'] in fused_ids, len(new_facets),
                       max(matches[c['id']].values(), default=(False, False, 0)),
                       key(c))
            proposals.append((utility, c, group, new_ids))
        if not proposals:
            break
        _, c, group, new_ids = max(proposals, key=lambda p: p[0])
        chunk = [r for r in ordered if r['id'] in new_ids]
        additions.append(dict(candidate_id=c['id'], ids=[r['id'] for r in chunk],
            group_ids=sorted(group), requirement_ids=sorted(matches[c['id']]),
            reason='path_completion' if c['id'] in chosen_ids else 'novel_visible_evidence',
            unscored=c.get('_ce_score') is None))
        chosen.extend(chunk)
        pending = [r for r in pending if r['id'] != c['id']]
    chosen_ids = {c['id'] for c in chosen}
    path_groups = set()
    for c in chosen:
        group, missing = closure(c, by_id)
        if len(group) > 1 and not missing and group <= chosen_ids:
            path_groups.add(tuple(sorted(group)))
    trace['adaptive'] = dict(core_ids=[c['id'] for c in core], extra_limit=extra_limit,
                             additions=additions, rejected=rejected,
                             path_groups=[list(g) for g in sorted(path_groups)],
                             proposed_ids=[c['id'] for c in chosen],
                             support_labels_unchanged=True)
    extra_ids = {c['id'] for c in chosen} - {c['id'] for c in core}
    for decision in trace['decisions']:
        if decision['id'] in extra_ids:
            decision.update(selected=True, reason='novel_visible_evidence', represented_by=[])
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
                              admitted=reason is None, reason=reason or 'novel_visible_evidence'))
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
