"""Bounded, source-backed additions to a CE core; relevance is not entailment.

No gold labels, provider calls, inferred personal facts or blended CE/graph scores.
The caller must admit additions against the actual whole-request budget.
"""
from collections import Counter

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


def propose(rows, count, requirements, *, rescue_ids=(), fused_ids=(), extra_limit=8):
    """Keep the existing bounded selection core; add novel relevant groups.

    Extra capacity is earned by visible evidence gain, not requirement count.
    This is a proposal: no downstream capacity or token budget is bypassed.
    """
    ordered = sorted(rows, key=key, reverse=True)
    by_id = {c['id']: c for c in ordered}
    visible = {c['id']: selection.fragments(c) for c in ordered}
    matches = {c['id']: relevance(c, requirements, visible[c['id']]) for c in ordered}
    sparse = sparse_matches(ordered, requirements, visible)
    sparse_ids = {mid for mid in matches if not matches[mid] and sparse[mid]}
    for mid in matches:
        for rid, value in sparse[mid].items():
            matches[mid][rid] = max(matches[mid].get(rid, (False, False, 0)), value)
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
            reason=('path_completion' if c['id'] in chosen_ids else
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
                             additions=additions, rejected=rejected,
                             path_groups=[list(g) for g in sorted(path_groups)],
                             proposed_ids=[c['id'] for c in chosen], capacity=capacity, unused_capacity=capacity-len(chosen),
                             support_labels_unchanged=True)
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
