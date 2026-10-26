"""Bounded reviews of fine units omitted solely by prompt bytes.

Main listwise decisions are immutable. Only validated useful additions whose
recorded dependencies survive may enter the final packet competition.
"""
import time
from . import budget, config, evidence_selection, fine_evidence, listwise_recovery

MAX_CANDIDATES = 4
MAX_EXTRA_CANDIDATES = 2
EXTENDED_PROMPT_BYTES = 16000
MAX_REVIEW_BATCHES = 3


def merge_groups(groups, selected):
    """Packers require disjoint components, even when paths share an owner."""
    components = []
    for group in groups:
        members = set(group)
        if len(members) < 2:
            continue
        overlapping = [c for c in components if c & members]
        components = [c for c in components if not c & members]
        for component in overlapping:
            members.update(component)
        components.append(members)
    order = [c['id'] for c in selected]
    return [[mid for mid in order if mid in group] for group in components]


def _propose(req, fine, submitted, selected, omissions, *, seed=(), seed_groups=(),
             candidate_limit=MAX_CANDIDATES, prompt_limit=None):
    by_id = {c['id']: c for c in fine}
    submitted_ids = {c['id'] for c in submitted}
    kept = {c['id'] for c in selected}
    eligible = {o['id'] for o in omissions
                if o['stage'] == 'listwise' and o['reason'] == 'prompt_budget'} - submitted_ids
    pending, groups, decisions = list(seed), list(seed_groups), []
    prompt_limit = min(listwise_recovery.MAX_PROMPT_BYTES if prompt_limit is None else prompt_limit,
                       config.RERANK_MAX_PROMPT_BYTES)
    limits = budget.current.get()
    for candidate in fine:
        if candidate['id'] not in eligible or candidate in pending:
            continue
        group, missing = fine_evidence.closure(candidate, by_id)
        extra = group - kept
        reason = None
        if missing or not extra <= eligible:
            reason = 'unavailable_dependency'
        elif evidence_selection.represented_by(candidate, selected):
            reason = 'represented'
        trial_ids = {c['id'] for c in pending} | extra
        trial = ([*pending, *[c for c in fine if c['id'] in trial_ids and c not in pending]]
                 if seed else [c for c in fine if c['id'] in trial_ids])
        if not reason and len(trial) > candidate_limit:
            reason = 'candidate_limit'
        if not reason:
            cards = [dict(candidate_id=c['id'], sources=listwise_recovery.sources_for(c)) for c in trial]
            if any(not c['sources'] for c in cards):
                reason = 'no_visible_source'
            else:
                request = listwise_recovery.request_for(req, cards)
                if not config.LISTWISE_SPAN_REFS:
                    request = listwise_recovery.candidate_contract(request, cards)
                if len(request['prompt'].encode()) > prompt_limit:
                    reason = 'prompt_budget'
                elif limits and request['reservation'] > limits.max_tokens - limits.tokens - limits.reserved_tokens:
                    reason = 'token_budget'
        decisions.append(dict(id=candidate['id'], group_ids=sorted(group), reason=reason or 'admitted'))
        if not reason:
            pending = trial
            groups.append(sorted(group))
    return pending, groups, decisions


def proposal(req, fine, submitted, selected, omissions):
    baseline, groups, decisions = _propose(req, fine, submitted, selected, omissions)
    baseline_ids = [c['id'] for c in baseline]
    pending = baseline
    prompt_limit = min(listwise_recovery.MAX_PROMPT_BYTES, config.RERANK_MAX_PROMPT_BYTES)
    if config.LISTWISE_OVERFLOW_EXTEND:
        prompt_limit = min(EXTENDED_PROMPT_BYTES, config.RERANK_MAX_PROMPT_BYTES)
        pending, groups, extra_decisions = _propose(
            req, fine, submitted, selected, omissions, seed=baseline, seed_groups=groups,
            candidate_limit=len(baseline) + MAX_EXTRA_CANDIDATES, prompt_limit=prompt_limit)
        decisions += [dict(d, phase='extension') for d in extra_decisions]
    metadata = dict(baseline_ids=baseline_ids,
                    extra_ids=[c['id'] for c in pending if c['id'] not in baseline_ids],
                    prompt_limit_bytes=prompt_limit,
                    candidate_limit=len(baseline) + MAX_EXTRA_CANDIDATES
                        if config.LISTWISE_OVERFLOW_EXTEND else MAX_CANDIDATES)
    return pending, groups, decisions, metadata


def propose(req, fine, submitted, selected, omissions):
    return proposal(req, fine, submitted, selected, omissions)[:3]


async def _review_once(req, fine, submitted, selected, omissions, *, deadline=None):
    pending, groups, decisions, metadata = proposal(req, fine, submitted, selected, omissions)
    kept = {c['id'] for c in selected}
    trace = dict(status='not_needed', proposed_ids=[c['id'] for c in pending],
                 decisions=decisions, groups=groups, appended_ids=[], **metadata)
    if not pending:
        return selected, trace
    _, reviewed = await listwise_recovery._recover_batch(
        req, pending, selected, set(range(len(pending))), deadline=deadline,
        stage='search.listwise.overflow', prompt_limit_bytes=metadata['prompt_limit_bytes'])
    useful = set(reviewed['restored_ids'])
    # Transitive closure also covers dependencies of dependencies, including
    # units rejected by the primary listwise or this review.
    by_id = {c['id']: c for c in fine}
    accepted = set(useful)
    for mid in useful:
        group, missing = fine_evidence.closure(by_id[mid], by_id)
        if missing or not group <= kept | useful:
            accepted.discard(mid)
    additions = [c for c in pending if c['id'] in accepted]
    trace.update(status=reviewed['status'], review=reviewed,
                 appended_ids=[c['id'] for c in additions],
                 dependency_blocked_ids=sorted(useful - accepted))
    return selected + additions, trace


async def review(req, fine, submitted, selected, omissions, *, deadline=None):
    if not config.LISTWISE_OVERFLOW_BATCHED:
        return await _review_once(req, fine, submitted, selected, omissions, deadline=deadline)
    # Share a deadline across batches, including calls without an outer scope.
    deadline = min(deadline if deadline is not None else float('inf'),
                   time.monotonic() + config.RERANK_DEADLINE_SECONDS)
    attempted = list(submitted)
    by_id = {c['id']: c for c in fine}
    batches = []
    for _ in range(MAX_REVIEW_BATCHES):
        selected, trace = await _review_once(
            req, fine, attempted, selected, omissions, deadline=deadline)
        batches.append(trace)
        attempted.extend(by_id[mid] for mid in trace['proposed_ids'])
        # Never repeat negative/invalid judgments. Only fresh byte omissions
        # can reach subsequent requests, under the original token/call limits.
        if not trace['proposed_ids'] or trace['status'] in ('error', 'budget_skipped'):
            break
        limits = budget.current.get()
        if time.monotonic() >= deadline or (limits and (
                limits.cancelled.is_set() or time.monotonic() >= limits.deadline - 2
                or limits.calls + limits.reserved_calls >= limits.max_calls)):
            break
    merged = dict(batches[0], batches=batches, batch_limit=MAX_REVIEW_BATCHES)
    for field in ('proposed_ids', 'decisions', 'groups', 'appended_ids', 'dependency_blocked_ids'):
        merged[field] = [v for b in batches for v in b.get(field, [])]
    merged['extra_ids'] = [mid for mid in merged['proposed_ids'] if mid not in merged['baseline_ids']]
    reviews = [b['review'] for b in batches if 'review' in b]
    if reviews:
        combined = dict(reviews[0], batches=reviews)
        for field in ('candidate_ids', 'sources', 'reviewed_ids', 'restored_ids', 'unreviewed_ids',
                      'represented', 'judgments', 'errors', 'omitted'):
            combined[field] = [v for r in reviews for v in r.get(field, [])]
        # Byte counts and hashes belong to individual calls, not this aggregate.
        for field in ('prompt_sha256', 'input_bytes', 'request_bound', 'time_limit_seconds', 'output_contract'):
            combined.pop(field, None)
        combined['status'] = ('partial' if combined['errors'] else
                              'recovered' if combined['restored_ids'] else 'ok')
        merged.update(status=combined['status'], review=combined)
    return selected, merged
