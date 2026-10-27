"""Append at most one scored coarse outsider within the original main budget.

Original fine and main membership, order, sources and qualifiers stay intact.
The first feasible source-novel singleton in existing CE order gets review;
admission does not establish relevance or personal-fact support.
"""
import math


def propose(coarse, fine, submitted, *, helpers, request_for, count_limit,
            byte_limit, token_headroom):
    original = list(submitted)
    fine_ids = {row['id'] for row in fine}
    submitted_ids = {row['id'] for row in original}
    visible = {row['id']: helpers.selection.fragments(row) for row in coarse}
    ordered = sorted((row for row in coarse
                      if type(row.get('_ce_score')) in (int, float)
                      and math.isfinite(row['_ce_score'])), key=helpers.key, reverse=True)
    decisions = []
    for rank, row in enumerate(ordered, 1):
        if row['id'] in fine_ids or row['id'] in submitted_ids:
            continue
        reason = None
        size, reservation = None, None
        if not visible[row['id']]:
            reason = 'no_visible_source'
        elif helpers.dependencies(row):
            reason = 'not_singleton'
        elif helpers.selection.represented_by(row, original, visible):
            reason = 'represented_source'
        elif len(original) + 1 > count_limit:
            reason = 'count_budget'
        else:
            request = request_for(original + [row])
            size = len(request['prompt'].encode('utf-8'))
            reservation = request['reservation']
            if size > byte_limit:
                reason = 'prompt_budget'
            elif token_headroom is not None and reservation > token_headroom:
                reason = 'token_budget'
        decisions.append(dict(id=row['id'], ce_rank=rank, reason=reason or 'admitted',
                              trial_bytes=size, trial_reservation=reservation))
        if reason is None:
            return original + [row], dict(addition=dict(id=row['id'], ce_rank=rank,
                ce_score=row['_ce_score'], type=row.get('type')), decisions=decisions)
    return original, dict(addition=None, decisions=decisions)
