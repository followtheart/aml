"""Append whole eligible fine units while preserving the actual endpoint prefix.

No extra source reads or relevance judgments. Keep original prepared content,
roles, inference labels and dependencies. Only append groups with novel visible
user catalog text, within the original total byte/item budget. Existing packet
items may change only their shared packet digest. The feature is default-off until paired HTTPS evaluation.
"""
import copy
import math
import time
from . import budget


def card_identity(card):
    return (card.get('request_id'), card.get('message_index'), card.get('role'),
            card.get('timestamp'), card.get('declared'), card.get('legacy_rule', False),
            card['id'][:1])


def visible_cards(items, answer):
    return [card for section in answer.build_catalog(items) for card in section.values()]


def contains(card, known):
    return any(card_identity(card) == card_identity(other) and card['text'] in other['text']
               for other in known)


def without_digest(item):
    return {key: value for key, value in item.items() if key != 'packet_hash'}


def complete(packed, rows, groups, packet, schemas, answer, *, top_k, byte_budget):
    original = copy.deepcopy(packed)
    if len({item['id'] for item in packed}) != len(packed):
        raise ValueError('Duplicate baseline IDs')
    result = copy.deepcopy(packed)
    used = sum(len(item['content'].encode()) for item in result) + max(0, len(result) - 1)
    if len(result) > top_k or used > byte_budget:
        raise ValueError('Baseline exceeds capacity')
    known = visible_cards(packed, answer)
    fine = {row['id']: row for row in rows}
    if len(fine) != len(rows):
        raise ValueError('Duplicate fine IDs')
    membership = {}
    for group in groups:
        if len(set(group)) != len(group) or any(mid not in fine or mid in membership for mid in group):
            raise ValueError('Groups must be disjoint known fine IDs')
        for mid in group:
            membership[mid] = set(group)
    kept = {item['id']: item for item in packed}
    preview = {}
    for row in rows:
        budget.check()
        score = row.get('_ce_score')
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError('Missing finite CE score')
        item = dict(row['_packet_item'], score=score, score_kind='cross_encoder', packet_hash_version=4)
        preview[row['id']] = schemas.SearchItem.model_validate(item).model_dump()
    decisions, visited, appended = [], set(), []
    for row in rows:
        budget.check()
        mid = row['id']
        if mid in visited or mid in kept:
            continue
        members = membership.get(mid, {mid})
        visited.update(members)
        additions = [preview[r['id']] for r in rows if r['id'] in members - kept.keys()]
        # A context-only item with the same ID is not the complete prepared unit.
        # Do not count such a shortened old item as satisfying a new group.
        unavailable = [other for other in members & kept.keys()
                       if any(kept[other].get(key) != preview[other].get(key)
                              for key in ('content', 'sources', 'memory_type', 'is_constraint',
                                          'personal_evidence', 'coverage_ids'))]
        temporary_hash = packet.digest(additions)
        additions = [dict(item, packet_hash=temporary_hash) for item in additions]
        cards = visible_cards(additions, answer)
        novel = [card for card in cards if card.get('role') == 'user' and not contains(card, known)]
        cost = sum(len(item['content'].encode()) for item in additions) + len(additions) - (not result)
        reason = ('unavailable_dependency' if unavailable else 'no_novel_user_text' if not novel else
                  'top_k' if len(result) + len(additions) > top_k else
                  'token_budget' if used + cost > byte_budget else 'admitted')
        decisions.append(dict(candidate_id=mid, group_ids=sorted(members),
            addition_ids=[item['id'] for item in additions], reason=reason, cost=cost,
            novel_user_cards=len(novel), unavailable_dependency_ids=unavailable))
        if reason != 'admitted':
            continue
        result.extend(additions)
        used += cost
        appended.extend(item['id'] for item in additions)
        kept.update((item['id'], item) for item in additions)
        known.extend(cards)
    digest = packet.digest(result)
    result = [dict(item, packet_hash=digest) for item in result]
    if packed != original or [without_digest(item) for item in result[:len(original)]] != [
            without_digest(item) for item in original]:
        raise ValueError('Original endpoint prefix changed')
    actual = visible_cards(result, answer)
    if any(not contains(card, actual) for card in visible_cards(original, answer)):
        raise ValueError('Original endpoint catalog not retained')
    return result, digest, dict(policy='endpoint_preserved_whole_fine_completion_v1',
        original_count=len(packed), total_bytes=used, added_bytes=used - sum(
            len(item['content'].encode()) for item in original) - max(0, len(original) - 1),
        appended_ids=appended, decisions=decisions, groups=groups)


def attach(req, plan, ranked, packed, packet_hash, manifest):
    """Run after existing packet/context stages; failures preserve their result."""
    from . import answer_choice, evidence_packet, fine_evidence, listwise_overflow, schemas
    if evidence_packet.digest(packed) != packet_hash:
        raise ValueError('Baseline packet hash mismatch')
    limits = budget.current.get()
    details = dict(policy='endpoint_preserved_whole_fine_completion_v1',
                   base_packet_ids=[item['id'] for item in packed], base_packet_hash=packet_hash,
                   base_token_upper_bound=manifest['token_upper_bound'], appended_ids=[])
    if limits and (limits.cancelled.is_set() or limits.deadline - time.monotonic() < .5):
        return packed, packet_hash, dict(manifest, endpoint_fine_completion=dict(details, status='budget_skipped'))
    pool = {row['id']: row for row in ranked}
    fine_ids = plan['_cascade'].get('fine', {}).get('selected_ids', [])
    if any(mid not in pool or '_packet_item' not in pool[mid] for mid in fine_ids):
        return packed, packet_hash, dict(manifest, endpoint_fine_completion=dict(details, status='missing_fine_unit'))
    rows = [pool[mid] for mid in fine_ids]
    groups = list(plan['_cascade']['fine'].get('selection', {}).get('adaptive', {}).get('path_groups', []))
    fine_pool = {row['id']: row for row in rows}
    for row in rows:
        budget.check()
        group, missing = fine_evidence.closure(row, fine_pool)
        if missing:
            return packed, packet_hash, dict(manifest, endpoint_fine_completion=dict(details, status='missing_dependency'))
        if len(group) > 1:
            groups.append(list(group))
    groups = listwise_overflow.merge_groups(groups, rows)
    from .search_pipeline import _foresight_filter
    eligible = {row['id'] for row in _foresight_filter(rows, plan) if row.get('view_status') != 'stale'}
    broken = {mid for group in groups if not set(group) <= eligible for mid in group}
    rows = [row for row in rows if row['id'] in eligible - broken]
    groups = [group for group in groups if not set(group) & broken]
    if any(type(row.get('_ce_score')) not in (int, float) or not math.isfinite(row['_ce_score']) for row in rows):
        return packed, packet_hash, dict(manifest, endpoint_fine_completion=dict(details, status='missing_ce_score'))
    result, digest, completion = complete(packed, rows, groups, evidence_packet, schemas,
                                        answer_choice, top_k=req.top_k, byte_budget=manifest['token_budget'])
    details.update(completion, status='appended' if completion['appended_ids'] else 'not_needed',
                   original_fine_order=fine_ids, eligible_fine_ids=[row['id'] for row in rows])
    added = set(details['appended_ids'])
    additions = result[len(packed):]
    covered = set(manifest.get('covered_requirements', []))
    covered.update(cid for item in additions for cid in item.get('coverage_ids', []))
    updated = dict(manifest, included_ids=[item['id'] for item in result],
        token_upper_bound=details['total_bytes'], endpoint_fine_completion=details,
        evidence_count=manifest['evidence_count'] + sum(not item.get('is_constraint') for item in additions),
        constraint_count=manifest['constraint_count'] + sum(bool(item.get('is_constraint')) for item in additions),
        constraint_bytes=manifest.get('constraint_bytes', 0) + sum(len(item['content'].encode()) + bool(len(packed) + index)
            for index, item in enumerate(additions) if item.get('is_constraint')),
        covered_requirements=sorted(covered),
        missing_requirements=sorted(set(manifest.get('missing_requirements', [])) - covered),
        omitted=[entry for entry in manifest.get('omitted', []) if entry['id'] not in added])
    accepted_groups = [group for group in groups if set(group) & added]
    updated['evidence_groups'] = listwise_overflow.merge_groups(
        manifest.get('evidence_groups', []) + accepted_groups, result)
    updated['requested_evidence_groups'] = manifest.get('requested_evidence_groups', []) + groups
    return result, digest, updated
