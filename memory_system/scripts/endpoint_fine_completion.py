"""Append whole eligible fine units while preserving the actual endpoint prefix.

No extra source reads or relevance judgments. Keep original prepared content,
roles, inference labels and dependencies. Only append groups with novel visible
user catalog text, within the original total byte/item budget. Existing packet
items may change only their shared packet digest. This is an offline hypothesis.
"""
import copy
import json
import math


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
        score = row.get('_ce_score')
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError('Missing finite CE score')
        item = dict(row['_packet_item'], score=score, score_kind='cross_encoder', packet_hash_version=4)
        preview[row['id']] = schemas.SearchItem.model_validate(item).model_dump()
    decisions, visited, appended = [], set(), []
    for row in rows:
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
