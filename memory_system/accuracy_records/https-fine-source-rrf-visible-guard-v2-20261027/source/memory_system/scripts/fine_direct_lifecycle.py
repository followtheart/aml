"""Prioritize existing whole fine units that carry uncovered reader constraints.

Use the same visible-source constraint classification as the answer reader.
Keep complete units and dependency groups; do not extract new claims, change
roles, promote assistant/fictional text, or promise retention beyond capacity.
"""
import hashlib
import json
import re


def constraint_key(card):
    return json.dumps([card.get('request_id'), card.get('message_index'), card.get('role'),
        card.get('timestamp'), card.get('legacy_rule', False), card['text']], ensure_ascii=False)


def keys(cards):
    return {constraint_key(card) for card in cards}


def digests(values):
    return sorted(hashlib.sha256(value.encode()).hexdigest() for value in values)


def item_for(row, schemas, *, rule=False):
    item = dict(row['_packet_item'], score_kind='constraint' if rule else 'cross_encoder')
    if not rule:
        item['score'] = row.get('_ce_score')
    else:
        item.pop('score', None)
    item['packet_hash_version'] = 4
    return schemas.SearchItem.model_validate(item).model_dump()


def prioritize(rows, rules, packet, schemas, answer, *, top_k, byte_budget):
    # Seed only constraints the original capped-rule prefix actually packs.
    prefix, _, _ = packet.pack_ranked([item_for(row, schemas, rule=True) for row in rules],
                                      top_k, byte_budget)
    seen = keys(answer.build_catalog(prefix)[1].values())
    seeded = set(seen)
    promoted, ordinary, observations = [], [], []
    # The reader's attribution expression is case-sensitive. For ordering only,
    # conservatively exclude its case variants too; never relabel reader cards.
    attributed = re.compile(answer._ATTRIBUTED.pattern, answer._ATTRIBUTED.flags | re.I)
    for row in rows:
        preview = item_for(row, schemas)
        preview['packet_hash'] = packet.digest([preview])
        cards = list(answer.build_catalog([preview])[1].values())
        excluded = [card for card in cards if attributed.search(card.get('source_text', card['text']))]
        constraints = keys(card for card in cards if card not in excluded)
        uncovered = constraints - seen
        (promoted if uncovered else ordinary).append(row)
        seen.update(constraints)
        observations.append(dict(candidate_id=row['id'], constraint_hashes=digests(constraints),
                                 attribution_excluded_hashes=digests(keys(excluded)),
                                 newly_represented_hashes=digests(uncovered)))
    return promoted + ordinary, dict(policy='first_existing_fine_owner_of_reader_constraint',
        prioritized_ids=[row['id'] for row in promoted], observations=observations,
        seeded_constraint_hashes=digests(seeded), expected_constraint_hashes=digests(seen),
        original_fine_order=[row['id'] for row in rows])
