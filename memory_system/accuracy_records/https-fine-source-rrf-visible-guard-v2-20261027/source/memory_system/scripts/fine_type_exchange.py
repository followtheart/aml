"""One optional CE exchange preserving the stored memory-type distribution.

Offline hypothesis, not a production selector. Donor eligibility is supplied by
the existing assistant-only weak-addition predicate. No reference data is used.
"""
import copy
import math
from collections import Counter


def choose(coarse, fine, selection, eligible_ids, helpers):
    adaptive = selection['adaptive']
    selected = {row['id'] for row in fine}
    protected = set(adaptive['core_ids'])
    protected.update(mid for row in fine for mid in helpers.dependencies(row))
    protected.update(r['candidate_id'] for r in selection['reservations'] if r['selected'])
    visible = {row['id']: helpers.selection.fragments(row) for row in coarse}
    valid = lambda row: type(row.get('_ce_score')) in (int, float) and math.isfinite(row['_ce_score'])
    ordered = sorted((row for row in coarse if valid(row)), key=helpers.key, reverse=True)
    ranks = {row['id']: i for i, row in enumerate(ordered, 1)}
    donors = [row for row in fine if row['id'] in eligible_ids and row['id'] not in protected
              and valid(row) and isinstance(row.get('type'), str) and row['type']]
    pairs = []
    for nominee in ordered[:adaptive['capacity']]:
        if (nominee['id'] in selected or not visible[nominee['id']] or helpers.dependencies(nominee)
                or not isinstance(nominee.get('type'), str) or not nominee['type']
                or helpers.selection.represented_by(nominee, fine, visible)):
            continue
        compatible = [row for row in donors if row['type'] == nominee['type']
                      and row['_ce_score'] < nominee['_ce_score']]
        if compatible:
            donor = min(compatible, key=helpers.key)
            pairs.append(dict(removed_id=donor['id'], added_id=nominee['id'], type=nominee['type'],
                removed_ce=donor['_ce_score'], added_ce=nominee['_ce_score'],
                removed_rank=ranks[donor['id']], added_rank=ranks[nominee['id']]))
    return dict(swap=pairs[0] if pairs else None, eligible_pairs=pairs,
                eligible_donors=[row['id'] for row in donors], protected_ids=sorted(protected),
                effective_ce_cutoff=adaptive['capacity'])


def apply(rows, chosen, metadata, proposal):
    exchange = proposal['swap']
    if exchange is None:
        return chosen, metadata
    metadata = copy.deepcopy(metadata)
    by_id = {row['id']: row for row in rows}
    old, new = exchange['removed_id'], exchange['added_id']
    assert by_id[old]['type'] == by_id[new]['type'] == exchange['type']
    result = [by_id[new] if row['id'] == old else row for row in chosen]
    additions = metadata['adaptive']['additions']
    index = next(i for i, value in enumerate(additions) if value['candidate_id'] == old)
    assert additions[index]['ids'] == additions[index]['group_ids'] == [old]
    additions[index] = dict(candidate_id=new, ids=[new], group_ids=[new], requirement_ids=[],
                            reason='same_type_ce_exchange', unscored=False)
    metadata['adaptive']['proposed_ids'] = [row['id'] for row in result]
    metadata['adaptive']['type_exchange'] = copy.deepcopy(exchange)
    metadata['adaptive']['rejected'] = [row for row in metadata['adaptive']['rejected'] if row['id'] != new]
    for decision in metadata['decisions']:
        if decision['id'] == old:
            decision.update(selected=False, reason='same_type_lower_ce_exchange', represented_by=[])
        elif decision['id'] == new:
            decision.update(selected=True, reason='same_type_ce_exchange', represented_by=[])
    assert len(result) == len(chosen)
    assert Counter(row['type'] for row in result) == Counter(row['type'] for row in chosen)
    assert {row['id'] for row in result} == ({row['id'] for row in chosen} - {old}) | {new}
    assert set(proposal['protected_ids']) <= {row['id'] for row in result}
    return result, metadata
