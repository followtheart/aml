"""One whole coarse source with an unseen rare user-query term, in spare 5 KiB.

Preserve all fine/main selections and the original three context seats. Use
the recorded original retrieval requirements; source rarity never establishes
personal support. The source reserve enforces visibility, whole user messages,
novel source identity, original global bytes/items and existing spare bytes.
No provider calls. References are loaded only after every selection is fixed.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import config, evidence_packet, fine_evidence, fine_sparse_reserve, search_pipeline
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve
from app.fine_context_reserve import full_user
from app.conversation_context import source_key, usable

MODES = ('rare_term_spare', 'rare_source_representative')


def prepare(t):
    ctx = t['cascade']['conversation_context']
    if ctx['policy'] != 'fine_user_lexical_context_v1':
        raise ValueError('Require original three context seats')
    base = [dict(p, packet_hash=ctx['base_packet_hash'])
            for p in t['returned'][:len(ctx['base_packet_ids'])]]
    assert evidence_packet.digest(base) == ctx['base_packet_hash']
    pool = {c['id']: c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']), copy.deepcopy(t['plan']))
            if c.get('view_status') != 'stale' and '_packet_item' in c}
    fine = [pool[mid] for mid in t['cascade']['fine']['selected_ids']
            if mid in pool and mid not in t['selection']['selected_ids']]
    kwargs = dict(user_id=t['user_id'], top_k=t['top_k'], byte_budget=t['coverage_manifest']['token_budget'])
    prefix, old = lexical(base, fine, **kwargs, reserve_bytes=5120, query=t['query'], options=t['options'])
    digest = evidence_packet.digest(prefix)
    assert [dict(p, packet_hash=digest) for p in prefix] == t['returned']
    coarse = [pool[mid] for mid in t['cascade']['coarse']['selected_ids'] if mid in pool]
    kept = [dict(id=p['id'], _rank_text=p['content'], _packet_item=p) for p in prefix]
    visible = {c['id']: fine_evidence.selection.fragments(c) for c in coarse}
    visible.update({c['id']: fine_evidence.selection.fragments(c) for c in kept})
    nominations = fine_sparse_reserve.nominate(coarse, kept, t['coverage_manifest']['coverage']['requirements'], visible)
    kept_ids = {p['id'] for p in prefix}
    excluded = kept_ids | set(t['cascade']['fine']['selected_ids'])
    eligible = []
    inventory = []
    for c in sorted(coarse, key=fine_evidence.key, reverse=True):
        if c['id'] in excluded or c['id'] not in nominations:
            continue
        group, missing = fine_evidence.closure(c, pool)
        reason = 'missing_dependency' if missing or not group <= kept_ids | {c['id']} else 'nominated'
        inventory.append(dict(candidate_id=c['id'], **nominations[c['id']], reason=reason))
        if reason == 'nominated':
            eligible.append(c)
    remaining = 5120-old['added_bytes']
    packet, proposal = reserve(prefix, eligible, **kwargs, reserve_bytes=remaining, limit=1)
    assert packet[:len(prefix)] == prefix
    # Rank a rare source by its best carrier; use an existing complete carrier
    # that fits. This changes neither any source text nor any candidate score.
    identities = {}
    costs = {}
    for c in eligible:
        _, trial = reserve(prefix, [c], **kwargs, reserve_bytes=remaining, limit=1)
        if not trial['proposed']:
            continue
        costs[c['id']] = trial['proposed'][0]['added_bytes']
        hits = set(nominations[c['id']]['terms'])
        identities[c['id']] = {source_key(s) for s in c['_packet_item']['sources']
            if usable(s, t['user_id']) and full_user(s)
            and hits & fine_evidence.selection._topic_concepts(s['content'])}
    source_order = []
    for c in eligible:
        for key in sorted(identities.get(c['id'], ())):
            if key not in source_order:
                source_order.append(key)
    representatives = []
    for key in source_order:
        suppliers = sorted([c for c in eligible if key in identities.get(c['id'], ())],
            key=lambda c: (costs[c['id']], -c['_ce_score'], c['id']))
        for c in suppliers:
            if c not in representatives:
                representatives.append(c)
    revised, revised_proposal = reserve(prefix, representatives, **kwargs, reserve_bytes=remaining, limit=1)
    assert revised[:len(prefix)] == prefix
    for row in inventory:
        row.update(source_keys=sorted(identities.get(row['candidate_id'], ())),
                   cost=costs.get(row['candidate_id']))
    return dict(packets=dict(baseline=prefix, rare_term_spare=packet, rare_source_representative=revised),
                proposals=dict(rare_term_spare=proposal, rare_source_representative=revised_proposal),
                inventory=inventory, remaining_bytes=remaining)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    with patch('socket.socket.connect', side_effect=AssertionError('Offline')):
        score(p.parse_args(), preparer=prepare, modes=MODES, limitation=__doc__, script_path=Path(__file__))
