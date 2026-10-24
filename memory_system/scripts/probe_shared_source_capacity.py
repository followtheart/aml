"""Offline capacity bound for residual context with already-delivered sources.

Preserve the actual complete packet and its three context seats. Compare one
ordinary residual unit with one whose repeated full source bodies are counted
only once. Keep full provenance and require identical local reader catalogs
to the corresponding uncompressed addition. No claim is made that a generic
content-only consumer preserves conversational associations after this layout
change. This is a capacity diagnostic, not a production packing policy.
"""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import answer_choice, answer_context, evidence_packet, search_pipeline
from app.conversation_context import source_key, usable
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import full_user, propose as reserve

MODES = ('whole_residual', 'shared_source_bound')


def catalog(packet):
    digest = evidence_packet.digest(packet)
    return answer_choice.build_catalog([dict(p, packet_hash=digest) for p in packet])


def prepare(t):
    ctx = t['cascade']['conversation_context']
    if ctx['policy'] != 'fine_user_lexical_context_v1':
        raise ValueError('Require unchanged three-seat context')
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
    remaining = 5120 - old['added_bytes']
    packets = dict(baseline=prefix)
    proposals = {}
    packets['whole_residual'], proposals['whole_residual'] = reserve(prefix, fine, **kwargs,
        reserve_bytes=remaining, limit=1)
    known = {source_key(s) for p in prefix for s in p.get('sources', []) if usable(s, t['user_id'])}
    kept = {p['id'] for p in prefix}
    used = sum(len(p['content'].encode()) for p in prefix) + max(0, len(prefix)-1)
    inventory, result, accepted = [], list(prefix), []
    for candidate in fine:
        if candidate['id'] in kept:
            continue
        # Use original full-unit eligibility before changing the cost model.
        trial, _ = reserve([], [candidate], user_id=t['user_id'], top_k=1,
                           byte_budget=10**9, reserve_bytes=10**9, limit=1)
        if not trial:
            continue
        whole = trial[0]
        novel = [s for s in whole['sources'] if source_key(s) not in known]
        shared = [s for s in whole['sources'] if source_key(s) in known]
        if not any(full_user(s) for s in novel):
            continue
        compact = dict(whole, content=answer_context.with_evidence('Conversation excerpt:', novel))
        # Sources are retained with original ownership/order. Every excluded
        # rendered body must be visible elsewhere, with the same provenance.
        assert catalog(prefix + [whole]) == catalog(prefix + [compact])
        cost = len(compact['content'].encode()) + bool(prefix)
        whole_cost = len(whole['content'].encode()) + bool(prefix)
        reason = ('context_limit' if accepted else 'top_k' if len(prefix) >= t['top_k']
                  else 'reserve_bytes' if cost > remaining
                  else 'token_budget' if used + cost > kwargs['byte_budget'] else 'admitted')
        inventory.append(dict(candidate_id=candidate['id'], whole_bytes=whole_cost,
            compact_bytes=cost, shared_sources=len(shared), novel_sources=len(novel),
            duplicate_saved_bytes=whole_cost-cost, reason=reason,
            can_fit=cost <= remaining and used+cost <= kwargs['byte_budget'] and len(prefix)<t['top_k'],
            source_event_ids=[s['source_event_id'] for s in whole['sources']]))
        if reason == 'admitted':
            result.append(compact)
            accepted.append(candidate['id'])
    assert result[:len(prefix)] == prefix
    packets['shared_source_bound'] = result
    proposals['shared_source_bound'] = dict(appended_ids=accepted,
        added_bytes=sum(len(p['content'].encode())+bool(prefix) for p in result[len(prefix):]))
    return dict(packets=packets, proposals=proposals, inventory=inventory, remaining_bytes=remaining)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    with patch('socket.socket.connect', side_effect=AssertionError('Offline only')):
        score(parser.parse_args(), preparer=prepare, modes=MODES, limitation=__doc__, script_path=Path(__file__))
