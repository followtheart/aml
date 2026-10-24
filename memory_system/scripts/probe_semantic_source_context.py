"""One complete visible coarse source by measured source-semantic rank.

Use the previously defined source route: top six per distinct original query,
RRF60, top eight source messages. Try existing complete carriers in that order,
smaller carriers of the same source first. Retain the original packet/three
context seats and spend only remaining original 5 KiB, at most one addition.
No new measurements here. Gold is read only after all selections are fixed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import evidence_packet, fine_evidence, search_pipeline
from app.conversation_context import usable
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve, full_user

MODES = ('whole', 'window2')


def ranked_sources(queries):
    scores, seen = {}, set()
    for row in queries:
        text = row['query']['text']
        if text in seen:
            continue
        seen.add(text)
        for rank, hit in enumerate(row['hits'][:6], 1):
            si = hit['source_index']
            scores[si] = scores.get(si, 0) + 1 / (60 + rank)
    return sorted(scores, key=lambda si: (-scores[si], si))


def preparer(plan, rankings, modes=MODES):
    sources = plan['sources']
    # Query signatures are paired with the original user, never a benchmark ID.
    by_query = {(r['user_id'], tuple(q['query']['text'] for q in r['modes'][modes[0]])): r for r in rankings['rankings']}
    if len(by_query) != rankings['questions']:
        raise ValueError('Duplicate user/query signatures')

    def prepare(t):
        measured = by_query[t['user_id'], tuple(s['text'] for s in t['query_specs'])]
        ctx = t['cascade']['conversation_context']
        if ctx['policy'] != 'fine_user_lexical_context_v1':
            raise ValueError('Require original lexical prefix')
        base = [dict(p, packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
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
        kept = {p['id'] for p in prefix}
        def whole_identity(source):
            if not usable(source, t['user_id']):
                return None
            span = source.get('content_span')
            if span and not (span.get('start') == 0 and span.get('end') == span.get('original_length') == len(source['content'])):
                return None
            return source['request_id'], source['message_index'], source['content']
        known = {key for p in prefix for s in p['sources'] if (key := whole_identity(s))}
        eligible, costs, identities = [], {}, {}
        for mid in t['cascade']['coarse']['selected_ids']:
            if mid not in pool or mid in kept:
                continue
            candidate = pool[mid]
            group, missing = fine_evidence.closure(candidate, pool)
            if missing or not group <= kept | {mid}:
                continue
            _, trial = reserve(prefix, [candidate], **kwargs, reserve_bytes=remaining, limit=1)
            if not trial['proposed']:
                continue
            costs[mid] = trial['proposed'][0]['added_bytes']
            identities[mid] = {key for s in candidate['_packet_item']['sources'] if (key := whole_identity(s))}
            eligible.append(candidate)
        packets, proposals, inventory = {'baseline': prefix}, {}, []
        for mode in modes:
            selected_sources = ranked_sources(measured['modes'][mode])[:8]
            ordered, seen_ids = [], set()
            for rank, si in enumerate(selected_sources, 1):
                source = sources[si]
                if source['user_id'] != t['user_id']:
                    raise ValueError('Cross-user source ranking')
                key = source['request_id'], source['message_index'], source['content']
                exchange = plan.get('exchange_sources', {}).get(str(si), [source])
                required = {(s['request_id'], s['message_index'], s['content']) for s in exchange}
                carriers = sorted([c for c in eligible if key in identities[c['id']]
                                   and required <= identities[c['id']] | known],
                                  key=lambda c: (costs[c['id']], -c['_ce_score'], c['id']))
                inventory.append(dict(mode=mode, source_index=si, source_rank=rank,
                    candidate_ids=[c['id'] for c in carriers], costs=[costs[c['id']] for c in carriers]))
                for c in carriers:
                    if c['id'] not in seen_ids:
                        ordered.append(c)
                        seen_ids.add(c['id'])
            packets[mode], proposals[mode] = reserve(prefix, ordered, **kwargs, reserve_bytes=remaining, limit=1)
            proposals[mode]['semantic_source_indices'] = selected_sources
            assert packets[mode][:len(prefix)] == prefix
        return dict(packets=packets, proposals=proposals, inventory=inventory, remaining_bytes=remaining)
    return prepare


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    args = parser.parse_args()
    plan_path, rankings_path = args.probe / 'plan.json', args.probe / 'rankings.json'
    plan, rankings = json.loads(plan_path.read_text()), json.loads(rankings_path.read_text())
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != rankings['plan_sha256']:
        raise ValueError('Measured source universe changed')
    modes = tuple(rankings['rankings'][0]['modes'])
    if modes not in (MODES, ('exchange',)) or any(tuple(r['modes']) != modes for r in rankings['rankings']):
        raise ValueError('Unexpected or mixed measured modes')
    with patch('socket.socket.connect', side_effect=AssertionError('Offline proposal')):
        score(args, preparer=preparer(plan, rankings, modes), modes=modes, limitation=__doc__, script_path=Path(__file__))
    result = json.loads(args.output.read_text())
    result['measured_inputs'] = dict(probe=str(args.probe), plan_sha256=rankings['plan_sha256'],
        rankings_sha256=hashlib.sha256(rankings_path.read_bytes()).hexdigest(), vectors_sha256=rankings['vectors_sha256'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
