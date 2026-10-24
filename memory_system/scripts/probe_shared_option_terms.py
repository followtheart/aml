"""One residual complete carrier for an unseen literal option topic.

Compare terms shared by at least two original options with option-unique terms.
Source rarity counts original observations across prepared coarse candidates,
not memory projections. Both roles may anchor topical context; neither proves
personal facts. Preserve all original packet content, three context seats,
source spans, CE scores, dependencies and the remaining original 5 KiB budget.
All choices precede gold scoring, and networking is disabled.
"""
import argparse
from collections import Counter
import copy
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import main as score
from app import evidence_packet, fine_evidence, personal_evidence, search_pipeline
from app.conversation_context import usable
from app.fine_context_lexical import propose as lexical
from app.fine_context_reserve import propose as reserve

MODES = ('shared_options', 'unique_options')


def source_identity(source):
    return tuple(source.get(k) for k in ('user_id', 'request_id', 'message_index', 'source_event_id', 'role'))


def prepare(trace):
    context = trace['cascade']['conversation_context']
    if context['policy'] != 'fine_user_lexical_context_v1':
        raise ValueError('Require original lexical prefix')
    base = [dict(p, packet_hash=context['base_packet_hash'])
        for p in trace['returned'][:len(context['base_packet_ids'])]]
    if evidence_packet.digest(base) != context['base_packet_hash']:
        raise ValueError('Base packet changed')
    pool = {c['id']: c for c in search_pipeline._foresight_filter(
        copy.deepcopy(trace['ranked']), copy.deepcopy(trace['plan']))
        if c.get('view_status') != 'stale' and '_packet_item' in c}
    fine = [pool[mid] for mid in trace['cascade']['fine']['selected_ids']
        if mid in pool and mid not in trace['selection']['selected_ids']]
    kwargs = dict(user_id=trace['user_id'], top_k=trace['top_k'],
        byte_budget=trace['coverage_manifest']['token_budget'])
    prefix, original = lexical(base, fine, **kwargs, reserve_bytes=5120,
        query=trace['query'], options=trace['options'])
    if [dict(p, packet_hash=evidence_packet.digest(prefix)) for p in prefix] != trace['returned']:
        raise ValueError('Full historical packet did not reproduce')
    remaining = 5120 - original['added_bytes']
    kept = {p['id'] for p in prefix}
    coarse = [pool[mid] for mid in trace['cascade']['coarse']['selected_ids'] if mid in pool]
    corpus = {}
    for candidate in coarse:
        for source in candidate['_packet_item'].get('sources', []):
            if usable(source, trace['user_id']) and source['role'] in ('user', 'assistant'):
                corpus.setdefault(source_identity(source), set()).update(personal_evidence.terms(source['content']))
    frequency = Counter(word for words in corpus.values() for word in words)
    rare_limit = max(1, len(corpus)//10)
    option_frequency = Counter(word for option in trace['options'] for word in personal_evidence.terms(option))
    represented = set().union(*(personal_evidence.terms(s['content']) for p in prefix
        for s in p['sources'] if usable(s, trace['user_id'])))
    qualified = {w for w, n in frequency.items() if n <= rare_limit and len(w) >= 4
        and w not in represented and option_frequency[w] > 0}
    eligible, inventory = [], []
    for candidate in coarse:
        mid = candidate['id']
        if mid in kept:
            continue
        closure, missing = fine_evidence.closure(candidate, pool)
        if missing or not closure <= kept | {mid}:
            continue
        _, trial = reserve(prefix, [candidate], **kwargs, reserve_bytes=remaining, limit=1)
        if not trial['proposed']:
            continue
        sources = [s for s in candidate['_packet_item']['sources'] if s.get('content')]
        anchors = []
        for source in sources:
            hits = qualified & personal_evidence.terms(source['content'])
            if hits:
                anchors.append(dict(source_identity=list(source_identity(source)),
                    content_span=source.get('content_span'),
                    source_sha256=hashlib.sha256(source['content'].encode()).hexdigest(),
                    hits=sorted(hits)))
        hits = set().union(*(set(a['hits']) for a in anchors))
        scores, matched = {}, {}
        for mode in MODES:
            words = {w for w in hits if (option_frequency[w] >= 2 if mode == 'shared_options'
                else option_frequency[w] == 1)}
            matched[mode] = sorted(words)
            scores[mode] = sum(math.log(1 + (len(corpus)-frequency[w]+.5)/(frequency[w]+.5))
                for w in sorted(words))
        eligible.append(candidate)
        inventory.append(dict(candidate_id=mid, ce_score=candidate['_ce_score'],
            cost=trial['proposed'][0]['added_bytes'], anchors=anchors, scores=scores, matched_terms=matched))
    by_id = {r['candidate_id']: r for r in inventory}
    packets, proposals = {'baseline': prefix}, {}
    for mode in MODES:
        ordered = sorted([c for c in eligible if by_id[c['id']]['scores'][mode] > 0],
            key=lambda c: (-by_id[c['id']]['scores'][mode], -c['_ce_score'], c['id']))
        packets[mode], proposals[mode] = reserve(prefix, ordered, **kwargs, reserve_bytes=remaining, limit=1)
        proposals[mode].update(ordered_ids=[c['id'] for c in ordered], corpus_observations=len(corpus),
            rare_limit=rare_limit, option_frequency=dict(option_frequency),
            source_frequency={w: frequency[w] for w in sorted(qualified)},
            qualified_terms=sorted(qualified))
        if packets[mode][:len(prefix)] != prefix:
            raise ValueError('Original packet changed')
    return dict(packets=packets, proposals=proposals, inventory=inventory, remaining_bytes=remaining)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    with patch('socket.socket.connect', side_effect=AssertionError('Offline shared-option study')):
        score(parser.parse_args(), preparer=prepare, modes=MODES, limitation=__doc__, script_path=Path(__file__))
