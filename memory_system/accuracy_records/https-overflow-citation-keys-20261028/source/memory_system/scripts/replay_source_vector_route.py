"""Diagnostic source-level recall using original queries and a fixed packet budget.

Uses only previously measured source vectors. No gold-conditioned source picks,
new summaries, facts or translations; downstream reader remains unchanged.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, config, eval_scoring, evidence_packet
from replay_query_context import records


def validate_probe_granularity(probe):
    """Window keys may rank sources, but the reader always gets original messages."""
    if not probe['user_only']:
        raise ValueError('Require original user messages')
    if not probe['sentence_window']:
        return
    if probe.get('source_aggregation') != 'max_window_per_source':
        raise ValueError('Window ranks must be collapsed per source before replay')
    sources, documents = probe['sources'], probe['documents']
    for document in documents:
        si, start, end = (document.get(k) for k in ('source_index', 'start', 'end'))
        if (type(si) is not int or not 0 <= si < len(sources)
                or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(sources[si]['content'])
                or document['text'] != sources[si]['content'][start:end]):
            raise ValueError('Window must be an exact original source span')
    for kinds in probe['rankings'].values():
        for kind in ('all', 'user'):
            seen, previous = set(), math.inf
            for hit in kinds[kind]:
                di, si, score = hit['document_index'], hit['source_index'], hit['score']
                if (type(di) is not int or not 0 <= di < len(documents)
                        or documents[di]['source_index'] != si or si in seen
                        or not isinstance(score, (int, float)) or not math.isfinite(score)
                        or score > previous or sources[si]['role'] != 'user'):
                    raise ValueError('Invalid collapsed window ranking')
                previous = score
                seen.add(si)
            if len(seen) != len(sources):
                raise ValueError('Collapsed ranking must cover every source exactly once')


def source_route(probe, question, packet, byte_limit):
    scores = {}
    for query in dict.fromkeys(q['text'] for q in question['query_specs']):
        hits = probe['rankings'][query]['user'][:6]
        if len({h['source_index'] for h in hits}) != len(hits):
            raise ValueError('Route requires whole-message ranks without duplicate source units')
        for rank, hit in enumerate(hits, 1):
            index = hit['source_index']
            scores[index] = scores.get(index, 0) + 1 / (60 + rank)
    selected = sorted(scores, key=lambda i: (-scores[i], i))[:8]
    additions = []
    for index in selected:
        source = copy.deepcopy(probe['sources'][index])
        if source['role'] != 'user' or not source['amu_ids'] or not source['content']:
            raise ValueError('Source is not eligible original user evidence')
        linked_ids = source.pop('amu_ids')
        identity = json.dumps([source['request_id'], source['message_index']], separators=(',', ':'))
        additions.append(dict(id='source_route_' + hashlib.sha256(identity.encode()).hexdigest()[:20],
            content=answer_context.with_evidence('Original user message (retrieval evidence, not a verified personal fact).', [source]),
            memory_type='episode', sources=[source], source_count=1, personal_evidence=None,
            is_constraint=False, equivalent_ids=[], coverage_ids=[], score=None,
            score_kind='source_vector_diagnostic', linked_memory_ids=linked_ids))
    constraints = [copy.deepcopy(m) for m in packet if m.get('is_constraint')]
    rest = [copy.deepcopy(m) for m in packet if not m.get('is_constraint')]
    packed, digest, manifest = evidence_packet.pack(constraints + additions + rest, 50, byte_limit)
    if not {m['id'] for m in constraints} <= {m['id'] for m in packed}:
        raise ValueError('Source route displaced a required forget constraint')
    answer_context.build(packed)
    return packed, dict(selected_source_indices=selected, packet_hash=digest, manifest=manifest,
                        admitted_source_ids=[m['id'] for m in packed if m['id'].startswith('source_route_')])


async def main(args):
    if config.FAKE:
        raise ValueError('Real reader required')
    probe = json.loads(args.probe.read_text())
    validate_probe_granularity(probe)
    results = {r['search_id']: r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')}
    wanted = {q['search_id'] for q in probe['questions']}
    traces = {t['search_id']: t for t in records(args.run / 'server-search-debug.jsonl') if t.get('search_id') in wanted}
    questions = {(d['conversation_id'], q['id']): q for d in records(args.prepared) for q in d['qa']}
    source_hash = hashlib.sha256(json.dumps([s['content'] for s in probe['sources']], ensure_ascii=False).encode()).hexdigest()
    # Verify the source universe is the same one embedded by the probe.
    if source_hash != probe['source_text_sha256']:
        raise ValueError('Probe source content changed')
    with args.output.open('x') as out:
        for index, question in enumerate(probe['questions']):
            result = results[question['search_id']]
            trace = traces[question['search_id']]
            if (trace['user_id'] != probe['user_id'] or trace['query_specs'] != question['query_specs']
                    or result['qa_id'] != question['qa_id']
                    or evidence_packet.digest(trace['returned']) != result['packet_hash']):
                raise ValueError('Frozen question/query/packet mismatch')
            treatment, route = source_route(probe, question, trace['returned'], trace['evidence_token_budget'])
            qa = questions[result['conversation_id'], result['qa_id']]
            for repetition in range(args.repeat):
                for enabled in ([False, True] if (index + repetition) % 2 == 0 else [True, False]):
                    packet = treatment if enabled else trace['returned']
                    with patch.object(config, 'CHOICE_QUERY_CONTEXT', bool(args.query_context and enabled)):
                        prediction, score, diagnostics = await eval_scoring.evaluate(qa, copy.deepcopy(packet))
                    row = dict(qa_id=result['qa_id'], repetition=repetition+1, enabled=enabled,
                        source_sentence_window=probe['sentence_window'],
                        source_aggregation=probe.get('source_aggregation', 'whole_message'),
                        prediction=prediction, score=score, diagnostics=diagnostics,
                        query_context=bool(args.query_context and enabled),
                        baseline_packet_hash=result['packet_hash'], packet_hash=evidence_packet.digest(packet),
                        route=route if enabled else None,
                        caveat='Selected recall/packet diagnostic cases. All reader calls live. '
                        'Source vectors and query generation costs omitted; no HTTPS accuracy claim. '
                        'Top6/query, RRF60, top8 original sources before original ranked packet, same byte budget.')
                    out.write(json.dumps(row, ensure_ascii=False)+'\n');out.flush()
                    print(json.dumps({k:row[k] for k in ('qa_id','repetition','enabled','prediction','score')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run','probe','prepared','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--query-context', action='store_true', help='Enable existing question-only source focus in treatment')
    p.add_argument('--repeat', type=int, choices=(1,2,3), default=1)
    asyncio.run(main(p.parse_args()))
