"""Replay a bounded CE semantic seat through real fine/main admission.

All completed questions are included, including partial CE responses. No new
model calls or downstream judgments; references enter only after selection.
Packet replay freezes historical downstream selections. Overflow proposals use
optimistic admission headroom, so neither is a live final-recall measurement.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, evidence_packet, fine_evidence, listwise_overflow, schemas, search_pipeline
from app.fine_context_lexical import propose as lexical
import audit_https_funnel as audit
from evaluation_model_guard import check_versions
from probe_source_ce import select, records
from probe_coarse_fine_core import admit


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.run / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require completed successful arm')
    results = {r['search_id']: r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')}
    scored = []
    for t in records(args.run / 'server-search-debug.jsonl'):
        r = results[t['search_id']]
        check_versions(t['versions'], 'source trace')
        if r.get('error_stage') or r['fake'] or r['transport'] != 'https' or r['packet_hash'] != t['packet_hash']:
            raise ValueError('Incomplete or mismatched result')
        by_id = {c['id']: c for c in t['ranked']}
        coarse = [by_id[mid] for mid in t['cascade']['coarse']['selected_ids']]
        variants = {}
        original_propose = fine_evidence.propose
        before = copy.deepcopy(coarse)
        with patch.multiple(config, **t['versions']['settings']), patch('socket.socket.connect', side_effect=AssertionError('Offline')):
            for name, enabled in [('baseline', False), ('semantic', True)]:
                captured = []
                def propose(*a, **kw):
                    value = original_propose(*a, **kw, semantic_reserve=enabled)
                    captured.append(value[1])
                    return value
                with patch.object(fine_evidence, 'propose', side_effect=propose):
                    value = select(t, coarse)
                if len(captured) != 1:
                    raise ValueError('Expected one fine selection')
                value['selection'] = captured[0]
                main, size, omissions = admit(t, value['fine'], captured[0], with_omissions=True)
                assert main == value['main'] and size == value['main_bytes']
                main_ids = {c['id'] for c in main}
                selected = [by_id[mid] for mid in t['cascade']['listwise']['selected_ids'] if mid in main_ids]
                limits = t['cascade']['admission_budget']
                req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'])
                with budget.scope(seconds=limits['deadline_seconds'], calls=limits['call_headroom'],
                                  tokens=limits['token_headroom']):
                    pending, groups, decisions, metadata = listwise_overflow.proposal(
                        req, value['fine'], main, selected, omissions)
                if not enabled:
                    assert [c['id'] for c in pending] == t['cascade']['overflow']['proposed_ids']
                value['overflow_input'] = selected + pending
                value['overflow_proposal'] = dict(ids=[c['id'] for c in pending], decisions=decisions)
                context = t['cascade']['conversation_context']
                if context['policy'] != 'fine_user_lexical_context_v1':
                    raise ValueError('Require historical lexical context policy')
                base_packet = [dict(p, packet_hash=context['base_packet_hash'])
                               for p in t['returned'][:len(context['base_packet_ids'])]]
                assert evidence_packet.digest(base_packet) == context['base_packet_hash']
                available = {c['id']: c for c in search_pipeline._foresight_filter(
                    copy.deepcopy(t['ranked']), copy.deepcopy(t['plan']))
                    if c.get('view_status') != 'stale' and '_packet_item' in c}
                fine_context = [available[c['id']] for c in value['fine']
                                if c['id'] in available and c['id'] not in t['selection']['selected_ids']]
                packet, context_trace = lexical(base_packet, fine_context, user_id=t['user_id'],
                    top_k=t['top_k'], byte_budget=t['coverage_manifest']['token_budget'],
                    reserve_bytes=5120, query=t['query'], options=t['options'])
                if not enabled:
                    digest = evidence_packet.digest(packet)
                    assert [dict(p, packet_hash=digest) for p in packet] == t['returned']
                value['fixed_packet'] = [dict(id=p['id'], _packet_item=p) for p in packet]
                value['fixed_packet_context'] = context_trace
                variants[name] = value
        base, new = variants['baseline'], variants['semantic']
        if ([c['id'] for c in base['fine']] != t['cascade']['fine']['selected_ids'] or
                [c['id'] for c in base['main']] != t['cascade']['listwise']['candidate_ids'] or
                base['main_bytes'] != t['cascade']['listwise']['input_bytes']):
            raise ValueError('Historical fine/main not reproduced: ' + r['qa_id'])
        assert before == coarse
        assert new['selection']['adaptive']['core_ids'] == base['selection']['adaptive']['core_ids']
        assert len(new['fine']) <= base['selection']['adaptive']['capacity']
        assert new['main_bytes'] <= t['versions']['settings']['RERANK_MAX_PROMPT_BYTES']
        scored.append(dict(qa_id=r['qa_id'], t=t, variants=variants))
    if len(scored) != len(results) or len({r['qa_id'] for r in scored}) != len(results):
        raise ValueError('Cohort mismatch')
    # Selection and admission are fixed before reference scoring.
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    raw = {}
    for event in records(args.run / 'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'], {}).update({audit.identity(s): s for s in event['source_messages']})
    rows = []
    for record in scored:
        snippet = gold[int(record['qa_id'])]['related_conversation_snippet']
        try:
            snippet = json.loads(snippet)
        except ValueError:
            snippet = ast.literal_eval(snippet)
        variants = {}
        for name, value in record['variants'].items():
            item = dict(main_bytes=value['main_bytes'], selection=value['selection'],
                        fixed_packet_context=value['fixed_packet_context'], overflow_proposal=value['overflow_proposal'])
            for stage in ('fine', 'main', 'overflow_input', 'fixed_packet'):
                cards = [dict(s, text=s.get('content', ''), id=f'{i}:{j}')
                         for i, c in enumerate(value[stage])
                         for j, s in enumerate(c['_packet_item'].get('sources', []))]
                item[stage] = dict(ids=[c['id'] for c in value[stage]])
                for role in ('user', 'assistant'):
                    item[stage][role] = [v['full'] for v in audit.turn_coverage(
                        [audit.norm(s['content']) for s in snippet if s['role'] == role],
                        cards, raw[record['t']['user_id']], role=role)]
            variants[name] = item
        rows.append(dict(qa_id=record['qa_id'], ce_status=record['t']['cascade']['cross_encoder']['status'], variants=variants))
    summary = dict(questions=len(rows), reproduced_fine_and_main=len(rows),
        partial_ce_questions=[r['qa_id'] for r in rows if r['ce_status'] != 'ok'],
        activated=[r['qa_id'] for r in rows if r['variants']['semantic']['selection']['adaptive']['semantic_reserve']['used']],
        variants={})
    for name in ('baseline', 'semantic'):
        summary['variants'][name] = {}
        for stage in ('fine', 'main', 'overflow_input', 'fixed_packet'):
            summary['variants'][name][stage] = {}
            for role in ('user', 'assistant'):
                summary['variants'][name][stage][role] = dict(
                    reference_questions=sum(any(r['variants'][name][stage][role]) for r in rows),
                    turns=sum(sum(r['variants'][name][stage][role]) for r in rows),
                    gained=[r['qa_id'] for r in rows if any(r['variants'][name][stage][role]) and not any(r['variants']['baseline'][stage][role])],
                    lost=[r['qa_id'] for r in rows if any(r['variants']['baseline'][stage][role]) and not any(r['variants'][name][stage][role])],
                    turn_losses=[r['qa_id'] for r in rows if any(a and not b for a, b in zip(r['variants']['baseline'][stage][role], r['variants'][name][stage][role]))])
    with args.output.open('x') as f:
        json.dump(dict(summary=summary, rows=rows, limitation=__doc__,
            trace_sha256=hashlib.sha256((args.run/'server-search-debug.jsonl').read_bytes()).hexdigest(),
            implementation_sha256=hashlib.sha256(Path(fine_evidence.__file__).read_bytes()).hexdigest()), f, indent=2)
        f.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
