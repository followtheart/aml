"""Locate fine/main transport losses in an audited complete CE diagnostic.

Offline reference attribution only. This script proposes no retrieval rule and
never submits reference text to a provider. Candidate-level full-turn carriers
are distinguished from coverage assembled across multiple candidates.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import evidence_selection, fine_evidence, rerank_query
import audit_https_funnel as funnel
from evaluate_recall_context import reference_turns
from probe_full_option_ce import sha, base


def analyze(directory, gold_csv):
    audit = json.loads((directory / 'audit.json').read_text())
    assert audit['status'] == 'passed'
    for name, expected in audit['inputs'].items():
        assert sha(Path(name)) == expected, name
    plan = json.loads((directory / 'plan.json').read_text())
    report = json.loads((directory / 'result.json').read_text())
    run = Path(plan['run'])
    assert sha(run / 'server-search-debug.jsonl') == plan['trace_sha256']
    assert sha(Path(plan['prepared'])) == plan['data_sha256']
    traces = {t['search_id']: t for t in base.records(run / 'server-search-debug.jsonl')}
    conversations = {c['conversation_id']: c for c in base.records(Path(plan['prepared']))}
    events = {(e['qa_id'], e['arm']): e for e in base.records(directory / 'responses.jsonl')}
    raw = {}
    with sqlite3.connect((run / 'memory.db').resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            row = dict(row)
            raw.setdefault(row['user_id'], {})[funnel.identity(row)] = row
    csv.field_size_limit(32 * 1024 * 1024)
    with gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    rows = []
    for job, result in zip(plan['jobs'], report['rows']):
        assert job['qa_id'] == result['qa_id']
        t = traces[job['search_id']]
        conv = conversations[job['conversation_id']]
        q = next(q for q in conv['qa'] if q['id'] == job['qa_id'])
        turns = reference_turns(q, conv['source_history'], gold[int(job['qa_id'])])
        lookup = {c['id']: c for c in t['ranked']}
        coarse = [lookup[mid] for mid in job['candidate_ids']]
        visible = {c['id']: evidence_selection.fragments(c) for c in coarse}
        user_raw = raw[t['user_id']]
        def cover(items):
            cards = [dict(s, text=s.get('content', ''), id=f'{i}:{j}') for i, c in enumerate(items)
                     for j, s in enumerate(c['_packet_item'].get('sources', []))]
            return [v['full'] for v in funnel.turn_coverage(turns, cards, user_raw)]
        def cover_raw():
            cards = [dict(s, text=s['content'], id=str(i)) for i, s in enumerate(user_raw.values())]
            return [v['full'] for v in funnel.turn_coverage(turns, cards, user_raw)]
        scores = {}
        orders = {}
        core_pools = {}
        for arm in ('control', 'candidate'):
            values = events[job['qa_id'], arm]['scores'] if job['changed'] else [c['_ce_score'] for c in coarse]
            scores[arm] = dict(zip(job['candidate_ids'], values))
            orders[arm] = {mid: index + 1 for index, mid in enumerate(sorted(job['candidate_ids'],
                key=lambda mid: (scores[arm][mid], lookup[mid].get('_fused', 0)), reverse=True))}
            settings = t['versions']['settings']
            baseline = set(t['cascade']['coarse'].get('baseline_ids', job['candidate_ids']))
            extras = sorted(set(job['candidate_ids']) - baseline, key=lambda mid: orders[arm][mid])
            core_pools[arm] = (baseline | set(extras[:1]) if settings['COARSE_EVIDENCE_EXTEND']
                              and settings['FINE_COARSE_CORE_QUOTA'] else set(job['candidate_ids']))
        decisions = {d['id']: d for d in t['cascade']['fine']['selection']['decisions']}
        requirements = t['coverage_manifest']['coverage']['requirements']
        sparse = fine_evidence.sparse_matches(coarse, requirements, visible)
        carriers = []
        for c in coarse:
            coverage = cover([c])
            if not any(coverage):
                continue
            carrier = dict(id=c['id'], type=c['type'], full_turns=coverage,
                rank_text_bytes=len(c.get('_rank_text', c['content']).encode()),
                historical_decision=decisions[c['id']],
                lexical_requirement_ids=sorted(fine_evidence.relevance(c, requirements, visible[c['id']])),
                sparse_requirement_ids=sorted(sparse[c['id']]),
                in_historical_core_pool=c['id'] in t['cascade']['fine']['selection']['adaptive'].get('core_pool_ids', job['candidate_ids']),
                visible_source_count=len(visible[c['id']]), arms={})
            for arm in ('control', 'candidate'):
                fine_ids = result['variants'][arm]['fine']['ids']
                chosen = [lookup[mid] for mid in fine_ids]
                carrier['arms'][arm] = dict(ce_rank=orders[arm][c['id']], ce_score=scores[arm][c['id']],
                    in_core_pool=c['id'] in core_pools[arm],
                    in_fine=c['id'] in fine_ids,
                    in_main=c['id'] in result['variants'][arm]['main']['ids'],
                    represented_by=evidence_selection.represented_by(c, chosen, visible) if c['id'] not in fine_ids else [])
            carriers.append(carrier)
        rows.append(dict(qa_id=job['qa_id'], conversation_id=job['conversation_id'], changed=job['changed'],
            query=q['question'], explicit_premises=sum(bool(rerank_query._premise(o)) for o in q['options']),
            raw=cover_raw(), coarse=cover(coarse), fine_capacity=t['cascade']['fine']['selection']['adaptive']['capacity'],
            fine_size={arm:len(result['variants'][arm]['fine']['ids']) for arm in ('control', 'candidate')},
            main_bytes={arm:result['variants'][arm]['main_bytes'] for arm in ('control','candidate')},
            variants={arm:{stage:result['variants'][arm][stage]['user'] for stage in ('fine','main')}
                      for arm in ('control', 'candidate')}, carriers=carriers))
    summary = dict(questions=len(rows), reference_available=sum(any(r['raw']) for r in rows),
                   coarse_reference_questions=sum(any(r['coarse']) for r in rows), arms={})
    for arm in ('control', 'candidate'):
        gaps = [r for r in rows if any(r['coarse']) and not any(r['variants'][arm]['fine'])]
        main_gaps = [r for r in rows if any(r['variants'][arm]['fine']) and not any(r['variants'][arm]['main'])]
        summary['arms'][arm] = dict(fine_gap_questions=[r['qa_id'] for r in gaps],
            main_gap_questions=[r['qa_id'] for r in main_gaps],
            fine_gap_min_carrier_ranks={r['qa_id']: min((c['arms'][arm]['ce_rank'] for c in r['carriers']), default=None) for r in gaps},
            fine_gap_with_spare_capacity=[r['qa_id'] for r in gaps if r['fine_size'][arm]<r['fine_capacity']],
            fine_gap_without_literal_premise=[r['qa_id'] for r in gaps if not r['explicit_premises']])
    summary['historical_fine_gap_carrier_reasons'] = dict(Counter(c['historical_decision']['reason']
        for r in rows if any(r['coarse']) and not any(r['variants']['control']['fine']) for c in r['carriers']))
    return dict(summary=summary, rows=rows, limitation=__doc__,
        inputs={str(p):sha(p) for p in (directory/'audit.json', directory/'result.json',
            run/'server-search-debug.jsonl', gold_csv, Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    args = parser.parse_args()
    result = analyze(args.probe, args.gold_csv)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False))
