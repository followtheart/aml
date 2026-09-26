"""Offline fine/listwise admission replay; no new scoring or answer predictions."""
import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import cascade_rerank as ranking


def replay(search, fine_limit, llm_limit):
    trace = search['cascade']
    by_id = {r['id']: r for r in search['ranked']}
    coarse = [by_id[mid] for mid in trace['coarse']['selected_ids']]
    assert trace['cross_encoder']['status'] == 'ok', 'Partial CE needs separate reservation replay'
    assert all(c.get('_ce_score') is not None for c in coarse)
    requirements = [{k: r[k] for k in ('id', 'text', 'origin')}
                    for r in search['coverage_manifest']['coverage']['requirements']]
    fused = {r['candidate_id'] for r in trace['fine']['selection']['reservations'] if 'fused_head' in r['reasons']}
    selection = {}
    fine = ranking.shortlist(coarse, fine_limit, requirements,
        lambda c: (c['_ce_score'], c.get('_fused', 0)), fused_ids=fused, trace=selection)
    reservations = [r for r in selection['reservations'] if r['selected']]
    carried = {r['candidate_id'] for r in reservations}
    soft = {r['candidate_id'] for r in reservations if r['reasons'] == ['fused_head']}
    positions = {r['id']: len(fine) - i for i, r in enumerate(fine)}
    ordered = ranking.shortlist(fine, llm_limit, [],
        lambda c: (c['id'] in carried and c['id'] not in soft, positions[c['id']]), carry_ids=carried)
    plan = copy.deepcopy(search['plan'])
    plan.update(_coverage_requirements=requirements, _fusion_triples=search['fusion_triples'],
                _fusion_edges=search['fusion_edges'])
    request = SimpleNamespace(query=search['query'], options=search['options'])
    submitted, rejected = [], []
    limit = search['versions']['settings']['RERANK_MAX_PROMPT_BYTES']
    for row in sorted(ordered, key=lambda c: c['id'] not in carried):
        size = len(ranking.listwise_prompt(request, plan, submitted + [row]).encode())
        if size > limit:
            rejected.append(row['id'])
        else:
            submitted.append(row)
    submitted.sort(key=lambda c: -positions[c['id']])
    return dict(fine_ids=[c['id'] for c in fine], input_ids=[c['id'] for c in submitted],
                prompt_rejected=rejected,
                prompt_bytes=len(ranking.listwise_prompt(request, plan, submitted).encode()) if submitted else 0)


def main(args):
    audit = json.loads((args.run / 'funnel/audit.json').read_text())
    searches = {s['search_id']: s for s in map(json.loads,
        (args.run / 'completed/logs/search-debug.jsonl').read_text().splitlines())}
    results = {r['qa_id']: r for r in map(json.loads,
        (args.run / 'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines())}
    rows = []
    for question in audit['rows']:
        search = searches[results[question['qa_id']]['search_id']]
        settings = search['versions']['settings']
        baseline = replay(search, settings['CASCADE_FINE_LIMIT'], settings['CASCADE_LLM_LIMIT'])
        assert baseline['fine_ids'] == search['cascade']['fine']['selected_ids'], question['qa_id']
        assert baseline['input_ids'] == search['cascade']['listwise']['candidate_ids'], question['qa_id']
        assert baseline['prompt_bytes'] == search['cascade']['listwise']['input_bytes'], question['qa_id']
        candidate = replay(search, args.fine, args.listwise)
        probes = []
        for p in question['probes']:
            carriers = set(p['stage_visible_quote_carriers']['coarse'])
            probes.append(dict(label=p['label'], quote=p['quote'],
                baseline=bool(carriers & set(baseline['input_ids'])),
                candidate=bool(carriers & set(candidate['input_ids']))))
        rows.append(dict(qa_id=question['qa_id'], baseline=baseline, candidate=candidate, probes=probes))
    out = dict(caveat='Frozen CE and candidate rows; baseline fine IDs, listwise IDs and prompt bytes exactly reproduced. Candidate admission only, ignores dynamic token headroom and does not run listwise/packet/answer.',
        fine_limit=args.fine, listwise_limit=args.listwise, rows=rows,
        baseline_visible=sum(p['baseline'] for r in rows for p in r['probes']),
        candidate_visible=sum(p['candidate'] for r in rows for p in r['probes']))
    with args.output.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--fine', type=int, default=16)
    p.add_argument('--listwise', type=int, default=14)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
