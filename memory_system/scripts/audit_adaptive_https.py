"""Audit adaptive admission and per-candidate deletion outcomes in a finished run.

Literal source transport is not semantic entailment. This audit checks runtime
invariants, not whether a model's valid negative judgment is correct.
"""
import argparse
from collections import Counter
import copy
import json
from datetime import datetime
import re
import statistics
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import cascade_rerank, listwise_recovery
import audit_https_funnel as audit


def classify_deletion(mid, lw):
    recovery = lw.get('recovery', {})
    if mid in recovery.get('restored_ids', []):
        return 'review_restored'
    if mid in recovery.get('provisional_ids', []):
        return 'unresolved_provisional'
    if any(j['candidate_id'] == mid and j.get('valid') and not j.get('useful')
           for j in recovery.get('judgments', [])):
        return 'valid_negative'
    if any(r['candidate_id'] == mid for r in recovery.get('represented', [])):
        return 'source_represented'
    if mid in recovery.get('unreviewed_ids', []):
        return 'unresolved_deleted'
    return 'unclassified'


def analyze(run):
    if (run / 'exit-code').read_text().strip() != '0':
        raise ValueError('A successful completed run is required')
    searches = {s['search_id']: s for s in audit.records(run / 'server-search-debug.jsonl')}
    results = list(audit.records(run / 'completed/data/results/personamem-v2-32k.jsonl'))
    stages = {r['qa_id']: r for r in json.loads((run / 'funnel/stage-transport.json').read_text())['rows']}
    rows, violations = [], []
    def check(condition, qid, message):
        if not condition:
            violations.append(dict(qa_id=qid, invariant=message))
    for result in results:
        qid = result['qa_id']
        search = searches[result['search_id']]
        check(search['packet_hash'] == result['packet_hash'], qid, 'search/result packet identity')
        settings = search['versions']['settings']
        c = search['cascade']
        fine, lw = c['fine'], c['listwise']
        adaptive = fine['selection'].get('adaptive', {})
        admission = c.get('adaptive_admission', {})
        by_id = {r['id']: r for r in search['ranked']}
        fine_ids, input_ids = set(fine['selected_ids']), set(lw['candidate_ids'])
        check(set(adaptive.get('core_ids', [])) <= fine_ids, qid, 'fine core preserved')
        check(set(admission.get('core_ids', [])) <= input_ids, qid, 'admitted core preserved')
        check(len(fine_ids) <= settings['CASCADE_FINE_LIMIT'] + settings.get('FINE_EXTRA_LIMIT', 0),
              qid, 'fine count bound')
        check(len(input_ids) <= settings['CASCADE_LLM_LIMIT'] + settings.get('FINE_EXTRA_LIMIT', 0),
              qid, 'listwise count bound')
        for d in admission.get('decisions', []):
            if d['admitted']:
                check(set(d['group_ids']) <= input_ids, qid, 'joint admission is whole')
        plan = copy.deepcopy(search['plan'])
        plan.update(_coverage_requirements=[{k: r[k] for k in ('id', 'text', 'origin')}
                    for r in search['coverage_manifest']['coverage']['requirements']],
                    _fusion_triples=search['fusion_triples'], _fusion_edges=search['fusion_edges'],
                    _fine_path_groups=adaptive.get('path_groups', []))
        request = cascade_rerank.listwise_request(
            SimpleNamespace(query=search['query'], options=search['options']), plan,
            [by_id[mid] for mid in lw['candidate_ids']], shared_sources=lw['shared_sources'],
            keyed=settings.get('LISTWISE_KEYED_OUTPUT', False))
        byte_count = len(request['prompt'].encode()) if input_ids else 0
        check(byte_count == lw['input_bytes'], qid, 'archived request bytes reproduce')
        check(byte_count <= settings['RERANK_MAX_PROMPT_BYTES'], qid, 'prompt byte budget')
        headroom = c.get('admission_budget', {}).get('token_headroom')
        if headroom is not None and input_ids:
            check(request['reservation'] <= headroom, qid, 'whole request token bound')
        recovery = lw.get('recovery', {})
        provisional = set(recovery.get('provisional_ids', []))
        confirmed_policy = recovery.get('deletion_policy') == 'confirmed_negative_only'
        bound = len(input_ids) if confirmed_policy else settings.get('LISTWISE_UNREVIEWED_LIMIT', 0)
        check(len(provisional) <= bound, qid, 'provisional count bound')
        if confirmed_policy:
            check(set(recovery.get('unreviewed_ids', [])) <= set(lw['selected_ids']),
                  qid, 'unresolved candidates never silently deleted')
        check(provisional <= set(recovery.get('unreviewed_ids', [])), qid, 'only unresolved reviews fall back')
        check(not provisional.intersection(recovery.get('reviewed_ids', [])), qid, 'valid decisions not overridden')
        check(provisional <= set(lw['selected_ids']), qid, 'provisional trace agrees with selection')
        if not confirmed_policy:
            for mid in provisional:
                check(bool(listwise_recovery.sources_for(by_id[mid])), qid, 'provisional source fully visible')
        deletions = {mid: classify_deletion(mid, lw) for mid in lw.get('primary_irrelevant_ids', [])}
        stage = stages[qid]
        target_deletions = {mid: deletions[mid] for mid in stage['carrier_ids']['listwise_input']
                            if mid in deletions}
        rows.append(dict(qa_id=qid, score=result['score'], fine_count=len(fine_ids),
            core_count=len(adaptive.get('core_ids', fine_ids)),
            listwise_input_count=len(input_ids), added_input_count=len(input_ids - set(admission.get('core_ids', input_ids))),
            addition_decisions=admission.get('decisions', []),
            whole_request_bound=request['reservation'] if input_ids else 0, token_headroom=headroom,
            provisional_ids=sorted(provisional), deletion_outcomes=deletions,
            target_deletion_outcomes=target_deletions,
            full_turns={k: v['full_turns'] for k, v in stage['visible'].items()}))
    elapsed = sorted((datetime.fromisoformat(s['finished_at']) - datetime.fromisoformat(s['started_at'])).total_seconds()
                     for s in searches.values())
    provider = {}
    for name, prefix in (('server.log', 'search.'), ('output.log', 'eval.')):
        for line in (run / name).read_text().splitlines():
            if 'provider_call ' not in line:
                continue
            fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
            stage = fields.get('stage', '')
            if not stage.startswith(prefix):
                continue
            entry = provider.setdefault(stage, dict(calls=0, total_tokens=0, duration_seconds=0.0))
            entry['calls'] += 1
            entry['total_tokens'] += int(fields.get('total_tokens', 0))
            entry['duration_seconds'] += float(fields.get('duration_s', 0))
    summary = dict(questions=len(rows), violations=violations,
        added_input_candidates=sum(r['added_input_count'] for r in rows),
        provisional_candidates=sum(len(r['provisional_ids']) for r in rows),
        deletion_outcomes=dict(Counter(v for r in rows for v in r['deletion_outcomes'].values())),
        rejected_additions=dict(Counter(d['reason'] for r in rows for d in r['addition_decisions'] if not d['admitted'])),
        fine_count_range=[min(r['fine_count'] for r in rows), max(r['fine_count'] for r in rows)],
        search_seconds=dict(mean=statistics.mean(elapsed), median=statistics.median(elapsed),
                            maximum=max(elapsed)),
        listwise_input_bytes=sum(s['cascade']['listwise']['input_bytes'] for s in searches.values()),
        observed_provider_metrics=provider,
        caveat='Whole-source probes do not prove answer sufficiency; valid negative means schema/citation-valid, not semantically correct.')
    return dict(summary=summary, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    out = analyze(args.run)
    (args.run / 'funnel/adaptive-audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out['summary'], ensure_ascii=False, indent=2))
    if out['summary']['violations']:
        raise SystemExit(1)
