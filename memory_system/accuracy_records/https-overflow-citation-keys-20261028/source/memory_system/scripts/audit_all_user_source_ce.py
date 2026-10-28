"""Reconstruct every source CE request, selection and reference transport score.

Offline verification of the complete two-cohort diagnostic. Historical whole
scores versus fresh source scores do not establish a causal answer improvement.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, cross_encoder
import audit_https_funnel as funnel
from probe_source_ce import build_views, records, select
from evaluate_recall_context import reference_turns


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    registration = json.loads((root / 'preregistration.json').read_text())
    for name, digest in registration['script_hashes'].items():
        if sha(root / 'scripts' / name) != digest or sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Study implementation changed: ' + name)
    csv.field_size_limit(32 * 1024 * 1024)
    with Path('data/raw/personamem-v2.csv').open() as stream:
        gold = list(csv.DictReader(stream))
    cohorts = {}
    for arm in ('validation', 'regression'):
        if (root / f'{arm}.exit-code').read_text().strip() != '0':
            raise ValueError('Diagnostic is incomplete: ' + arm)
        plan = json.loads((root / f'{arm}.plan.json').read_text())
        report = json.loads((root / f'{arm}.json').read_text())
        live = records(root / f'{arm}.live.jsonl')
        source = Path(registration['source_runs'][arm])
        if plan['run'] != str(source) and Path(plan['run']).resolve() != source:
            raise ValueError('Source run changed')
        if (plan['source_policy'] != 'all_eligible_complete_user_sources'
                or plan['llm_model'] != 'gpt-4o-mini' or plan['ce_model'] != 'qwen3-rerank'
                or plan['script_sha256'] != registration['script_hashes']['probe_source_ce.py']):
            raise ValueError('Source policy or models changed')
        for name, digest in registration['source_hashes'][arm].items():
            if sha(source / name) != digest:
                raise ValueError('Frozen input changed')
        manifest = json.loads((source / 'isolated-manifest.json').read_text())
        prepared_path = Path(manifest['data'])
        if sha(prepared_path) != plan['prepared_sha256']:
            raise ValueError('Prepared data hash differs')
        conversations = {c['conversation_id']: c for c in records(prepared_path)}
        results = {r['qa_id']: r for r in records(source / 'completed/data/results/personamem-v2-32k.jsonl')}
        traces = {t['search_id']: t for t in records(source / 'server-search-debug.jsonl')}
        expected = registration['expected_questions'][arm]
        if (len(plan['jobs']) != expected or len(report['rows']) != expected or len(results) != expected
                or {j['qa_id'] for j in plan['jobs']} != set(results)):
            raise ValueError('Incomplete cohort')
        expected_calls = [(j['qa_id'], i) for j in plan['jobs'] for i in range(len(j['batches']))]
        if [(r['qa_id'], r['batch']) for r in live] != expected_calls or len(live) != registration['expected_calls'][arm]:
            raise ValueError('Call inventory differs')
        live_by_key = {(r['qa_id'], r['batch']): r for r in live}
        raw = {}
        for event in records(source / 'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'], {}).update({funnel.identity(s): s for s in event['source_messages']})
        for job, row in zip(plan['jobs'], report['rows']):
            if row['qa_id'] != job['qa_id']:
                raise ValueError('Question order mismatch')
            result = results[row['qa_id']]
            trace = traces[result['search_id']]
            coarse_by_id = {c['id']: c for c in trace['ranked']}
            coarse = [coarse_by_id[mid] for mid in trace['cascade']['coarse']['selected_ids']]
            query = trace['cascade']['cross_encoder']['query']
            with patch.multiple(config, **trace['versions']['settings']), patch('socket.socket.connect', side_effect=AssertionError('Offline')):
                views, members, batches, omitted = build_views(coarse, trace['user_id'], query, max_sources=None)
                if (job['query'] != query or job['views'] != list(views.values()) or job['members'] != members
                        or job['batches'] != [[v['id'] for v in batch] for batch in batches] or job['omitted'] != omitted):
                    raise ValueError('Source request did not reconstruct')
                scores = {}
                for i, batch in enumerate(batches):
                    call = live_by_key[job['qa_id'], i]
                    values = call.get('scores', [])
                    if (call.get('error') or call['calls'] != 1 or len(values) != len(batch)
                            or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
                            or call['request_bound'] != cross_encoder.request_size(query, [v['text'] for v in batch])):
                        raise ValueError('Invalid source score/cost record')
                    scores.update(zip(call['view_ids'], values))
                    if call['view_ids'] != [v['id'] for v in batch]:
                        raise ValueError('Provider score mapping changed')
                if scores != row['source_scores'] or set(scores) != set(views):
                    raise ValueError('Incomplete source scores')
                variants = {'baseline': select(trace, coarse)}
                for mode in ('source_only','maximum'):
                    revised = copy.deepcopy(coarse)
                    ranking = {}
                    for candidate in revised:
                        values = [scores[k] for k in members.get(candidate['id'], [])]
                        if values:
                            candidate['_ce_score'] = max(values + ([candidate['_ce_score']] if mode == 'maximum' else []))
                        ranking[candidate['id']] = candidate['_ce_score']
                    if ranking != row['rankings'][mode]:
                        raise ValueError('Candidate score aggregation differs')
                    variants[mode] = select(trace, revised)
            conv = conversations[result['conversation_id']]
            qa = next(q for q in conv['qa'] if q['id'] == row['qa_id'])
            reference_turns(qa, conv['source_history'], gold[int(row['qa_id'])])
            snippet = gold[int(row['qa_id'])]['related_conversation_snippet']
            try:
                snippet = json.loads(snippet)
            except ValueError:
                snippet = ast.literal_eval(snippet)
            for mode, variant in variants.items():
                recorded = row['variants'][mode]
                if recorded['main_bytes'] != variant['main_bytes']:
                    raise ValueError('Main request bytes differ')
                for stage in ('fine', 'main'):
                    if recorded[stage]['ids'] != [c['id'] for c in variant[stage]]:
                        raise ValueError('Selection differs')
                    cards = [dict(s, text=s.get('content',''), id=f'{i}:{j}') for i,c in enumerate(variant[stage])
                             for j,s in enumerate(c['_packet_item'].get('sources', []))]
                    for role in ('user','assistant'):
                        coverage = funnel.turn_coverage([funnel.norm(s['content']) for s in snippet if s['role'] == role], cards, raw[trace['user_id']], role=role)
                        if recorded[stage][role] != [t['full'] for t in coverage]:
                            raise ValueError('Reference transport score differs')
        events = [dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in (root / f'{arm}.log').read_text().splitlines() if 'provider_call ' in line]
        if (len(events) != len(live) or any(e['model'] != 'qwen3-rerank' or e['kind'] != 'rerank' or e['status'] != 'ok' for e in events)
                or sum(int(e['total_tokens']) for e in events) != sum(r['tokens'] for r in live)
                or report['summary']['calls'] != len(live) or report['summary']['tokens'] != sum(r['tokens'] for r in live)):
            raise ValueError('Provider ledger mismatch')
        if report['live'] != live:
            raise ValueError('Final report differs from append-only call log')
        for mode in ('baseline','source_only','maximum'):
            for stage in ('fine','main'):
                recomputed = dict(reference_questions=sum(any(r['variants'][mode][stage]['user']) for r in report['rows']),
                    user_turns=sum(sum(r['variants'][mode][stage]['user']) for r in report['rows']),
                    gained=[r['qa_id'] for r in report['rows'] if any(r['variants'][mode][stage]['user']) and not any(r['variants']['baseline'][stage]['user'])],
                    lost=[r['qa_id'] for r in report['rows'] if any(r['variants']['baseline'][stage]['user']) and not any(r['variants'][mode][stage]['user'])])
                if report['summary']['variants'][mode][stage] != recomputed:
                    raise ValueError('Aggregate coverage differs from per-question records')
        cohorts[arm] = dict(questions=expected, calls=len(live), tokens=report['summary']['tokens'],
            full_requests_reconstructed=True, selections_and_transport_recomputed=True, summary=report['summary'],
            artifact_hashes={name:sha(root / f'{arm}.{name}') for name in ('plan.json','live.jsonl','json','log')})
    return dict(cohorts=cohorts, provider_calls_during_audit=0, auditor_sha256=sha(Path(__file__)), limitation=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.study)
    with (args.study / 'audit.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({arm:{k:v for k,v in value.items() if k not in ('summary','artifact_hashes')} for arm,value in report['cohorts'].items()}))
