"""Targeted explanatory capacity checks on archived reference-linked carriers.

This intentionally uses already-known carrier identities to explain a rejected
full-cohort experiment. It does not select a new variant, score answers, or claim
that linkage alone delivers an entire reference turn. Both original-main and
actual-main-after-one budgets are measured without changing any thresholds.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch

from batched_records import file_digest, json_records
from audit_retained_porter_https import require, same, selection_code
from probe_assistant_admission_order import replay
from inventory_main_residual_capacity import jobs_by_search_id
import local_candidate_ids


def read(path):
    data = bytearray()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            data.extend(block)
    return json.loads(data)


def main(args):
    experiment = args.experiment.resolve()
    plan = read(experiment / 'plan.json')
    for name,digest in {**plan['source_sha256'], **plan['input_sha256']}.items():
        same(file_digest(Path(name)), digest, 'Frozen experiment')
    run = Path(plan['cohorts'][args.cohort]['run'])
    original_plan = read(run.parent / 'plan.json')
    for name,digest in original_plan['source_files'].items():
        same(file_digest(run.parent / 'source' / name), digest, 'Frozen runtime')
    report_path = experiment / (args.cohort + '.json')
    terminal = read(experiment / 'final-decision.json')
    same(file_digest(report_path), terminal['input_sha256'][str(report_path)], 'Terminal report')
    report = read(report_path)
    jobs = jobs_by_search_id(report['rows'], json_records(run/'completed/data/results/personamem-v2-32k.jsonl'))
    inventory_path = run.parents[1] / (args.cohort + '-fine-type-inventory.json')
    inventory = read(inventory_path)
    targets = {str(row['qa_id']): row['omitted_carriers'] for row in inventory['gap_carrier_observations']}
    sys.path.insert(0, str(run.parent / 'source/memory_system'))
    from app import cascade_rerank, config, fine_evidence, schemas
    modules = cascade_rerank, config, fine_evidence, schemas
    code, code_hash = selection_code(cascade_rerank)
    seen, rows = set(), []
    with patch('socket.socket.connect', side_effect=AssertionError('Offline targeted diagnosis')):
        for trace in json_records(run/'server-search-debug.jsonl'):
            job = jobs[trace['search_id']]
            qa_id = str(job['qa_id'])
            if qa_id not in targets:
                continue
            require(qa_id not in seen, 'Repeated target trace')
            seen.add(qa_id)
            with patch.multiple(config, **trace['versions']['settings']):
                baseline, _, size = replay(trace, modules, code)
                same(size, job['original_bytes'], 'Original prompt bytes')
                same([c['id'] for c in baseline['submitted']], job['original_main_ids'], 'Original main order')
                same(baseline['trace']['fine'], trace['cascade']['fine'], 'Original fine metadata')
                coarse = {c['id']: c for c in baseline['coarse']}
                actual = [coarse[mid] for mid in job['proposed_main_ids']]
                fine_ids = {c['id'] for c in baseline['fine']}
                before_request, before_audit = local_candidate_ids.request(
                    baseline['req'], baseline['plan'], baseline['submitted'], cascade_rerank)
                actual_request, actual_audit = local_candidate_ids.request(
                    baseline['req'], baseline['plan'], actual, cascade_rerank)
                same(len(before_request['prompt'].encode()), job['compact_original_bytes'], 'Encoded original bytes')
                same(len(actual_request['prompt'].encode()), job['proposed_bytes'], 'Actual encoded bytes')
                same(before_audit, job['display_audit']['original'], 'Original display audit')
                same(actual_audit, job['display_audit']['proposed'], 'Actual display audit')
                visible = {mid: fine_evidence.selection.fragments(c) for mid,c in coarse.items()}
                ranked = sorted(coarse.values(), key=fine_evidence.key, reverse=True)
                ranks = {c['id']: rank for rank,c in enumerate(ranked,1)}
                for carrier in targets[qa_id]:
                    candidate = coarse[carrier['id']]
                    same(ranks[candidate['id']], carrier['ce_rank'], 'Archived carrier rank')
                    require(candidate['id'] not in fine_ids, 'Archived fine omission')
                    measurements = {}
                    for name,prefix in (('original_main',baseline['submitted']),('after_actual_addition',actual)):
                        present = candidate['id'] in {c['id'] for c in prefix}
                        trial = prefix if present else prefix + [candidate]
                        request,audit = local_candidate_ids.request(baseline['req'],baseline['plan'],trial,cascade_rerank)
                        reasons = []
                        if not present:
                            if not visible[candidate['id']]: reasons.append('no_visible_source')
                            if fine_evidence.dependencies(candidate): reasons.append('not_singleton')
                            if fine_evidence.selection.represented_by(candidate,prefix,visible):
                                reasons.append('represented_source')
                        if len(trial) > config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT:
                            reasons.append('count_budget')
                        trial_bytes = len(request['prompt'].encode())
                        if trial_bytes > config.RERANK_MAX_PROMPT_BYTES: reasons.append('prompt_budget')
                        headroom = trace['cascade']['admission_budget']['token_headroom']
                        if headroom is not None and request['reservation'] > headroom: reasons.append('token_budget')
                        missing = local_candidate_ids.missing_graph(before_audit['decoded_graph'],audit['decoded_graph'])
                        if missing: reasons.append('original_graph_context_loss')
                        measurements[name] = dict(already_present=present,feasible=not reasons,
                            reasons=reasons,trial_bytes=trial_bytes,trial_count=len(trial),
                            prompt_excess_bytes=max(0,trial_bytes-config.RERANK_MAX_PROMPT_BYTES),
                            missing_original_graph=missing)
                    rows.append(dict(qa_id=qa_id,**carrier,actual_addition=job['proposal']['addition'],
                        actual_main_reference_turns=job['reference_turns']['proposed'],measurements=measurements))
    same(seen,set(targets),'All archived target questions')
    summary = dict(target_questions=len(seen),linked_carriers=len(rows),provider_calls=0,
        original_main_feasible=sum(r['measurements']['original_main']['feasible'] for r in rows),
        after_one_feasible=sum(r['measurements']['after_actual_addition']['feasible'] for r in rows),
        already_selected=sum(r['measurements']['after_actual_addition']['already_present'] for r in rows),
        original_main_reasons=dict(Counter(reason for r in rows for reason in r['measurements']['original_main']['reasons'])),
        after_one_reasons=dict(Counter(reason for r in rows for reason in r['measurements']['after_actual_addition']['reasons'])))
    inputs = [experiment/'plan.json',experiment/'final-decision.json',report_path,inventory_path,
              run/'server-search-debug.jsonl',Path(__file__)]
    result = dict(status='complete_targeted_explanatory_capacity_diagnosis',cohort=args.cohort,
        summary=summary,rows=rows,limitation=__doc__,runtime_changed=False,provider_calls=0,
        selection_ast_sha256=code_hash,input_sha256={str(p):file_digest(p) for p in inputs})
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(cohort=args.cohort,**summary)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--cohort',choices=('regression','validation'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
