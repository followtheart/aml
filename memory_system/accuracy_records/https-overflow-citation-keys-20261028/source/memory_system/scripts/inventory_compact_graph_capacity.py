"""Measure every coarse outsider against compact graph metadata with the original main prefix.

This is a descriptive capacity inventory, not another selector or answer run.
Archived reference-linked carrier identities are joined only after all input
measurements for the cohort have been fixed. No thresholds are changed.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

from batched_records import file_digest, json_records
from audit_retained_porter_https import require, same, selection_code
from probe_assistant_admission_order import replay
import compact_graph_metadata


def read(path):
    content = bytearray()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            content.extend(block)
    return json.loads(content)


def jobs_by_search_id(selections, results):
    fixed = {}
    for row in selections:
        identity = row['conversation_id'], str(row['qa_id'])
        require(identity not in fixed, 'Duplicate fixed question identity')
        fixed[identity] = row
    jobs, identities = {}, set()
    for result in results:
        identity = result['conversation_id'], str(result['qa_id'])
        sid = result['search_id']
        require(sid not in jobs and identity not in identities and identity in fixed,
                'Unexpected or duplicate result identity')
        same(result['packet_hash'], fixed[identity]['packet_hash'], 'Packet join identity')
        jobs[sid] = fixed[identity]
        identities.add(identity)
    same(identities, set(fixed), 'Complete result to fixed-selection join')
    return jobs


def main(args):
    parent = args.experiment.resolve()
    plan = read(parent / 'plan.json')
    for path, digest in plan['input_sha256'].items():
        same(file_digest(Path(path)), digest, 'Frozen experiment input')
    run = Path(plan['cohorts'][args.cohort]['run'])
    snapshot = run.parent
    runtime_plan = read(snapshot / 'plan.json')
    for path, digest in runtime_plan['source_files'].items():
        same(file_digest(snapshot / 'source' / path), digest, 'Frozen runtime')
    sys.path.insert(0, str(snapshot / 'source/memory_system'))
    from app import cascade_rerank, config, fine_evidence, schemas
    modules = cascade_rerank, config, fine_evidence, schemas
    compact_render = compact_graph_metadata.renderer(cascade_rerank.prompts.render)
    code, code_hash = selection_code(cascade_rerank)
    selections = parent / (args.cohort + '-selections.jsonl')
    fixed = read(selections.with_suffix('.manifest.json'))
    same(file_digest(selections), fixed['selection_file_sha256'], 'Fixed selections')
    jobs = jobs_by_search_id(json_records(selections),
        json_records(run / 'completed/data/results/personamem-v2-32k.jsonl'))
    rows, seen = [], set()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline inventory')):
        for trace in json_records(run / 'server-search-debug.jsonl'):
            sid = trace['search_id']
            require(sid in jobs and sid not in seen, 'Trace identity')
            seen.add(sid)
            job = jobs[sid]
            with patch.multiple(config, **trace['versions']['settings']):
                baseline, _, original_bytes = replay(trace, modules, code)
                same([r['id'] for r in baseline['submitted']], job['original_main_ids'], 'Original main IDs')
                same(original_bytes, job['original_bytes'], 'Original main bytes')
                same(baseline['trace']['fine'], trace['cascade']['fine'], 'Original fine metadata')
                fine_ids = {r['id'] for r in baseline['fine']}
                main_ids = {r['id'] for r in baseline['submitted']}
                visible = {r['id']: fine_evidence.selection.fragments(r) for r in baseline['coarse']}
                ordered = sorted((r for r in baseline['coarse']
                    if type(r.get('_ce_score')) in (int, float) and math.isfinite(r['_ce_score'])),
                    key=fine_evidence.key, reverse=True)
                count_limit = config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT
                token_headroom = trace['cascade']['admission_budget']['token_headroom']
                candidates = []
                for rank, candidate in enumerate(ordered, 1):
                    mid = candidate['id']
                    if mid in fine_ids or mid in main_ids:
                        continue
                    with patch.object(cascade_rerank.prompts, 'render', compact_render):
                        request = cascade_rerank.listwise_request(baseline['req'], baseline['plan'],
                                                                  baseline['submitted'] + [candidate])
                    trial_bytes = len(request['prompt'].encode('utf-8'))
                    reasons = []
                    if not visible[mid]: reasons.append('no_visible_source')
                    if fine_evidence.dependencies(candidate): reasons.append('not_singleton')
                    if fine_evidence.selection.represented_by(candidate, baseline['submitted'], visible):
                        reasons.append('represented_source')
                    if len(baseline['submitted']) + 1 > count_limit: reasons.append('count_budget')
                    if trial_bytes > config.RERANK_MAX_PROMPT_BYTES: reasons.append('prompt_budget')
                    if token_headroom is not None and request['reservation'] > token_headroom:
                        reasons.append('token_budget')
                    candidates.append(dict(id=mid, ce_rank=rank, type=candidate.get('type'),
                        reasons=reasons, feasible=not reasons, trial_bytes=trial_bytes,
                        incremental_bytes=trial_bytes-original_bytes, reservation=request['reservation']))
                chosen = next((r for r in candidates if r['feasible']), None)
                recorded = job['proposal']['addition']
                same(chosen['id'] if chosen else None, recorded['id'] if recorded else None,
                     'Independent first feasible choice')
                prefix = job['proposal']['decisions']
                same([r['id'] for r in candidates[:len(prefix)]], [r['id'] for r in prefix], 'Attempt order')
                for candidate, decision in zip(candidates, prefix):
                    same(candidate['reasons'][0] if candidate['reasons'] else 'admitted', decision['reason'],
                         'Original eligibility and budget decisions')
                rows.append(dict(qa_id=job['qa_id'], search_id=sid, original_main_bytes=original_bytes,
                    original_main_count=len(main_ids), byte_limit=config.RERANK_MAX_PROMPT_BYTES,
                    chosen_id=chosen['id'] if chosen else None, candidates=candidates))
    same(seen, set(jobs), 'All cohort traces')
    # Measurements above are complete before linking archived reference carriers.
    inventory_path = run.parents[1] / (args.cohort + '-fine-type-inventory.json')
    inventory = read(inventory_path)
    by_qa = {str(row['qa_id']): row for row in rows}
    linked = []
    for gap in inventory['gap_carrier_observations']:
        job = by_qa[str(gap['qa_id'])]
        candidates = {r['id']: r for r in job['candidates']}
        for carrier in gap['omitted_carriers']:
            candidate = candidates.get(carrier['id'])
            require(candidate is not None, 'Archived scored outsider identity')
            linked.append(dict(qa_id=gap['qa_id'], **candidate,
                original_main_bytes=job['original_main_bytes'], chosen_id=job['chosen_id']))
    cgroup = Path('/sys/fs/cgroup/system.slice/compact-graph-capacity-20261026.service')
    resources = {name: (cgroup/name).read_text() for name in
                 ('memory.peak', 'memory.events', 'memory.max', 'cpu.max', 'memory.swap.max')}
    summary = dict(questions=len(rows), baseline_exact=len(rows), first_feasible_exact=len(rows),
        outsiders=sum(len(r['candidates']) for r in rows),
        feasible=sum(c['feasible'] for r in rows for c in r['candidates']),
        feasible_questions=sum(any(c['feasible'] for c in r['candidates']) for r in rows),
        linked_carriers=len(linked), linked_feasible=sum(r['feasible'] for r in linked),
        linked_reason_counts=dict(Counter(reason for r in linked for reason in r['reasons'])))
    inputs = [Path(__file__), parent/'plan.json', selections, selections.with_suffix('.manifest.json'),
              run/'server-search-debug.jsonl', inventory_path]
    report = dict(status='complete_descriptive_capacity_inventory', cohort=args.cohort, summary=summary,
        rows=rows, linked_carriers=linked, provider_calls=0, runtime_changed=False, resources=resources,
        selection_ast_sha256=code_hash, limitation=__doc__,
        input_sha256={str(p.resolve()): file_digest(p) for p in inputs})
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(cohort=args.cohort, **summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--cohort', choices=('regression', 'validation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
