"""Reconstruct every joint source admission from its own live coarse inputs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batched_records import file_digest, json_records, physical_records
from audit_retained_porter_https import selection_code, same, require
from probe_assistant_admission_order import replay
from app import cascade_rerank, config, fine_evidence, schemas
from app import joint_source_admission as execution, joint_source_selection as selection


def read(path):
    with path.open('rb') as f:
        return json.loads(b''.join(iter(lambda: f.read(4*1024*1024), b'')))


def main(args):
    run = args.run.resolve()
    plan = read(run.parent/'plan.json')
    require(plan['toggle'] == 'JOINT_SOURCE_RANK_ADMISSION', 'Correct experiment')
    for name, sha in plan['source_files'].items():
        same(file_digest(run.parent/'source'/name), sha, 'Frozen source')
    inputs = [run.parent/'plan.json', run.parent/'fixture.json',
              run/'completed/data/results/personamem-v2-32k.jsonl',
              run/'server-search-debug.jsonl', run/'server.log']
    input_sha256 = {str(p): file_digest(p) for p in inputs}
    code, _ = selection_code(cascade_rerank)
    results = {r['search_id']: r for r in json_records(run/'completed/data/results/personamem-v2-32k.jsonl')}
    rows, seen = [], set()
    batches = successes = 0
    with patch('socket.socket.connect', side_effect=AssertionError('Offline HTTPS admission audit')):
        for trace in json_records(run/'server-search-debug.jsonl'):
            sid = trace['search_id']
            require(sid in results and sid not in seen, 'Unique terminal search')
            seen.add(sid)
            c = trace['cascade']
            details = c.get('joint_source_admission')
            with patch.multiple(config, **trace['versions']['settings']):
                baseline, _, old_bytes = replay(trace, (cascade_rerank, config, fine_evidence, schemas), code)
                fine, main, working = baseline['fine'], baseline['submitted'], baseline['plan']
                if not config.JOINT_SOURCE_RANK_ADMISSION:
                    require(details is None, 'Disabled hook not executed')
                else:
                    same(details['original_fine_ids'], [v['id'] for v in fine], 'Own baseline fine')
                    same(details['original_main_ids'], [v['id'] for v in main], 'Own baseline main')
                    if 'scheduling' in details:
                        job = selection.prepare(baseline['req'], working, baseline['coarse'], main,
                                                details['original_headroom'], cascade_rerank)
                        same(job['ce_query'], details['query'], 'Actual source query')
                        same(job['protected_ids'], details['protected_score_ids'], 'Score pin set')
                        same(list(job['source']['views']), details['source_view_ids'], 'Complete source namespace')
                        same(job['source']['members'], details['source_members'], 'Source membership')
                        same(job['main_reservation'], details['main_reservation'], 'Main reservation')
                        scores = {}
                        for index, batch in enumerate(details['batches']):
                            keys = job['batches'][index]
                            same(batch['index'], index, 'Sequential unique batch')
                            same(batch['view_ids'], keys, 'Actual ordered batch IDs')
                            texts = [job['source']['views'][k]['text'] for k in keys]
                            same(execution.body_hash(job['ce_query'], texts), batch['body_sha256'], 'Full request body')
                            same(batch['bound'], job['batch_bounds'][index], 'Input bound')
                            batches += 1
                            if batch['status'] == 'ok':
                                successes += 1
                                require(len(batch['scores']) == len(keys), 'Complete score mapping')
                                scores.update(zip(keys, batch['scores'], strict=True))
                            else:
                                require(batch['status'] == 'error', 'Terminal batch failure')
                        if details['status'] == 'selected':
                            require(len(details['batches']) == len(job['batches'])
                                    and all(b['status'] == 'ok' for b in details['batches']), 'Complete source scoring')
                            require(all(type(v) in (float, int) and math.isfinite(v) for v in scores.values()), 'Finite scores')
                            same(details['original_stages']['fine'], baseline['trace']['fine'], 'Original fine diagnostics')
                            # Audit deterministic membership without spending a second live deadline.
                            # The actual execution's elapsed time and post-selection headroom are
                            # checked separately below; this patch is confined to an offline process.
                            with patch('time.monotonic', return_value=0):
                                proposal = selection.select(baseline['req'], working, baseline['coarse'], fine, main,
                                    c['coarse'], baseline['carried_ids'], job, scores,
                                    details['selection_headroom'], cascade_rerank)
                            same(proposal['status'], 'selected', 'Reconstructed selection')
                            for key in ('proposal', 'guard', 'decisions', 'rejections', 'mandatory_main_ids', 'protected_fine_ids'):
                                same(proposal[key], details[key], 'Actual '+key)
                            require(details['selection_seconds'] <= max(0, details['selection_headroom']['deadline_seconds']
                                    - config.RERANK_DEADLINE_SECONDS), 'Selection time reserved for main')
                            require(details['remaining_budget']['call_headroom'] >= 1, 'Main call retained')
                            require(details['remaining_budget']['token_headroom'] >= proposal['request']['reservation'], 'Main tokens retained')
                            fine, main, working = proposal['fine'], proposal['submitted'], proposal['plan']
                            same(c['fine']['output_count'], len(fine), 'Actual fine count')
                            same({o['id'] for o in c['omitted'] if o['stage'] == 'fine'},
                                 {v['id'] for v in baseline['coarse']} - {v['id'] for v in fine}, 'Fine omissions')
                    if details['status'] != 'selected':
                        same(c['fine'], baseline['trace']['fine'], 'Fallback fine diagnostics')
                same(c['fine']['selected_ids'], [v['id'] for v in fine], 'Actual fine membership/order')
                same(c['listwise']['candidate_ids'], [v['id'] for v in main], 'Actual main membership/order')
                request = cascade_rerank.listwise_request(baseline['req'], working, main)
                actual_bytes = len(request['prompt'].encode())
                same(actual_bytes, c['listwise']['input_bytes'], 'Actual whole main prompt bytes')
                require(actual_bytes <= config.RERANK_MAX_PROMPT_BYTES and len(main) <= 24 and len(fine) <= 24, 'Original caps')
                rows.append(dict(search_id=sid, qa_id=results[sid]['qa_id'], baseline_exact=True,
                    status=details['status'] if details else 'disabled', original_bytes=old_bytes,
                    actual_bytes=actual_bytes, fine_ids=[v['id'] for v in fine], main_ids=[v['id'] for v in main]))
    same(seen, set(results), 'Complete terminal coverage')
    ledger = []
    for record in physical_records(run/'server.log'):
        text = record.decode(errors='replace')
        if 'provider_call ' in text and 'stage=search.joint_source_rerank ' in text:
            fields = dict(re.findall(r'(\w+)=([^\s]+)', text))
            require(fields['model'] == 'qwen3-rerank' and fields['kind'] == 'rerank'
                    and fields['attempt'] == '1' and fields['retries'] == '0', 'Fixed source provider policy')
            ledger.append(fields)
    require(len(ledger) <= batches, 'No untracked source retries')
    require(sum(r['status'] == 'ok' for r in ledger) >= successes, 'Successful source batches accounted')
    summary = dict(questions=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
        source_batches=batches, source_calls=len(ledger), source_reported_tokens=sum(int(r['total_tokens']) for r in ledger),
        source_provider_errors=sum(r['status'] != 'ok' for r in ledger),
        failed_without_trustworthy_usage=sum(r['status'] != 'ok' and int(r['total_tokens']) == 0 for r in ledger))
    for path, sha in input_sha256.items():
        same(file_digest(Path(path)), sha, 'Audit input changed during replay')
    with (run/'joint-source-audit.json').open('x') as f:
        json.dump(dict(status='passed', summary=summary, rows=rows, provider_calls_by_audit=0,
            input_sha256=input_sha256,
            limitation='Actual admissions and budgets; no causal or final answer correctness claim. Unknown failed usage is not zero cost.'), f, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args())
