"""Extract every supplemental main candidate from a terminal HTTPS arm.

This is review input, not a relevance or transport audit. Hide answer scores,
predictions and reference labels so the first semantic pass reads the question,
options and full visible candidate text. Retain selection paths separately.
Run serially after evaluation and the registered terminal audit queue finish.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    with path.open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'), 'Incomplete terminal record')
            if line.strip():
                yield json.loads(line), hashlib.sha256(line).hexdigest()


def main(args):
    run = args.run.resolve()
    root = run.parent
    require(not args.output.exists(), 'Refusing to overwrite review input')
    plan_path, manifest_path = root / 'plan.json', run / 'isolated-manifest.json'
    plan, manifest = (json.loads(p.read_text()) for p in (plan_path, manifest_path))
    require((run / 'exit-code').exists() and manifest.get('audit_exit_code') == 0
            and manifest.get('server_exit_code') in (0, -15), 'Arm is not terminal and stage audited')
    process = Path('/proc') / str(manifest['server_pid']) / 'cmdline'
    require(not process.exists() or str(root).encode() not in process.read_bytes(), 'Service still live')
    for name, digest in plan['source_files'].items():
        require(sha(root / 'source' / name) == digest, 'Frozen source mismatch: ' + name)
    result_path = run / 'completed/data/results/personamem-v2-32k.jsonl'
    identities = {}
    for result, _ in records(result_path):
        sid = result['search_id']
        require(sid and sid not in identities, 'Missing or duplicate search identity')
        identities[sid] = {k: result[k] for k in ('qa_id', 'conversation_id', 'packet_hash')}
    require(len(identities) == plan['questions'], 'Incomplete cohort')
    trace_path = run / 'server-search-debug.jsonl'
    seen, rows, paths = set(), [], []
    for trace, digest in records(trace_path):
        sid = trace['search_id']
        require(sid in identities and sid not in seen, 'Unexpected or duplicate trace')
        seen.add(sid)
        require(trace['packet_hash'] == identities[sid]['packet_hash'], 'Packet identity mismatch')
        require(all(plan['source_files'].get(name) == value
                    for name, value in trace['versions']['file_hashes'].items()), 'Trace source mismatch')
        require(trace['versions']['settings']['RECALL_PORTER_SUPPLEMENT'], 'Expected treatment arm')
        pool = {c['id']: c for c in trace['ranked']}
        cascade = trace['cascade']
        main = cascade['listwise']
        admitted = set(trace['porter_recall'].get('admitted_ids', []))
        new_main = [mid for mid in main['candidate_ids'] if mid in admitted]
        if not new_main:
            continue
        adaptive = cascade['fine']['selection'].get('adaptive', {})
        coarse = [pool[mid] for mid in cascade['coarse']['selected_ids']]
        ordered = sorted(coarse, key=lambda c: (
            c['_ce_score'] if c.get('_ce_score') is not None else -1e30, c.get('_fused', 0)), reverse=True)
        ranks = {c['id']: i + 1 for i, c in enumerate(ordered)}
        identity = {k: identities[sid][k] for k in ('qa_id', 'conversation_id')}
        candidates, selections = [], []
        for mid in new_main:
            c = pool[mid]
            candidates.append(dict(candidate_id=mid, rank_text=c.get('_rank_text', c.get('content', '')),
                                   review_sources=c.get('_review_sources', [])))
            selections.append(dict(candidate_id=mid, ce_score=c.get('_ce_score'), full_pool_ce_rank=ranks[mid],
                main_position=main['candidate_ids'].index(mid), in_core=mid in adaptive.get('core_ids', []),
                additions=[a for a in adaptive.get('additions', []) if mid in a['ids']],
                primary_relevance=main.get('candidate_relevance', {}).get(mid),
                selected=mid in main['selected_ids'], in_packet=mid in trace['coverage_manifest']['included_ids'],
                recovery=[r for r in main.get('recovery', {}).get('judgments', []) if r.get('candidate_id') == mid]))
        rows.append(dict(**identity, search_id=sid, trace_record_sha256=digest,
                         query=trace['query'], options=trace['options'], candidates=candidates))
        paths.append(dict(**identity, candidates=selections))
    require(seen == set(identities), 'Incomplete trace inventory')
    inputs = [Path(__file__), plan_path, manifest_path, result_path, trace_path]
    report = dict(status='review_input_only', scope='Every admitted Porter supplemental main candidate',
                  cohort_questions=len(seen), questions=len(rows), candidates=sum(len(r['candidates']) for r in rows),
                  answers_and_references_hidden=True, provider_calls=0, rows=rows, selection_paths=paths,
                  input_sha256={str(p): sha(p) for p in inputs}, limitation=__doc__)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'cohort_questions', 'questions', 'candidates')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
