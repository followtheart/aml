"""Freeze diagnostic sources/artifact hashes and check the final study ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main(args):
    root = args.run.resolve()
    project = Path(__file__).resolve().parents[1]
    read = lambda name:json.loads((root/name).read_text())
    before = read('semantic-new-fine-context-proposal.json')
    after = read('semantic-new-fine-context-complete-exchange.json')
    if len(before['rows']) != 56 or len(after['rows']) != 56:
        raise ValueError('Missing residual cases')
    old = {r['qa_id']:r for r in before['rows']}
    for row in after['rows']:
        for mode in ('whole','window2'):
            if row['proposals'][mode] != old[row['qa_id']]['proposals'][mode] or row['coverage'][mode] != old[row['qa_id']]['coverage'][mode]:
                raise ValueError('Exchange-only correction affected other views')
        for mode in ('whole','window2','exchange'):
            if any(a and not b for role in ('user','assistant') for a,b in zip(row['coverage']['baseline'][role],row['coverage'][mode][role])):
                raise ValueError('Residual proposal lost reference content')
            eligible = {v['candidate_id'] for v in row['inventory'] if v['mode']==mode and v['eligible']}
            if not set(row['proposals'][mode]['appended_ids']) <= eligible:
                raise ValueError('Appended ineligible carrier')
    exchange = next(s for s in after['summary'] if s['mode']=='exchange')
    if exchange['gained'] != ['40'] or exchange['appended_questions'] != 18:
        raise ValueError('Unexpected corrected outcome')
    originals = [json.loads(line) for line in (root/'control/completed/data/results/personamem-v2-32k.jsonl').open()]
    q40 = next(r for r in originals if r['qa_id']=='40')
    if q40['score'] != 1 or q40['prediction'] != 'A':
        raise ValueError('Historical Q40 answer differs')
    verification = read('semantic-upstream-verification.json')
    if sha(project/'scripts/verify_semantic_upstream.py') != verification['verifier_sha256']:
        raise ValueError('Verifier source changed')
    frozen = read('plan.json')['source_files']
    for name,expected in frozen.items():
        if name.startswith('memory_system/app/') and sha(project/name.removeprefix('memory_system/')) != expected:
            raise ValueError('Runtime changed')
    events = []
    for directory in ('source-vector-diagnostic','exchange-vector-diagnostic','source-upstream-ce-diagnostic'):
        events += [dict(p.split('=',1) for p in line.split() if '=' in p)
            for line in (root/directory/'provider.log').read_text().splitlines() if 'provider_call ' in line]
    if len(events) != 181 or sum(int(e['total_tokens']) for e in events) != 199165 or any(e['status']!='ok' for e in events):
        raise ValueError('Provider ledger mismatch')
    archive = root/'source-semantic-study-scripts'
    archive.mkdir(exist_ok=False)
    names = ('probe_current_source_vectors.py','probe_source_exchange_vectors.py',
        'probe_semantic_source_context.py','probe_semantic_query_balance.py','audit_current_source_vectors.py',
        'verify_source_semantic_measurements.py','audit_semantic_source_proposals.py','probe_semantic_upstream.py',
        'probe_semantic_upstream_ce.py','verify_semantic_upstream.py','probe_semantic_new_fine_context.py',
        'audit_source_exchange_gaps.py','probe_source_rank_residual.py','probe_source_ce.py',
        'probe_coarse_fine_core.py','audit_https_funnel.py','evaluation_model_guard.py',
        'evaluate_recall_context.py','finalize_semantic_study.py')
    for name in names:
        shutil.copy2(project/'scripts'/name,archive/name)
    paths = []
    for directory in ('source-vector-diagnostic','exchange-vector-diagnostic','source-upstream-diagnostic',
                      'source-upstream-ce-diagnostic','source-semantic-study-scripts'):
        paths.extend(p for p in (root/directory).rglob('*') if p.is_file())
    for pattern in ('semantic-*.json','semantic-*.log','source-*diagnostic*.log','source-exchange-gap-audit.*'):
        paths.extend(root.glob(pattern))
    paths.extend(root/n for n in ('exchange-source-context-proposal.json','exchange-balanced-context-proposal.json'))
    artifacts = {str(p.relative_to(root)):sha(p) for p in sorted(set(paths))}
    result = dict(provider_calls=181,recorded_tokens=199165,
        current_continuation_provider_calls=55,current_continuation_recorded_tokens=111416,
        latest_https_accuracy=dict(control='31/56',treatment='34/56'),new_https_run=False,
        upstream=read('source-upstream-diagnostic/summary.json')['summary'],
        fine_main=read('source-upstream-ce-diagnostic/summary.json')['summary'],
        bounded_context=after['summary'],exchange_gaps=read('source-exchange-gap-audit.json')['reasons'],
        verification=verification,script_snapshots=len(names),artifacts=artifacts,
        limitation='Diagnostics only. No new production policy, final live packet, or reader accuracy measurement.')
    with (root/'source-semantic-study.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('upstream','fine_main','bounded_context','artifacts')},ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    main(p.parse_args())
