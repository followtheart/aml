"""Verify the source-identity correction against every actual HTTPS packet.

The trial uses frozen v40 code. Current v41 only canonicalizes profile quote
identities for sparse rarity. Exact packet/catalog equality on this cohort is
not another live provider evaluation or proof of equality on other inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

from verify_sparse_source_context import replay, records
from app import answer_choice


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    root = args.run.resolve()
    plan = json.loads((root/'plan.json').read_text())
    assert (root/'exit-code').read_text().strip() in ('0', '1')
    current = Path(__file__).resolve().parents[1]
    app_hashes = {str(p.relative_to(current.parent)): sha(p) for p in (current/'app').glob('*.py')}
    changes = {name:dict(frozen=plan['source_files'][name], current=value)
               for name, value in app_hashes.items() if plan['source_files'][name] != value}
    assert set(changes) == {'memory_system/app/fine_sparse_reserve.py', 'memory_system/app/run_metadata.py'}
    rows = []
    for arm in ('control', 'treatment'):
        run = root/arm
        results = {r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        traces = records(run/'server-search-debug.jsonl')
        assert len(traces) == len(results) == plan['questions']
        for t in traces:
            actual, digest, manifest = replay(t, arm == 'treatment')
            assert actual == t['returned'] and digest == t['packet_hash'], (arm, t['search_id'])
            result = results[t['search_id']]
            catalogs = answer_choice.build_catalog(actual)
            if not result.get('error_stage'):
                assert list(catalogs[0].values()) == result['answer_source_catalog']
                assert list(catalogs[1].values()) == result['answer_constraint_catalog']
            rows.append(dict(arm=arm, qa_id=result['qa_id'], packet_equal=True,
                catalog_equal=not result.get('error_stage'),
                assembly_manifest_equal=all(value == t['coverage_manifest'].get(key) for key,value in manifest.items()),
                post_assembly_fields=sorted(t['coverage_manifest'].keys()-manifest.keys())))
    report = dict(questions=len(rows), packets_equal=len(rows), reader_catalogs_equal=sum(r['catalog_equal'] for r in rows),
        assembly_manifest_differences=[dict(arm=r['arm'],qa_id=r['qa_id']) for r in rows if not r['assembly_manifest_equal']],
        app_changes=changes, current_app_hashes=app_hashes, rows=rows, provider_calls=0,
        auditor_sha256=sha(Path(__file__)), limitation=__doc__)
    with (root/'identity-fix-replay.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows', 'current_app_hashes')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    main(p.parse_args())
