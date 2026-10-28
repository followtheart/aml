"""Verify focal query interventions against a successful archive before paid probes."""
import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_question_source_filter import load_cohort
from probe_ce_source_units import records


def check(run, prepared, module_name, wanted):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', module_name):
        raise ValueError('Expected a diagnostic module name')
    module = importlib.import_module(module_name)
    cohort = load_cohort(run, prepared)
    selected = {r['search_id']: r['qa_id'] for r, *_ in cohort if r['qa_id'] in wanted}
    if set(selected.values()) != set(wanted):
        raise ValueError('Unknown or missing focal QA')
    rows = []
    for trace in records(run / 'server-search-debug.jsonl'):
        if trace['search_id'] not in selected:
            continue
        queries = module.queries_for(trace)
        if set(queries) != {'control', 'candidate'} or queries['control'] != trace['cascade']['cross_encoder']['query']:
            raise ValueError('Original CE query mismatch')
        rows.append(dict(qa_id=selected[trace['search_id']], changed=queries['candidate'] != queries['control'],
                         bytes={k: len(v.encode()) for k, v in queries.items()},
                         hashes={k: hashlib.sha256(v.encode()).hexdigest() for k, v in queries.items()}))
    if len(rows) != len(wanted):
        raise ValueError('Duplicate or missing focal search')
    return dict(status='passed' if all(r['changed'] for r in rows) else 'focus_not_triggered',
                rows=rows, module=module_name,
                module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                caveat='Read-only intervention preflight. A changed query is necessary for this query intervention '
                       'to test the stated focus, but does not establish retrieval or answer benefit. No provider calls.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'prepared', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--qa', required=True, help='Comma-separated focal QA IDs')
    args = parser.parse_args()
    result = check(args.run, args.prepared, args.module, args.qa.split(','))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    sys.exit(0 if result['status'] == 'passed' else 2)
