"""Audit a preregistered isolated HTTPS baseline without altering its results.

Final mode requires the launcher, client and automatic funnel audit to finish.
Progress reports remain partial and do not report an overall accuracy. Logged
usage is not a billing guarantee; internal reader validation failures are kept
separate from successful provider transports and terminal answer failures.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config
import audit_https_funnel as funnel
from audit_https_ingestion import audit as audit_ingestion
from audit_frozen_runtime import runtime_failures
from audit_residual_reader import citations
from analyze_overflow_https import provider_stage_costs
from evaluation_model_guard import check_versions, check_provider_log


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(args):
    run = args.run.resolve()
    inputs = {}

    def read(path, jsonl=False):
        data = path.read_bytes()
        if args.progress and jsonl and data and not data.endswith(b'\n'):
            data = data[:data.rfind(b'\n') + 1]
        inputs[str(path.resolve())] = dict(bytes=len(data), sha256=digest(data))
        return [json.loads(line) for line in data.splitlines() if line.strip()] if jsonl else data.decode()

    manifest = json.loads(read(run / 'isolated-manifest.json'))
    registration = json.loads(read(args.registration))
    prepared_path = Path(manifest['data'])
    prepared = read(prepared_path, True)
    if inputs[str(prepared_path.resolve())]['sha256'] != registration['dataset_sha256'] or manifest['data_sha256'] != registration['dataset_sha256']:
        raise ValueError('Dataset identity changed')
    expected = [(c['conversation_id'], str(q['id'])) for c in prepared for q in c['qa']]
    if len(expected) != registration['questions'] or len(set(expected)) != len(expected):
        raise ValueError('Prepared question inventory differs from registration')
    if [(c['conversation_id'], len(c['qa'])) for c in prepared] != [(c['conversation_id'], c['questions']) for c in registration['selected_histories']]:
        raise ValueError('History inventory changed')
    if manifest.get('fake') is not False or manifest['evaluation']['AML_EVAL_LIMIT'] != 0 or manifest['evaluation']['AML_EVAL_CONVS'] != len(prepared):
        raise ValueError('Baseline must evaluate all registered complete histories')
    process = Path('/proc') / str(manifest['server_pid']) / 'cmdline'
    live = process.exists() and str(run).encode() in process.read_bytes()
    if not args.progress:
        if (live or manifest.get('client_exit_code') not in (0, 1) or manifest.get('audit_exit_code') != 0
                or 'server_exit_code' not in manifest):
            raise ValueError('Launcher/client/audit have not finished')
        if int((run / 'exit-code').read_text().strip()) != manifest['client_exit_code']:
            raise ValueError('Client exit records disagree')
    frozen_files = manifest['files']
    for name, sha in frozen_files.items():
        if digest((run / 'source' / name).read_bytes()) != sha:
            raise ValueError('Frozen source changed: ' + name)
    project = Path(__file__).resolve().parents[2]
    current = {n: sha for n, sha in frozen_files.items() if n.startswith(('memory_system/app/', 'prompts/'))}
    if any(digest((project / n).read_bytes()) != sha for n, sha in current.items()):
        raise ValueError('Current app/prompts differ from baseline')
    for name, value in registration['historical_boolean_defaults'].items():
        if getattr(config, name) != value:
            raise ValueError('Historical default changed: ' + name)
    ingestion = audit_ingestion(run, run / 'ingestion-audit-plan.json')
    if not args.progress and not ingestion['complete']:
        raise ValueError('Incomplete ingestion')
    results = read(run / 'completed/data/results/personamem-v2-32k.jsonl', True)
    actual = [(r['conversation_id'], str(r['qa_id'])) for r in results]
    if not results or actual != expected[:len(actual)] or not args.progress and actual != expected:
        raise ValueError('Result identity/order/count differs')
    if {r['run_id'] for r in results} != {ingestion['run_id']}:
        raise ValueError('Mixed run identities')
    terminal = [dict(qa_id=r['qa_id'], stage=r['error_stage'], error_type=r.get('error_type')) for r in results if r.get('error_stage')]
    if not args.progress and manifest['client_exit_code'] != int(bool(terminal)):
        raise ValueError('Terminal errors disagree with exit code')
    searches = read(run / 'server-search-debug.jsonl', True)
    by_search = {s['search_id']: s for s in searches}
    if len(by_search) != len(searches):
        raise ValueError('Duplicate Search identities')
    for r in results:
        if r.get('fake') is not False or r.get('transport') != 'https':
            raise ValueError('Expected real HTTPS result')
    for row in results + searches:
        check_versions(row['versions'], 'baseline')
        versions = row['versions']
        if versions.get('embedding_model') != registration['models']['embedding'] or versions.get('cross_encoder_model') != registration['models']['ce']:
            raise ValueError('Runtime provider models differ')
        if any(frozen_files.get(name) != sha for name, sha in versions['file_hashes'].items()):
            raise ValueError('Dispatched code/prompt hashes differ from frozen manifest')
        settings = row['versions']['settings']
        for name, value in {**registration['historical_boolean_defaults'], **registration['extra_context_flags']}.items():
            if settings.get(name) != value:
                raise ValueError('Runtime setting differs: ' + name)
    for trace in searches:
        if trace['cascade']['cross_encoder']['model'] != registration['models']['ce']:
            raise ValueError('CE model changed')
    memory = read(run / 'server-memory-debug.jsonl', True)
    space = f"{registration['models']['embedding']}:float32:{registration['models']['dimensions']}"
    if any(e['fake'] is not False or e['configured_embedding_model'] != registration['models']['embedding'] or e['configured_embedding_space'] != space for e in memory):
        raise ValueError('Embedding model/dimensions or real ingestion changed')
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    rows = funnel.analyze(results, memory, searches, {c['conversation_id']: c for c in prepared}, gold)
    stage_path = run / 'funnel' / ('progress-stage-transport.json' if args.progress else 'stage-transport.json')
    stage = json.loads(read(stage_path))
    if bool(stage['summary']['progress_snapshot']) != args.progress:
        raise ValueError('Stage audit has the wrong progress/final status')
    stage_rows = {r['qa_id']: r for r in stage['rows']}
    if [r['qa_id'] for r in results] != [r['qa_id'] for r in stage['rows']]:
        raise ValueError('Stage audit snapshot differs; refresh stages first')
    enriched = []
    for row, result in zip(rows, results):
        if row['qa_id'] != result['qa_id']:
            raise ValueError('Funnel result order mismatch')
        covered = row['catalog_turn_coverage'] + row['constraint_turn_coverage']
        full = any(t['full'] for t in covered)
        if full != (stage_rows[row['qa_id']]['visible']['catalog']['full_turns'] > 0):
            raise ValueError('Reference transport audits disagree')
        reference_ids = {sid for t in covered if t['full'] for sid in t['source_ids']}
        cards = {c['id']: c for c in result.get('answer_source_catalog', []) + result.get('answer_constraint_catalog', [])}
        reference_cards = {sid: cards[sid] for sid in reference_ids}
        uses = [dict(stage=c['stage'], status=c.get('status'), **cite) for c in result.get('answer_calls', [])
                for cite in citations(c.get('response'), reference_cards) if cite['exact_quote']]
        enriched.append(dict(qa_id=row['qa_id'], conversation_id=row['conversation_id'], score=row['score'],
            reference_available=stage_rows[row['qa_id']]['raw_full_turns'] > 0,
            full_reference_delivered=full, exact_reference_source_quotes=uses,
            reader_gate=row['reader_gate_detail'], gold_eligible=row['gold'] in row['eligible']))
    server_log = read(run / 'server.log')
    reader_log = read(run / 'output.log')
    costs = provider_stage_costs(server_log)
    search_costs = {k:v for k,v in costs.items() if k.startswith('search.')}
    add_costs = {k:v for k,v in costs.items() if k.startswith('add.')}
    reader_costs = provider_stage_costs(reader_log)
    unknown = [k for k in costs if not k.startswith(('search.', 'add.'))] + [k for k in reader_costs if not k.startswith('eval.')]
    if unknown:
        raise ValueError('Unclassified provider stages: ' + repr(unknown))
    checks = dict(search_calls=sum(r['coverage_manifest'].get('provider_calls', 0) for r in results) == sum(v['calls'] for v in search_costs.values()),
                  search_tokens=sum(r['coverage_manifest'].get('provider_tokens', 0) for r in results) == sum(v['tokens'] for v in search_costs.values()),
                  reader_stage_calls=dict(Counter(c['stage'] for r in results for c in r.get('answer_calls', []))) == {k:v['calls'] for k,v in reader_costs.items()})
    if not args.progress and not all(checks.values()):
        raise ValueError('Provider accounting mismatch: ' + repr(checks))
    errors = [dict(qa_id=r['qa_id'], stage=c['stage'], error_type=c.get('error_type'))
              for r in results for c in r.get('answer_calls', []) if c.get('status') == 'error']
    search_errors = [dict(qa_id=r['qa_id'], **error) for r in results if r.get('search_id') in by_search
                     for error in runtime_failures(by_search[r['search_id']]['cascade'])]
    wrong = [r for r in enriched if r['score'] < 1]
    groups = dict(missing_reference=[r['qa_id'] for r in wrong if r['reference_available'] and not r['full_reference_delivered']],
                  unavailable_reference=[r['qa_id'] for r in wrong if not r['reference_available']],
                  delivered_reference=[r['qa_id'] for r in wrong if r['full_reference_delivered']],
                  delivered_without_explicit_quote=[r['qa_id'] for r in wrong if r['full_reference_delivered'] and not r['exact_reference_source_quotes']])
    summary = dict(progress_snapshot=args.progress, questions=len(results), expected_questions=len(expected),
                   server_live=live, terminal_errors=terminal, frozen_files=len(frozen_files),
                   current_app_prompt_files=len(current), historical_defaults=len(registration['historical_boolean_defaults']),
                   wrong_reader_gates=dict(Counter(r['reader_gate'] for r in wrong)), wrong_groups=groups,
                   stage_summary=stage['summary'])
    if not args.progress:
        summary.update(score=sum(r['score'] for r in results), accuracy=sum(r['score'] for r in results) / len(results))
    return dict(summary=summary, ingestion=ingestion, rows=enriched, terminal_manifest=manifest if not args.progress else None,
                accounting=dict(checks=checks, add=add_costs, search=search_costs, reader=reader_costs,
                    server_llm_dispatches=check_provider_log(server_log, 'server'), reader_llm_dispatches=check_provider_log(reader_log, 'reader'),
                    reader_internal_errors=errors, search_runtime_errors=search_errors,
                    limitation='Progress logs may include in-flight work. Recorded usage only; failed calls may have unreported usage.'),
                inputs=inputs, script_sha256=digest(Path(__file__).read_bytes()), limitation=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result['summary'].items() if k != 'stage_summary'}, ensure_ascii=False))
