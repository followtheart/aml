"""Finalize complete paired HTTPS answers, reference turns and provider ledgers.

Only terminal complete cohorts are scored. A single stochastic paired run on
previously inspected cohorts is not causal or held-out proof. Main admission
does not guarantee final delivery, semantic support or an improved answer.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics

from batched_records import file_digest, json_records, physical_records


def read(path):
    payload = bytearray()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            payload.extend(block)
    return json.loads(payload)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(row):
    return row['conversation_id'], str(row['qa_id'])


def reference_changes(before, after):
    require(len(before) == len(after), 'Reference count mismatch')
    gains, losses = [], []
    for old, new in zip(before, after, strict=True):
        require(old['turn_index'] == new['turn_index'] and old['normalized_chars'] == new['normalized_chars'],
                'Reference identity mismatch')
        require(type(old['full']) is bool and type(new['full']) is bool, 'Boolean full coverage required')
        if new['full'] and not old['full']:
            gains.append(new['turn_index'])
        if old['full'] and not new['full']:
            losses.append(old['turn_index'])
    return dict(gained=gains, lost=losses)


def provider_ledger(path, *, arm=None):
    by_stage = {}
    for record in physical_records(path):
        message = record.decode('utf-8', errors='strict')
        if 'provider_call ' not in message:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', message))
        if arm is not None and fields.get('reader_arm') != arm:
            continue
        require(fields.get('status') != 'fake', 'Fake provider dispatch')
        if fields['kind'] == 'llm':
            require(fields['model'] == 'gpt-4o-mini', 'LLM model drift')
        elif fields['kind'] == 'rerank':
            require(fields['model'] == 'qwen3-rerank', 'CE model drift')
        elif fields['kind'] == 'embedding':
            require(fields['model'] == 'openai/text-embedding-v4', 'Embedding model drift')
        stage = fields['stage']
        summary = by_stage.setdefault(stage, dict(calls=0, reported_tokens=0, retry_attempts=0,
            statuses={}, failed_without_trustworthy_usage=0))
        summary['calls'] += 1
        summary['reported_tokens'] += int(fields['total_tokens'])
        summary['retry_attempts'] += int(fields['attempt']) > 1
        status = fields['status']
        summary['statuses'][status] = summary['statuses'].get(status, 0) + 1
        # Error/cancellation without positive reported usage is unknown billing,
        # not a zero-cost request. Report it even if terminal answer succeeds.
        summary['failed_without_trustworthy_usage'] += status != 'ok' and (
            fields.get('usage_known') == 'false' or int(fields['total_tokens']) == 0)
    return dict(by_stage=by_stage, **{field: sum(row[field] for row in by_stage.values())
        for field in ('calls', 'reported_tokens', 'retry_attempts', 'failed_without_trustworthy_usage')})


def reader_statuses(rows):
    statuses, phases, errors, partial = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        for call in row.get('answer_calls', []):
            status = call.get('status', 'unfinished')
            statuses[status] += 1
            if status == 'partial':
                partial[call['stage']] += 1
            if status in ('invalid', 'error', 'unfinished'):
                phases[call.get('failure_phase', 'unspecified')] += 1
                errors[call.get('error_type', 'unspecified')] += 1
    return dict(statuses=dict(statuses), failure_phases=dict(phases), error_types=dict(errors),
                partial_by_stage=dict(partial))


def main(root):
    registered = read(root/'preregistered-plan.json')
    state = read(root/'state.json')
    require(state['status'] in ('complete', 'complete_with_failures'), 'Launcher is not terminal')
    require(state.get('child_pid') is None, 'Launcher still has a child')
    require(not (root/'final-decision.json').exists(), 'Never overwrite a final decision')
    for name, digest in registered['prerequisite_sha256'].items():
        require(file_digest(Path(name)) == digest, 'Prerequisite changed: '+name)
    inputs = [root/'preregistered-plan.json', root/'state.json']
    reports = {}
    for cohort, spec in registered['cohorts'].items():
        directory = root/cohort
        plan = read(directory/'plan.json')
        fixture = read(directory/'fixture.json')
        require(plan['toggle'] == 'JOINT_SOURCE_RANK_ADMISSION' and
                plan['setting_values'] == dict(control=False, treatment=True), 'Single intended intervention')
        require(plan['concurrency_per_arm'] == 1 and plan['arm_schedule'] == 'sequential', 'Serial execution')
        require(file_digest(directory/'fixture.json') == plan['fixture_sha256'], 'Frozen fixture')
        expected = {identity(r) for r in spec['identities']}
        require(len(expected) == spec['questions'], 'Unique preregistered questions')
        require({identity(r) for r in fixture['requests']} == expected, 'All preregistered requests')
        for name, digest in plan['source_files'].items():
            require(file_digest(directory/'source'/name) == digest, 'Frozen runtime source changed')
            require(registered['source_files'].get(name, digest) == digest, 'Runtime differs from preregistration')
        pair_path = directory/'funnel-comparison.json'
        comparison = read(pair_path)
        require(comparison['settings_differences'] == {'JOINT_SOURCE_RANK_ADMISSION': [False, True]}, 'Unexpected settings drift')
        require({identity(r) for r in comparison['pairs']} == expected, 'Complete paired funnel analysis')
        inputs.extend([directory/'plan.json', directory/'fixture.json', pair_path, directory/'reader-provider.log'])
        arms, paired, stage_rows = {}, {}, {}
        for arm in ('control', 'treatment'):
            run = directory/arm
            manifest = read(run/'isolated-manifest.json')
            require(manifest.get('server_exit_code') in (0, -15), 'TLS service not terminated normally')
            require(manifest.get('audit_exit_code') == manifest.get('source_audit_exit_code') == 0,
                    'Complete stage and source audits required')
            process = Path('/proc')/str(manifest['server_pid'])/'cmdline'
            require(not process.exists() or str(directory).encode() not in process.read_bytes(), 'TLS service still live')
            raw_rows = list(json_records(run/'completed/data/results/personamem-v2-32k.jsonl'))
            result_rows = {identity(r): r for r in raw_rows}
            require(len(result_rows) == len(raw_rows) == spec['questions'] and set(result_rows) == expected,
                    'All original attempts, including failures, required')
            source = read(run/'joint-source-audit.json')
            require(source['status'] == 'passed' and source['summary']['questions'] == spec['questions'], 'Source audit coverage')
            for name, digest in source['input_sha256'].items():
                require(file_digest(Path(name)) == digest, 'Source audit input changed')
            funnel = read(run/'funnel/stage-transport.json')
            stages = {identity(r): r for r in funnel['rows']}
            require(len(stages) == spec['questions'] and set(stages) == expected, 'Funnel identities')
            search = provider_ledger(run/'server.log')
            reader = provider_ledger(directory/'reader-provider.log', arm=arm)
            require(search['calls'] == sum(r['coverage_manifest']['provider_calls'] for r in raw_rows),
                    'Search call ledger mismatch')
            require(search['reported_tokens'] == sum(r['coverage_manifest']['provider_tokens'] for r in raw_rows),
                    'Search token ledger mismatch')
            extra = search['by_stage'].get('search.joint_source_rerank', dict(calls=0, reported_tokens=0))
            require(extra['calls'] == source['summary']['source_calls'] and
                    extra['reported_tokens'] == source['summary']['source_reported_tokens'], 'Source ledger mismatch')
            errors = [dict(qa_id=r['qa_id'], stage=r['error_stage'], type=r.get('error_type'))
                      for r in raw_rows if r.get('error_stage')]
            score = sum(r['score'] for r in raw_rows)
            require(comparison['summaries'][arm]['score'] == score, 'Paired answer score mismatch')
            arms[arm] = dict(questions=len(raw_rows), score=score, accuracy=score/len(raw_rows), terminal_errors=errors,
                joint_source_admission=source['summary'], search_ledger=search, reader_ledger=reader,
                reader_judgments=reader_statuses(raw_rows), funnel=funnel['summary'],
                search_degraded=sum(bool(r.get('search_degraded')) for r in raw_rows),
                median_search_seconds=statistics.median(r['search_seconds'] for r in raw_rows),
                reported_tokens=search['reported_tokens']+reader['reported_tokens'])
            paired[arm] = result_rows
            stage_rows[arm] = stages
            inputs.extend([run/'isolated-manifest.json', run/'server.log', run/'joint-source-audit.json',
                           run/'funnel/stage-transport.json', run/'completed/data/results/personamem-v2-32k.jsonl'])
        changes = {}
        for stage in ('coarse', 'fine', 'listwise_input', 'listwise_selected', 'selected', 'packet', 'catalog'):
            gains, losses = [], []
            for key in sorted(expected):
                delta = reference_changes(stage_rows['control'][key]['visible'][stage]['turn_coverage'],
                                          stage_rows['treatment'][key]['visible'][stage]['turn_coverage'])
                gains.extend(dict(conversation_id=key[0], qa_id=key[1], turn=i) for i in delta['gained'])
                losses.extend(dict(conversation_id=key[0], qa_id=key[1], turn=i) for i in delta['lost'])
            changes[stage] = dict(gained=gains, lost=losses)
        wins = [dict(conversation_id=k[0], qa_id=k[1]) for k in sorted(expected)
                if paired['treatment'][k]['score'] > paired['control'][k]['score']]
        losses = [dict(conversation_id=k[0], qa_id=k[1]) for k in sorted(expected)
                  if paired['treatment'][k]['score'] < paired['control'][k]['score']]
        gates = dict(answer_nonregression=arms['treatment']['score'] >= arms['control']['score'],
            no_final_reference_loss=not changes['packet']['lost'] and not changes['catalog']['lost'],
            no_terminal_errors=all(not v['terminal_errors'] for v in arms.values()))
        reports[cohort] = dict(arms=arms, reference_changes=changes, wins=wins, losses=losses, gates=gates,
            reported_token_delta=arms['treatment']['reported_tokens']-arms['control']['reported_tokens'],
            upstream_matches={k:comparison[k] for k in ('same_precoarse_questions','same_fine_questions','same_main_input_questions')})
    gain = any(r['reference_changes']['packet']['gained'] for r in reports.values())
    all_gates = all(all(r['gates'].values()) for r in reports.values()) and gain
    report = dict(status='complete_paired_https_analysis', completed_at=datetime.now(timezone.utc).isoformat(),
        decision='numeric_gates_passed_pending_joint_review' if all_gates else 'reject_answer_or_reference_or_error_gate',
        actual_final_reference_gain=gain, default_enabled=False, cohorts=reports,
        original_requests=sum(sum(a['questions'] for a in r['arms'].values()) for r in reports.values()),
        reported_tokens=sum(sum(a['reported_tokens'] for a in r['arms'].values()) for r in reports.values()),
        resources=state.get('resources'), provider_calls_by_finalizer=0,
        limitation=__doc__+' Missing failed-request usage is not zero billing. Default stays off even when numeric gates pass.',
        input_sha256={str(p):file_digest(p) for p in inputs})
    require(report['original_requests'] == registered['total_https_requests'], 'Complete experiment request count')
    with (root/'final-decision.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(dict(decision=report['decision'], cohorts={k:dict(gates=v['gates'],
        answers={a:(x['score'],x['questions']) for a,x in v['arms'].items()},
        packet_changes=v['reference_changes']['packet']) for k,v in reports.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    main(parser.parse_args().experiment.resolve())
