"""Audit a completed two-arm HTTPS fine-topic experiment, without providers.

Rebuild stage coverage from raw traces, verify the planned toggle and report
transport separately from answer accuracy. Fresh ingestion remains a confounder.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from compare_https_experiments import compare


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(directory):
    plan = json.loads((directory / 'plan.json').read_text())
    manifests, settings, summaries = {}, {}, {}
    for arm in ('control', 'treatment'):
        run = directory / arm
        m = manifests[arm] = json.loads((run / 'isolated-manifest.json').read_text())
        if 'server_exit_code' not in m or m.get('audit_exit_code') != 0:
            raise ValueError(f'{arm} is not a completed, audited run')
        if hashlib.sha256(Path(m['data']).read_bytes()).hexdigest() != m['data_sha256']:
            raise ValueError('Prepared dataset changed since execution')
        for path, expected in m['files'].items():
            if hashlib.sha256((run / 'source' / path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'{arm} frozen source has changed: {path}')
        subprocess.run([sys.executable, str(Path(__file__).with_name('audit_https_stages.py')),
                        '--run', str(run), '--prepared', m['data']], check=True,
                       stdout=subprocess.DEVNULL)
        stage = json.loads((run / 'funnel/stage-transport.json').read_text())
        results = records(run / 'completed/data/results/personamem-v2-32k.jsonl')
        traces = {t['search_id']: t for t in records(run / 'server-search-debug.jsonl')}
        if len(results) != plan['expected_questions_per_arm'] or len(traces) != len(results):
            raise ValueError('Incomplete or extra searches')
        versions = {json.dumps(t['versions']['settings'], sort_keys=True) for t in traces.values()}
        if len(versions) != 1:
            raise ValueError('Mixed effective settings')
        settings[arm] = json.loads(next(iter(versions)))
        if settings[arm]['FINE_TOPIC_CONTEXT'] != (arm == 'treatment'):
            raise ValueError('Wrong experimental toggle')
        if not all(settings[arm][k] for k in ('RECALL_TOPIC_ANCHORS', 'FINE_EVIDENCE_ADAPTIVE',
                                             'SOURCE_RECALL_DIVERSITY', 'LISTWISE_CONTEXT_LABEL')):
            raise ValueError('Missing planned prerequisite')
        for result in results:
            t = traces[result['search_id']]
            if t['versions']['file_hashes'] != m['files']:
                raise ValueError('Trace code does not match frozen service')
            if not result.get('fake') is False or t.get('fake') is not False:
                raise ValueError('Expected real providers')
        times = sorted((datetime.fromisoformat(t['finished_at']) -
                        datetime.fromisoformat(t['started_at'])).total_seconds() for t in traces.values())
        summary = stage['summary']
        visible = summary['visible_full_turn_question_counts']
        present = sum(r['raw_full_turns'] > 0 for r in stage['rows'])
        summaries[arm] = dict(questions=len(results), score=sum(r['score'] for r in results),
            reference_present=present, visible=visible,
            conditional_packet_coverage=visible['packet'] / present if present else None,
            available_set_complete=summary['available_set_complete_question_counts'],
            strict_reference_set_complete=summary['strict_reference_set_complete_question_counts'],
            delivered_available_turn_counts=summary['delivered_available_turn_counts'],
            conditional_complete_set_coverage=summary['available_set_complete_question_counts']['packet'] / present if present else None,
            full_turns=summary['visible_full_turn_counts'], transitions=summary['transport_transitions'],
            errors=dict(Counter(r['error_stage'] for r in results if r.get('error_stage'))),
            degraded=sum(bool(r.get('search_degraded')) for r in results),
            ce_status=dict(Counter(t['cascade']['cross_encoder']['status'] for t in traces.values())),
            listwise_status=dict(Counter(t['cascade']['listwise']['status'] for t in traces.values())),
            search_median_seconds=statistics.median(times),
            search_p95_seconds=times[math.ceil(.95*len(times))-1],
            context_facet_questions=sum(bool(t['cascade']['fine']['selection'].get('adaptive', {}).get('context_facets'))
                                        for t in traces.values()),
            review_validation_errors=dict(Counter(error for t in traces.values()
                for judgment in t['cascade']['listwise'].get('recovery', {}).get('judgments', [])
                for error in judgment.get('validation_errors', []))),
            unreviewed_candidate_count=sum(len(t['cascade']['listwise'].get('recovery', {}).get('unreviewed_ids', []))
                                          for t in traces.values()),
            unreviewed_retention_violations=[t['search_id'] for t in traces.values()
                if not set(t['cascade']['listwise'].get('recovery', {}).get('unreviewed_ids', []))
                <= set(t['cascade']['listwise']['selected_ids'])])
    if manifests['control']['files'] != manifests['treatment']['files']:
        raise ValueError('Arms ran different source snapshots')
    if manifests['control']['data_sha256'] != manifests['treatment']['data_sha256']:
        raise ValueError('Arms used different datasets')
    differences = {k: [settings['control'].get(k), settings['treatment'].get(k)]
                   for k in settings['control'].keys() | settings['treatment'].keys()
                   if settings['control'].get(k) != settings['treatment'].get(k)}
    if differences != {'FINE_TOPIC_CONTEXT': [False, True]}:
        raise ValueError(f'Unexpected settings differences: {differences}')
    comparison = compare(directory / 'control', directory / 'treatment')
    # Preserve the full raw score. A separate matched subset excludes the
    # UNION of runtime-error questions so the two denominators stay identical.
    # This subset is descriptive, not an unbiased estimate of repaired runs.
    answered = [pair for pair in comparison['pairs']
                if not pair['control_error'] and not pair['treatment_error']]
    error_free_pairs = dict(questions=len(answered),
        control_score=sum(pair['control_score'] for pair in answered),
        treatment_score=sum(pair['treatment_score'] for pair in answered),
        wins=sum(pair['treatment_score'] > pair['control_score'] for pair in answered),
        losses=sum(pair['treatment_score'] < pair['control_score'] for pair in answered),
        excluded=[dict(conversation_id=pair['conversation_id'], qa_id=pair['qa_id'],
                       control_error=pair['control_error'], treatment_error=pair['treatment_error'])
                  for pair in comparison['pairs'] if pair['control_error'] or pair['treatment_error']],
        caveat='Descriptive matched subset; raw scores retain every runtime failure as recorded.')
    (directory / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+'\n')
    control = json.loads((directory / 'control/funnel/stage-transport.json').read_text())['rows']
    treatment = json.loads((directory / 'treatment/funnel/stage-transport.json').read_text())['rows']
    old = {(r['conversation_id'], r['qa_id']): r for r in control}
    gained, lost = [], []
    for row in treatment:
        key = row['conversation_id'], row['qa_id']
        before = bool(old[key]['visible']['packet']['full_turns'])
        after = bool(row['visible']['packet']['full_turns'])
        if after and not before:
            gained.append(dict(conversation_id=key[0], qa_id=key[1]))
        if before and not after:
            lost.append(dict(conversation_id=key[0], qa_id=key[1]))
    report = dict(summaries=summaries, settings_differences=differences,
        error_free_answer_pairs=error_free_pairs,
        gained_packet_questions=gained, lost_packet_questions=lost,
        wins=comparison['wins'], losses=comparison['losses'],
        changed_source_catalogs=comparison['changed_source_catalogs'],
        caveat=__doc__, default_decision='Keep disabled pending replicated final-evidence gains on frozen upstream inputs.')
    (directory / 'funnel-comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    analyze(p.parse_args().run)
