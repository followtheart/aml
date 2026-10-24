"""Audit completed HTTPS evaluations. Gold snippets are used only offline."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from .evaluate_https import records, save


def norm(text):
    return ' '.join((text or '').split()).casefold()


def full_turns(turns, sources, text_key='content'):
    texts = [norm(s.get(text_key)) for s in sources if s.get('role') == 'user']
    return [i for i, turn in enumerate(turns) if norm(turn) and any(norm(turn) in text for text in texts)]


def audit(run, gold_csv):
    if (run / 'validation-failure.json').exists():
        raise ValueError('This run has an explicit validation failure; do not compare its scores')
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Only complete evaluations may be compared')
    summary = json.loads((run / 'summary.json').read_text())
    results = records(run / 'results.jsonl')
    baseline = records(run / 'baseline-results.jsonl')
    response_rows = records(run / 'responses.jsonl')
    calls = records(run / 'requests.jsonl')
    cohort = {r['conversation_id']: r for r in records(run / 'prepared.jsonl')}
    identity = lambda row: (row['conversation_id'], row['qa_id'])
    assert len(results) == len(set(map(identity, results))) == manifest['questions']
    assert set(map(identity, results)) == set(map(identity, baseline))
    assert hashlib.sha256((run / 'prepared.jsonl').read_bytes()).hexdigest() == manifest['data_sha256']
    reversed_users = {new: old for old, new in manifest['users'].items()}
    actual = []
    for row in calls:
        if row['path'] not in {'/add', '/search'}:
            continue
        body = dict(row['payload'])
        new, old = body['user_id'], reversed_users[body['user_id']]
        body['user_id'] = old
        for field in ['request_id', 'session_id']:
            if field in body:
                body[field] = body[field].replace(new, old)
        actual.append(dict(path=row['path'].lstrip('/'), payload=body))
    expected = records(run / 'baseline-requests.jsonl')
    if manifest.get('fresh_ingestion') is False:
        expected = [r for r in expected if r['path'] == 'search']
    # Requests interleave because independent tenants and searches run in parallel.
    canonical = lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False)
    assert Counter(map(canonical, actual)) == Counter(map(canonical, expected)), 'Request payloads differ beyond tenant identifiers'
    assert len(response_rows) == len(calls) and all(r['success'] for r in response_rows)
    assert all(r.get('success') for r in json.loads((run / 'cleanup.json').read_text()))
    source_changed = [p for p, h in manifest['source_hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert not source_changed, source_changed
    searches = {r['identity']: r['response'] for r in response_rows if r['path'] == '/search'}
    assert len(searches) == len(results)
    csv.field_size_limit(32 * 1024 * 1024)
    with gold_csv.open() as source:
        gold = list(csv.DictReader(source))
    previous_audit = {identity(r): r for r in records_json(Path(manifest['baseline']) / 'funnel/https-rows.json')}
    details = []
    for result in results:
        key = identity(result)
        conv = cohort[result['conversation_id']]
        qa = next(q for q in conv['qa'] if q['id'] == result['qa_id'])
        row = gold[int(result['qa_id'])]
        assert norm(ast.literal_eval(row['user_query'])['content']) == norm(qa['question'])
        assert row['correct_answer'] == qa['answer']
        assert conv['source_history'] in (row['chat_history_32k_link'], row['chat_history_128k_link'])
        try:
            snippet = json.loads(row['related_conversation_snippet'])
        except ValueError:
            snippet = ast.literal_eval(row['related_conversation_snippet'])
        turns = [s['content'] for s in snippet if s['role'] == 'user']
        original = [s for session in conv['sessions'] for s in session]
        response = searches[':'.join(key)]
        sources = [s for item in response['data'] for s in item['sources']]
        diagnostics = result.get('reader_diagnostics', {})
        catalog = diagnostics.get('answer_source_catalog', [])
        available = full_turns(turns, original)
        packet = full_turns(turns, sources)
        reader = full_turns(turns, catalog, 'text')
        prior = previous_audit[key]
        # These are observed delivery stages, not causal explanations of wrong answers.
        stage = ('no_reference_in_input' if not available else 'reference_not_returned' if not packet
                 else 'returned_but_not_visible_to_reader' if not reader else 'reference_visible_to_reader')
        details.append(dict(conversation_id=key[0], qa_id=key[1], score=result['score'],
            baseline_score=result['baseline_score'], prediction=result.get('prediction'), gold_labels=qa['gold_labels'],
            reference_turns=len(turns), available_turns=available, returned_turns=packet, reader_turns=reader,
            baseline_reader_full=any(t['full'] for t in prior['catalog_turn_coverage']),
            returned_items=len(response['data']), source_count=len(sources), reader_sources=len(catalog),
            observed_stage=stage, reader_error=diagnostics.get('error_stage'),
            witness_status=diagnostics.get('answer_witness_retrieval', {}).get('status'),
            answer_status=diagnostics.get('answer_status')))
    metrics = records(run / 'reader-calls.jsonl')
    input_tokens = sum(int(c.get('prompt_tokens', 0)) for c in metrics)
    output_tokens = sum(int(c.get('completion_tokens', 0)) for c in metrics)
    usage = dict(calls=len(metrics), status=dict(Counter(c.get('status') for c in metrics)),
        models=dict(Counter(c.get('model') for c in metrics)), prompt_tokens=input_tokens,
        completion_tokens=output_tokens, total_tokens=input_tokens+output_tokens,
        estimated_usd=input_tokens*.15/1_000_000 + output_tokens*.60/1_000_000,
        note='Reader only, using repository GPT-4o-mini price assumptions. Retrieval costs and unknown failed-call usage are excluded.')
    result = dict(integrity=dict(complete_question_set=True, duplicate_questions=0, expected_questions=len(baseline),
        observed_questions=len(results), dataset_hash_matches=True, request_payloads_match_baseline=True,
        source_code_unchanged=True, http_calls=len(calls), http_errors=0, cleanup_complete=True),
        reference_coverage=dict(available_questions=sum(bool(r['available_turns']) for r in details),
            returned_questions=sum(bool(r['returned_turns']) for r in details),
            reader_questions=sum(bool(r['reader_turns']) for r in details),
            baseline_reader_questions=sum(r['baseline_reader_full'] for r in details),
            wrong_stages=dict(Counter(r['observed_stage'] for r in details if r['score'] < 1)),
            caveat='At least one complete normalized original user turn; literal transport is not semantic entailment.'),
        reader_usage=usage, details=details)
    save(run / 'audit.json', result)
    return result


def records_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--gold-csv', type=Path, default=Path('memory_system/data/raw/personamem-v2.csv'))
    args = parser.parse_args()
    result = audit(args.run, args.gold_csv)
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
