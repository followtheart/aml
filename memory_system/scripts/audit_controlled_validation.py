"""Prepare a matched-output-cap control and audit immutable replay journals."""
import argparse
import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import controlled_validation as replay


def prepare_matched(out):
    original = replay.read_sealed(out / 'fixtures.json')
    data = copy.deepcopy(original)
    data['parent_fixture_sha256'] = replay.digest(original)
    data['method'] = 'Admitted 10 with max_tokens=288, matched to the original full12 arm; only output cap changes from admitted10 reference.'
    data['fixtures'] = {q: f for q, f in data['fixtures'].items() if q in replay.LISTWISE_QS}
    data['requests'] = [r for r in data['requests'] if r['arm'] == 'listwise_admitted10']
    for request in data['requests']:
        request['arm'] = 'listwise_admitted10_cap288'
        request['max_tokens'] = 288
        request['request_sha256'] = replay.digest({k: request.get(k) for k in
            ('kind', 'prompt', 'system', 'schema', 'max_tokens', 'settings')})
        full = next(r for r in original['requests'] if r['qid'] == request['qid'] and r['arm'] == 'listwise_full12')
        assert full['max_tokens'] == request['max_tokens']
        assert full['candidate_ids'][:len(request['candidate_ids'])] == request['candidate_ids']
    replay.write_new(out / 'matched-cap/fixtures.json', replay.seal(data))
    print(json.dumps(dict(planned_calls=len(data['requests']) * data['repetitions'],
                         fixture_sha256=replay.digest(data))))


def audit(out):
    calls, aggregates = [], []
    for directory in [out, out / 'matched-cap']:
        if not (directory / 'fixtures.json').exists():
            continue
        data = replay.read_sealed(directory / 'fixtures.json')
        requests = {(r['qid'], r['arm']): r for r in data['requests']}
        journal = replay.journal_rows(directory)
        finished = [r for r in journal if r['event'] == 'finished']
        started = [r['call_id'] for r in journal if r['event'] == 'started']
        if len(set(started)) != len(started) or len({r['call_id'] for r in finished}) != len(finished):
            raise ValueError('Duplicate call IDs')
        for row in finished:
            request = requests[(row['qid'], row['arm'])]
            if any(row[k] != request[k] for k in ('prompt_sha256', 'request_sha256')) or row['fixture_sha256'] != replay.digest(data):
                raise ValueError('Replay hash mismatch')
            response = row.get('response')
            errors = []
            if request['kind'] == 'json' and response is not None:
                errors = [dict(path=list(e.absolute_path), message=e.message)
                          for e in Draft202012Validator(request['schema']).iter_errors(response)]
            result = dict(call_id=row['call_id'], qid=row['qid'], arm=row['arm'],
                status=row['status'], schema_errors=errors,
                schema_valid=not errors if request['kind'] == 'json' and response is not None else None,
                provider_calls=row['provider_calls'], tokens=row['tokens'],
                prompt_sha256=request['prompt_sha256'], request_sha256=request['request_sha256'],
                max_tokens=request['max_tokens'])
            if row['arm'].startswith('answer'):
                result.update(answer=row.get('answer'), score=row.get('score'))
                if row['arm'] == 'answer_diagnostic' and isinstance(response, dict):
                    check = replay.check_diagnostic(response, data['fixtures'][row['qid']])
                    result.update(literal_citations_valid=check['valid'], literal_citation_errors=check['errors'],
                                  model_options=response.get('options', []))
                    chosen = next((e for e in response.get('options', []) if e.get('letter') == response.get('answer')), {})
                    result['selected_despite_own_constraint'] = chosen.get('constraint', {}).get('applies') is True
            else:
                fixture = data['fixtures'][row['qid']]
                targets = []
                for target in fixture['targets']:
                    carriers = replay.source_carriers(fixture['fine'], target)
                    targets.append(dict(source=f"S{target['chunk']}:{target['message_index']}",
                        input_carriers=[m for m in carriers if m in request['candidate_ids']],
                        output_carriers=[m for m in carriers if m in row['selected_ids']]
                            if row.get('status') == 'ok' and 'selected_ids' in row else None))
                result.update(targets=targets, selected_ids=row.get('selected_ids'),
                              exceeds_production_prompt_limit=request['exceeds_production_prompt_limit'])
            calls.append(result)
        for (qid, arm), request in requests.items():
            samples = [r for r in calls if (r['qid'], r['arm']) == (qid, arm)]
            item = dict(qid=qid, arm=arm, completed=len(samples), expected=data['repetitions'],
                        errors=sum(r['status'] != 'ok' for r in samples),
                        schema_valid=sum(r['schema_valid'] is True for r in samples),
                        prompt_sha256=request['prompt_sha256'], request_sha256=request['request_sha256'],
                        max_tokens=request['max_tokens'], prompt_bytes=len(request['prompt'].encode()))
            if arm.startswith('answer'):
                item.update(answers=[r.get('answer') for r in samples], scores=[r.get('score') for r in samples])
                if arm == 'answer_diagnostic':
                    item.update(literal_valid=sum(r.get('literal_citations_valid') is True for r in samples),
                                own_constraint_violations=sum(r.get('selected_despite_own_constraint') is True for r in samples))
            else:
                target_names = [f"S{t['chunk']}:{t['message_index']}" for t in data['fixtures'][qid]['targets']]
                item['source_retention'] = {name: dict(
                    input=sum(any(t['source'] == name and t['input_carriers'] for t in r.get('targets', [])) for r in samples),
                    output_evaluable=sum(any(t['source'] == name and t['output_carriers'] is not None for t in r.get('targets', [])) for r in samples),
                    output=sum(any(t['source'] == name and t['output_carriers'] for t in r.get('targets', [])) for r in samples))
                    for name in target_names}
                item['selected_counts'] = [len(r['selected_ids']) if r['selected_ids'] is not None else None for r in samples]
                item['exceeds_production_prompt_limit'] = request['exceeds_production_prompt_limit']
            aggregates.append(item)
    record = dict(completed=len(calls), expected=sum(a['expected'] for a in aggregates),
                  provider_calls=sum(r['provider_calls'] for r in calls), tokens=sum(r['tokens'] for r in calls),
                  schema_note='Schema validity and literal quotation validity do not prove subject attribution, relevance, or constraint applicability.',
                  arms=aggregates, calls=calls)
    (out / 'audited-results.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in record.items() if k not in ('arms', 'calls')}))
    for arm in aggregates:
        print(json.dumps(arm, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-matched', 'audit'])
    parser.add_argument('--output', type=Path, default=replay.DEFAULT_OUT)
    args = parser.parse_args()
    (prepare_matched if args.command == 'prepare-matched' else audit)(args.output)
