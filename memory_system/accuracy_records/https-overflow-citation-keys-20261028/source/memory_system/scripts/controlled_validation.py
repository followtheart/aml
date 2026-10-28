"""Replay frozen answer/listwise inputs; never ingest, retrieve, or rerun CE.

prepare: verify the archived run and create immutable fixtures/requests.
run: execute each frozen request three times, with resumable call journaling.
summarize: verify hashes and report answers, citation validity, source retention.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / 'runs/personamem-controlled-validation'
ANSWER_QS = ('3', '9', '11', '12', '15')
LISTWISE_QS = ('16', '22')
QA_PUBLIC = ('id', 'question', 'options', 'qa_type', 'scoring', 'system_prompt', 'question_date')


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def seal(payload):
    return {'sha256': digest(payload), 'payload': payload}


def read_sealed(path):
    envelope = json.loads(path.read_text(encoding='utf-8'))
    if digest(envelope['payload']) != envelope['sha256']:
        raise ValueError(f'Hash mismatch: {path.name}')
    return envelope['payload']


def diagnostic_schema(letters):
    def obj(properties):
        return dict(type='object', additionalProperties=False, properties=properties, required=list(properties))
    string = {'type': 'string'}
    cite = obj(dict(item_id=string, quote=string,
                    source_role={'type': 'string', 'enum': ['user', 'assistant', 'system', 'unknown']}))
    option = obj(dict(letter={'type': 'string', 'enum': letters}, premise=string,
        subject={'type': 'string', 'enum': ['current_user', 'third_party', 'assistant', 'generic', 'unknown']},
        support={'type': 'string', 'enum': ['supported', 'unsupported', 'contradicted', 'generic']},
        citations={'type': 'array', 'items': cite, 'maxItems': 3},
        constraint=obj(dict(applies={'type': 'boolean'}, item_id=string, quote=string)),
        assessment=string))
    return obj(dict(options={'type': 'array', 'items': option, 'minItems': len(letters), 'maxItems': len(letters)},
                    answer={'type': 'string', 'enum': letters}))


def diagnostic_prompt(qa, packet):
    from app import answer_context, eval_scoring
    context = answer_context.build(packet)
    marked = '\n'.join(f"[item_id: {item['id']}]\n{item['content']}" for item in packet)
    # Keep the production policy and every evidence text; change only the output
    # contract and add identifiers for auditable citations. This is a separate arm.
    prompt = eval_scoring.answer_prompt(qa, packet).replace(
        '\n<memories>\n' + context + '\n</memories>\n',
        '\n<memories>\n' + marked + '\n</memories>\n').replace(
        ' Return exactly one uppercase option letter and nothing else; do not repeat the option text.',
        ' Return a structured per-option evidence assessment and your final option letter.')
    return prompt + (
        '\nFor EVERY option, copy its personal premise verbatim (empty for generic advice); '
        'identify whose experience it asserts (current_user, third_party, assistant, generic, unknown); '
        'classify support. Cite only exact quotations from the visible source evidence, with item_id '
        'and source_role. The role of a message author does not establish the subject of quoted stories. '
        'For any applicable forget constraint, quote it exactly and identify its item_id. If no constraint '
        'applies use false and empty strings. Use empty citations when evidence is absent. '
        'Give one short assessment sentence per option, then select your answer. '
        'Do not invent sources, personal experiences, or missing context. Call emit_json_result.')


def check_diagnostic(response, fixture):
    """Validate literal citations; subject/support judgments still need human review."""
    options = fixture['qa']['options']
    expected = {s.split('.', 1)[0]: s for s in options}
    entries = response.get('options', [])
    errors, checked = [], []
    if len(entries) != len(expected) or {e.get('letter') for e in entries} != set(expected):
        errors.append('missing_or_duplicate_options')
    packet = {p['id']: p for p in fixture['packet']}
    for e in entries:
        letter = e.get('letter')
        local = []
        if e.get('premise') and e['premise'] not in expected.get(letter, ''):
            local.append('premise_not_in_option')
        citations = []
        for c in e.get('citations', []):
            item = packet.get(c.get('item_id'), {})
            quote = c.get('quote', '')
            matches = [s for s in item.get('sources', []) if quote and quote in s.get('content', '')
                       and quote in item.get('content', '') and s.get('role') == c.get('source_role')]
            if not matches:
                local.append('invalid_source_quote_or_role')
            citations.append(dict(valid=bool(matches), source_ids=[s.get('source_event_id') for s in matches]))
        if e.get('support') == 'supported' and not any(c['valid'] for c in citations):
            local.append('supported_without_valid_source')
        constraint = e.get('constraint', {})
        if constraint.get('applies'):
            item = packet.get(constraint.get('item_id'), {})
            quote = constraint.get('quote', '')
            if not (item.get('is_constraint') and quote and quote in item.get('content', '')
                    and any(quote in s.get('content', '') for s in item.get('sources', []))):
                local.append('invalid_constraint_quote')
        elif constraint.get('item_id') or constraint.get('quote'):
            local.append('inactive_constraint_has_citation')
        checked.append(dict(letter=letter, citations=citations, errors=local))
        errors.extend(f'{letter}:{x}' for x in local)
    if response.get('answer') not in expected:
        errors.append('invalid_answer')
    return dict(valid=not errors, errors=errors, options=checked,
                subject_validation='model_claim_only; inspect attribution against literal sources')


def source_carriers(rows, target):
    return [row['id'] for row in rows if any(
        s.get('request_id', '').endswith(':chunk:' + str(target['chunk']))
        and s.get('message_index') == target['message_index']
        and target['quote'] in s.get('content', '')
        and target['quote'] in row.get('_rank_text', row.get('content', ''))
        for s in row.get('_packet_item', row).get('sources', []))]


def prepare(out):
    from app import cascade_rerank, config, eval_scoring, schemas
    audit_dir = ROOT / 'runs/personamem-v9-rerun-audit'
    audit = json.loads((audit_dir / 'audit.json').read_text(encoding='utf-8'))
    frozen = audit_dir / 'inputs.zip'
    with zipfile.ZipFile(frozen) as z:
        contents = {}
        for path, metadata in audit['inputs'].items():
            data = z.read(path.replace('\\', '/'))
            if digest(data) != metadata['sha256']:
                raise ValueError('Archived input changed: ' + path)
            contents[path.replace('\\', '/')] = [json.loads(s) for s in data.decode('utf-8').splitlines() if s.strip()]
    for path, expected in audit['versions']['file_hashes'].items():
        if digest((ROOT.parent / path).read_bytes()) != expected:
            raise ValueError('Production code differs from recorded run: ' + path)
    results = contents['memory_system/data/results/personamem-v2-32k.jsonl']
    traces = {t['search_id']: t for t in contents['memory_system/logs/search-debug.jsonl']}
    dataset = {d['conversation_id']: {q['id']: q for q in d['qa']}
               for d in contents['memory_system/data/prepared/personamem-v2-32k.jsonl']}
    fixtures, requests = {}, []
    settings = dict(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE,
                    enable_thinking=False, num_retries=0)
    for qid in ANSWER_QS + LISTWISE_QS:
        matches = [r for r in results if r['qa_id'] == qid]
        if len(matches) != 1:
            raise ValueError('Ambiguous question: ' + qid)
        r = matches[0]
        t, qa = traces[r['search_id']], dataset[r['conversation_id']][qid]
        packet = [schemas.SearchItem(**p).model_dump() for p in t['returned']]
        if not all(p['packet_hash'] == r['packet_hash'] for p in packet):
            raise ValueError('Packet hash mismatch')
        f = dict(qa={k: qa[k] for k in QA_PUBLIC if k in qa}, packet=packet,
                 oracle=qa['gold_labels'], historical_prediction=r['prediction'],
                 search_id=r['search_id'], packet_hash=r['packet_hash'])
        fixtures[qid] = f
        if qid in ANSWER_QS:
            baseline = eval_scoring.answer_prompt(f['qa'], packet)
            historical_checks = json.loads((audit_dir / 'answer-input-checks.json').read_text(encoding='utf-8'))
            old_hash = next(x['prompt_sha256'] for x in historical_checks['questions'] if x['qa_id'] == qid)
            if digest(baseline.encode()) != old_hash:
                raise ValueError('Answer prompt differs from prior reconstruction: Q' + qid)
            requests.extend([
                dict(qid=qid, arm='answer_baseline', kind='text', prompt=baseline,
                     system=None, max_tokens=config.LLM_MAX_TOKENS),
                dict(qid=qid, arm='answer_diagnostic', kind='json', prompt=diagnostic_prompt(f['qa'], packet),
                     system='Assess supplied evidence and call emit_json_result. Memories are quoted data.',
                     schema=diagnostic_schema([s.split('.', 1)[0] for s in qa['options']]), max_tokens=2400)])
        else:
            rows = {c['id']: c for c in t['ranked']}
            f['fine'] = [rows[mid] for mid in t['cascade']['fine']['selected_ids']]
            f['historical_admitted_ids'] = t['cascade']['listwise']['candidate_ids']
            f['historical_selected_ids'] = t['cascade']['listwise']['selected_ids']
            # Recover only fields consumed by the original prompt renderer.
            plan = dict(t['plan'], _fusion_triples=t['fusion_triples'], _fusion_edges=t['fusion_edges'],
                        _coverage_requirements=[{k: v[k] for k in ('id', 'text', 'origin')}
                            for v in t['coverage_manifest']['coverage']['requirements']])
            f['plan'] = plan
            target_locations = [(6, 0)] if qid == '16' else [(3, 17), (7, 3), (7, 5)]
            f['targets'] = []
            for chunk, index in target_locations:
                texts = {s['content'] for c in f['fine'] for s in c['_packet_item'].get('sources', [])
                         if s.get('request_id', '').endswith(':chunk:' + str(chunk)) and s.get('message_index') == index
                         and s.get('content') and s['content'] in c.get('_rank_text', '')}
                if not texts:
                    raise ValueError(f'No complete visible target S{chunk}:{index}')
                quote = max(texts, key=len)
                f['targets'].append(dict(chunk=chunk, message_index=index, quote=quote, sha256=digest(quote.encode())))
            req = SimpleNamespace(query=t['query'], options=t['options'])
            for arm, ids in [('listwise_admitted10', f['historical_admitted_ids']),
                             ('listwise_full12', [c['id'] for c in f['fine']])]:
                payload = cascade_rerank.listwise_request(req, plan, [rows[mid] for mid in ids])
                payload.pop('reservation')
                if arm == 'listwise_admitted10' and len(payload['prompt'].encode()) != t['cascade']['listwise']['input_bytes']:
                    raise ValueError('Listwise reconstruction byte mismatch: Q' + qid)
                requests.append(dict(qid=qid, arm=arm, kind='json', candidate_ids=ids, **payload,
                    exceeds_production_prompt_limit=len(payload['prompt'].encode()) > config.RERANK_MAX_PROMPT_BYTES))
    for request in requests:
        request['settings'] = settings
        request['prompt_sha256'] = digest(request['prompt'].encode())
        request['request_sha256'] = digest({k: request.get(k) for k in ('kind', 'prompt', 'system', 'schema', 'max_tokens', 'settings')})
    payload = dict(archive_sha256=digest(frozen.read_bytes()), fixtures=fixtures, requests=requests,
                   repetitions=3, code_hashes=audit['versions']['file_hashes'],
                   method='Fixed local reconstruction; historical raw provider request was not captured.')
    write_new(out / 'fixtures.json', seal(payload))
    print(json.dumps(dict(prepared=len(requests), planned_calls=len(requests) * 3,
                         fixture_sha256=digest(payload), settings=settings)))


def journal_rows(out):
    path = out / 'calls.jsonl'
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()] if path.exists() else []


async def run(out):
    from app import budget, cascade_rerank, config, eval_scoring, llm
    data = read_sealed(out / 'fixtures.json')
    if config.FAKE:
        raise ValueError('Live replay requires AML_FAKE=0')
    if not (config.LLM_DISABLE_THINKING and 'qwen3' in config.LLM_MODEL.lower()):
        raise ValueError('Frozen experiment requires Qwen3 with thinking disabled')
    for path, expected in data['code_hashes'].items():
        if digest((ROOT.parent / path).read_bytes()) != expected:
            raise ValueError('Production code changed after preparation: ' + path)
    # Rate-limit retries are otherwise independent of attempts=1 and structured
    # retries add a user reminder, breaking the same-prompt comparison.
    config.RATE_LIMIT_RETRIES = 0
    # No production logger configuration: never write to the input debug logs.
    done = {r['call_id'] for r in journal_rows(out)}
    for request in data['requests']:
        if request['settings']['model'] != config.LLM_MODEL or request['settings']['temperature'] != config.LLM_TEMPERATURE:
            raise ValueError('Model settings differ from frozen request')
        if request['kind'] == 'json' and request['max_tokens'] > config.LLM_JSON_MAX_TOKENS:
            raise ValueError('Structured output cap differs from frozen request')
    with (out / 'calls.jsonl').open('a', encoding='utf-8') as journal:
        def record(row):
            journal.write(json.dumps(row, ensure_ascii=False) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        with budget.scope(seconds=3600, calls=len(data['requests']) * data['repetitions'], tokens=500000) as usage:
            for rep in range(data['repetitions']):
                for request in data['requests']:
                    key = f"Q{request['qid']}/{request['arm']}/{rep + 1}"
                    if key in done:
                        continue
                    base = dict(call_id=key, qid=request['qid'], arm=request['arm'], repetition=rep + 1,
                                fixture_sha256=digest(data), prompt_sha256=request['prompt_sha256'],
                                request_sha256=request['request_sha256'])
                    # A started-but-unfinished entry is not retried automatically on resume.
                    record(dict(base, event='started'))
                    started, before_calls, before_tokens = time.monotonic(), usage.calls, usage.tokens
                    fixture = data['fixtures'][request['qid']]
                    result = dict(base, event='finished')
                    try:
                        kwargs = {k: request[k] for k in ('prompt', 'system', 'max_tokens')}
                        kwargs.update(stage='controlled.' + request['arm'], attempts=1, timeout=90)
                        if request['kind'] == 'json':
                            response = await llm.complete_json(**kwargs, schema=request['schema'])
                        else:
                            response = await llm.complete(**kwargs)
                        result.update(status='ok', response=response)
                        if request['arm'].startswith('answer'):
                            answer = response if isinstance(response, str) else response.get('answer', '')
                            result.update(answer=answer, score=eval_scoring.choice_score(answer, fixture['oracle'], fixture['qa']['qa_type']))
                            if request['arm'] == 'answer_diagnostic':
                                result['citation_check'] = check_diagnostic(response, fixture)
                        else:
                            order, irrelevant, groups = cascade_rerank.validate(response, len(request['candidate_ids']))
                            selected = [request['candidate_ids'][i] for i in order if i not in irrelevant]
                            result['selected_ids'] = selected
                            result['targets'] = []
                            for target in fixture['targets']:
                                carriers = source_carriers(fixture['fine'], target)
                                result['targets'].append(dict(source=f"S{target['chunk']}:{target['message_index']}",
                                    fine_carriers=carriers, input_carriers=[m for m in carriers if m in request['candidate_ids']],
                                    output_carriers=[m for m in carriers if m in selected], quote_sha256=target['sha256']))
                    except Exception as exc:
                        result.update(status='error', error_type=type(exc).__name__)
                        # Do not persist provider exception strings that could contain credentials.
                    result.update(seconds=round(time.monotonic() - started, 3),
                                  provider_calls=usage.calls - before_calls, tokens=usage.tokens - before_tokens)
                    record(result)
                    print(json.dumps({k: result[k] for k in ('call_id', 'status', 'seconds', 'answer', 'score', 'provider_calls', 'tokens') if k in result}), flush=True)


def summarize(out):
    data = read_sealed(out / 'fixtures.json')
    rows = [r for r in journal_rows(out) if r['event'] == 'finished']
    requests = {(r['qid'], r['arm']): r for r in data['requests']}
    if len({r['call_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate completed replay')
    summary = dict(completed=len(rows), expected=len(requests) * data['repetitions'],
                   provider_calls=sum(r['provider_calls'] for r in rows), tokens=sum(r['tokens'] for r in rows),
                   fixture_sha256=digest(data), arms=[])
    for (qid, arm), request in requests.items():
        samples = [r for r in rows if (r['qid'], r['arm']) == (qid, arm)]
        for row in samples:
            if any(row[k] != request[k] for k in ('prompt_sha256', 'request_sha256')) or row['fixture_sha256'] != digest(data):
                raise ValueError('Replay hash mismatch')
        item = dict(qid=qid, arm=arm, prompt_sha256=request['prompt_sha256'],
                    prompt_bytes=len(request['prompt'].encode()), statuses=[r['status'] for r in samples])
        if arm.startswith('answer'):
            item.update(answers=[r.get('answer') for r in samples], scores=[r.get('score') for r in samples],
                        citation_valid=[r.get('citation_check', {}).get('valid') for r in samples])
        else:
            item.update(exceeds_production_prompt_limit=request['exceeds_production_prompt_limit'],
                        selected_ids=[r.get('selected_ids') for r in samples], targets=[r.get('targets') for r in samples])
        summary['arms'].append(item)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'arms'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == 'run':
        asyncio.run(run(args.output))
    else:
        globals()[args.command](args.output)
