"""Isolate membership, candidate aliases, source order and source aliases.

Requires historical batches differing by one redundant-source candidate.
No source text, role, target or citation policy changes. This is a fixed-request
diagnostic, not a production selection policy or a full HTTPS evaluation.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, listwise_recovery as recovery, llm, recovery_spans as spans, schemas
from evaluation_model_guard import REQUIRED_MODEL, check_runtime, check_versions


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_identity(source):
    return dict(role=source['role'], timestamp=source.get('timestamp'), content_span=source.get('content_span'),
                parts=[part['text'] for part in source['parts']])


def payload(built):
    prefix, data = built['prompt'].rsplit('\n', 1)
    return prefix + '\n', json.loads(data)


def presentation(original, *, drop=(), candidate_names=None, source_names=None, source_order=None):
    """Rewrite structured identifiers only; original cards/local owners stay intact."""
    result = copy.deepcopy(original)
    prefix, data = payload(result)
    for key in drop:
        mid = result['candidate_map'].pop(key)
        result['owners'].pop(key)
        result['schema']['properties'].pop(key)
        result['schema']['required'].remove(key)
        result['cards'] = [card for card in result['cards'] if card['candidate_id'] != mid]
        data['candidates'] = [card for card in data['candidates'] if card['id'] != key]
    if candidate_names is not None:
        if set(candidate_names) != set(result['candidate_map']) or len(set(candidate_names.values())) != len(candidate_names):
            raise ValueError('Candidate renaming must be a bijection')
        for key in ('candidate_map', 'owners'):
            result[key] = {candidate_names[k]: v for k, v in result[key].items()}
        result['schema']['properties'] = {candidate_names[k]: v for k, v in result['schema']['properties'].items()}
        result['schema']['required'] = [candidate_names[k] for k in result['schema']['required']]
        for card in data['candidates']:
            card['id'] = candidate_names[card['id']]
    if source_order is not None:
        by_id = {s['id']: s for s in data['sources']}
        if set(source_order) != set(by_id) or len(source_order) != len(by_id):
            raise ValueError('Source order must be a permutation')
        data['sources'] = [by_id[sid] for sid in source_order]
    if source_names is not None:
        ids = {s['id'] for s in data['sources']}
        if set(source_names) != ids or len(set(source_names.values())) != len(ids):
            raise ValueError('Source renaming must be a bijection')
        part_names = {pid: source_names[value['global_source']] + pid[len(value['global_source']):]
                      for pid, value in result['spans'].items()}
        for source in data['sources']:
            source['id'] = source_names[source['id']]
            for part in source['parts']:
                part['id'] = part_names[part['id']]
        for card in data['candidates']:
            card['sources'] = [source_names[sid] for sid in card['sources']]
        result['spans'] = {part_names[pid]: dict(value, global_source=source_names[value['global_source']])
                           for pid, value in result['spans'].items()}
        result['owners'] = {key: {part_names[pid]: local for pid, local in owners.items()}
                            for key, owners in result['owners'].items()}
        for decision in result['schema']['properties'].values():
            enum = decision['properties']['citation_parts']['items']['enum']
            decision['properties']['citation_parts']['items']['enum'] = [part_names[pid] for pid in enum]
    result['prompt'] = prefix + json.dumps(data, ensure_ascii=False)
    return result


def request_for(built, guidance=False):
    system = built['system'] + (recovery.TOPIC_INTEREST_GUIDANCE if guidance else '')
    request = dict(prompt=built['prompt'], schema=built['schema'], system=system, max_tokens=recovery.OUTPUT_TOKENS)
    request['reservation'] = (len(request['prompt'].encode()) + len(json.dumps(request['schema']).encode())
                              + len(system.encode()) + 1024 + recovery.OUTPUT_TOKENS)
    return request


def load_batches(run, qa_id):
    if (run / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require a completed successful arm')
    rows = [r for r in records(run / 'completed/data/results/personamem-v2-32k.jsonl') if r['qa_id'] == qa_id]
    if len(rows) != 1:
        raise ValueError('Question identity missing or ambiguous')
    trace = next(t for t in records(run / 'server-search-debug.jsonl') if t['search_id'] == rows[0]['search_id'])
    check_versions(trace['versions'], 'source trace')
    req = schemas.SearchRequest(**{k: trace[k] for k in ('user_id', 'query', 'options', 'top_k', 'reference_time', 'evidence_token_budget')})
    settings = {'LISTWISE_TOPIC_INTEREST': False, **trace['versions']['settings']}
    batches = []
    with patch.multiple(config, **settings):
        check_runtime()
        if not config.LISTWISE_SPAN_REFS or any((config.LISTWISE_ROLE_BOUND_REFS, config.LISTWISE_TRANSFORM_CONTEXT,
                                                config.LISTWISE_ABBREVIATION_PARTS, config.LISTWISE_TOPIC_INTEREST)):
            raise ValueError('Use baseline flat span-reference requests with other review experiments off')
        for i, review in enumerate(trace['cascade']['listwise'].get('recovery', {}).get('batches', [])):
            cards = review.get('sources', [])
            if not cards or not review.get('prompt_sha256'):
                continue
            built = spans.build(req, cards)
            request = recovery.request_for(req, cards)
            if (request != request_for(built) or request['reservation'] != review['request_bound']
                    or hashlib.sha256(request['prompt'].encode()).hexdigest() != review['prompt_sha256']):
                raise ValueError('Historical request did not reproduce exactly')
            batches.append((i, built))
    return req, batches


def variants(before, after):
    old_ids, new_ids = set(before['candidate_map'].values()), set(after['candidate_map'].values())
    if not new_ids < old_ids or len(old_ids - new_ids) != 1:
        raise ValueError('Require exactly one removed candidate')
    _, old_data = payload(before)
    _, new_data = payload(after)
    old_sources = {fingerprint(source_identity(s)): s['id'] for s in old_data['sources']}
    new_sources = {fingerprint(source_identity(s)): s['id'] for s in new_data['sources']}
    if (old_sources.keys() != new_sources.keys() or len(old_sources) != len(old_data['sources'])
            or len(new_sources) != len(new_data['sources'])):
        raise ValueError('Source inventories differ or source identities are ambiguous')
    removed = [k for k, mid in before['candidate_map'].items() if mid not in new_ids]
    dropped = presentation(before, drop=removed)
    by_mid = {mid: key for key, mid in after['candidate_map'].items()}
    candidate_names = {key: by_mid[mid] for key, mid in dropped['candidate_map'].items()}
    compact = presentation(dropped, candidate_names=candidate_names)
    source_names = {sid: new_sources[identity] for identity, sid in old_sources.items()}
    source_order = [old_sources[fingerprint(source_identity(s))] for s in new_data['sources']]
    result = dict(original=before, membership_only=dropped, candidate_aliases=compact,
                  source_order_only=presentation(compact, source_order=source_order),
                  source_aliases_only=presentation(compact, source_names=source_names),
                  current=presentation(compact, source_order=source_order, source_names=source_names))
    if request_for(result['current']) != request_for(after):
        raise ValueError('Isolated transformations do not reconstruct the actual current request')
    for built in result.values():
        _, data = payload(built)
        if sorted(fingerprint(source_identity(s)) for s in data['sources']) != sorted(old_sources):
            raise ValueError('Source content or roles changed')
        if data['targets'] != old_data['targets']:
            raise ValueError('Target text changed')
    return result


async def main(args):
    check_runtime()
    paths = [args.output, args.output.with_suffix('.plan.json'), args.output.with_suffix('.live.jsonl')]
    if any(path.exists() for path in paths):
        raise FileExistsError(args.output)
    req, old_batches = load_batches(args.before, args.qa_id)
    new_req, new_batches = load_batches(args.after, args.qa_id)
    if req != new_req:
        raise ValueError('Questions differ')
    matches = []
    for old_index, old in old_batches:
        for new_index, new in new_batches:
            try:
                built = variants(old, new)
            except ValueError:
                continue
            matches.append((old_index, new_index, built))
    if len(matches) != 1:
        raise ValueError('Require exactly one unambiguous batch transition')
    old_index, new_index, built = matches[0]
    requests = {(name, guidance): request_for(value, guidance) for name, value in built.items() for guidance in (False, True)}
    plan = dict(model=REQUIRED_MODEL, before=str(args.before), after=str(args.after), qa_id=args.qa_id,
                old_batch=old_index, new_batch=new_index, repeats=args.repeats,
                expected_calls=len(requests)*args.repeats if args.live else 0,
                variants=[dict(name=name, guidance=guidance, request_sha256=fingerprint(request),
                               reservation=request['reservation'], candidate_map=built[name]['candidate_map'])
                          for (name, guidance), request in requests.items()], limitation=__doc__)
    paths[1].write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    rows = []
    if args.live:
        if config.FAKE:
            raise ValueError('Real provider required')
        keys = list(requests)
        for repeat in range(args.repeats):
            order = keys if repeat % 2 == 0 else list(reversed(keys))
            for name, guidance in order:
                check_runtime()
                request = copy.deepcopy(requests[name, guidance])
                reservation = request.pop('reservation')
                row = dict(name=name, guidance=guidance, repeat=repeat, model=REQUIRED_MODEL)
                with budget.scope(seconds=45, calls=1, tokens=reservation) as limits:
                    try:
                        async with asyncio.timeout(45):
                            answer = await llm.complete_json(**request, attempts=1, timeout=45,
                                stage='diagnostic.presentation.' + name)
                        selected, judgments = spans.decode_partial(answer, built[name])
                        row.update(selected_ids=sorted(selected), judgments=judgments, payload=answer)
                    except Exception as exc:
                        row.update(error=type(exc).__name__, detail=str(exc))
                row.update(calls=limits.calls, tokens=limits.tokens)
                rows.append(row)
                with paths[2].open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('payload','judgments','detail')}), flush=True)
    summary = dict(variants=len(built), live_calls=sum(r['calls'] for r in rows),
                   errors=sum('error' in r for r in rows), exact_current_request_reconstructed=True)
    paths[0].write_text(json.dumps(dict(summary=summary, plan=plan, live=rows), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if len(rows) != plan['expected_calls'] or summary['errors']:
        raise RuntimeError('Diagnostic incomplete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--qa-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Positive repeat count required')
    asyncio.run(main(args))
