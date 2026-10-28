"""Fixed review requests: inspect citation-first output order without changing sources.

No gold labels are loaded. Optional live pairs are a development diagnostic,
not a full Search/HTTPS evaluation or proof of answer accuracy.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, listwise_recovery as recovery, llm, recovery_spans, schemas


ORDER = ('target_id', 'citation_parts', 'evidence_kind', 'useful')
GUIDANCE = (' For each candidate, inspect the complete sources and targets first. '
            'Emit target_id, citation_parts, evidence_kind, and useful in that order: '
            'identify an actual contextual connection and its exact source parts before '
            'the final usefulness decision. Classify the cited source role and scope '
            'without turning interest, quotations or assistant text into personal facts. '
            'If there is no useful connection, emit an empty target_id and citation_parts, '
            'evidence_kind="none", and useful=false. Do not invent a connection to fill fields.')


def evidence_first(request):
    result = copy.deepcopy(request)
    for decision in result['schema']['properties'].values():
        if set(decision.get('properties', {})) != set(ORDER):
            raise ValueError('Requires the flat span-reference decision schema')
        decision['properties'] = {name: decision['properties'][name] for name in ORDER}
        decision['required'] = list(ORDER)
    result['system'] += GUIDANCE
    result['reservation'] += len(GUIDANCE.encode())
    return result


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def main(args):
    live_path = args.output.with_suffix('.live.jsonl')
    if args.output.exists() or live_path.exists():
        raise FileExistsError(args.output)
    if not (args.run / 'exit-code').exists():
        raise ValueError('Use a completed arm')
    traces = {t['search_id']: t for t in records(args.run / 'server-search-debug.jsonl')}
    checks, live = [], []
    selected_qas = set(args.live_qas.split(',')) if args.live_qas else set()
    for result in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl'):
        t = traces[result['search_id']]
        req = schemas.SearchRequest(**{k: t[k] for k in
            ('user_id', 'query', 'options', 'top_k', 'reference_time', 'evidence_token_budget')})
        reviews = [('main', i, r) for i, r in enumerate(
            t['cascade']['listwise'].get('recovery', {}).get('batches', []))]
        reviews.append(('overflow', 0, t['cascade'].get('overflow', {}).get('review', {})))
        with patch.multiple(config, **t['versions']['settings']):
            if not config.LISTWISE_SPAN_REFS or config.LISTWISE_ROLE_BOUND_REFS:
                raise ValueError('Requires span references and flat role-unconstrained schema')
            for stage, batch, review in reviews:
                cards = review.get('sources', [])
                if not cards or not review.get('prompt_sha256'):
                    continue
                before = recovery.request_for(req, cards)
                after = evidence_first(before)
                if (hashlib.sha256(before['prompt'].encode()).hexdigest() != review['prompt_sha256']
                        or before['reservation'] != review['request_bound']):
                    raise ValueError('Historical request did not reproduce')
                # Required-field order and property order do not change the JSON contract.
                canonical = copy.deepcopy(after['schema'])
                for key, decision in canonical['properties'].items():
                    decision['required'] = before['schema']['properties'][key]['required']
                if canonical != before['schema'] or before['prompt'] != after['prompt']:
                    raise ValueError('Sources or output contract changed')
                if len(json.dumps(before['schema']).encode()) != len(json.dumps(after['schema']).encode()):
                    raise ValueError('Schema size changed')
                checks.append(dict(qa_id=result['qa_id'], stage=stage, batch=batch,
                    candidates=len(cards), prompt_sha256=review['prompt_sha256'],
                    input_bytes=review['input_bytes'], reservation_growth=after['reservation']-before['reservation']))
                if (result['qa_id'] not in selected_qas or stage not in args.stages.split(',')
                        or args.first_batch_only and batch != 0):
                    continue
                if config.FAKE:
                    raise ValueError('Live diagnostic requires real providers')
                built = recovery_spans.build(req, cards)
                for repeat in range(args.repeats):
                    for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                        request = copy.deepcopy(after if enabled else before)
                        reservation = request.pop('reservation')
                        row = dict(qa_id=result['qa_id'], stage=stage, batch=batch, repeat=repeat,
                            enabled=enabled, candidate_ids=[c['candidate_id'] for c in cards],
                            prompt_sha256=review['prompt_sha256'], request_bound=reservation)
                        with budget.scope(seconds=45, calls=1, tokens=reservation) as limits:
                            try:
                                async with asyncio.timeout(45):
                                    payload = await llm.complete_json(**request, attempts=1, timeout=45,
                                        stage='diagnostic.review_order.'+('treatment' if enabled else 'control'))
                                restored, judgments = recovery_spans.decode_partial(payload, built)
                                row.update(selected_ids=sorted(restored), judgments=judgments,
                                    raw_field_orders={key: list(value) if isinstance(value, dict) else None
                                                      for key, value in payload.items()})
                            except Exception as exc:
                                row.update(error=type(exc).__name__, detail=str(exc))
                        row.update(calls=limits.calls, tokens=limits.tokens)
                        live.append(row)
                        # Keep completed calls recoverable even if a later request fails.
                        with live_path.open('a') as stream:
                            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                        print(json.dumps({k: v for k, v in row.items()
                            if k not in ('judgments', 'raw_field_orders', 'detail')}), flush=True)
    summary = dict(reproduced_requests=len(checks), stages=dict(Counter(r['stage'] for r in checks)),
        source_prompts_unchanged=True, output_contract_unchanged=True,
        reservation_growths=sorted({r['reservation_growth'] for r in checks}),
        live_calls=len(live), live_errors=sum('error' in r for r in live),
        matched_live_qas=sorted({r['qa_id'] for r in live}, key=int))
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary, checks=checks, live=live,
            guidance=GUIDANCE, order=ORDER, limitation=__doc__,
            source_trace_sha256=hashlib.sha256((args.run/'server-search-debug.jsonl').read_bytes()).hexdigest()),
            stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary))
    if selected_qas != {r['qa_id'] for r in live} or summary['live_errors']:
        raise RuntimeError('Live diagnostic incomplete or failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live-qas', help='Comma-separated QA IDs; omit for offline checks only')
    parser.add_argument('--stages', choices=['main', 'overflow', 'main,overflow'], default='main')
    parser.add_argument('--first-batch-only', action='store_true', help='Limit live calls, not offline checks')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('Positive repeat count required')
    asyncio.run(main(args))
