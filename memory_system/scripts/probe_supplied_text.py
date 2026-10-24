"""Fixed-request diagnostic for non-personal supplied-text context.

No gold labels reach the reviewer. This does not change attribution validators,
relabel historical judgments, or establish full Search/answer improvements.
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


GUIDANCE = ''' Evaluate material supplied for translation, polishing or rewriting as contextual
material in its own right. If a concrete detail in that material helps assess a target, cite its
substantive original passage and use third_party_context, meaning non-personal supplied-text context.
Use this contextual label for the supplied passage even if the outer request claims authorship or
the passage uses first person. Authorship of a submitted text does not certify its autobiographical
truth. Do not classify that passage as user_context, topic_interest or counterevidence solely from
its wording. This instruction does not relabel ordinary direct user statements or topical questions.
Preserve and read the complete outer request, source role, speaker, conditions, negation and fictional
or quoted framing. Context does not establish a personal fact or prove an answer option. An assistant
rewrite remains assistant material, not independent user proof. A generic editing request, shared
words, or an unrelated passage is not useful. If useful, select only one or two substantive exact
source parts owned by that candidate; never cite only an authorship statement or task instruction.
If no such connection is supported, return the normal negative decision.'''


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def main(args):
    live_path = args.output.with_suffix('.live.jsonl')
    plan_path = args.output.with_suffix('.plan.json')
    if any(p.exists() for p in (args.output, live_path, plan_path)):
        raise FileExistsError(args.output)
    if not (args.run/'exit-code').exists():
        raise ValueError('Use a completed arm')
    traces = {t['search_id']: t for t in records(args.run/'server-search-debug.jsonl')}
    checks, jobs, live = [], [], []
    qas = set(args.live_qas.split(',')) if args.live_qas else set()
    for result in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
        t = traces[result['search_id']]
        req = schemas.SearchRequest(**{k: t[k] for k in
            ('user_id', 'query', 'options', 'top_k', 'reference_time', 'evidence_token_budget')})
        reviews = [('main', i, r) for i, r in enumerate(
            t['cascade']['listwise'].get('recovery', {}).get('batches', []))]
        reviews.append(('overflow', 0, t['cascade'].get('overflow', {}).get('review', {})))
        with patch.multiple(config, **t['versions']['settings']):
            if not config.LISTWISE_SPAN_REFS or config.LISTWISE_TRANSFORM_CONTEXT:
                raise ValueError('Requires baseline span references without transform guidance')
            for stage, batch, review in reviews:
                cards = review.get('sources', [])
                if not cards or not review.get('prompt_sha256'):
                    continue
                before = recovery.request_for(req, cards)
                after = copy.deepcopy(before)
                after['system'] += GUIDANCE
                after['reservation'] += len(GUIDANCE.encode())
                if (hashlib.sha256(before['prompt'].encode()).hexdigest() != review['prompt_sha256']
                        or before['reservation'] != review['request_bound']):
                    raise ValueError('Historical request did not reproduce')
                for field in ('prompt', 'schema', 'max_tokens'):
                    if before[field] != after[field]:
                        raise ValueError('Source or output contract changed')
                built = recovery_spans.build(req, cards, role_bound=config.LISTWISE_ROLE_BOUND_REFS)
                # Revalidate every decodable historical decision, including negative
                # controls, without changing any label or accepting invalid citations.
                reproduced = 0
                for judgment in review.get('judgments', []):
                    if 'span_reference_error' in judgment['validation_errors']:
                        continue
                    fields = recovery.Decision.model_fields
                    card = next(c for c in cards if c['candidate_id'] == judgment['candidate_id'])
                    _, checked = recovery.validate(dict(decisions=[{k: judgment[k] for k in fields}]),
                                                   [card], built['targets'])
                    if any(checked[0][key] != judgment[key] for key in ('valid', 'validation_errors')):
                        raise ValueError('Historical validator outcome differs')
                    reproduced += 1
                meta = dict(qa_id=result['qa_id'], stage=stage, batch=batch,
                    candidates=len(cards), reproduced_judgments=reproduced,
                    prompt_sha256=review['prompt_sha256'],
                    reservation_growth=after['reservation']-before['reservation'])
                checks.append(meta)
                if result['qa_id'] in qas and stage in args.stages.split(',') and batch == 0:
                    jobs.append((meta, before, after, built))
    if qas != {m['qa_id'] for m, *_ in jobs}:
        raise ValueError('Requested live questions not all found')
    plan = dict(run=str(args.run), guidance=GUIDANCE, repeats=args.repeats,
                expected_calls=len(jobs)*args.repeats*2,
                paired_requests=[m for m, *_ in jobs], checks=checks,
                limitation=__doc__)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    if jobs and config.FAKE:
        raise ValueError('Live diagnostic requires real providers')
    for meta, before, after, built in jobs:
        for repeat in range(args.repeats):
            for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                request = copy.deepcopy(after if enabled else before)
                reservation = request.pop('reservation')
                row = dict(**meta, repeat=repeat, enabled=enabled, request_bound=reservation)
                with budget.scope(seconds=45, calls=1, tokens=reservation) as limits:
                    try:
                        async with asyncio.timeout(45):
                            payload = await llm.complete_json(**request, attempts=1, timeout=45,
                                stage='diagnostic.supplied_text.'+('treatment' if enabled else 'control'))
                        restored, judgments = recovery_spans.decode_partial(payload, built)
                        row.update(selected_ids=sorted(restored), judgments=judgments, payload=payload)
                    except Exception as exc:
                        row.update(error=type(exc).__name__, detail=str(exc))
                row.update(calls=limits.calls, tokens=limits.tokens)
                live.append(row)
                with live_path.open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                print(json.dumps({k: v for k, v in row.items()
                    if k not in ('judgments', 'payload', 'detail')}), flush=True)
    summary = dict(reproduced_requests=len(checks),
        reproduced_judgments=sum(c['reproduced_judgments'] for c in checks),
        stages=dict(Counter(c['stage'] for c in checks)),
        source_prompts_and_schemas_unchanged=True,
        reservation_growths=sorted({c['reservation_growth'] for c in checks}),
        live_calls=sum(r['calls'] for r in live), live_errors=sum('error' in r for r in live))
    args.output.write_text(json.dumps(dict(summary=summary, checks=checks, live=live,
                                          limitation=__doc__), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    if len(live) != plan['expected_calls'] or summary['live_errors']:
        raise RuntimeError('Live diagnostic incomplete or provider failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--live-qas', default='')
    p.add_argument('--stages', choices=['main', 'overflow', 'main,overflow'], default='overflow')
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1:
        p.error('Positive repeats required')
    asyncio.run(main(args))
