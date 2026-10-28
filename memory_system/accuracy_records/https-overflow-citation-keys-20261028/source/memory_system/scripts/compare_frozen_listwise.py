"""Live paired protocols on identical archived listwise inputs; no CE/ingestion.

Not an end-to-end accuracy evaluation. Calls use current repair/review policy in
both arms, isolating the output protocol. Alternate arm order across questions.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank, config, listwise_contract, listwise_recovery, llm, fine_evidence, run_metadata


async def evaluate(trace, keyed, *, request_transform=None):
    by_id = {c['id']: c for c in trace['ranked']}
    submitted = [copy.deepcopy(by_id[mid]) for mid in trace['cascade']['listwise']['candidate_ids']]
    req = SimpleNamespace(query=trace['query'], options=trace['options'])
    plan = dict(trace['plan'], _fusion_triples=trace['fusion_triples'], _fusion_edges=trace['fusion_edges'],
        _coverage_requirements=[{k: r[k] for k in ('id', 'text', 'origin')}
            for r in trace['coverage_manifest']['coverage']['requirements']],
        _fine_path_groups=trace['cascade']['fine']['selection'].get('adaptive', {}).get('path_groups', []))
    original = cascade_rerank.listwise_request(req, plan, submitted, keyed=True,
        shared_sources=trace['cascade']['listwise']['shared_sources'])
    if len(original['prompt'].encode()) != trace['cascade']['listwise']['input_bytes']:
        raise ValueError('Archived primary prompt was not reproduced')
    row = dict(protocol='keyed' if keyed else 'permutation', candidate_ids=[c['id'] for c in submitted],
               input_hash=hashlib.sha256(json.dumps(submitted, sort_keys=True).encode()).hexdigest(), attempts=[])
    state = listwise_contract.State(len(submitted), keyed=keyed,
        tie_order=sorted(range(len(submitted)), key=lambda i: fine_evidence.key(submitted[i]), reverse=True))
    field, feedback = None, None
    started = time.monotonic()
    with budget.scope(seconds=90, calls=6, tokens=256000) as usage:
        for attempt in range(3):
            dispatched = False
            try:
                request = cascade_rerank.listwise_request(req, plan, submitted, keyed=keyed,
                    repair=bool(attempt), repair_field=field, feedback=feedback,
                    locked={k: v for k, v in state.payload.items() if k != field and k not in state.errors})
                if request_transform is not None:
                    request = request_transform(request)
                if len(request['prompt'].encode()) > config.RERANK_MAX_PROMPT_BYTES:
                    raise budget.BudgetExceeded('prompt_bytes')
                reservation = request.pop('reservation')
                usage.reserve_tokens(reservation)
                try:
                    dispatched = True
                    response = await llm.complete_json(request.pop('prompt'), **request,
                        stage='probe.frozen_listwise' + ('.repair' if attempt else ''), attempts=1, timeout=30)
                finally:
                    usage.release_tokens(reservation)
                row.setdefault('responses', []).append(dict(field=field, response=copy.deepcopy(response)))
                state.accept(response, field)
                order, irrelevant, groups = state.require_valid()
                row['attempts'].append(dict(status='ok', field=field, dispatched=dispatched))
                row['primary_status'] = 'recovered' if attempt else 'ok'
                break
            except Exception as exc:
                row['attempts'].append(dict(status='error', field=field, dispatched=dispatched,
                    error=type(exc).__name__, code=getattr(exc, 'validation_code', None)))
                field = state.next_field()
                feedback = getattr(exc, 'validation_details', None)
                if cascade_rerank.listwise_retry_kind(exc) != 'structure' or attempt == 2:
                    order, irrelevant, groups = state.ranking or list(range(len(submitted))), set(), []
                    row['primary_status'] = 'ranking_only' if state.ranking is not None else 'fallback'
                    break
        row['accepted_payload'] = copy.deepcopy(state.payload)
        selected = [submitted[i] for i in order if i not in irrelevant]
        row['primary_selected_ids'] = [c['id'] for c in selected]
        row['irrelevant_ids'] = [submitted[i]['id'] for i in sorted(irrelevant)]
        selected, review = await listwise_recovery.recover(req, submitted, selected, irrelevant,
                                                          deadline=usage.deadline)
        row.update(selected_ids=[c['id'] for c in selected], recovery=review,
                   calls=usage.calls, tokens=usage.tokens, seconds=time.monotonic() - started)
    return row


async def main(args):
    if config.FAKE:
        raise RuntimeError('Real provider required')
    config.MEMORY_DEBUG_LOG = config.SEARCH_DEBUG_LOG = ''
    searches = {s['search_id']: s for s in map(json.loads,
        (args.run / 'completed/logs/search-debug.jsonl').read_text().splitlines())}
    results = list(map(json.loads, (args.run / 'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines()))
    args.output.with_suffix('.metadata.json').write_text(json.dumps(run_metadata.versions(), indent=2))
    with args.output.open('x') as output:
        for index, result in enumerate(results):
            for keyed in ([False, True] if index % 2 == 0 else [True, False]):
                row = await evaluate(searches[result['search_id']], keyed)
                row.update(qa_id=result['qa_id'], source_search_id=result['search_id'])
                output.write(json.dumps(row, ensure_ascii=False) + '\n'); output.flush()
                print(json.dumps({k: row[k] for k in ('qa_id', 'protocol', 'primary_status', 'seconds')}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
