"""Run live listwise/recovery on frozen admission sets, without rerunning Add."""
import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, llm, cascade_rerank as ranking, listwise_recovery
from probe_ranking_capacity import replay


async def main(args):
    if config.FAKE:
        raise RuntimeError('Requires real provider')
    config.MEMORY_DEBUG_LOG = config.SEARCH_DEBUG_LOG = ''
    records = {r['qa_id']: r for r in map(json.loads,
        (args.run / 'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines())}
    searches = {s['search_id']: s for s in map(json.loads,
        (args.run / 'completed/logs/search-debug.jsonl').read_text().splitlines())}
    audit = {r['qa_id']: r for r in json.loads((args.run / 'funnel/audit.json').read_text())['rows']}
    with args.output.open('x') as output:
        for qid in args.questions.split(','):
            search = searches[records[qid]['search_id']]
            by_id = {r['id']: r for r in search['ranked']}
            req = SimpleNamespace(query=search['query'], options=search['options'])
            plan = copy.deepcopy(search['plan'])
            plan.update(_coverage_requirements=[{k: r[k] for k in ('id', 'text', 'origin')}
                for r in search['coverage_manifest']['coverage']['requirements']],
                _fusion_triples=search['fusion_triples'], _fusion_edges=search['fusion_edges'])
            settings = search['versions']['settings']
            for mode, fine, limit in [('baseline', settings['CASCADE_FINE_LIMIT'], settings['CASCADE_LLM_LIMIT']),
                                      ('capacity16', 16, 16)]:
                admission = replay(search, fine, limit)
                submitted = [by_id[mid] for mid in admission['input_ids']]
                row = dict(qa_id=qid, mode=mode, admission=admission, calls=[],
                    limitation='Frozen CE/candidates; live listwise and recovery. Excludes earlier search budget and final packing/answer.')
                with budget.scope(seconds=90, calls=4, tokens=128000) as usage:
                    try:
                        for attempt in range(2):
                            request = ranking.listwise_request(req, plan, submitted, repair=bool(attempt))
                            reserve = request.pop('reservation')
                            prompt = request.pop('prompt')
                            usage.reserve_tokens(reserve)
                            try:
                                response = await llm.complete_json(prompt, **request,
                                    stage='probe.capacity_listwise', attempts=1, timeout=20)
                            finally:
                                usage.release_tokens(reserve)
                            row['calls'].append(dict(prompt=prompt, response=response))
                            try:
                                order, irrelevant, groups = ranking.validate(response, len(submitted))
                                row['listwise_status'] = 'recovered' if attempt else 'ok'
                                break
                            except (ValueError, TypeError, KeyError) as exc:
                                row.setdefault('validation_errors', []).append(str(exc))
                                if attempt:
                                    # Production retains the submitted CE order when both
                                    # structured ranking attempts fail validation.
                                    order, irrelevant, groups = list(range(len(submitted))), set(), []
                                    row['listwise_status'] = 'fallback'
                        selected = [submitted[i] for i in order if i not in irrelevant]
                        row['primary_selected_ids'] = [r['id'] for r in selected]
                        if irrelevant:
                            selected, recovery = await listwise_recovery.recover(req, submitted, selected,
                                irrelevant, deadline=min(usage.deadline, time.monotonic() + config.RERANK_DEADLINE_SECONDS))
                            row['recovery'] = recovery
                        selected_ids = {r['id'] for r in selected}
                        row.update(selected_ids=[r['id'] for r in selected],
                            groups=[[submitted[i]['id'] for i in g] for g in groups],
                            probes=[dict(label=p['label'], quote=p['quote'],
                                admitted=bool(set(p['stage_visible_quote_carriers']['coarse']) & set(admission['input_ids'])),
                                retained=bool(set(p['stage_visible_quote_carriers']['coarse']) & selected_ids))
                                for p in audit[qid]['probes']])
                    except Exception as exc:
                        row.update(error_type=type(exc).__name__, error=str(exc)[:1000])
                    row['budget_usage'] = dict(calls=usage.calls, max_calls=usage.max_calls,
                                              tokens=usage.tokens, max_tokens=usage.max_tokens)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                print(json.dumps({k: row.get(k) for k in ('qa_id', 'mode', 'probes', 'error')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--questions', default='2,10,20,8')
    p.add_argument('--output', type=Path, required=True)
    asyncio.run(main(p.parse_args()))
