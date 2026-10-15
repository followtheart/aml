"""Paired downstream replay using previously measured focused-window CE scores."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, eval_scoring
from replay_https_ranking_budget import records, run_ranking


async def main(args):
    if config.FAKE:
        raise ValueError('Real downstream provider required')
    probe = json.loads(args.probe.read_text())
    result = next(r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')
                  if r['qa_id'] == probe['qa_id'])
    trace = next(t for t in records(args.run / 'server-search-debug.jsonl')
                 if t.get('search_id') == result['search_id'])
    if probe['search_id'] != trace['search_id'] or probe['query'] != trace['cascade']['cross_encoder']['query']:
        raise ValueError('Probe query/search identity mismatch')
    by_id = {c['id']: c for c in trace['ranked']}
    for c in probe['candidates']:
        old = by_id[c['id']]
        if c['whole_document'] != old.get('_rank_text', old['content']):
            raise ValueError('Probe whole-document mismatch')
    scores = {c['id']: c['focused_score'] for c in probe['candidates']}
    views = {c['id']: c['focused_document'] for c in probe['candidates']}
    if len(scores) != len(probe['candidates']):
        raise ValueError('Duplicate candidate scores')
    baseline = trace['versions']['settings']['RERANK_MAX_PROMPT_BYTES']
    checked = await run_ranking(trace, baseline, frozen=True)
    if (checked['packet_hash'] != trace['packet_hash'] or result['packet_hash'] != trace['packet_hash']
            or checked['cascade']['listwise']['candidate_ids'] != trace['cascade']['listwise']['candidate_ids']):
        raise ValueError('Frozen baseline packet/admission mismatch')
    print(json.dumps(dict(qa_id=result['qa_id'], frozen_verified=True)), flush=True)
    qa = next(q for d in records(args.prepared) if d['conversation_id'] == result['conversation_id']
              for q in d['qa'] if q['id'] == result['qa_id'])
    with args.output.open('x') as stream:
        for repetition in range(args.repeat):
            for enabled in ([False, True] if repetition % 2 == 0 else [True, False]):
                row = await run_ranking(trace, baseline, ce_scores=scores if enabled else None,
                    ranking_texts=views if enabled and args.focus_listwise else None)
                with budget.scope(seconds=240, calls=20, tokens=128000):
                    prediction, score, diagnostics = await eval_scoring.evaluate(qa, row['packet'])
                row.update(qa_id=result['qa_id'], repetition=repetition+1, enabled=enabled,
                           prediction=prediction, score=score, diagnostics=diagnostics,
                           frozen_verified=True, probe=str(args.probe),
                           focused_listwise=enabled and args.focus_listwise,
                           caveat='Original retrieval fixed; treatment uses measured focused-view CE scores. '
                           'Listwise view selected by focused_listwise flag; original packet construction '
                           'and full reader live. Embedding/CE latency and tokens omitted; no HTTPS claim.')
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                stream.flush()
                print(json.dumps(dict(qa_id=row['qa_id'], repetition=repetition+1, enabled=enabled,
                                      prediction=prediction, score=score)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run','probe','prepared','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--repeat', type=int, choices=(1,2,3), default=2)
    p.add_argument('--focus-listwise', action='store_true',
                   help='Use exact focused views for listwise too; retain original packet items')
    asyncio.run(main(p.parse_args()))
