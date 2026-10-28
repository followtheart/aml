"""Compare the capacity simulator with the actual cascade, without networking."""
import argparse
import asyncio
import copy
import json
from contextlib import ExitStack
from unittest.mock import patch

from probe_ranking_capacity import audit, coarse_rows, replay
from app import cascade_rerank, config, cross_encoder, llm, schemas


async def verify(run):
    results = []
    for search in audit.records(run / 'server-search-debug.jsonl'):
        settings = search['versions']['settings']
        # The fixture is the frozen coarse boundary, not a new retrieval run.
        rows = coarse_rows(search)
        scores = {}
        for c in rows:
            text = c.get('_rank_text', c['content'])
            if text in scores and scores[text] != c['_ce_score']:
                raise ValueError('Ambiguous frozen CE document')
            scores[text] = c['_ce_score']
        if any(score is None for score in scores.values()):
            raise ValueError('Use failure-specific unit tests for partial CE')
        for adaptive in (False, True):
            policy = 'adaptive_shared' if adaptive else 'bounded'
            expected = replay(search, 24 if adaptive else 16, 24 if adaptive else 16, policy)
            plan = copy.deepcopy(search['plan'])
            plan.update(_coverage_requirements=[{k: r[k] for k in ('id', 'text', 'origin')}
                        for r in search['coverage_manifest']['coverage']['requirements']],
                        _query_specs=search['query_specs'],
                        _fusion_triples=search['fusion_triples'], _fusion_edges=search['fusion_edges'])
            req = schemas.SearchRequest(user_id=search['user_id'], query=search['query'],
                                        options=search['options'], top_k=50)
            async def ce(query, documents, **kwargs):
                if query != search['cascade']['cross_encoder']['query']:
                    raise ValueError('CE query changed')
                return [scores[text] for text in documents]
            async def classify(prompt, **kwargs):
                if kwargs['stage'] != 'search.listwise':
                    raise ValueError('Unexpected provider stage')
                count = len(plan['_cascade']['fine']['selected_ids'])
                # The provider schema, not fine capacity, gives admitted count.
                properties = kwargs['schema']['properties']
                if 'candidates' in properties:
                    return dict(candidates={key: dict(priority=i, relevance='useful')
                                for i, key in enumerate(properties['candidates']['required'])}, groups=[])
                count = properties['ranking']['maxItems']
                return dict(ranking=list(range(count)), irrelevant=[], groups=[])
            with ExitStack() as stack:
                for name, value in settings.items():
                    if hasattr(config, name):
                        stack.enter_context(patch.object(config, name, value))
                stack.enter_context(patch.multiple(config, CASCADE_COARSE_LIMIT=50,
                    CASCADE_FINE_LIMIT=16, CASCADE_LLM_LIMIT=16,
                    FINE_EVIDENCE_ADAPTIVE=adaptive, FINE_EXTRA_LIMIT=8,
                    LISTWISE_KEYED_OUTPUT=settings.get('LISTWISE_KEYED_OUTPUT', False),
                    LISTWISE_UNREVIEWED_LIMIT=0,
                    LISTWISE_SHARED_SOURCES=True if adaptive else settings.get('LISTWISE_SHARED_SOURCES', False)))
                stack.enter_context(patch.object(cross_encoder, 'rerank', side_effect=ce))
                stack.enter_context(patch.object(llm, 'complete_json', side_effect=classify))
                stack.enter_context(patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')))
                await cascade_rerank.rank(req, plan, copy.deepcopy(rows))
            actual = plan['_cascade']
            checks = dict(fine_ids=actual['fine']['selected_ids'],
                          input_ids=actual['listwise']['candidate_ids'],
                          prompt_bytes=actual['listwise']['input_bytes'])
            for name, value in checks.items():
                if value != expected[name]:
                    raise ValueError(f"Runtime/simulator mismatch: {search['search_id']} {policy} {name}")
            if actual['cross_encoder']['status'] != 'ok' or actual['listwise']['errors']:
                raise ValueError('Unexpected degraded fixture')
            for d in actual.get('adaptive_admission', {}).get('decisions', []):
                if d['admitted']:
                    continue
                for mid in set(d['group_ids']) - set(actual['listwise']['candidate_ids']):
                    if not any(o['id'] == mid and o['stage'] == 'listwise' and o['reason'] == d['reason']
                               for o in actual['omitted']):
                        raise ValueError('Adaptive admission reason was lost from omissions')
            results.append(dict(search_id=search['search_id'], policy=policy, verified=True))
    return dict(count=len(results), checks=['fine_ids', 'listwise_input_ids', 'prompt_bytes'],
                network_calls=0, rows=results)


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = asyncio.run(verify(args.run))
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}))
