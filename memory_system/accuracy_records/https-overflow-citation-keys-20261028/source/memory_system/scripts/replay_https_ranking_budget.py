"""Replay real ranking/packing with frozen CE scores and live downstream calls.

First reproduce archived admission and packet hash without network access.
Then compare prompt-byte budgets using the actual cascade implementation.
Upstream retrieval, ingestion, elapsed time and token consumption are frozen
or omitted; this is a diagnostic, not a fresh HTTPS accuracy measurement.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank, config, cross_encoder, eval_scoring
from app import listwise_recovery, llm, schemas, search_pipeline
from replay_query_context import records


async def run_ranking(trace, byte_limit, *, frozen=False, shared_sources=False, ce_scores=None,
                      ranking_texts=None, capacity_limit=None):
    if capacity_limit is not None and not 1 <= capacity_limit <= 50:
        raise ValueError('Capacity must be between 1 and 50')
    if frozen and (ce_scores is not None or ranking_texts is not None or capacity_limit is not None):
        raise ValueError('Frozen baseline validation must use archived CE scores')
    by_id = {r['id']: copy.deepcopy(r) for r in trace['ranked']}
    if ranking_texts is not None:
        if set(ranking_texts) != set(trace['cascade']['coarse']['selected_ids']):
            raise ValueError('Ranking views must cover exactly the original coarse pool')
    order = [r['id'] for r in trace['fused'] if r['id'] in by_id]
    order += [mid for mid in by_id if mid not in order]
    candidates = [by_id[mid] for mid in order]
    req = schemas.SearchRequest(**{k: trace[k] for k in
        ('query', 'options', 'user_id', 'top_k', 'reference_time', 'evidence_token_budget')})
    plan = copy.deepcopy(trace['plan'])
    plan.update(_coverage_requirements=[{k: r[k] for k in ('id', 'text', 'origin')}
        for r in trace['coverage_manifest']['coverage']['requirements']],
        _fusion_triples=trace['fusion_triples'], _fusion_edges=trace['fusion_edges'])
    ce = trace['cascade']['cross_encoder']
    if ce['status'] != 'ok':
        raise ValueError('Partial CE traces require separate replay handling')
    scores = {}
    if ce_scores is not None and set(ce_scores) != set(trace['cascade']['coarse']['selected_ids']):
        raise ValueError('CE overrides must cover exactly the original coarse pool')
    for mid in trace['cascade']['coarse']['selected_ids']:
        row = by_id[mid]
        text = row.get('_rank_text', row['content'])
        value = row['_ce_score'] if ce_scores is None else ce_scores[mid]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid CE score')
        if text in scores and scores[text] != value:
            raise ValueError('Ambiguous document score')
        scores[text] = value
        if ranking_texts is not None:
            focused = ranking_texts[mid]
            if focused in scores and scores[focused] != value:
                raise ValueError('Ambiguous focused document score')
            scores[focused] = value
    violations, calls = [], []
    provider = llm.complete_json
    original_shortlist = cascade_rerank.shortlist
    coarse_checked = False

    def focused_shortlist(*args, **kwargs):
        nonlocal coarse_checked
        selected = original_shortlist(*args, **kwargs)
        if not coarse_checked:
            coarse_checked = True
            if [c['id'] for c in selected] != trace['cascade']['coarse']['selected_ids']:
                raise ValueError('Coarse candidate pool changed before focused-view intervention')
            for candidate in selected:
                candidate['_rank_text'] = ranking_texts[candidate['id']]
        return selected

    async def frozen_ce(query, documents, **kwargs):
        if query != ce['query'] or any(doc not in scores for doc in documents):
            violations.append('cross_encoder_input')
            raise ValueError('Cross-encoder inputs changed')
        budget.current.get().before_call()
        return [scores[doc] for doc in documents]

    async def recorded(prompt, *positional, **kwargs):
        if frozen:
            if kwargs['stage'] != 'search.listwise':
                violations.append(kwargs['stage'])
                raise ValueError('Unexpected network request in frozen validation')
            old = trace['cascade']['listwise']
            ids = old['candidate_ids']
            expected = cascade_rerank.listwise_request(req, plan, [by_id[mid] for mid in ids])
            if prompt != expected['prompt']:
                violations.append('listwise_prompt')
                raise ValueError('Archived listwise prompt changed')
            primary = old['primary_selected_ids']
            permutation = primary + [mid for mid in ids if mid not in primary]
            budget.current.get().before_call()
            return dict(ranking=[ids.index(mid) for mid in permutation],
                        irrelevant=[ids.index(mid) for mid in old['primary_irrelevant_ids']],
                        groups=[[ids.index(mid) for mid in group] for group in trace['evidence_groups']])
        call = dict(stage=kwargs['stage'], prompt=prompt,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        calls.append(call)
        try:
            response = await provider(prompt, *positional, **kwargs)
            call['response'] = response
            return response
        except Exception as exc:
            call['error_type'] = type(exc).__name__
            raise

    async def frozen_recovery(req, submitted, selected, irrelevant, **kwargs):
        old = trace['cascade']['listwise']
        return [by_id[mid] for mid in old['selected_ids']], copy.deepcopy(old['recovery'])

    from contextlib import ExitStack
    with ExitStack() as stack:
        if capacity_limit is not None:
            stack.enter_context(patch.object(config, 'CASCADE_FINE_LIMIT', capacity_limit))
            stack.enter_context(patch.object(config, 'CASCADE_LLM_LIMIT', capacity_limit))
        stack.enter_context(patch.object(config, 'RERANK_MAX_PROMPT_BYTES', byte_limit))
        stack.enter_context(patch.object(config, 'LISTWISE_SHARED_SOURCES', shared_sources))
        stack.enter_context(patch.object(cross_encoder, 'rerank', frozen_ce))
        if ranking_texts is not None:
            # Apply the representation change after original coarse admission.
            # Otherwise source equivalence during coarse selection can change
            # the pool before the measured scores are available.
            stack.enter_context(patch.object(cascade_rerank, 'shortlist', focused_shortlist))
        # The final search plan can include later follow-up subquestions. CE
        # was run before those additions; freeze its actual archived query.
        stack.enter_context(patch.object(cascade_rerank, 'query_text', lambda req, plan: ce['query']))
        stack.enter_context(patch.object(llm, 'complete_json', recorded))
        if frozen:
            stack.enter_context(patch.object(listwise_recovery, 'recover', frozen_recovery))
        with budget.scope(seconds=config.SEARCH_DEADLINE_SECONDS,
                          calls=config.SEARCH_MAX_CALLS, tokens=config.SEARCH_MAX_TOKENS) as usage:
            ranked = await cascade_rerank.rank(req, plan, candidates)
            selected = search_pipeline._select_evidence(req, plan, ranked)
            packet, packet_hash, manifest = search_pipeline._pack_evidence(
                None, req, plan, selected, trace['anchor_time'])
    if violations:
        raise ValueError(str(violations))
    return dict(packet=packet, packet_hash=packet_hash, manifest=manifest,
                cascade=plan['_cascade'], calls=calls, usage=dict(calls=usage.calls, tokens=usage.tokens))


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    if args.capacity_limit is not None:
        if args.shared_sources or args.questions != 'all':
            raise ValueError('Capacity comparison requires all questions and unchanged source formatting')
        summary = json.loads((args.run / 'funnel/https-summary.json').read_text())
        manifest = json.loads((args.run / 'isolated-manifest.json').read_text())
        if summary['partial'] or 'client_exit_code' not in manifest:
            raise ValueError('Capacity comparison requires a completed source experiment')
    results = [r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')
               if args.questions == 'all' or r['qa_id'] in args.questions.split(',')]
    if not results:
        raise ValueError('No selected results')
    if args.capacity_limit is not None:
        if (len(results) != summary['expected_qa']
                or len({(r['conversation_id'], r['qa_id']) for r in results}) != len(results)):
            raise ValueError('Incomplete or duplicate source cohort')
    wanted = {r['search_id'] for r in results}
    traces = {}
    with (args.run / 'server-search-debug.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('search_id') in wanted:
                traces[row['search_id']] = row
    qa = {(d['conversation_id'], q['id']): q
          for d in records(args.prepared) for q in d['qa']}
    for result in results:
        trace = traces[result['search_id']]
        baseline = trace['versions']['settings']['RERANK_MAX_PROMPT_BYTES']
        check = await run_ranking(trace, baseline, frozen=True)
        if (check['packet_hash'] != trace['packet_hash'] or trace['packet_hash'] != result['packet_hash']
                or check['cascade']['listwise']['candidate_ids'] != trace['cascade']['listwise']['candidate_ids']):
            raise ValueError('Frozen baseline packet/admission reproduction failed')
        print(dict(qa_id=result['qa_id'], frozen_packet_verified=True), flush=True)
    if args.inspect:
        return
    if args.capacity_limit is not None:
        args.output.with_suffix('.plan.json').write_text(json.dumps(dict(
            run=str(args.run), prepared=str(args.prepared), questions=[r['qa_id'] for r in results],
            capacity_limit=args.capacity_limit, repeat=args.repeat,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            intervention='Patch fine and LLM caps together; original byte budget and CE scores retained',
            caveat='Frozen retrieval/CE, live listwise and reader; not fresh HTTPS'), indent=2)+'\n')
    with args.output.open('x') as output:
        for repetition in range(args.repeat):
            for index, result in enumerate(results):
                trace = traces[result['search_id']]
                baseline = trace['versions']['settings']['RERANK_MAX_PROMPT_BYTES']
                arms = ([(baseline, False), (baseline, True)] if args.shared_sources
                        else [(baseline, False), (args.prompt_bytes, False)])
                if (index + repetition) % 2:
                    arms.reverse()
                arms = [(limit, shared, None) for limit, shared in arms]
                if args.capacity_limit is not None:
                    arms = [(baseline, False, None), (baseline, False, args.capacity_limit)]
                    if (index + repetition) % 2:
                        arms.reverse()
                for limit, shared_sources, capacity in arms:
                    row = await run_ranking(trace, limit, shared_sources=shared_sources,
                                            capacity_limit=capacity)
                    if args.answer:
                        with budget.scope(seconds=240, calls=20, tokens=128000):
                            prediction, score, diagnostics = await eval_scoring.evaluate(
                                qa[result['conversation_id'], result['qa_id']], row['packet'])
                        row.update(prediction=prediction, score=score, diagnostics=diagnostics)
                    row.update(qa_id=result['qa_id'], repetition=repetition + 1, prompt_bytes=limit,
                               capacity_limit=capacity,
                               shared_sources=shared_sources,
                               baseline_packet_hash=trace['packet_hash'], frozen_packet_verified=True,
                               caveat='Frozen retrieval and CE scores; live listwise/recovery/answer. '
                                      'Earlier token and elapsed-time use omitted; no end-to-end claim.')
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    output.flush()
                    print(dict(qa_id=row['qa_id'], repetition=repetition + 1, prompt_bytes=limit,
                               admitted=len(row['cascade']['listwise']['candidate_ids']),
                               prediction=row.get('prediction'), score=row.get('score')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--capacity-limit', type=int, choices=range(1, 51),
                        help='Compare original caps against this fine/LLM cap on --questions all')
    parser.add_argument('--prompt-bytes', type=int, default=32000)
    parser.add_argument('--repeat', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--answer', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--shared-sources', action='store_true',
                        help='Compare shared-source mode at the same original byte budget')
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
