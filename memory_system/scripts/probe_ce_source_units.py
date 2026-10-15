"""Live CE-only granularity diagnostic; preserve all source text and speaker labels.

Compare whole ranking units with max scores of their visible source messages.
This is not a retrieval/reader replay. No gold labels enter provider requests.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, budget, config, cross_encoder


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


async def score_documents(query, documents):
    batches, pending = [], []
    for document in documents:
        if (len(document.encode()) > config.CE_MAX_DOCUMENT_BYTES or
                cross_encoder.request_size(query, [document]) > config.CE_MAX_REQUEST_BYTES):
            raise ValueError('Untruncated source exceeds CE bounds')
        if pending and (len(pending) >= config.CE_BATCH_SIZE or
                        cross_encoder.request_size(query, pending + [document]) > config.CE_MAX_REQUEST_BYTES):
            batches.append(pending)
            pending = []
        pending.append(document)
    if pending:
        batches.append(pending)
    scores = []
    for batch in batches:
        scores.extend(await cross_encoder.rerank(query, batch, stage='probe.ce_source_units'))
    return scores, len(batches)


async def main(args):
    if config.FAKE:
        raise ValueError('Real cross encoder required')
    wanted = set(args.questions.split(','))
    query_plan = json.loads(args.query_plan.read_text()) if args.query_plan else None
    if query_plan and wanted != {query_plan['qa_id']}:
        raise ValueError('Query plan must match the single requested question')
    result_ids = {r['search_id']: r['qa_id'] for r in records(
        args.run / 'completed/data/results/personamem-v2-32k.jsonl') if r['qa_id'] in wanted}
    if set(result_ids.values()) != wanted:
        raise ValueError('All selected questions need completed results')
    with args.output.open('x') as output:
        for trace in records(args.run / 'server-search-debug.jsonl'):
            if trace.get('search_id') not in result_ids:
                continue
            by_id = {r['id']: r for r in trace['ranked']}
            candidates = [by_id[mid] for mid in trace['cascade']['coarse']['selected_ids']]
            query = trace['cascade']['cross_encoder']['query']
            whole = [r.get('_rank_text', r['content']) for r in candidates]
            if query_plan:
                if (trace['search_id'] != query_plan['search_id'] or
                        [c['id'] for c in candidates] != query_plan['candidate_ids'] or
                        [hashlib.sha256(d.encode()).hexdigest() for d in whole] != query_plan['document_hashes']):
                    raise ValueError('Query probe source identity changed')
                query = query_plan['queries']['treatment']
            documents, lookup, mappings = [], {}, []
            for candidate, text in zip(candidates, whole):
                refs = []
                for source in (candidate.get('_packet_item') or {}).get('sources', []):
                    content = source.get('content')
                    if not content:
                        continue
                    if content not in text:
                        raise ValueError('Visible source is not in the actual ranking unit')
                    document = answer_context.with_evidence('', [source])
                    if document not in lookup:
                        lookup[document] = len(documents)
                        documents.append(document)
                    refs.append(dict(index=lookup[document], role=source.get('role'),
                                     request_id=source.get('request_id'),
                                     message_index=source.get('message_index'),
                                     content_span=source.get('content_span')))
                mappings.append(refs)
            with budget.scope(seconds=240, calls=100, tokens=1000000) as usage:
                original_scores, whole_batches = await score_documents(query, whole)
                source_scores, source_batches = await score_documents(query, documents)
            rows = []
            for candidate, original, refs in zip(candidates, original_scores, mappings):
                all_scores = [source_scores[r['index']] for r in refs]
                user_scores = [source_scores[r['index']] for r in refs if r['role'] == 'user']
                rows.append(dict(id=candidate['id'], type=candidate['type'], refs=refs,
                                 whole_score=original, source_max=max(all_scores, default=original),
                                 user_source_max=max(user_scores, default=original)))
            orders = {mode: [r['id'] for r in sorted(rows, key=lambda r: -r[mode])]
                      for mode in ('whole_score', 'source_max', 'user_source_max')}
            row = dict(qa_id=result_ids[trace['search_id']], query=query, candidates=rows,
                       query_plan=str(args.query_plan) if args.query_plan else None,
                       whole_documents=whole, source_documents=documents, source_scores=source_scores,
                       orders=orders, whole_batches=whole_batches, source_batches=source_batches,
                       calls=usage.calls, tokens=usage.tokens,
                       caveat='Live CE-only probe, expanded diagnostic budget. Max aggregation may '
                              'favor multi-source candidates. No listwise, packing or reader score claim; '
                              'complete visible messages/excerpts retain role and timestamp, no gold supplied.')
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps(dict(qa_id=row['qa_id'], candidates=len(rows), sources=len(documents),
                                  whole_batches=whole_batches, source_batches=source_batches)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--questions', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--query-plan', type=Path,
                   help='Use the option-text probe treatment query after validating all candidate hashes')
    asyncio.run(main(p.parse_args()))
