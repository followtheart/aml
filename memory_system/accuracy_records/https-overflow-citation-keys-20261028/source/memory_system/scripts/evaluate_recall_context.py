"""Frozen ingestion/planner, live search and reader; paired via separate arms.

All writes go to a new output directory. Gold snippets are loaded only by the
offline coverage scorer and never enter retrieval. Concurrent latency is not a
single-request service benchmark. No HTTPS or fresh-ingestion quality claim.
"""
import argparse
import ast
import asyncio
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, eval_scoring, retrieval_queries, run_metadata, schemas, search_debug, search_pipeline, store
from audit_https_funnel import turn_coverage
from funnel_gold_snippet import CSV, norm


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reference_turns(qa, history, metadata):
    """CSV row identity, not question text: repeated questions have different gold."""
    query = ast.literal_eval(metadata['user_query'])['content']
    if (norm(query) != norm(qa['question']) or metadata['correct_answer'] != qa['answer']
            or history not in (metadata['chat_history_32k_link'], metadata['chat_history_128k_link'])):
        raise ValueError('Gold reference identity mismatch')
    try:
        snippet = json.loads(metadata['related_conversation_snippet'])
    except ValueError:
        snippet = ast.literal_eval(metadata['related_conversation_snippet'])
    return [norm(m['content']) for m in snippet if m.get('role') == 'user' and m.get('content')]


def packet_coverage(turns, packet, raw):
    cards = [dict(s, id=f'{i}:{j}', text=s.get('content', ''))
             for i, item in enumerate(packet) for j, s in enumerate(item.get('sources', []))]
    matches = turn_coverage(turns, cards, raw)
    return dict(visible=any(m['full'] for m in matches), full_turns=sum(m['full'] for m in matches),
                turns=len(turns), reference_present=any(m['provenance_origins'] for m in matches))


async def main(args):
    if config.FAKE:
        raise ValueError('Real providers required; fake scores are not quality evidence')
    args.output.mkdir(parents=True, exist_ok=False)
    traces = {t['search_id']: t for t in records(args.run / 'completed/logs/search-debug.jsonl')}
    original = records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')
    if args.questions:
        wanted = set(args.questions.split(','))
        original = [r for r in original if r['qa_id'] in wanted]
        if {r['qa_id'] for r in original} != wanted:
            raise ValueError('Question subset is incomplete')
    conversations = {d['conversation_id']: d for d in records(args.prepared)}
    qa = {(d['conversation_id'], q['id']): q for d in conversations.values() for q in d['qa']}
    settings = traces[original[0]['search_id']]['versions']['settings']
    for row in original:
        if traces[row['search_id']]['versions']['settings'] != settings:
            raise ValueError('Mixed effective settings')
    for key, value in settings.items():
        setattr(config, key, value)
    config.RECALL_TOPIC_ANCHORS = args.arm in ('optimized', 'anchors')
    config.SOURCE_RECALL_DIVERSITY = args.arm in ('optimized', 'diversity')
    config.LISTWISE_CONTEXT_LABEL = args.arm in ('optimized', 'context')
    run_metadata.versions.cache_clear()
    manifest = json.loads((args.run / 'isolated-manifest.json').read_text())
    if hashlib.sha256(args.prepared.read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Prepared data differs from frozen run')
    # SQLite backup includes committed WAL pages and never opens the source writable.
    with sqlite3.connect(f'file:{(args.run / "memory.db").resolve()}?mode=ro', uri=True) as src:
        with sqlite3.connect(args.output / 'memory.db') as dst:
            src.backup(dst)
    st = store.Store(str(args.output / 'memory.db'))
    plans = {(t['user_id'], t['query'], tuple(t.get('options') or [])): t['plan'] for t in traces.values()}

    async def frozen_understand(_st, req, anchor):
        return copy.deepcopy(plans[req.user_id, req.query, tuple(req.options or [])])

    metadata = dict(arm=args.arm, run=str(args.run), questions=len(original), concurrency=args.concurrency,
        versions=run_metadata.versions(), frozen_planner=True, fresh_ingestion=False,
        prepared_sha256=manifest['data_sha256'],
        database_sha256=hashlib.sha256((args.output / 'memory.db').read_bytes()).hexdigest(),
        source_trace_sha256=hashlib.sha256((args.run / 'completed/logs/search-debug.jsonl').read_bytes()).hexdigest(),
        limitation='Single stochastic run per arm; concurrent provider latency; not an HTTPS benchmark.')
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    captured = {}
    def capture(trace):
        captured[trace['search_id']] = trace
        with (args.output / 'search-debug.jsonl').open('a') as out:
            out.write(json.dumps(trace, ensure_ascii=False) + '\n')

    # Reference metadata stays exclusively in this scorer, outside request/plan.
    csv.field_size_limit(32 * 1024 * 1024)
    with CSV.open() as source:
        gold = list(csv.DictReader(source))
    raw = {}
    for user in {traces[r['search_id']]['user_id'] for r in original}:
        raw[user] = {(r['request_id'], r['message_index']): dict(r) for r in
                    st.conn.execute('SELECT * FROM source_messages WHERE user_id=?', (user,))}

    def coverage(q, result, user, packet):
        turns = reference_turns(q, conversations[result['conversation_id']]['source_history'], gold[int(result['qa_id'])])
        return packet_coverage(turns, packet, raw[user])

    semaphore = asyncio.Semaphore(args.concurrency)
    rows = []
    async def one(result):
        async with semaphore:
            q = qa[result['conversation_id'], result['qa_id']]
            old = traces[result['search_id']]
            req = schemas.SearchRequest(user_id=old['user_id'], query=old['query'], options=old.get('options'),
                top_k=old['top_k'], reference_time=old.get('reference_time'),
                evidence_token_budget=old['evidence_token_budget'])
            row = dict(qa_id=result['qa_id'], conversation_id=result['conversation_id'], arm=args.arm)
            started = time.monotonic()
            try:
                response = await search_pipeline.run_search(st, req)
                row['search_seconds'] = time.monotonic() - started
                packet = [x.model_dump() for x in response.data]
                trace = captured.pop(response.search_id)
                row.update(search_id=response.search_id, coverage=coverage(q, result, req.user_id, packet),
                    fused_coverage=coverage(q, result, req.user_id, [x['_packet_item'] for x in trace['ranked']
                        if x.get('_packet_item')]),
                    degraded=response.coverage_manifest['search_degraded'],
                    provider_calls=response.coverage_manifest['provider_calls'],
                    provider_tokens=response.coverage_manifest['provider_tokens'],
                    context_count=sum(v == 'context' for v in trace['cascade']['listwise'].get('candidate_relevance', {}).values()))
                pred, score, diagnostics = await eval_scoring.evaluate(q, packet)
                row.update(prediction=pred, score=score, diagnostics=diagnostics)
            except Exception as exc:
                row.update(error_type=type(exc).__name__, error_detail=str(exc)[:500], score=0,
                           search_seconds=time.monotonic() - started)
            row['total_seconds'] = time.monotonic() - started
            rows.append(row)
            with (args.output / 'results.jsonl').open('a') as out:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps({k: v for k, v in row.items() if k not in ('diagnostics', 'error_detail')}, ensure_ascii=False), flush=True)
    try:
        with patch.object(search_pipeline, '_understand', frozen_understand), patch.object(search_debug, 'append', capture):
            await asyncio.gather(*(one(r) for r in original))
    finally:
        st.conn.close()
    times = sorted(r['search_seconds'] for r in rows)
    summary = dict(arm=args.arm, questions=len(rows), score=sum(r['score'] for r in rows),
        errors=sum('error_type' in r for r in rows), reader_errors=sum(r.get('diagnostics', {}).get('answer_status') == 'error' for r in rows),
        reference_present=sum(r.get('coverage', {}).get('reference_present', False) for r in rows),
        visible_questions=sum(r.get('coverage', {}).get('visible', False) for r in rows),
        full_turns=sum(r.get('coverage', {}).get('full_turns', 0) for r in rows),
        degraded=sum(r.get('degraded', True) for r in rows),
        latency=dict(mean=statistics.mean(times), median=statistics.median(times), p95=times[min(len(times)-1, int(len(times)*.95))]),
        search_calls=sum(r.get('provider_calls', 0) for r in rows),
        search_tokens=sum(r.get('provider_tokens', 0) for r in rows),
        limitation=metadata['limitation'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('memory_system/data/prepared/personamem-v2-32k.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['baseline', 'optimized', 'anchors', 'diversity', 'context'], required=True)
    parser.add_argument('--questions')
    parser.add_argument('--concurrency', type=int, default=2)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('concurrency must be positive')
    asyncio.run(main(args))
