#!/usr/bin/env python3
"""HTTPS Add/Search with the same local Answer/Judge protocol as local_eval.

Run from memory_system:
  python scripts/https_eval.py --data data/sample_eval.json --inspect
  python scripts/https_eval.py --url https://localhost:8443 \
      --ca ../https_client/certs/server.crt --data data/sample_eval.json \
      --output results/https.jsonl

AML_API_KEY (or --token-file) authenticates to the memory service. Answer/Judge
uses the client's AML_LLM_* / provider credentials. AML_FAKE affects only the
client; the server's model configuration and ablations are configured separately.
"""
import argparse
import asyncio
from collections import defaultdict
import json
import logging
import math
import os
from pathlib import Path
import ssl
import sys
from urllib.parse import urlsplit
import uuid

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, eval_data, eval_scoring, run_metadata, schemas
from local_eval import Progress


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--url', default='https://localhost:8443')
    p.add_argument('--ca', help='Trusted CA PEM, including self-signed certificates')
    p.add_argument('--token-file', type=Path, help='Overrides AML_API_KEY')
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--limit', type=int, default=0, help='Maximum QA pairs across all conversations; 0 = all')
    p.add_argument('--convs', type=int, default=0, help='Maximum conversations; 0 = all')
    p.add_argument('--dataset', choices=eval_data.DATASETS, default='normalized')
    p.add_argument('--history-dir', help='PersonaMem/ScriptMem source histories')
    p.add_argument('--size', choices=['32k', '128k'], default='32k')
    p.add_argument('--output', help='Write per-question results as JSONL')
    p.add_argument('--inspect', action='store_true', help='Validate/count only; no network or model calls')
    p.add_argument('--chunk-messages', type=int, default=20)
    p.add_argument('--chunk-words', type=int, default=2000)
    p.add_argument('--answer-top-k', type=int, default=50)
    p.add_argument('--evidence-token-budget', type=int, default=32000,
                   help='Final evidence packet budget in UTF-8 bytes (256..32000)')
    p.add_argument('--no-sensitive', action='store_true')
    p.add_argument('--no-progress', action='store_true')
    p.add_argument('--official-fields', action='store_true',
                   help='Send only AML Add/Search contract fields; use --answer-top-k 100 for formal settings')
    p.add_argument('--request-log', type=Path, help='Archive request payloads without authentication headers')
    args = p.parse_args(argv)
    u = urlsplit(args.url)
    if (u.scheme != 'https' or not u.hostname or u.username is not None
            or u.password is not None or u.query or u.fragment):
        p.error('--url must be HTTPS without credentials, query or fragment')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error('--timeout must be finite and positive')
    if min(args.limit, args.convs) < 0 or min(args.chunk_messages, args.chunk_words) < 1:
        p.error('Limits must be nonnegative and chunk sizes positive')
    if not 1 <= args.answer_top_k <= 100 or not 256 <= args.evidence_token_budget <= 32000:
        p.error('--answer-top-k must be 1..100; --evidence-token-budget must be 256..32000')
    if args.output and Path(args.output).resolve() == Path(args.data).resolve():
        p.error('--output must not overwrite input data')
    return args


def selected(args):
    data = eval_data.convert_records(args.data, args.dataset, args.history_dir, args.size)
    return eval_data.iter_plan(data, args.limit, args.convs)


def search_request(args, uid, qa, revision=None):
    return schemas.HTTPSearchRequest(
        query=qa['question'], options=qa.get('options'), user_id=uid,
        top_k=args.answer_top_k, evidence_token_budget=args.evidence_token_budget,
        include_sensitive=not args.no_sensitive, reference_time=qa.get('question_date'),
        as_of=qa.get('as_of'), min_revision=revision)


def request_payload(path, request, official_fields=False):
    payload = request.model_dump(exclude_none=True)
    if official_fields:
        if path == 'add':
            payload = {k: payload[k] for k in ('request_id', 'user_id', 'session_id', 'messages')}
            payload['messages'] = [{k: v for k, v in m.items() if k in ('role', 'content', 'timestamp')}
                                   for m in payload['messages']]
        elif path == 'search':
            payload = {k: v for k, v in payload.items() if k in ('query', 'options', 'user_id', 'top_k')}
        else:
            raise ValueError('Unknown official contract endpoint')
    return payload


async def post(client, path, request, response_type, *, official_fields=False, request_log=None):
    payload = request_payload(path, request, official_fields)
    if request_log:
        request_log.parent.mkdir(parents=True, exist_ok=True)
        with request_log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(path=path, payload=payload), ensure_ascii=False) + '\n')
    response = await client.post(path, json=payload)
    response.raise_for_status()
    return response_type.model_validate(response.json())


async def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%H:%M:%S')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    # Preflight the entire selected subset before writing remote memory or paying
    # for model calls. The second pass streams the same subset for execution.
    conv_count = qa_count = add_count = 0
    protocols = set()
    for conv, qas in selected(args):
        protocols.add(conv.get('protocol', eval_scoring.PROTOCOL))
        conv_count += 1
        qa_count += len(qas)
        for _, _, session in eval_data.ingestion_chunks(conv, args.chunk_messages, args.chunk_words):
            for message in session:
                schemas.HTTPMessage(**message)
            add_count += 1
        for qa in qas:
            search_request(args, 'preflight', qa)
    print(f'protocol={",".join(sorted(protocols))} client_fake={config.FAKE} '
          f'server_fake=unknown conversations={conv_count} add_chunks={add_count} qa={qa_count}')
    if args.inspect:
        return 0
    token = (args.token_file.read_text(encoding='utf-8').strip() if args.token_file
             else os.environ.get('AML_API_KEY', '').strip())
    if not token:
        raise ValueError('Provide --token-file or AML_API_KEY for the memory service')
    context = ssl.create_default_context(cafile=args.ca)
    run_id = uuid.uuid4().hex
    print(f'run_id={run_id} url={args.url}')
    progress = Progress(add_count + 2 * qa_count, enabled=not args.no_progress)
    output = None
    per_cat = defaultdict(lambda: [0.0, 0])
    n_qa = errors = 0
    score_sum = 0.0
    try:
        async with httpx.AsyncClient(
                base_url=args.url.rstrip('/') + '/', verify=context,
                timeout=args.timeout, follow_redirects=False,
                headers={'Authorization': f'Bearer {token}', 'Accept': 'application/json'}) as client:
            health = await client.get('health')
            health.raise_for_status()
            if health.json().get('status') != 'ok':
                raise ValueError('Service health check failed')
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                output = open(args.output, 'w', encoding='utf-8')
            for ci, (conv, qas) in enumerate(selected(args), 1):
                cid = conv['conversation_id']
                dataset = conv.get('dataset', args.dataset)
                uid = f'https:{run_id}:{ci}:{dataset}:{cid}'
                revision = 0
                for sid, chunk_id, session in eval_data.ingestion_chunks(
                        conv, args.chunk_messages, args.chunk_words):
                    req = schemas.HTTPAddRequest(
                        request_id=f'{uid}:session:{sid}:chunk:{chunk_id}', user_id=uid,
                        session_id=f'{uid}:session:{sid}', messages=session)
                    # Fail fast on ingestion failure; scoring partial histories
                    # would silently change the evaluation task. No HTTP retries.
                    added = await progress.run(post(client, 'add', req, schemas.AddResponse,
                                               official_fields=args.official_fields, request_log=args.request_log),
                                               f'Add conv {ci}/{conv_count} session {sid} chunk {chunk_id}')
                    if (not added.success or added.request_id != req.request_id
                            or added.user_id != uid or added.session_id != req.session_id):
                        raise ValueError('Add failed or returned mismatched identifiers')
                    revision = max(revision, added.write_revision)
                for qi, qa in enumerate(qas, 1):
                    before = progress.done
                    try:
                        resp = await progress.run(
                            post(client, 'search', search_request(args, uid, qa, revision),
                                 schemas.SearchResponse, official_fields=args.official_fields,
                                 request_log=args.request_log),
                            f'Search conv {ci}/{conv_count} QA {qi}/{len(qas)}')
                    except Exception as exc:
                        progress.done = before + 2  # Search failed; Answer/Judge skipped.
                        diagnostics = {'error_stage': 'search', 'error_type': type(exc).__name__}
                        if isinstance(exc, httpx.HTTPStatusError):
                            diagnostics['http_status'] = exc.response.status_code
                        pred, score = '', 0.0
                    else:
                        pred, score, diagnostics = await progress.run(
                            eval_scoring.evaluate(qa, [d.model_dump() for d in resp.data]),
                            f'Answer/Judge conv {ci}/{conv_count} QA {qi}/{len(qas)}')
                        rerank = resp.coverage_manifest.get('rerank_status')
                        degraded = bool(resp.coverage_manifest.get('search_degraded',
                                                                  rerank in ('partial', 'fallback')))
                        diagnostics.update(search_id=resp.search_id, packet_hash=resp.packet_hash,
                                           evidence_status=resp.evidence_status, search_degraded=degraded,
                                           coverage_manifest=resp.coverage_manifest,
                                           read_revision=resp.read_revision)
                        if degraded:
                            progress.write(f'[search degraded: {resp.evidence_status}, rerank {rerank}]')
                    errors += int(bool(diagnostics.get('error_stage')))
                    n_qa += 1
                    score_sum += score
                    cat = f"{dataset}/{qa.get('category', 'unknown')}"
                    per_cat[cat][0] += score
                    per_cat[cat][1] += 1
                    if output:
                        result = {
                            'dataset': dataset, 'conversation_id': cid,
                            'qa_id': qa.get('id', str(qi - 1)), 'prediction': pred, 'predicted_answer': pred,
                            'score': score, 'scoring': qa.get('scoring', 'binary'),
                            'category': qa.get('category', 'unknown'),
                            'protocol': conv.get('protocol', eval_scoring.PROTOCOL),
                            'fake': config.FAKE, 'model': config.LLM_MODEL, 'versions': run_metadata.versions(),
                            'transport': 'https', 'run_id': run_id, 'user_id': uid,
                            'request_contract': 'official' if args.official_fields else 'extended',
                            'request_top_k': args.answer_top_k,
                            'server_url': args.url, 'server_fake': None, 'server_versions': None,
                            **diagnostics}
                        output.write(json.dumps(result, ensure_ascii=False) + '\n')
                        output.flush()
                    progress.write(f'[score={score:.2f}] {cat}: {qa["question"][:60]} -> {pred[:60]}'
                                   + (f' [error: {diagnostics["error_stage"]}]' if diagnostics.get('error_stage') else ''))
    finally:
        if output:
            output.close()
        progress.finish()
    print('\n===== RESULTS =====')
    for cat, (score, count) in sorted(per_cat.items()):
        print(f'  {cat:20s} {score}/{count} = {score/count:.2%}')
    if n_qa:
        print(f'  {"MEAN SCORE":20s} {score_sum:.2f}/{n_qa} = {score_sum/n_qa:.2%}')
    print(f'errors={errors}; remote memories retained; run_id={run_id}')
    print('HTTPS retrieval + local proxy scoring; not official AML leaderboard scores.'
          + (' Client FAKE mode: scores are meaningless.' if config.FAKE else ''))
    return int(errors > 0)


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
