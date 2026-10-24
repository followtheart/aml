"""Isolated write-view ablation through coarse/fine/main-input admission.

Uses a completed HTTPS run's immutable originals, query plans and recall routes.
Real embedding and CE calls; stops BEFORE main LLM review and answering. This
is candidate transport diagnostics, never an end-to-end HTTPS quality score.
Gold snippets are read only by the separate analyze phase.
"""
import argparse
import asyncio
import copy
import csv
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import (association_views as av, budget, cascade_rerank, config, embeddings,
                 llm, run_metadata, schemas, search_pipeline as search, store)


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def identity(st):
    # Semantic original state must not change when only an index is added.
    result = {}
    for table in ('amu', 'source_messages', 'amu_sources', 'triples', 'requests', 'claim_versions'):
        rows = sorted(repr(tuple(row)) for row in st.conn.execute(f'SELECT * FROM {table}'))
        result[table] = hashlib.sha256('\n'.join(rows).encode()).hexdigest()
    return result


def cohort(args):
    manifest = json.loads((args.baseline/'isolated-manifest.json').read_text())
    prepared = Path(manifest['data'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Baseline input hash mismatch')
    expected = {(c['conversation_id'], str(q['id'])): q for c in records(prepared) for q in c['qa']}
    results = list(records(args.baseline/'completed/data/results/personamem-v2-32k.jsonl'))
    actual = [(r['conversation_id'], str(r['qa_id'])) for r in results]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('Full cohort required')
    traces = {t['search_id']: t for t in records(args.baseline/'server-search-debug.jsonl')}
    jobs = []
    for result in results:
        trace = traces[result['search_id']]
        qa = expected[result['conversation_id'], str(result['qa_id'])]
        if trace['query'] != qa['question'] or trace['options'] != qa['options'] or trace['user_id'] != result['user_id']:
            raise ValueError('Question identity mismatch')
        jobs.append((result, trace))
    return jobs


async def index(args):
    if config.FAKE:
        raise ValueError('Real providers required')
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = args.baseline/'memory.db'
    destination = args.output/'memory.db'
    with sqlite3.connect(source.resolve().as_uri()+'?mode=ro', uri=True) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
    st = store.Store(str(destination))
    before = identity(st)
    jobs = cohort(args)
    write(args.output/'manifest.json', dict(baseline=str(args.baseline.resolve()),
        baseline_db=str(source.resolve()), questions=len(jobs), versions=run_metadata.versions(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        identity=before, preregistered_gate='Improve main-input reference transport without losing baseline references; otherwise do not deploy or expand to query-side.',
        scope='Frozen upstream, new source-view index, live embeddings and CE, no main review/answer; equal fresh ranking budgets.'))
    (args.output/'index-script.py').write_bytes(Path(__file__).read_bytes())
    requests = [tuple(r) for r in st.conn.execute('SELECT user_id,request_id FROM requests ORDER BY user_id,created_at,request_id')]
    totals = []
    try:
        for user, rid in requests:
            with st.staged(user) as work:
                stats = await av.index_sources(work, user, rid)
            totals.append(dict(request_id=rid, user_id=user, **stats))
            write(args.output/'index-progress.json', totals)
            print(json.dumps(dict(phase='index', completed=len(totals), requests=len(requests), **stats)), flush=True)
        assert identity(st) == before, 'Original indexed state changed'
        write(args.output/'index-complete.json', dict(requests=totals, original_state_unchanged=True))
    finally:
        st.conn.close()


class AdmissionReached(BaseException):
    pass


async def replay_one(st, trace, vectors, enabled):
    plan = copy.deepcopy(trace['plan'])
    plan.update(_anchor=trace['anchor_time'], _include_history=trace['include_history'],
        _include_sensitive=trace['include_sensitive'], _query_specs=trace['query_specs'],
        _used_queries=[q['text'] for q in trace['query_specs']],
        _fusion_query_vectors=vectors.tolist(), _routes=copy.deepcopy(trace['routes']),
        _coverage_requirements=[{k: r[k] for k in ('id', 'text', 'origin')}
            for r in trace['coverage_manifest']['coverage']['requirements']])
    req = schemas.SearchRequest(user_id=trace['user_id'], query=trace['query'], options=trace['options'],
        top_k=trace['top_k'], reference_time=trace['reference_time'], include_sensitive=trace['include_sensitive'],
        evidence_token_budget=trace['evidence_token_budget'])
    routes = [copy.deepcopy(r['candidates']) for r in trace['routes']]
    all_ids = list({c['id'] for route in routes for c in route})
    live = {c['id']: c for c in st.get_amus_by_ids(all_ids, True, True)}
    for route in routes:
        for c in route:
            c['embedding'] = live[c['id']].get('embedding')
    proposals, diagnostics, requests, admission = [], {}, {}, {}
    real_request = cascade_rerank.listwise_request

    def capture(req, plan, rows, **kwargs):
        result = real_request(req, plan, rows, **kwargs)
        requests[result['prompt']] = [c['id'] for c in rows]
        return result

    async def stop(prompt, **kwargs):
        if kwargs['stage'] != 'search.listwise':
            raise AssertionError('Unexpected LLM stage')
        admission.update(ids=requests[prompt], prompt=prompt)
        raise AdmissionReached()

    with budget.scope(seconds=config.SEARCH_DEADLINE_SECONDS,
                      calls=config.SEARCH_MAX_CALLS, tokens=config.SEARCH_MAX_TOKENS) as limits:
        if enabled:
            proposals, diagnostics = av.recall(st, req.user_id, vectors,
                include_history=trace['include_history'], include_sensitive=trace['include_sensitive'])
            search._route(plan, routes, 'association', proposals, ' | '.join(plan['_used_queries']))
        fused = search._fuse_candidates(st, req, plan, routes)
        candidates = search._prepare_candidates(st, req, plan, fused)
        candidates = search._foresight_filter(candidates, plan)
        with patch.object(cascade_rerank, 'listwise_request', capture), patch.object(llm, 'complete_json', stop):
            try:
                await cascade_rerank.rank(req, plan, candidates)
            except AdmissionReached:
                pass
        by_id = {c['id']: c for c in candidates}
        cascade = plan['_cascade']
        stages = dict(coarse=cascade['coarse'].get('selected_ids', []),
                      fine=cascade['fine'].get('selected_ids', []), main_input=admission.get('ids', []))
        # Complete source bodies here measure carrier coverage; rank_text is the
        # actual prepared representation available to CE/main-input rendering.
        sources = st.sources_for_amus(list(by_id))
        return dict(stages=stages, associated_ids=[c['id'] for c in proposals],
            association_diagnostics=diagnostics, cascade=cascade, calls=limits.calls, tokens=limits.tokens,
            main_prompt=admission.get('prompt'),
            candidates={mid: dict(sources=sources[mid], rank_text=c.get('_rank_text', c['content'])) for mid,c in by_id.items()})


async def replay(args):
    if config.FAKE:
        raise ValueError('Real providers required')
    if not (args.output/'index-complete.json').exists():
        raise ValueError('Complete index required')
    jobs = cohort(args)
    manifest = json.loads((args.output/'manifest.json').read_text())
    current = run_metadata.versions()
    changed = [name for name, sha in manifest['versions']['file_hashes'].items()
               if current['file_hashes'].get(name) != sha]
    # Ranking may be implemented after the index is built, but generation and
    # original storage must remain identical. Freeze the replay separately,
    # before any score is observed; retain both scripts and all fingerprints.
    if set(changed) - {'memory_system/app/graph_fusion.py'}:
        raise ValueError('Index dependencies changed since generation')
    if hashlib.sha256((args.output/'index-script.py').read_bytes()).hexdigest() != manifest['script_sha256']:
        raise ValueError('Index script archive mismatch')
    if (args.output/'replay-manifest.json').exists():
        raise ValueError('Replay already frozen; use a new isolated run for another experiment')
    write(args.output/'replay-manifest.json', dict(versions=current, changed_since_index=changed,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (args.output/'replay-script.py').write_bytes(Path(__file__).read_bytes())
    for key, value in jobs[0][1]['versions']['settings'].items():
        setattr(config, key, value)
    st = store.Store(str(args.output/'memory.db'))
    before = identity(st)
    assert before == manifest['identity'], 'Frozen original state changed before replay'
    try:
        with (args.output/'replay.jsonl').open('x') as output:
            for i, (result, trace) in enumerate(jobs):
                vectors = await embeddings.embed([s['text'] for s in trace['query_specs']], stage='diagnostic.shared_query_embeddings')
                rows = {}
                for enabled in ([False, True] if i % 2 == 0 else [True, False]):
                    with st.snapshot(trace['user_id']) as snap:
                        rows['treatment' if enabled else 'control'] = await replay_one(snap, trace, vectors, enabled)
                row = dict(qa_id=str(result['qa_id']), conversation_id=result['conversation_id'], user_id=result['user_id'],
                    historical_coarse_ids=trace['cascade']['coarse']['selected_ids'], **rows)
                output.write(json.dumps(row, ensure_ascii=False)+'\n'); output.flush()
                print(json.dumps(dict(phase='replay', completed=i+1, questions=len(jobs), qa_id=row['qa_id'],
                    control_main=len(rows['control']['stages']['main_input']), treatment_main=len(rows['treatment']['stages']['main_input']))), flush=True)
        assert identity(st) == before
        write(args.output/'replay-complete.json', dict(questions=len(jobs), original_state_unchanged=True))
    finally:
        st.conn.close()


def analyze(args):
    import ast
    import audit_https_funnel as audit
    if not (args.output/'replay-complete.json').exists():
        raise ValueError('Complete replay required')
    csv.field_size_limit(32*1024*1024)
    with (ROOT/'data/raw/personamem-v2.csv').open() as stream:
        gold = list(csv.DictReader(stream))
    rows = list(records(args.output/'replay.jsonl'))
    report = []
    for row in rows:
        metadata = gold[int(row['qa_id'])]
        try:
            snippet = json.loads(metadata['related_conversation_snippet'])
        except ValueError:
            snippet = ast.literal_eval(metadata['related_conversation_snippet'])
        turns = [audit.norm(s['content']) for s in snippet if s['role'] == 'user']
        scores = {}
        for arm in ('control', 'treatment'):
            data = row[arm]
            scores[arm] = {}
            for stage, ids in data['stages'].items():
                texts = [audit.norm(s['content']) for mid in ids for s in data['candidates'][mid]['sources'] if s['role']=='user']
                rendered = [audit.norm(data['candidates'][mid]['rank_text']) for mid in ids]
                # Only user reference turns; literal matches are transport proxies,
                # not relevance/entailment scores or answer correctness.
                scores[arm][stage] = dict(carrier=any(t and any(t in x for x in texts) for t in turns),
                    visible=any(t and any(t in x for x in rendered) for t in turns))
        report.append(dict(qa_id=row['qa_id'], scores=scores,
            control_coarse_matches_history=row['control']['stages']['coarse']==row['historical_coarse_ids']))
    summary = {arm: {stage: {kind: sum(r['scores'][arm][stage][kind] for r in report)
        for kind in ('carrier','visible')} for stage in ('coarse','fine','main_input')} for arm in ('control','treatment')}
    gains = [r['qa_id'] for r in report if r['scores']['treatment']['main_input']['visible'] and not r['scores']['control']['main_input']['visible']]
    losses = [r['qa_id'] for r in report if r['scores']['control']['main_input']['visible'] and not r['scores']['treatment']['main_input']['visible']]
    ce_failures = sum(r[arm]['cascade']['cross_encoder'].get('status') != 'ok'
                      for r in rows for arm in ('control','treatment'))
    write(args.output/'analysis.json', dict(questions=len(rows), summary=summary, gains=gains, losses=losses,
        ce_failures=ce_failures, gate_passed=bool(gains) and not losses and not ce_failures, rows=report,
        caveat='Candidate transport only; main reviewer and answerer were not called. No HTTPS accuracy or latency claim.'))
    print(json.dumps(dict(summary=summary,gains=gains,losses=losses)), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('index','replay','analyze'),required=True)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.phase=='analyze': analyze(args)
    else: asyncio.run(index(args) if args.phase=='index' else replay(args))


if __name__=='__main__':
    main()
