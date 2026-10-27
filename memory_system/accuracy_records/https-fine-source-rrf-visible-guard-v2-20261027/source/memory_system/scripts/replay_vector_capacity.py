"""Full-cohort paired vector-route capacity replay on isolated database copies."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, cross_encoder, embeddings, eval_scoring, llm
from app import run_metadata, schemas, search_debug, search_pipeline, store


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


async def main(args):
    preflight = json.loads(args.probe.read_text())
    assert preflight['include_sensitive'] and all(r['archived_route_exact'] for r in preflight['rows'])
    questions = [q for line in args.prepared.read_text().splitlines() for q in json.loads(line)['qa']]
    records = [json.loads(line) for line in (args.run / 'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines()]
    assert [q['id'] for q in questions] == [r['qa_id'] for r in records] == [r['qa_id'] for r in preflight['rows']]
    traces = {t['search_id']: t for t in map(json.loads, (args.run / 'server-search-debug.jsonl').open())}
    assert all(t['versions']['settings']['RECALL_VECTOR_LIMIT'] == 40 for t in traces.values())
    assert args.capacity > 40 and not config.FAKE
    manifest = json.loads((args.run / 'isolated-manifest.json').read_text())
    assert [manifest[k] for k in ['client_exit_code', 'audit_exit_code', 'server_exit_code']] == [0, 0, -15]
    providers = dict(llm=llm.complete_json, embedding=embeddings.embed, ce=cross_encoder.rerank)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        seed = directory / 'seed.db'
        source = sqlite3.connect(f'file:{(args.run / "memory.db").resolve()}?mode=ro', uri=True)
        destination = sqlite3.connect(seed)
        source.backup(destination)
        destination.close()
        source.close()
        seed_hash = hashlib.sha256(seed.read_bytes()).hexdigest()
        plan = dict(run=str(args.run), prepared=str(args.prepared), probe=str(args.probe),
                    prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
                    probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(), seed_sha256=seed_hash,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    qa_ids=[q['id'] for q in questions], capacities=[40, args.capacity],
                    caveat='Each arm starts from the same private post-ingestion database snapshot. Original query plan and anchor frozen; one planning call charged, planning latency/tokens omitted equally. Other search budgets and ranking/reader settings preserved. Fresh embeddings, CE, listwise, recovery and full reader in both arms. Alternating arm order. Not a fresh-ingestion HTTPS result; provider variability remains.')
        with args.output.with_suffix('.plan.json').open('x') as stream:
            json.dump(plan, stream, indent=2)
        with args.output.open('x') as output:
            for index, (question, result) in enumerate(zip(questions, records)):
                original = traces[result['search_id']]
                for arm in (['control', 'candidate'] if index % 2 == 0 else ['candidate', 'control']):
                    path = directory / f'{index}-{arm}.db'
                    shutil.copyfile(seed, path)
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == seed_hash
                    events, searches = [], []

                    async def capture(kind, *pos, **kw):
                        if kind == 'llm':
                            identity = dict(prompt_sha256=digest(pos[0]), stage=kw.get('stage'),
                                            schema_sha256=digest(json.dumps(kw.get('schema'), sort_keys=True)))
                        elif kind == 'embedding':
                            identity = dict(texts=list(pos[0]), stage=kw.get('stage'))
                        else:
                            identity = dict(query=pos[0], document_hashes=[digest(t) for t in pos[1]], stage=kw.get('stage'))
                        event = dict(kind=kind, **identity)
                        events.append(event)
                        try:
                            response = await providers[kind](*pos, **kw)
                            event['response'] = response.tolist() if kind == 'embedding' else copy.deepcopy(response)
                            return response
                        except Exception as error:
                            event.update(error_type=type(error).__name__, error_detail=str(error))
                            raise

                    async def call_llm(*pos, **kw): return await capture('llm', *pos, **kw)
                    async def call_embed(*pos, **kw): return await capture('embedding', *pos, **kw)
                    async def call_ce(*pos, **kw): return await capture('ce', *pos, **kw)
                    async def understand(*unused):
                        budget.current.get().before_call()
                        return copy.deepcopy(original['plan'])

                    settings = dict(original['versions']['settings'], RECALL_VECTOR_LIMIT=40 if arm == 'control' else args.capacity,
                                    SENSITIVE_RECALL_ENABLED=True, SEARCH_DEBUG_LOG='', MEMORY_DEBUG_LOG='')
                    with patch.multiple(config, **settings):
                        run_metadata.versions.cache_clear()
                        database = store.Store(str(path))
                        try:
                            request = schemas.SearchRequest(**{k: original[k] for k in
                                ['query', 'options', 'user_id', 'top_k', 'reference_time', 'evidence_token_budget']}, include_sensitive=True)
                            with patch.object(search_pipeline, '_understand', understand), \
                                 patch.object(search_pipeline, '_anchor_time', lambda *unused: original['anchor_time']), \
                                 patch.object(search_debug, 'append', lambda t: searches.append(copy.deepcopy(t))), \
                                 patch.object(llm, 'complete_json', call_llm), \
                                 patch.object(search_pipeline, 'embed', call_embed), \
                                 patch.object(cross_encoder, 'rerank', call_ce):
                                response = await search_pipeline.run_search(database, request)
                                packet = [item.model_dump() for item in response.data]
                                with budget.scope(seconds=300, calls=40, tokens=200000):
                                    prediction, score, diagnostics = await eval_scoring.evaluate(question, packet)
                        finally:
                            database.conn.close()
                    assert len(searches) == 1
                    search = searches[0]
                    actual = next(r['candidates'] for r in search['routes'] if r['channel'] == 'vector')
                    expected = next(r['candidates'] for r in original['routes'] if r['channel'] == 'vector')
                    route_exact = [r['id'] for r in actual] == [r['id'] for r in expected]
                    if arm == 'control':
                        assert route_exact, 'Control vector route changed'
                    row = dict(qa_id=question['id'], arm=arm, prediction=prediction, score=score,
                               original_vector_route_exact=route_exact, response=response.model_dump(),
                               search=search, diagnostics=diagnostics, provider_events=events)
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    output.flush()
                    rows.append({k: row[k] for k in ['qa_id', 'arm', 'prediction', 'score']})
                    print(json.dumps(rows[-1]), flush=True)
        summary = dict(questions=len(questions), scores={arm: sum(r['score'] for r in rows if r['arm'] == arm)
                                                       for arm in ['control', 'candidate']}, caveat=plan['caveat'])
        args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'prepared', 'probe', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--capacity', type=int, default=80)
    asyncio.run(main(parser.parse_args()))
