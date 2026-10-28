"""Paired real HTTPS Search with identical frozen ingestion, plans and usage state.

This is a search ablation, not a fresh Add/Search benchmark. Real retrieval,
CE, listwise and the local reader run; only planning and recall usage writes
are frozen. Gold stays in the client scorer. Each arm owns its database.
"""
import argparse
import asyncio
import copy
from contextvars import ContextVar
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import ssl
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import config
from evaluation_model_guard import REQUIRED_MODEL, check_runtime, check_versions
from https_experiment_settings import arm_values, fixture_values, run_arms
from batched_records import file_digest

READER_ARM=ContextVar('frozen_https_reader_arm',default='none')


class ReaderArmFilter(logging.Filter):
    def filter(self,record):
        record.reader_arm=READER_ARM.get()
        return record.reader_arm!='none'


def records(path, block_bytes=4 * 1024 * 1024):
    """Parse JSONL in binary batches, retaining only the unfinished record.

    Decode complete records so UTF-8 characters may cross a block boundary.
    A record larger than a block must still be held until it is complete.
    """
    if block_bytes < 1:
        raise ValueError('Block size must be positive')
    pending = bytearray()
    with path.open('rb') as stream:
        while block := stream.read(block_bytes):
            batch = block.split(b'\n')
            if len(batch) == 1:
                pending.extend(block)
                continue
            pending.extend(batch[0])
            if pending.strip():
                yield json.loads(pending)
            pending.clear()
            for record in batch[1:-1]:
                if record.strip():
                    yield json.loads(record)
            pending.extend(batch[-1])
        if pending.strip():
            yield json.loads(pending)


def digest(path):
    return file_digest(path)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def serve(args):
    import uvicorn
    from app import run_metadata, search_pipeline, store
    fixture = json.loads(args.fixture.read_text())
    for key, value in fixture['settings'].items():
        setattr(config, key, value)
    setattr(config, fixture.get('toggle', 'LISTWISE_OVERFLOW_REVIEW'), fixture_values(fixture)[args.arm])
    check_runtime()
    run_metadata.versions.cache_clear()
    plans = {(r['request']['user_id'], r['request']['query'], tuple(r['request']['options'] or [])): r
             for r in fixture['requests']}
    def lookup(req):
        return plans[req.user_id, req.query, tuple(req.options or [])]
    async def understand(_st, req, anchor):
        return copy.deepcopy(lookup(req)['plan'])
    search_pipeline._understand = understand
    search_pipeline._anchor_time = lambda _st, req: lookup(req)['anchor_time']
    # Keep all questions on the same ingestion/usage snapshot, independent of
    # arm-specific returned evidence or request completion order.
    store.Store.record_recall = lambda self, amu_ids, scene_ids: None
    from app.main import app
    uvicorn.run(app, host='127.0.0.1', port=args.port, ssl_certfile=str(args.cert),
                ssl_keyfile=str(args.key), proxy_headers=False)


async def wait_resources():
    while True:
        with Path('/proc/meminfo').open('rb') as f:
            fields = dict(row.split(':', 1) for row in f.read(65536).decode().splitlines())
        available = int(fields['MemAvailable'].split()[0])/1024
        load = os.getloadavg()[0]
        if available >= 1536 and load <= 1.25:
            return
        print(json.dumps(dict(status='waiting_for_arm_resources', available_mib=available, load1=load)), flush=True)
        await asyncio.sleep(15)


async def evaluate(args):
    import httpx
    from app import eval_scoring, run_metadata, schemas
    if config.FAKE:
        raise ValueError('Real providers required')
    check_runtime()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=False)
    source_run = args.run.resolve()
    prepared = args.prepared.resolve()
    old_manifest = json.loads((source_run/'isolated-manifest.json').read_text())
    if digest(prepared) != old_manifest['data_sha256']:
        raise ValueError('Prepared data identity differs')
    # Keep only fixture inputs, not every historical candidate/answer snapshot.
    # This matters when the evaluator shares a 4 GB host with the public API.
    original = [{k: row[k] for k in ('search_id', 'qa_id', 'conversation_id')}
                for row in records(source_run/'completed/data/results/personamem-v2-32k.jsonl')]
    search_ids = {row['search_id'] for row in original}
    trace_fields = ('search_id', 'versions', 'user_id', 'query', 'options', 'top_k',
                    'reference_time', 'evidence_token_budget', 'requested_include_sensitive', 'plan', 'anchor_time')
    traces = {}
    for row in records(source_run/'server-search-debug.jsonl'):
        if row['search_id'] in search_ids:
            traces[row['search_id']] = {key: row[key] for key in trace_fields if key in row}
    conversation_ids = {row['conversation_id'] for row in original}
    conversations = {row['conversation_id']: row for row in records(prepared)
                     if row['conversation_id'] in conversation_ids}
    settings = traces[original[0]['search_id']]['versions']['settings']
    requests = []
    for r in original:
        t = traces[r['search_id']]
        check_versions(t['versions'], 'source trace')
        q = next(q for q in conversations[r['conversation_id']]['qa'] if q['id'] == r['qa_id'])
        if t['query'] != q['question'] or t['options'] != q['options'] or t['versions']['settings'] != settings:
            raise ValueError('Mixed settings or question identity')
        request={k:t[k] for k in ('user_id','query','options','top_k','reference_time','evidence_token_budget')}
        if 'requested_include_sensitive' in t:
            request['include_sensitive']=t['requested_include_sensitive']
        requests.append(dict(qa_id=r['qa_id'], conversation_id=r['conversation_id'],
            request=request,
            plan=t['plan'], anchor_time=t['anchor_time']))
    settings = dict(settings)
    if 'SENSITIVE_RECALL_ENABLED' not in settings:
        if args.legacy_sensitive_recall is None:
            raise ValueError('Historical sensitive policy missing; supply --legacy-sensitive-recall on/off')
        settings['SENSITIVE_RECALL_ENABLED'] = args.legacy_sensitive_recall == 'on'
    elif args.legacy_sensitive_recall is not None:
        raise ValueError('Do not override a recorded sensitive recall policy')
    if len({(r['conversation_id'],r['qa_id']) for r in requests}) != len(requests):
        raise ValueError('Duplicate question identity')
    fixture = run/'fixture.json'
    # No answers, snippet references, old ranks, or gold metadata reach server.
    setting_values = arm_values(args.toggle, settings)
    write(fixture, dict(settings=settings, requests=requests, toggle=args.toggle, setting_values=setting_values))
    snapshot = run/'source'
    project = snapshot/'memory_system'
    for directory in ('app','scripts'):
        shutil.copytree(ROOT/directory, project/directory, symlinks=True, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT.parent/'prompts', snapshot/'prompts', symlinks=True)
    lexicon = ROOT/'data/lexicons/wordnet.zip'
    if lexicon.is_file():
        target = project/'data/lexicons/wordnet.zip'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lexicon, target)
    elif args.toggle == 'PACKET_UNMATCHED_NOUN_CONTEXT':
        raise ValueError('Pinned WordNet is required for this experiment')
    (project/'.venv').symlink_to(ROOT/'.venv', target_is_directory=True)
    cert, key = run/'localhost.crt', run/'localhost.key'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(key),'-out',str(cert),
                    '-days','2','-subj','/CN=localhost','-addext','subjectAltName=DNS:localhost'],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    key.chmod(0o600)
    base_db = run/'frozen-memory.db'
    with sqlite3.connect((source_run/'memory.db').as_uri()+'?mode=ro', uri=True) as src:
        with sqlite3.connect(base_db) as dst:
            src.backup(dst)
    for k,v in settings.items():
        setattr(config,k,v)
    setattr(config,args.toggle,setting_values['control'])
    config.EVAL_CHOICE_POLICY='local'
    check_runtime()
    run_metadata.versions.cache_clear()
    metadata = dict(protocol='frozen-search-https-v1', required_llm_model=REQUIRED_MODEL, source_run=str(source_run),
        questions=len(requests), source_trace_sha256=digest(source_run/'server-search-debug.jsonl'),
        data_sha256=digest(prepared), database_sha256=digest(base_db), fixture_sha256=digest(fixture),
        frozen_planner=True, frozen_usage=True, frozen_anchor=True, fresh_ingestion=False,
        concurrency_per_arm=args.concurrency, arm_schedule='sequential' if args.sequential_arms else 'concurrent',
        toggle=args.toggle, setting_values=setting_values,
        reader_versions=run_metadata.versions(),
        source_files={str(p.relative_to(snapshot)):digest(p) for p in sorted(snapshot.rglob('*'))
                      if p.is_file() and p.suffix in ('.py','.txt','.zip')},
        limitation='Single stochastic paired search run; real HTTPS, frozen Add/planning/usage. Scheduling and host resource limits affect latency; not an unrestricted latency benchmark.')
    write(run/'plan.json', metadata)
    async def arm(name, port):
        await wait_resources()
        out=run/name
        out.mkdir()
        shutil.copyfile(base_db,out/'memory.db')
        shutil.copyfile(source_run/'server-memory-debug.jsonl',out/'server-memory-debug.jsonl')
        manifest=dict(metadata, arm=name, url=f'https://localhost:{port}', initial_database_sha256=digest(out/'memory.db'),
                      fake=False, data=str(prepared), files=metadata['source_files'])
        write(out/'isolated-manifest.json',manifest)
        token=secrets.token_urlsafe(32)
        env=os.environ.copy()
        env.update(AML_DB_PATH=str(out/'memory.db'), AML_API_KEY=token,
                   AML_SEARCH_DEBUG_LOG=str(out/'server-search-debug.jsonl'),
                   AML_MEMORY_DEBUG_LOG=str(out/'server-memory-debug.jsonl'),
                   AML_ACCESS_LOG=str(out/'server-access.jsonl'))
        env.pop('PYTHONPATH',None)
        rows=[]
        server=None
        try:
            with (out/'server.log').open('w') as log:
                server=subprocess.Popen([sys.executable,str(project/'scripts/evaluate_frozen_https.py'),
                    '--serve','--fixture',str(fixture),'--arm',name,'--port',str(port),'--cert',str(cert),'--key',str(key)],
                    cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT)
                manifest['server_pid']=server.pid
                write(out/'isolated-manifest.json',manifest)
                context=ssl.create_default_context(cafile=str(cert))
                async with httpx.AsyncClient(base_url=manifest['url'],verify=context,trust_env=False,
                        headers={'Authorization':'Bearer '+token},timeout=600,
                        # Reader calls leave Search sockets idle; avoid reusing
                        # a connection at the server's keep-alive expiry edge.
                        limits=httpx.Limits(max_keepalive_connections=0)) as client:
                    for _ in range(90):
                        if server.poll() is not None:
                            raise RuntimeError('TLS server exited')
                        try:
                            response=await client.get('/health',timeout=1)
                            response.raise_for_status()
                            break
                        except (httpx.HTTPError,OSError):
                            await asyncio.sleep(.5)
                    else:
                        raise RuntimeError('TLS server did not become healthy')
                    result_path=out/'completed/data/results/personamem-v2-32k.jsonl'
                    result_path.parent.mkdir(parents=True)
                    sem=asyncio.Semaphore(args.concurrency)
                    async def one(item):
                        async with sem:
                            conv=conversations[item['conversation_id']]
                            q=next(q for q in conv['qa'] if q['id']==item['qa_id'])
                            row=dict(qa_id=item['qa_id'],conversation_id=item['conversation_id'],user_id=item['request']['user_id'],
                                dataset=conv['dataset'],scoring=q['scoring'],fake=False,transport='https',run_id=name,
                                frozen_search=True,score=0,request_top_k=item['request']['top_k'])
                            started=time.monotonic()
                            try:
                                check_runtime()
                                response=await client.post('/search',json=item['request'])
                                response.raise_for_status()
                                payload=schemas.SearchResponse.model_validate(response.json())
                                row.update(search_seconds=time.monotonic()-started,search_id=payload.search_id,
                                    packet_hash=payload.packet_hash,coverage_manifest=payload.coverage_manifest,
                                    search_degraded=payload.coverage_manifest['search_degraded'])
                                reader_token=READER_ARM.set(name)
                                try:
                                    pred,score,diagnostics=await eval_scoring.evaluate(q,[v.model_dump() for v in payload.data])
                                finally:
                                    READER_ARM.reset(reader_token)
                                row.update(prediction=pred,score=score,**diagnostics)
                            except Exception as exc:
                                row.update(error_stage='search' if 'search_id' not in row else 'answer',
                                           error_type=type(exc).__name__,error_detail=str(exc)[:500])
                            row['total_seconds']=time.monotonic()-started
                            rows.append(row)
                            with result_path.open('a') as stream:
                                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                            print(json.dumps(dict(arm=name,qa_id=row['qa_id'],score=row['score'],
                                error=row.get('error_stage'),search_seconds=row.get('search_seconds'))),flush=True)
                    await asyncio.gather(*(one(item) for item in requests))
            errors=sum(bool(r.get('error_stage')) for r in rows)
            manifest['client_exit_code']=int(bool(errors))
            times=[r['search_seconds'] for r in rows if 'search_seconds' in r]
            write(out/'summary.json',dict(questions=len(rows),score=sum(r['score'] for r in rows),errors=errors,
                degraded=sum(r.get('search_degraded',True) for r in rows),
                search_calls=sum(r.get('coverage_manifest',{}).get('provider_calls',0) for r in rows),
                search_tokens=sum(r.get('coverage_manifest',{}).get('provider_tokens',0) for r in rows),
                median_search_seconds=statistics.median(times) if times else None))
        finally:
            if server:
                server.terminate()
                try:
                    await asyncio.to_thread(server.wait,10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    await asyncio.to_thread(server.wait)
                manifest['server_exit_code']=server.returncode
            write(out/'isolated-manifest.json',manifest)
            (out/'exit-code').write_text(str(manifest.get('client_exit_code',2))+'\n')
        if len(rows)==len(requests) and all('search_id' in r for r in rows):
            with (out/'audit.log').open('w') as log:
                audit=await asyncio.create_subprocess_exec(sys.executable,str(project/'scripts/audit_https_stages.py'),
                    '--run',str(out),'--prepared',str(prepared),'--gold-csv',str(ROOT/'data/raw/personamem-v2.csv'),
                    stdout=log,stderr=log)
                manifest['audit_exit_code']=await audit.wait()
            write(out/'isolated-manifest.json',manifest)
            if args.toggle in ('OVERFLOW_CITATION_KEYS', 'LISTWISE_VALID_CITATION_CONTRACT', 'FINE_SOURCE_RRF', 'OPTION_MARGIN_SOURCE_ADMISSION', 'SOURCE_CE_LATE_ADMISSION', 'JOINT_SOURCE_RANK_ADMISSION'):
                audit_script = ('audit_recovery_citation_keys.py' if args.toggle == 'OVERFLOW_CITATION_KEYS' else
                                'audit_recovery_citation_contract.py' if args.toggle == 'LISTWISE_VALID_CITATION_CONTRACT' else
                                'audit_fine_source_rrf_https.py' if args.toggle == 'FINE_SOURCE_RRF' else
                                'audit_option_margin_https.py' if args.toggle == 'OPTION_MARGIN_SOURCE_ADMISSION' else
                                'audit_joint_source_https.py' if args.toggle == 'JOINT_SOURCE_RANK_ADMISSION'
                                else 'audit_screened_source_https.py')
                with (out/'source-audit.log').open('w') as log:
                    audit=await asyncio.create_subprocess_exec(sys.executable,str(project/'scripts'/audit_script),
                        '--run',str(out),stdout=log,stderr=log)
                    manifest['source_audit_exit_code']=await audit.wait()
                write(out/'isolated-manifest.json',manifest)
        print(json.dumps(dict(arm=name,finished=True,client_exit_code=manifest.get('client_exit_code'),
                              audit_exit_code=manifest.get('audit_exit_code'))),flush=True)
        return bool(manifest.get('client_exit_code', 2) or manifest.get('audit_exit_code', 2)
                    or manifest.get('source_audit_exit_code', 0))
    logger=logging.getLogger('aml.metrics');previous_level=logger.level;logger.setLevel(logging.INFO)
    handler=logging.FileHandler(run/'reader-provider.log')
    handler.addFilter(ReaderArmFilter())
    handler.setFormatter(logging.Formatter('reader_arm=%(reader_arm)s %(message)s'))
    logger.addHandler(handler)
    try:
        failed = await run_arms(arm, args.port, sequential=args.sequential_arms)
    finally:
        logger.removeHandler(handler);handler.close();logger.setLevel(previous_level)
    code = int(any(failed))
    (run/'exit-code').write_text(str(code)+'\n')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serve',action='store_true')
    p.add_argument('--run',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--prepared',type=Path,default=ROOT/'data/prepared/personamem-v2-32k.jsonl')
    p.add_argument('--port',type=int,default=18546)
    p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--sequential-arms',action='store_true',help='Finish each arm and its audit before starting the next service')
    p.add_argument('--legacy-sensitive-recall',choices=['on','off'])
    p.add_argument('--toggle',choices=['OVERFLOW_CITATION_KEYS','LISTWISE_VALID_CITATION_CONTRACT','FINE_SOURCE_RRF','OPTION_MARGIN_SOURCE_ADMISSION','JOINT_SOURCE_RANK_ADMISSION','SOURCE_CE_LATE_ADMISSION','PACKET_ENDPOINT_FINE_COMPLETION','RECALL_PORTER_SUPPLEMENT','RERANK_MAX_PROMPT_BYTES','LISTWISE_OVERFLOW_BATCHED','PACKET_UNMATCHED_NOUN_CONTEXT','PACKET_EXCLUSIVE_SHARED_CONTEXT','PACKET_SPARSE_SOURCE_CONTEXT','PACKET_CORE_GAP_CONTEXT','FINE_NEW_COARSE_MISSING_ONLY','FINE_NEW_COARSE_INSURANCE','FINE_SEMANTIC_RESERVE','PACKET_QUERY_CONTEXT_COMBINED','PACKET_MARGINAL_QUERY_CONTEXT','PACKET_SOURCE_FIRST_RESIDUAL','PACKET_QUESTION_CONTEXT_CE','PACKET_SOURCE_QUERY_CONTEXT','LISTWISE_OVERFLOW_REVIEW','LISTWISE_ROLE_BOUND_REFS','LISTWISE_TRANSFORM_CONTEXT','LISTWISE_TOPIC_INTEREST','PACKET_ADJACENT_CONTEXT','PACKET_FINE_CONTEXT_RESERVE','PACKET_FINE_CONTEXT_5K','PACKET_FINE_CONTEXT_LEXICAL','PACKET_FINE_CONTEXT_FILL','LISTWISE_ABBREVIATION_PARTS','LISTWISE_OVERFLOW_EXTEND','COARSE_EVIDENCE_EXTEND','FINE_COARSE_CORE_QUOTA','FINE_WEAK_PROMOTION_ONE','SOURCE_RECALL_NOVELTY'],
                   default='LISTWISE_OVERFLOW_REVIEW')
    p.add_argument('--fixture',type=Path)
    p.add_argument('--arm',choices=['control','treatment'])
    p.add_argument('--cert',type=Path)
    p.add_argument('--key',type=Path)
    args=p.parse_args()
    if args.concurrency<1:
        p.error('Concurrency must be positive')
    if args.serve:
        serve(args)
    else:
        if not args.run or not args.output:
            p.error('--run and --output required')
        sys.exit(asyncio.run(evaluate(args)))
