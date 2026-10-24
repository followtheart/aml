"""Fresh real HTTPS Search once per question, two isolated readers per packet.

Frozen ingestion/plans/usage, live Search, identical exact HTTP response for both
readers. Only the selected reader toggle differs. This isolates reader changes
from Search variability; it is not two independent Search runs. Gold remains in
the scoring clients. No successful answer is retried or substituted.
An explicit optional core-gap Search setting is shared by both readers and
recorded separately; its answer effect is not isolated by the reader contrast.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import ssl
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import config
from evaluation_model_guard import check_runtime,check_versions

TOGGLE='CHOICE_WITNESS_KEYED_OUTPUT'
TOGGLES=(TOGGLE,'CHOICE_WITNESS_LEXICAL_FILL','CHOICE_CITED_SCOPE')


def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def fingerprint(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


async def reader(args):
    from app import eval_scoring,run_metadata,evidence_packet
    fixture=json.loads(args.fixture.read_text())
    for k,v in fixture['settings'].items():setattr(config,k,v)
    toggle=fixture.get('reader_toggle',TOGGLE)
    if toggle not in TOGGLES:raise ValueError('Unknown reader toggle')
    setattr(config,toggle,args.arm=='treatment')
    config.EVAL_CHOICE_POLICY='local';check_runtime()
    metrics_log=logging.getLogger('aml.metrics');metrics_log.setLevel(logging.INFO)
    metrics_log.addHandler(logging.StreamHandler(sys.stderr))
    if config.FAKE:raise ValueError('Real reader required')
    run_metadata.versions.cache_clear()
    out=args.output;write(out/'reader-versions.json',run_metadata.versions())
    path=out/'completed/data/results/personamem-v2-32k.jsonl';path.parent.mkdir(parents=True)
    semaphore=asyncio.Semaphore(args.concurrency);rows=[];jobs=[];seen=set()
    async def one(job):
        async with semaphore:
            qa=job['qa'];payload=job['payload'];packet=payload['data']
            if evidence_packet.digest(packet)!=payload['packet_hash']:raise ValueError('HTTP packet mismatch')
            row=dict(qa_id=qa['id'],conversation_id=job['conversation_id'],user_id=job['request']['user_id'],
                dataset=job['dataset'],scoring=qa['scoring'],fake=False,transport='https',run_id=args.arm,
                shared_search=True,score=0,request_top_k=job['request']['top_k'],
                search_seconds=job['search_seconds'],search_id=payload['search_id'],packet_hash=payload['packet_hash'],
                coverage_manifest=payload['coverage_manifest'],search_degraded=payload['coverage_manifest']['search_degraded'],
                http_payload_sha256=fingerprint(payload),reader_input_sha256=fingerprint(job),
                reader_toggle=toggle,reader_toggle_enabled=getattr(config,toggle),
                reader_keyed_output=config.CHOICE_WITNESS_KEYED_OUTPUT)
            started=time.monotonic()
            try:
                check_runtime()
                pred,score,diagnostics=await eval_scoring.evaluate(qa,packet)
                row.update(prediction=pred,score=score,**diagnostics)
            except Exception as exc:row.update(error_stage='answer',error_type=type(exc).__name__)
            row['reader_seconds']=time.monotonic()-started
            row['total_seconds']=row['search_seconds']+row['reader_seconds'];rows.append(row)
            with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps(dict(arm=args.arm,qa_id=qa['id'],score=row['score'],error=row.get('error_stage'))),flush=True)
    while line:=await asyncio.to_thread(sys.stdin.readline):
        job=json.loads(line);key=job['conversation_id'],job['qa']['id']
        if key in seen:raise ValueError('Duplicate reader input')
        seen.add(key);jobs.append(asyncio.create_task(one(job)))
    await asyncio.gather(*jobs)
    if len(rows)!=fixture['questions']:raise ValueError('Incomplete reader cohort')
    errors=sum(bool(r.get('error_stage')) for r in rows)
    write(out/'summary.json',dict(questions=len(rows),score=sum(r['score'] for r in rows),errors=errors,
        witness_fallbacks=sum(r.get('answer_witness_retrieval',{}).get('status')=='fallback' for r in rows)))
    (out/'exit-code').write_text(str(int(bool(errors)))+'\n')
    return int(bool(errors))


async def evaluate(args):
    import httpx
    from app import schemas
    check_runtime()
    if config.FAKE:raise ValueError('Real Search required')
    source=args.run.resolve();root=args.output.resolve();prepared=args.prepared.resolve()
    old=json.loads((source/'isolated-manifest.json').read_text())
    if digest(prepared)!=old['data_sha256']:raise ValueError('Prepared dataset changed')
    original=records(source/'completed/data/results/personamem-v2-32k.jsonl')
    traces={r['search_id']:r for r in records(source/'server-search-debug.jsonl')}
    conversations={r['conversation_id']:r for r in records(prepared)}
    settings=traces[original[0]['search_id']]['versions']['settings'];requests=[]
    for r in original:
        t=traces[r['search_id']];check_versions(t['versions'],'source trace')
        if t['versions']['settings']!=settings:raise ValueError('Mixed settings')
        qa=next(q for q in conversations[r['conversation_id']]['qa'] if q['id']==r['qa_id'])
        if qa['question']!=t['query'] or qa['options']!=t['options']:raise ValueError('Question identity mismatch')
        req={k:t[k] for k in ('user_id','query','options','top_k','reference_time','evidence_token_budget')}
        req['include_sensitive']=t['requested_include_sensitive']
        requests.append(dict(qa_id=r['qa_id'],conversation_id=r['conversation_id'],request=req,
            plan=t['plan'],anchor_time=t['anchor_time']))
    if len(requests)!=56:raise ValueError('Require full 56-question cohort')
    settings=dict(settings)
    search_overrides={}
    if args.search_core_gap_context:
        from app.core_gap_context import INCOMPATIBLE
        if not all(settings.get(k) for k in ('PACKET_FINE_CONTEXT_RESERVE','PACKET_FINE_CONTEXT_5K','PACKET_FINE_CONTEXT_LEXICAL')):
            raise ValueError('Core-gap Search requires the original 5 KiB lexical prefix')
        if any(settings.get(k,False) for k in INCOMPATIBLE):
            raise ValueError('Incompatible historical context policy')
        search_overrides['PACKET_CORE_GAP_CONTEXT']=True
        settings.update(search_overrides)
    root.mkdir(parents=True,exist_ok=False);search=root/'search';search.mkdir()
    fixture=root/'fixture.json'
    # Only requests/plans/settings reach the HTTPS server; no QA answers/snippets.
    write(fixture,dict(settings=settings,requests=requests,questions=len(requests),
        toggle=args.toggle,reader_toggle=args.toggle,search_overrides=search_overrides))
    snapshot=root/'source';project=snapshot/'memory_system'
    for directory in ('app','scripts'):
        shutil.copytree(ROOT/directory,project/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT.parent/'prompts',snapshot/'prompts');(project/'.venv').symlink_to(ROOT/'.venv',target_is_directory=True)
    base=root/'frozen-memory.db'
    with sqlite3.connect((source/'memory.db').as_uri()+'?mode=ro',uri=True) as src:
        with sqlite3.connect(base) as dst:src.backup(dst)
    shutil.copyfile(base,search/'memory.db');shutil.copyfile(source/'server-memory-debug.jsonl',search/'server-memory-debug.jsonl')
    cert,key=root/'localhost.crt',root/'localhost.key'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(key),'-out',str(cert),
        '-days','2','-subj','/CN=localhost','-addext','subjectAltName=DNS:localhost'],check=True,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);key.chmod(0o600)
    plan=dict(protocol='shared-search-reader-https-v1',source_run=str(source),questions=len(requests),
        reader_toggle=args.toggle,required_llm_model=config.LLM_MODEL,source_trace_sha256=digest(source/'server-search-debug.jsonl'),
        data_sha256=digest(prepared),database_sha256=digest(base),fixture_sha256=digest(fixture),
        source_files={str(p.relative_to(snapshot)):digest(p) for p in sorted(snapshot.rglob('*')) if p.is_file() and p.suffix in ('.py','.txt')},
        frozen_planner=True,frozen_usage=True,frozen_anchor=True,fresh_ingestion=False,shared_live_search=True,
        search_concurrency=args.concurrency,reader_concurrency_per_arm=args.concurrency,
        search_overrides=search_overrides,limitation=__doc__)
    write(root/'plan.json',plan)
    token=secrets.token_urlsafe(32);env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(AML_DB_PATH=str(search/'memory.db'),AML_API_KEY=token,AML_SEARCH_DEBUG_LOG=str(search/'server-search-debug.jsonl'),
        AML_MEMORY_DEBUG_LOG=str(search/'server-memory-debug.jsonl'),AML_ACCESS_LOG=str(search/'server-access.jsonl'))
    server=None;workers={};logs=[];manifest=dict(plan,server_url=f'https://localhost:{args.port}',fake=False,data=str(prepared))
    completed=[];failure=False
    try:
        for arm in ('control','treatment'):
            out=root/arm;out.mkdir();log=(out/'reader.log').open('w');logs.append(log)
            workers[arm]=await asyncio.create_subprocess_exec(sys.executable,str(project/'scripts/evaluate_shared_search_https.py'),
                '--reader','--fixture',str(fixture),'--arm',arm,'--output',str(out),'--concurrency',str(args.concurrency),
                cwd=project,env=env,stdin=asyncio.subprocess.PIPE,stdout=log,stderr=log)
        server_log=(search/'server.log').open('w');logs.append(server_log)
        server=subprocess.Popen([sys.executable,str(project/'scripts/evaluate_frozen_https.py'),'--serve',
            '--fixture',str(fixture),'--arm','control','--port',str(args.port),'--cert',str(cert),'--key',str(key)],
            cwd=project,env=env,stdout=server_log,stderr=subprocess.STDOUT)
        manifest.update(server_pid=server.pid,reader_pids={a:p.pid for a,p in workers.items()});write(root/'manifest.json',manifest)
        context=ssl.create_default_context(cafile=str(cert))
        async with httpx.AsyncClient(base_url=manifest['server_url'],verify=context,trust_env=False,
                headers={'Authorization':'Bearer '+token},timeout=600,
                limits=httpx.Limits(max_keepalive_connections=0)) as client:
            for _ in range(90):
                if server.poll() is not None:raise RuntimeError('TLS server exited')
                try:
                    response=await client.get('/health',timeout=1);response.raise_for_status();break
                except (httpx.HTTPError,OSError):await asyncio.sleep(.5)
            else:raise RuntimeError('TLS server did not become healthy')
            semaphore=asyncio.Semaphore(args.concurrency);send_lock=asyncio.Lock()
            async def one(item):
                async with semaphore:
                    start=time.monotonic();response=await client.post('/search',json=item['request']);response.raise_for_status()
                    payload=schemas.SearchResponse.model_validate(response.json()).model_dump()
                    conv=conversations[item['conversation_id']];qa=next(q for q in conv['qa'] if q['id']==item['qa_id'])
                    job=dict(qa=qa,conversation_id=item['conversation_id'],dataset=conv['dataset'],request=item['request'],
                        payload=payload,search_seconds=time.monotonic()-start)
                    encoded=(json.dumps(job,ensure_ascii=False)+'\n').encode()
                    async with send_lock:
                        with (root/'http-reader-inputs.jsonl').open('ab') as f:f.write(encoded)
                        for process in workers.values():process.stdin.write(encoded)
                        await asyncio.gather(*(p.stdin.drain() for p in workers.values()))
                    completed.append(item['qa_id']);print(json.dumps(dict(search_completed=len(completed),qa_id=item['qa_id'])),flush=True)
            await asyncio.gather(*(one(item) for item in requests))
        for process in workers.values():process.stdin.close()
        server.terminate();await asyncio.to_thread(server.wait,10);manifest['server_exit_code']=server.returncode
        codes=await asyncio.gather(*(p.wait() for p in workers.values()))
        manifest['reader_exit_codes']=dict(zip(workers,codes));failure=any(codes)
    finally:
        if server and server.poll() is None:
            server.terminate()
            try:await asyncio.to_thread(server.wait,10)
            except subprocess.TimeoutExpired:server.kill();await asyncio.to_thread(server.wait)
        if server:manifest['server_exit_code']=server.returncode
        for process in workers.values():
            if process.returncode is None:
                process.terminate();await process.wait()
        for log in logs:log.close()
        manifest['search_questions_completed']=len(completed);write(root/'manifest.json',manifest)
    manifest['audit_exit_codes']={}
    for arm in ('control','treatment'):
        out=root/arm
        for name in ('server-search-debug.jsonl','server-memory-debug.jsonl'):
            shutil.copyfile(search/name,out/name)
        with (out/'audit.log').open('w') as log:
            audit=await asyncio.create_subprocess_exec(sys.executable,str(project/'scripts/audit_https_stages.py'),
                '--run',str(out),'--prepared',str(prepared),'--gold-csv',str(ROOT/'data/raw/personamem-v2.csv'),stdout=log,stderr=log)
            manifest['audit_exit_codes'][arm]=await audit.wait()
    failure=failure or any(manifest['audit_exit_codes'].values()) or len(completed)!=len(requests)
    write(root/'manifest.json',manifest);(root/'exit-code').write_text(str(int(failure))+'\n')
    return int(failure)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reader',action='store_true')
    p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=ROOT/'data/prepared/personamem-v2-32k.jsonl')
    p.add_argument('--fixture',type=Path);p.add_argument('--arm',choices=['control','treatment'])
    p.add_argument('--port',type=int,default=18586);p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--toggle',choices=TOGGLES,default=TOGGLE)
    p.add_argument('--search-core-gap-context',action='store_true',
        help='Enable existing bounded core-gap context in the shared Search, keeping other historical settings')
    a=p.parse_args()
    if a.concurrency<1:p.error('Positive concurrency required')
    if a.reader and (not a.fixture or not a.arm):p.error('Reader fixture/arm required')
    if not a.reader and not a.run:p.error('Source run required')
    sys.exit(asyncio.run(reader(a) if a.reader else evaluate(a)))
