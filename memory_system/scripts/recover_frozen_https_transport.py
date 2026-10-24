"""Recover unserved transport failures into a separate, fully audited experiment.

The original run is immutable. Only failed Search rows with no service trace
may be replayed. Successful rows are copied byte for byte; frozen server/reader
code, request, database, model and planner are reused. Recovery is disclosed.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import ssl
import statistics
import subprocess
import sys
import time


def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


async def main(args):
    source=args.run.resolve();out=args.output.resolve()
    if not (source/'exit-code').exists():raise ValueError('Original run is still active')
    if out.exists():raise FileExistsError(out)
    plan=json.loads((source/'plan.json').read_text());fixture=json.loads((source/'fixture.json').read_text())
    if plan['protocol']!='frozen-search-https-v1':raise ValueError('Require frozen HTTPS protocol')
    expected={(r['conversation_id'],r['qa_id']):r for r in fixture['requests']}
    jobs={};original_hashes={}
    for arm in ('control','treatment'):
        path=source/arm;manifest=json.loads((path/'isolated-manifest.json').read_text())
        if 'server_exit_code' not in manifest:raise ValueError('Original server not terminal')
        proc=Path('/proc')/str(manifest['server_pid'])/'cmdline'
        if proc.exists() and str(source).encode() in proc.read_bytes():raise ValueError('Original server still running')
        rows=records(path/'completed/data/results/personamem-v2-32k.jsonl')
        traces=records(path/'server-search-debug.jsonl');failed=[]
        if len(rows)!=plan['questions']:raise ValueError('Original client did not finish all attempts')
        for index,row in enumerate(rows):
            if 'error_stage' not in row:continue
            if row['error_stage']!='search' or row['error_type']!='RemoteProtocolError' or 'search_id' in row:
                raise ValueError('Only unserved Search disconnects can be recovered')
            item=expected[row['conversation_id'],row['qa_id']]
            if any(t['user_id']==item['request']['user_id'] and t['query']==item['request']['query'] for t in traces):
                raise ValueError('Failed request has already reached the service')
            failed.append((index,row,item))
        if len(traces)+len(failed)!=len(rows):raise ValueError('Unexpected service trace count')
        jobs[arm]=failed
        original_hashes[arm]=digest(path/'completed/data/results/personamem-v2-32k.jsonl')
    if not any(jobs.values()):raise ValueError('No recoverable failures')
    for name,key in (('fixture.json','fixture_sha256'),('frozen-memory.db','database_sha256')):
        if digest(source/name)!=plan[key]:raise ValueError('Frozen input changed')
    for name,value in plan['source_files'].items():
        if digest(source/'source'/name)!=value:raise ValueError('Frozen code changed')
    shutil.copytree(source,out,symlinks=True)
    project=out/'source/memory_system'
    # Frozen snapshots intentionally omit credentials. Bootstrap the same
    # local environment as the original runner before importing frozen config.
    env_file=Path(__file__).resolve().parents[1]/'.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line:continue
            key,value=line.split('=',1)
            os.environ.setdefault(key.strip(),value.strip().strip('"').strip("'"))
    sys.path.insert(0,str(project));sys.path.insert(0,str(project/'scripts'))
    import httpx
    from app import config,eval_scoring,run_metadata,schemas
    from evaluation_model_guard import check_runtime
    for key,value in fixture['settings'].items():setattr(config,key,value)
    setattr(config,plan['toggle'],False)  # Original runner's reader configuration.
    config.EVAL_CHOICE_POLICY='local';check_runtime();run_metadata.versions.cache_clear()
    if run_metadata.versions()!=plan['reader_versions']:raise ValueError('Reader source or configuration changed')
    recovery=dict(original_run=str(source),original_plan_sha256=digest(source/'plan.json'),
        helper_sha256=digest(Path(__file__)),original_result_hashes=original_hashes,attempts=[],
        limitation='Only unserved transport failures replayed later, outside original concurrency. Report original-attempt matched subset separately.')
    write(out/'transport-recovery.json',recovery)
    plan['transport_recovery']='transport-recovery.json';write(out/'plan.json',plan)
    for arm,failed in jobs.items():
        if not failed:continue
        path=out/arm;manifest=json.loads((path/'isolated-manifest.json').read_text())
        prepared=Path(manifest['data'])
        if digest(prepared)!=plan['data_sha256']:raise ValueError('Prepared input changed')
        conversations={r['conversation_id']:r for r in records(prepared)}
        shutil.copyfile(out/'frozen-memory.db',path/'memory.db')
        result_path=path/'completed/data/results/personamem-v2-32k.jsonl'
        lines=result_path.read_text().splitlines(keepends=True)
        token=secrets.token_urlsafe(32);port=args.port+(arm=='treatment')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        env.update(AML_DB_PATH=str(path/'memory.db'),AML_API_KEY=token,
            AML_SEARCH_DEBUG_LOG=str(path/'server-search-debug.jsonl'),
            AML_MEMORY_DEBUG_LOG=str(path/'server-memory-debug.jsonl'),AML_ACCESS_LOG=str(path/'server-access.jsonl'))
        manifest['transport_recovery_original_manifest']=str(source/arm/'isolated-manifest.json')
        server=None
        try:
            with (path/'server.log').open('a') as log:
                server=subprocess.Popen([sys.executable,str(project/'scripts/evaluate_frozen_https.py'),'--serve',
                    '--fixture',str(out/'fixture.json'),'--arm',arm,'--port',str(port),
                    '--cert',str(out/'localhost.crt'),'--key',str(out/'localhost.key')],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT)
                manifest.update(server_pid=server.pid,url=f'https://localhost:{port}')
                write(path/'isolated-manifest.json',manifest)
                context=ssl.create_default_context(cafile=str(out/'localhost.crt'))
                async with httpx.AsyncClient(base_url=manifest['url'],verify=context,trust_env=False,
                        headers={'Authorization':'Bearer '+token},timeout=600,
                        limits=httpx.Limits(max_keepalive_connections=0)) as client:
                    for _ in range(90):
                        if server.poll() is not None:raise RuntimeError('Recovery TLS server exited')
                        try:
                            response=await client.get('/health',timeout=1);response.raise_for_status();break
                        except (httpx.HTTPError,OSError):await asyncio.sleep(.5)
                    else:raise RuntimeError('Recovery TLS health timeout')
                    for index,old,item in failed:
                        conv=conversations[old['conversation_id']]
                        q=next(q for q in conv['qa'] if q['id']==old['qa_id'])
                        if q['question']!=item['request']['query'] or q['options']!=item['request']['options']:
                            raise ValueError('Question identity changed')
                        row=dict(qa_id=old['qa_id'],conversation_id=old['conversation_id'],user_id=item['request']['user_id'],
                            dataset=conv['dataset'],scoring=q['scoring'],fake=False,transport='https',run_id=arm,
                            frozen_search=True,score=0,request_top_k=item['request']['top_k'],transport_recovered=True)
                        started=time.monotonic();check_runtime()
                        response=await client.post('/search',json=item['request']);response.raise_for_status()
                        payload=schemas.SearchResponse.model_validate(response.json())
                        row.update(search_seconds=time.monotonic()-started,search_id=payload.search_id,
                            packet_hash=payload.packet_hash,coverage_manifest=payload.coverage_manifest,
                            search_degraded=payload.coverage_manifest['search_degraded'])
                        pred,score,diagnostics=await eval_scoring.evaluate(q,[v.model_dump() for v in payload.data])
                        row.update(prediction=pred,score=score,**diagnostics,total_seconds=time.monotonic()-started)
                        lines[index]=json.dumps(row,ensure_ascii=False)+'\n'
                        recovery['attempts'].append(dict(arm=arm,qa_id=old['qa_id'],original_error=old,
                            replacement_search_id=row['search_id'],replacement_score=score))
                        result_path.write_text(''.join(lines));write(out/'transport-recovery.json',recovery)
                        print(json.dumps(dict(arm=arm,qa_id=old['qa_id'],score=score,recovered=True)),flush=True)
        finally:
            if server:
                server.terminate()
                try:await asyncio.to_thread(server.wait,10)
                except subprocess.TimeoutExpired:server.kill();await asyncio.to_thread(server.wait)
                manifest['server_exit_code']=server.returncode
            write(path/'isolated-manifest.json',manifest)
        rows=records(result_path)
        if any('error_stage' in r for r in rows):raise ValueError('Recovery incomplete')
        for index,old_line in enumerate((source/arm/'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines(keepends=True)):
            if index not in {j[0] for j in failed} and lines[index]!=old_line:raise ValueError('Successful row changed')
        manifest['client_exit_code']=0
        write(path/'summary.json',dict(questions=len(rows),score=sum(r['score'] for r in rows),errors=0,
            original_transport_errors=len(failed),degraded=sum(r.get('search_degraded',True) for r in rows),
            search_calls=sum(r['coverage_manifest']['provider_calls'] for r in rows),
            search_tokens=sum(r['coverage_manifest']['provider_tokens'] for r in rows),
            median_search_seconds=statistics.median(r['search_seconds'] for r in rows)))
        (path/'exit-code').write_text('0\n')
        with (path/'recovery-audit.log').open('w') as log:
            audit=subprocess.run([sys.executable,str(project/'scripts/audit_https_stages.py'),'--run',str(path),
                '--prepared',str(prepared),'--gold-csv',str(args.gold_csv.resolve())],stdout=log,stderr=log)
        manifest['audit_exit_code']=audit.returncode;write(path/'isolated-manifest.json',manifest)
        if audit.returncode:raise RuntimeError('Recovered stage audit failed')
    recovery['final_result_hashes']={a:digest(out/a/'completed/data/results/personamem-v2-32k.jsonl') for a in jobs}
    recovery['original_runs_unchanged']=all(digest(source/a/'completed/data/results/personamem-v2-32k.jsonl')==v for a,v in original_hashes.items())
    if not recovery['original_runs_unchanged']:raise ValueError('Original run changed')
    write(out/'transport-recovery.json',recovery);(out/'exit-code').write_text('0\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=18572)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    asyncio.run(main(p.parse_args()))
