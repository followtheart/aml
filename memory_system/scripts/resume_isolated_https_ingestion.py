"""Resume a terminal, unscored isolated run in a new artifact directory.

Copies the frozen application and SQLite database; original failure artifacts
remain untouched. Reuses the exact user/request IDs so committed Adds are noops.
Only the copied evaluator's run-ID allocation is instrumented.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import ssl
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import config, eval_data


def verify_interrupted(failed, manifest):
    """Require live OS evidence before recovering a run without exit records."""
    pid = manifest.get('server_pid')
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError('Interrupted run lacks a server PID')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Original server PID still exists')
    proc = Path('/proc')
    if not proc.is_dir():
        raise ValueError('Interrupted recovery requires /proc process inspection')
    expected = {f'AML_EXPERIMENT_DIR={failed.resolve()}'.encode(),
                f'AML_DB_PATH={failed.resolve() / "memory.db"}'.encode()}
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            environment = set((entry / 'environ').read_bytes().split(b'\0'))
            cwd = (entry / 'cwd').resolve(strict=True)
        except (FileNotFoundError, ProcessLookupError):
            continue
        if expected & environment or cwd.is_relative_to(failed.resolve()):
            raise ValueError(f'Original run process still exists: {entry.name}')
    return dict(server_pid=pid, server_pid_absent=True, matching_processes=0,
                observed_at=time.time(), exit_codes_unknown=True)


def preflight(failed, plan_path, allow_interrupted=False):
    manifest=json.loads((failed/'isolated-manifest.json').read_text())
    interruption = None
    if manifest.get('client_exit_code')!=1 or manifest.get('server_exit_code') is None:
        if not allow_interrupted or manifest.get('client_exit_code') is not None:
            raise ValueError('Require terminal failed client and server')
        interruption = verify_interrupted(failed, manifest)
    result=failed/'completed/data/results/personamem-v2-32k.jsonl'
    if result.exists() and result.stat().st_size:
        raise ValueError('Scoring already began; ingestion-only recovery refused')
    for filename,digest in manifest['files'].items():
        if hashlib.sha256((failed/'source'/filename).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen server source changed')
    plan=json.loads(plan_path.read_text());data=Path(manifest['data'])
    if data.resolve()!=Path(plan['input']).resolve() or hashlib.sha256(data.read_bytes()).hexdigest()!=plan['sha256']:
        raise ValueError('Input path/hash changed')
    records=[json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    if len(records)!=1 or len(records[0]['qa'])!=plan['expected_qa']:
        raise ValueError('Require one complete planned conversation')
    run_ids=re.findall(r'^run_id=([0-9a-f]{32}) ',(failed/'output.log').read_text(),re.M)
    if len(run_ids)!=1:raise ValueError('Ambiguous original run ID')
    rid=run_ids[0];conv=records[0];uid=f"https:{rid}:1:{conv.get('dataset','normalized')}:{conv['conversation_id']}"
    chunks=list(eval_data.ingestion_chunks(conv,20,2000))
    expected={f'{uid}:session:{sid}:chunk:{chunk}':messages for sid,chunk,messages in chunks}
    conn=sqlite3.connect(f'file:{(failed/"memory.db").resolve()}?mode=ro',uri=True);conn.row_factory=sqlite3.Row
    try:
        committed=[dict(r) for r in conn.execute('SELECT * FROM requests ORDER BY created_at')]
        ids=[r['request_id'] for r in committed]
        if not ids or ids!=list(expected)[:len(ids)] or len(ids)>=len(chunks):
            raise ValueError('Committed requests are not a proper prefix')
        stored=[dict(r) for r in conn.execute('SELECT * FROM source_messages')]
        if {r['request_id'] for r in stored}!=set(ids):raise ValueError('Uncommitted source messages exist')
        for request in committed:
            request_id=request['request_id'];actual=sorted([r for r in stored if r['request_id']==request_id],key=lambda r:r['message_index'])
            if request['user_id']!=uid or request['session_id']!=request_id.rsplit(':chunk:',1)[0]:
                raise ValueError('Request ownership mismatch')
            wanted=expected[request_id]
            if len(actual)!=len(wanted):raise ValueError('Source count mismatch')
            for i,(a,b) in enumerate(zip(actual,wanted)):
                if a['message_index']!=i or any(a[k]!=b.get(k) for k in ('role','content','timestamp')):
                    raise ValueError('Committed source differs from input')
        debug=[json.loads(l) for l in (failed/'server-memory-debug.jsonl').read_text().splitlines()]
        if [r['request_id'] for r in debug]!=ids:raise ValueError('Debug/DB commit mismatch')
    finally:conn.close()
    return dict(run_id=rid,user_id=uid,committed_chunks=len(ids),messages=len(stored),total_chunks=len(chunks),
                input=str(data),input_sha256=plan['sha256'],manifest=manifest,
                interruption=interruption)


def main(args):
    failed=args.failed_run.resolve();verified=preflight(failed,args.plan,args.allow_interrupted)
    if args.inspect:
        print(json.dumps({k:v for k,v in verified.items() if k!='manifest'}));return 0
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=False)
    shutil.copytree(failed/'source',run/'source',symlinks=True)
    src=sqlite3.connect(f'file:{(failed/"memory.db").resolve()}?mode=ro',uri=True)
    dst=sqlite3.connect(run/'memory.db')
    try:src.backup(dst)
    finally:src.close();dst.close()
    shutil.copy2(failed/'server-memory-debug.jsonl',run/'server-memory-debug.jsonl')
    project=run/'source/memory_system';evaluator=project/'scripts/https_eval.py'
    text=evaluator.read_text();needle='    run_id = uuid.uuid4().hex'
    if text.count(needle)!=1:raise ValueError('Unexpected evaluator run ID allocation')
    evaluator.write_text(text.replace(needle,"    run_id = os.environ['AML_HTTPS_RESUME_RUN_ID']",1))
    cert=run/'localhost.crt';key=run/'localhost.key'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-keyout',str(key),'-out',str(cert),
                    '-days','2','-subj','/CN=localhost','-addext','subjectAltName=DNS:localhost'],check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);key.chmod(0o600)
    env=os.environ.copy();env.update(AML_API_KEY=secrets.token_urlsafe(32),AML_DB_PATH=str(run/'memory.db'),
        AML_MEMORY_DEBUG_LOG=str(run/'server-memory-debug.jsonl'),AML_SEARCH_DEBUG_LOG=str(run/'server-search-debug.jsonl'),
        AML_ACCESS_LOG=str(run/'server-access.jsonl'),AML_HTTPS_URL=f'https://localhost:{args.port}',AML_HTTPS_CA=str(cert),
        AML_EXPERIMENT_DIR=str(run),AML_EVAL_DATA=verified['input'],AML_EVAL_CONVS='1',AML_EVAL_LIMIT='0',
        AML_EVAL_CHUNK_MESSAGES='20',AML_EVAL_CHUNK_WORDS='2000',AML_HTTPS_RESUME_RUN_ID=verified['run_id'])
    env.pop('AML_HTTPS_TOKEN_FILE',None);env.pop('PYTHONPATH',None)
    manifest=dict(verified['manifest'],url=env['AML_HTTPS_URL'],recovery_from=str(failed),
        recovery={k:v for k,v in verified.items() if k!='manifest'},
        evaluator_change='Copied evaluator uses verified original run ID; server app/prompts unchanged',
        evaluator_sha256=hashlib.sha256(evaluator.read_bytes()).hexdigest())
    for k in ('client_exit_code','audit_exit_code','server_exit_code','server_pid'):manifest.pop(k,None)
    def save():(run/'isolated-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save();server=None
    try:
        with (run/'server.log').open('w') as log:
            server=subprocess.Popen([sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port',str(args.port),
                '--ssl-certfile',str(cert),'--ssl-keyfile',str(key),'--no-proxy-headers'],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT)
            manifest['server_pid']=server.pid;save();context=ssl.create_default_context(cafile=str(cert))
            for _ in range(60):
                if server.poll() is not None:raise RuntimeError('Recovery server exited')
                try:
                    with urllib.request.urlopen(env['AML_HTTPS_URL']+'/health',context=context,timeout=1) as response:
                        if json.load(response).get('status')=='ok':break
                except OSError:pass
                time.sleep(.5)
            else:raise RuntimeError('Recovery service did not become healthy')
            print(json.dumps(dict(server_pid=server.pid,reusing_committed_chunks=verified['committed_chunks'],run_id=verified['run_id'])),flush=True)
            client=subprocess.run(['bash','scripts/run_experiment_https.sh','--no-progress'],cwd=project,env=env)
            manifest['client_exit_code']=client.returncode;save()
            result=run/'completed/data/results/personamem-v2-32k.jsonl'
            if result.exists() and result.stat().st_size:
                audit=subprocess.run([sys.executable,str(ROOT/'scripts/audit_https_funnel.py'),'--run',str(run),
                    '--prepared',verified['input'],'--memory-log',env['AML_MEMORY_DEBUG_LOG'],'--search-log',env['AML_SEARCH_DEBUG_LOG']],cwd=ROOT)
                manifest['audit_exit_code']=audit.returncode;save()
                if audit.returncode:return audit.returncode
            return client.returncode
    finally:
        if server is not None:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            manifest['server_exit_code']=server.returncode;save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('failed-run','run-dir','plan'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,default=18443);p.add_argument('--inspect',action='store_true')
    p.add_argument('--allow-interrupted',action='store_true',
                   help='Recover missing exit records only after confirming original processes are absent')
    sys.exit(main(p.parse_args()))
