"""Retry failed reader invocations on their exact completed HTTPS packets.

The original run is immutable. Successful result lines and all Search traces
are copied unchanged. Only terminal answer errors are replayed, once, with the
same frozen reader/model/settings. This is a disclosed recovery, not a new
HTTPS Search or an original-concurrency timing measurement.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


async def main(args):
    source=args.run.resolve();out=args.output.resolve()
    if not (source/'exit-code').exists():
        raise ValueError('Original runner is not terminal')
    if out.exists():
        raise FileExistsError(out)
    plan=json.loads((source/'plan.json').read_text())
    fixture=json.loads((source/'fixture.json').read_text())
    expected={(r['conversation_id'],r['qa_id']):r for r in fixture['requests']}
    jobs={};original={};manifests={}
    for name,key in (('fixture.json','fixture_sha256'),('frozen-memory.db','database_sha256')):
        if digest(source/name)!=plan[key]:raise ValueError('Frozen input changed')
    for name,value in plan['source_files'].items():
        if digest(source/'source'/name)!=value:raise ValueError('Frozen code changed')
    for arm in ('control','treatment'):
        path=source/arm;manifest=manifests[arm]=json.loads((path/'isolated-manifest.json').read_text())
        if 'server_exit_code' not in manifest or manifest.get('audit_exit_code')!=0:
            raise ValueError('Require terminal server and successful full Search audit')
        proc=Path('/proc')/str(manifest['server_pid'])/'cmdline'
        if proc.exists() and str(source).encode() in proc.read_bytes():raise ValueError('Original server is still running')
        rows=records(path/'completed/data/results/personamem-v2-32k.jsonl')
        traces={t['search_id']:t for t in records(path/'server-search-debug.jsonl')}
        if len(rows)!=plan['questions'] or len(traces)!=len(rows):raise ValueError('Incomplete Search cohort')
        failed=[]
        for index,row in enumerate(rows):
            if 'error_stage' not in row:continue
            if row['error_stage']!='answer' or row.get('error_type')!='LLMError' or row.get('search_id') not in traces:
                raise ValueError('Only terminal reader LLM errors can be recovered')
            failed.append((index,row,traces[row['search_id']]))
        jobs[arm]=failed
        original[arm]=dict(results=digest(path/'completed/data/results/personamem-v2-32k.jsonl'),
                           searches=digest(path/'server-search-debug.jsonl'))
    if not any(jobs.values()):raise ValueError('No failed reader to recover')
    shutil.copytree(source,out,symlinks=True)
    # Remove the copied terminal marker until recovery and audits finish.
    (out/'exit-code').unlink()
    env_file=Path(__file__).resolve().parents[1]/'.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line:continue
            key,value=line.split('=',1)
            os.environ.setdefault(key.strip(),value.strip().strip('"').strip("'"))
    project=out/'source/memory_system'
    sys.path.insert(0,str(project));sys.path.insert(0,str(project/'scripts'))
    from app import config,eval_scoring,run_metadata,evidence_packet
    from evaluation_model_guard import check_runtime
    for key,value in fixture['settings'].items():setattr(config,key,value)
    setattr(config,plan['toggle'],False)
    config.EVAL_CHOICE_POLICY='local';check_runtime();run_metadata.versions.cache_clear()
    if run_metadata.versions()!=plan['reader_versions']:raise ValueError('Reader identity changed')
    recovery=dict(original_run=str(source),original_hashes=original,attempts=[],
        helper_sha256=digest(Path(__file__)),limitation=__doc__)
    write(out/'reader-recovery.json',recovery)
    plan['reader_recovery']='reader-recovery.json';write(out/'plan.json',plan)
    for arm,failed in jobs.items():
        if not failed:continue
        path=out/arm;manifest=dict(manifests[arm]);prepared=Path(manifest['data'])
        if digest(prepared)!=plan['data_sha256']:raise ValueError('Prepared questions changed')
        conversations={r['conversation_id']:r for r in records(prepared)}
        result=path/'completed/data/results/personamem-v2-32k.jsonl'
        lines=result.read_text().splitlines(keepends=True)
        changed=set()
        for index,old,t in failed:
            item=expected[old['conversation_id'],old['qa_id']]
            qa=next(q for q in conversations[old['conversation_id']]['qa'] if q['id']==old['qa_id'])
            if qa['question']!=t['query'] or qa['options']!=t['options'] or t['query']!=item['request']['query']:
                raise ValueError('Question identity differs')
            if evidence_packet.digest(t['returned'])!=old['packet_hash']:raise ValueError('Packet changed')
            start=time.monotonic();check_runtime()
            pred,score,diagnostics=await eval_scoring.evaluate(qa,t['returned'])
            seconds=time.monotonic()-start
            recovery['attempts'].append(dict(arm=arm,qa_id=old['qa_id'],original_result=old,
                search_id=old['search_id'],packet_hash=old['packet_hash'],reader_seconds=seconds,
                score=score,error=diagnostics.get('error_stage')))
            write(out/'reader-recovery.json',recovery)
            if diagnostics.get('error_stage'):raise RuntimeError('Reader recovery failed; no further automatic retries')
            # Recreate only reader fields; retain exact Search accounting/identity.
            row={k:v for k,v in old.items() if not k.startswith('answer_') and k not in (
                'prediction','score','error_stage','error_type','error_detail','total_seconds')}
            row.update(prediction=pred,score=score,**diagnostics,reader_recovered=True,
                       recovery_reader_seconds=seconds,total_seconds=old['search_seconds']+seconds)
            lines[index]=json.dumps(row,ensure_ascii=False)+'\n';changed.add(index)
            result.write_text(''.join(lines))
            print(json.dumps(dict(arm=arm,qa_id=old['qa_id'],score=score,reader_recovered=True)),flush=True)
        original_lines=(source/arm/'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines(keepends=True)
        if any(lines[i]!=line for i,line in enumerate(original_lines) if i not in changed):
            raise ValueError('Successful result changed')
        rows=records(result)
        summary=json.loads((path/'summary.json').read_text())
        summary.update(score=sum(r['score'] for r in rows),errors=sum(bool(r.get('error_stage')) for r in rows),
                       original_reader_errors=len(failed),reader_recovered_questions=[r['qa_id'] for _,r,_ in failed])
        write(path/'summary.json',summary)
        manifest.update(client_exit_code=0,reader_recovery_original_manifest=str(source/arm/'isolated-manifest.json'))
        (path/'exit-code').write_text('0\n')
        with (path/'reader-recovery-audit.log').open('w') as log:
            audit=subprocess.run([sys.executable,str(project/'scripts/audit_https_stages.py'),'--run',str(path),
                '--prepared',str(prepared),'--gold-csv',str(args.gold_csv.resolve())],stdout=log,stderr=log)
        manifest['audit_exit_code']=audit.returncode;write(path/'isolated-manifest.json',manifest)
        if audit.returncode:raise RuntimeError('Recovered stage audit failed')
    for arm in jobs:
        if (digest(source/arm/'completed/data/results/personamem-v2-32k.jsonl')!=original[arm]['results'] or
                digest(source/arm/'server-search-debug.jsonl')!=original[arm]['searches'] or
                digest(out/arm/'server-search-debug.jsonl')!=original[arm]['searches']):
            raise ValueError('Original run or frozen Search trace changed')
    recovery['original_unchanged']=True
    recovery['search_traces_unchanged']=True
    recovery['final_result_hashes']={a:digest(out/a/'completed/data/results/personamem-v2-32k.jsonl') for a in jobs}
    write(out/'reader-recovery.json',recovery);(out/'exit-code').write_text('0\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    asyncio.run(main(p.parse_args()))
