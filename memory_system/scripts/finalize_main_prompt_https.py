"""Finish offline audits after the original HTTPS runners exit; never restart them."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def runner_live(pid,root):
    process=Path('/proc')/str(pid)
    try:
        args=[value.decode() for value in (process/'cmdline').read_bytes().split(b'\0') if value]
        if '--output' not in args:
            return False
        output=Path(args[args.index('--output')+1])
        if not output.is_absolute():output=(process/'cwd').resolve()/output
        return output.resolve()==root and any(a.endswith('/evaluate_frozen_https.py') for a in args)
    except FileNotFoundError:
        return False
    except (OSError,UnicodeDecodeError):
        return None  # A transient observation error is not a terminal state.


def main(args):
    project=Path(__file__).resolve().parents[1]
    pending={};failed=False
    for name,pid in zip(args.runs,args.runner_pids):
        root=name.resolve()
        plan=json.loads((root/'plan.json').read_text())
        if plan['toggle']!='RERANK_MAX_PROMPT_BYTES':raise ValueError('Wrong experiment')
        state_path=root/'finalization-state.json'
        if state_path.exists():
            if not args.resume:raise FileExistsError(state_path)
            state=json.loads(state_path.read_text())
            if state['status']!='waiting_for_original_runner' or state['steps']:
                raise ValueError('Only an untouched waiting audit queue may resume')
            if state['runner_pid']!=pid:raise ValueError('Original runner PID changed')
            state['previous_supervisor_pid']=state['supervisor_pid']
            state['supervisor_pid']=__import__('os').getpid()
            write(state_path,state);pending[root]=state
            continue
        support=root/'audit-support';support.mkdir(exist_ok=True)
        audit=project/'scripts/audit_main_prompt_https.py'
        destination=support/audit.name
        shutil.copyfile(audit,destination)
        state=dict(status='waiting_for_original_runner',runner_pid=pid,supervisor_pid=__import__('os').getpid(),
            admission_auditor_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),steps=[])
        write(state_path,state);pending[root]=state
    while pending:
        for root,state in list(pending.items()):
            marker=root/'exit-code';state_path=root/'finalization-state.json'
            if not marker.exists():
                if runner_live(state['runner_pid'],root) is False and not marker.exists():
                    state['status']='failed_runner_exited_without_terminal_record'
                    write(state_path,state);pending.pop(root);failed=True
                continue
            state.update(status='auditing',experiment_exit_code=marker.read_text().strip())
            write(state_path,state)
            # The scripts derive their reference-data path from their location.
            # Run byte-identical working copies, where the dataset exists; do
            # not add reference labels to the running service's source snapshot.
            plan=json.loads((root/'plan.json').read_text())
            for name,expected in plan['source_files'].items():
                if name.startswith(('memory_system/scripts/','memory_system/app/','prompts/')):
                    if hashlib.sha256((project.parent/name).read_bytes()).hexdigest()!=expected:
                        raise ValueError('Audit dependency differs from frozen code: '+name)
            steps=[('funnel',project/'scripts/analyze_overflow_https.py',['--run',str(root),'--include-reader-errors']),
                   ('runtime',project/'scripts/audit_frozen_runtime.py',['--run',str(root),'--include-reader-errors']),
                   ('admission',root/'audit-support/audit_main_prompt_https.py',
                    ['--run',str(root),'--output',str(root/'main-admission-audit.json'),
                     '--gold-csv',str(project/'data/raw/personamem-v2.csv')])]
            for name,script,options in steps:
                state['current_step']=name;write(state_path,state)
                with (root/f'final-{name}-audit.log').open('x') as log:
                    result=subprocess.run([sys.executable,str(script),*options],cwd=project,stdout=log,stderr=subprocess.STDOUT)
                state['steps'].append(dict(name=name,exit_code=result.returncode,
                    script_sha256=hashlib.sha256(script.read_bytes()).hexdigest()))
                write(state_path,state)
            state['status']='complete' if all(s['exit_code']==0 for s in state['steps']) else 'audit_failed'
            state.pop('current_step',None);write(state_path,state)
            failed=failed or state['status']!='complete';pending.pop(root)
            print(json.dumps(dict(run=str(root),**state)),flush=True)
        if pending:time.sleep(15)
    return int(failed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--runner-pids',nargs='+',type=int,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if len(args.runs)!=len(args.runner_pids):parser.error('Each run needs its original runner PID')
    sys.exit(main(args))
