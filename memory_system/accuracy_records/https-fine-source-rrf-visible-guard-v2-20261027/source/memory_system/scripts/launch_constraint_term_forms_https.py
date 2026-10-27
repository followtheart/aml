"""Launch isolated fresh paired HTTPS runs after runtime-equivalence validation."""
import copy,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();base=root/'runs';prefix='https-cycle-20261011-constraint-term-forms-https'
equivalence=json.loads((base/'https-cycle-20261011-constraint-term-forms-runtime-equivalence.json').read_text());assert equivalence['status']=='passed' and not equivalence['default_enabled']
original=json.loads((base/'https-cycle-20261011-cohort28-control-plan.json').read_text())
assert hashlib.sha256((root/original['input']).read_bytes()).hexdigest()==original['sha256']
# Each arm gets a separate frozen source snapshot, database and loopback port.
for arm,enabled,port in [('control',0,18444),('candidate',1,18445)]:
 plan=copy.deepcopy(original);plan['configuration']['AML_CHOICE_CONSTRAINT_TERM_FORMS']=enabled
 plan['selection']='Complete development cohort28, paired fresh HTTPS comparison; same current code and input, only constraint term forms flag differs. Fresh ingestion/retrieval stochasticity remains a confound.'
 plan_path=base/(prefix+'-'+arm+'-plan.json')
 with plan_path.open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2);f.write('\n')
 env=os.environ.copy();env.update({k:str(v) for k,v in plan['configuration'].items()});env.update(AML_EVAL_DATA=str(root/plan['input']),AML_EVAL_CONVS='1',AML_EVAL_LIMIT=str(plan['expected_qa']),AML_EVAL_CHUNK_MESSAGES=str(plan['chunk_messages']),AML_EVAL_CHUNK_WORDS=str(plan['chunk_words']))
 run='runs/'+prefix+'-'+arm;log=run+'.log'
 with (root/log).open('x') as f:
  process=subprocess.Popen(['.venv/bin/python','-u','scripts/run_isolated_https_experiment.py','--run-dir',run,'--port',str(port),'--','--no-progress'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 launch=dict(pid=process.pid,run=run,plan=str(plan_path.relative_to(root)),log=log,expected_qa=19,expected_messages=293,expected_chunks=16,status='running',arm=arm,port=port)
 (base/(prefix+'-'+arm+'-launch.json')).write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
