import json,subprocess,sys
from pathlib import Path
jobs=json.loads(Path('runs/https-cycle-20261011-constraint-term-forms-jobs.json').read_text())
for j in jobs:
 if j['name'] in ['cohort27','cohort28']:continue
 prefix='runs/https-cycle-20261011-constraint-term-forms-'+j['name']
 output=prefix+'-reader.json'
 with open(prefix+'-reader.log','x') as f:
  subprocess.run([sys.executable,'-u','scripts/replay_constraint_term_forms.py','--run',j['run'],'--prepared',j['prepared'],'--output',output],stdout=f,stderr=subprocess.STDOUT,check=True)
 with open(prefix+'-audit.log','x') as f:
  subprocess.run([sys.executable,'scripts/audit_constraint_term_forms.py','--replay',output,'--output',prefix+'-audit.json'],stdout=f,stderr=subprocess.STDOUT,check=True)
 a=json.loads(Path(prefix+'-audit.json').read_text());print(json.dumps(dict(cohort=j['name'],status=a['status'],scores=a['scores'],calls=a['live_calls'])),flush=True)
