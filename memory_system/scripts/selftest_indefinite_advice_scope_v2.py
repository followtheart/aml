import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from indefinite_advice_scope_v2 import supports
old=Path(__file__).with_name('selftest_indefinite_advice_scope.py').read_text().replace('from indefinite_advice_scope import supports','from indefinite_advice_scope_v2 import supports')
exec(compile(old,'original_controls','exec'))
rows=json.loads(Path('runs/https-cycle-20261011-indefinite-advice-context-probe.json').read_text())['rows']
# A story-writing frame is intentionally out of scope, although not a fact.
for r in rows:
 expected=r['expected'] and r['label']!='task'
 assert supports(r['claim'],r['option'])==expected,r['label']
print(json.dumps(dict(status='passed',additional_contexts=len(rows),scope='Direct advice or coordinated example list only')))
