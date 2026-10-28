"""Check whether source-valid quotations lose their surrounding task context."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac
from repeated_topic_interest_v2 import supports
CLAIM='your interest in Scandinavian crime novels'
QUESTIONS=['What gives Scandinavian crime novels their atmosphere?','Can you give an example of a Scandinavian crime novel?']
CASES=[('direct','',True),('translation','Translate this question into Korean; do not answer it: ',False),('exam_copy','This is an exam question that I need copied verbatim: ',False),('zero_interest','I have zero interest in this topic. ',False),('denial','I am not interested in this topic. ',False),('received','My colleague wrote: ',False)]
rows=[]
for name,prefix,expected in CASES:
 refs=[];validation=[]
 for i,q in enumerate(QUESTIONS):
  sid='s'+str(i);context=prefix+q;source=dict(id=sid,role='user',text=context)
  checked=ac._check_citation(ac.Citation(source_id=sid,quote=q,basis='topic_interest',subject='current_user'),CLAIM,{sid:source},'interest');validation.append(dict(valid=checked['valid'],anchor=checked['anchor'],errors=checked['validation_errors']))
  if checked['valid'] and checked['anchor']:refs.append(dict(id=sid,role='user',quote=q,context=context))
 actual=supports(dict(check_type='premise',premise_type='interest',claim=CLAIM,sources=refs))
 rows.append(dict(case=name,expected=expected,actual=actual,source_validation=validation,contexts=[prefix+q for q in QUESTIONS]))
report=dict(cases=rows,false_positives=[r['case'] for r in rows if r['actual'] and not r['expected']],false_negatives=[r['case'] for r in rows if r['expected'] and not r['actual']],caveat='Run through existing citation validation before candidate support. Synthetic attribution controls, not benchmark accuracy.')
Path('runs/https-cycle-20261011-repeated-interest-context-v2-probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
