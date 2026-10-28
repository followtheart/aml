"""Complete archived cohort constraint proposal differences; no semantic verdict."""
import argparse,json,sys,hashlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,config
from probe_question_source_filter import load_cohort
from constraint_term_forms import wrapper

def main(a):
 rows=[];original=ac._topic_terms
 for job in json.loads(a.jobs.read_text()):
  for r,q,p,s,c in load_cohort(Path(job['run']),Path(job['prepared'])):
   constraints={x['id']:x for x in c}
   with patch.multiple(config,**r['versions']['settings']):
    before=ac.constraint_pairs(q['options'],constraints)
    with patch.object(ac,'_topic_terms',wrapper(original)):after=ac.constraint_pairs(q['options'],constraints)
   bi={x['pair_id']:x for x in before};ai={x['pair_id']:x for x in after}
   rows.append(dict(cohort=job['name'],qa_id=r['qa_id'],added=[ai[k] for k in ai if k not in bi],removed=[bi[k] for k in bi if k not in ai]))
 result=dict(count=len(rows),changed=sum(bool(x['added'] or x['removed']) for x in rows),added=sum(len(x['added']) for x in rows),removed=sum(len(x['removed']) for x in rows),rows=rows,rule_sha256=hashlib.sha256(Path(__file__).with_name('constraint_term_forms.py').read_bytes()).hexdigest(),caveat='Exact full population packet/catalog input validation; proposal differences only, not semantic decisions or score gains.')
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','caveat']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
