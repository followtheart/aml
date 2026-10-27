"""Verify paired HTTPS provenance, terminal success, exact ingestion and complete scores."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from audit_https_ingestion import audit as ingestion_audit
from compare_https_experiments import compare
FLAG='AML_CHOICE_CONSTRAINT_TERM_FORMS'
PREFIX=Path('runs/https-cycle-20261011-constraint-term-forms-https')
def audit(preflight=False):
 arms={};files=[];configs=[]
 for arm in ['control','candidate']:
  run=Path(str(PREFIX)+'-'+arm);plan_path=Path(str(run)+'-plan.json');plan=json.loads(plan_path.read_text());manifest=json.loads((run/'isolated-manifest.json').read_text());inp=Path(plan['input'])
  assert Path(manifest['data']).resolve()==inp.resolve() and hashlib.sha256(inp.read_bytes()).hexdigest()==plan['sha256'],'Input mismatch'
  assert all(hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==digest for name,digest in manifest['files'].items()),'Snapshot file changed'
  files.append(manifest['files']);configs.append(plan['configuration']);arms[arm]=dict(run=str(run),plan=str(plan_path),input_sha256=plan['sha256'],expected_qa=plan['expected_qa'],qa_ids=plan['qa_ids'])
  if not preflight:
   assert [manifest.get(k) for k in ['client_exit_code','audit_exit_code','server_exit_code']]==[0,0,-15],'Run not successfully terminal'
   assert (run/'exit-code').read_text().strip()=='0'
   ing=ingestion_audit(run,plan_path);assert ing['complete'] and ing['original_messages_exact'] and not ing['orphan_sources'];arms[arm]['ingestion']=ing
   rows=[json.loads(l) for l in (run/'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines() if l.strip()]
   assert [r['qa_id'] for r in rows]==plan['qa_ids'],'Question order/count mismatch'
   for r in rows:
    assert r['versions']['settings']['CHOICE_CONSTRAINT_TERM_FORMS']==(arm=='candidate'),'Reader flag mismatch'
    assert not r.get('fake') and not r.get('server_fake') and not r.get('error_stage'),'Invalid provider/result state'
   subprocess.run([sys.executable,'scripts/audit_https_funnel.py','--run',str(run),'--prepared',str(inp),'--memory-log',str(run/'server-memory-debug.jsonl'),'--search-log',str(run/'server-search-debug.jsonl')],stdout=subprocess.DEVNULL,check=True)
 assert files[0]==files[1],'Frozen runtime files differ'
 assert set(configs[0])==set(configs[1]) and [k for k in configs[0] if configs[0][k]!=configs[1][k]]==[FLAG]
 assert str(configs[0][FLAG])=='0' and str(configs[1][FLAG])=='1'
 assert arms['control']['input_sha256']==arms['candidate']['input_sha256'] and arms['control']['qa_ids']==arms['candidate']['qa_ids']
 report=dict(status='preflight_passed' if preflight else 'complete_audited',arms=arms,only_setting_difference=FLAG,same_frozen_code=True,caveat='Fresh ingestion and retrieval remain stochastic; source catalog changes must be reviewed. This does not isolate the flag effect like fixed-packet replay.')
 if not preflight:
  report['comparison']=compare(Path(arms['control']['run']),Path(arms['candidate']['run']));assert not report['comparison']['partial']
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.preflight);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=r['status'],same_frozen_code=r['same_frozen_code'],comparison={k:v for k,v in r.get('comparison',{}).items() if k not in ['pairs','control','treatment','caveat']})))
