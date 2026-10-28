"""Offline exact provider-prompt replay of complete paired ranking/reader study."""
import argparse,asyncio,contextlib,copy,hashlib,io,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import llm
from replay_ce_isolated_options import main as replay
from replay_query_context import recorded_response

def records(p):return [json.loads(l) for l in p.open() if l.strip()]
def outcome(row):
    rank=row['ranking'];cascade=rank['cascade']
    return {'qa_id':row['qa_id'],'arm':row['arm'],'prediction':row['prediction'],'score':row['score'],'packet_hash':rank['packet_hash'],'fine_ids':cascade['fine']['selected_ids'],'admitted_ids':cascade['listwise']['candidate_ids'],'selected_ids':cascade['listwise']['selected_ids'],'eligible':row['diagnostics'].get('answer_eligible_options')}
async def audit(path):
    plan=json.loads(path.with_suffix('.plan.json').read_text());probe=Path(plan['probe']);assert hashlib.sha256(probe.read_bytes()).hexdigest()==plan['probe_sha256']
    rows=records(path);expected=[(q,arm) for i,q in enumerate(plan['qa_ids']) for arm in (['control','candidate'] if i%2==0 else ['candidate','control'])]
    assert [(r['qa_id'],r['arm']) for r in rows]==expected,'Require complete ordered paired population'
    assert plan['frozen_packets_exact']==len(plan['qa_ids'])
    pending=[copy.deepcopy(c) for r in rows for c in [*r['ranking']['calls'],*r['diagnostics'].get('answer_calls',[])]];count=len(pending);violations=[]
    async def frozen(prompt,*pos,**kw):
        if not pending:violations.append('extra');raise ValueError('Unexpected provider call')
        c=pending.pop(0)
        if c['stage']!=kw['stage'] or c['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():violations.append(kw['stage']);raise ValueError('Provider prompt changed')
        return recorded_response(c)
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'replayed.jsonl'
        with patch.object(llm,'complete_json',frozen),contextlib.redirect_stdout(io.StringIO()):await replay(SimpleNamespace(probe=probe,output=out))
        again=records(out)
    assert not pending and not violations,'Provider trace mismatch'
    assert [outcome(r) for r in rows]==[outcome(r) for r in again],'Outcomes or packet selection changed'
    controls={r['qa_id']:r for r in rows if r['arm']=='control'};candidates={r['qa_id']:r for r in rows if r['arm']=='candidate'}
    return {'status':'passed','questions':len(controls),'provider_calls':count,'scores':{arm:sum(r['score'] for r in rows if r['arm']==arm) for arm in ['control','candidate']},'wins':[q for q in controls if candidates[q]['score']>controls[q]['score']],'losses':[q for q in controls if candidates[q]['score']<controls[q]['score']],'provider_errors':[{'qa_id':r['qa_id'],'arm':r['arm'],'stage':c['stage'],'error_type':c.get('error_type')} for r in rows for c in [*r['ranking']['calls'],*r['diagnostics'].get('answer_calls',[])] if 'response' not in c],'caveat':plan['caveat']+' Offline audit reproduces original packet preflight, every live downstream full prompt, final packet hashes, fine/listwise selections, eligibility and outcomes. This validates trace integrity, not semantic truth.'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=asyncio.run(audit(a.replay))
    with a.output.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k!='caveat'}))
