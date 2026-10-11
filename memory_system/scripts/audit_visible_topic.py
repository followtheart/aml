"""Re-run complete preflight and all live branches using only recorded responses."""
import argparse,asyncio,contextlib,copy,io,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from replay_visible_topic_v2 import main as replay,llm,sha
from replay_query_context import recorded_response

def canonical_plan(plan):
    result=copy.deepcopy(plan)
    for pair in result['comparisons'].values():
        for arm in pair.values():
            reservations=arm['reservations']
            if len({r['candidate_id'] for r in reservations})!=len(reservations):raise ValueError('Duplicate reservation')
            for r in reservations:
                r['requirement_ids']=sorted(r['requirement_ids']);r['reasons']=sorted(r['reasons'])
            arm['reservations']=sorted(reservations,key=lambda r:r['candidate_id'])
    return result

async def audit(path):
    r=json.loads(path.read_text());pending=[copy.deepcopy(c) for row in r['rows'] for c in row['live_calls']];total=len(pending);errors=[]
    async def frozen(prompt,*pos,**kw):
        if not pending:errors.append('extra');raise ValueError('Network forbidden')
        c=pending.pop(0)
        if (kw['stage'],sha(prompt))!=(c['stage'],c['prompt_sha256']):errors.append('mismatch');raise ValueError('Changed prompt')
        return recorded_response(c)
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'audit.json'
        with patch.object(llm,'complete_json',frozen),contextlib.redirect_stdout(io.StringIO()):
            await replay(SimpleNamespace(run=Path(r['plan']['run']),prepared=Path(r['plan']['prepared']),output=out))
        again=json.loads(out.read_text())
    if errors or pending:raise ValueError('Provider call sequence mismatch')
    if canonical_plan(r['plan'])!=canonical_plan(again['plan']):raise ValueError('Population/design/input changed')
    fields=['qa_id','arm','reused','prediction','score','packet_hash']
    if [{k:row[k] for k in fields} for row in r['rows']]!=[{k:row[k] for k in fields} for row in again['rows']]:raise ValueError('Outcome changed')
    if r['scores']!=again['scores'] or total!=r['live_calls']:raise ValueError('Totals changed')
    original={row['qa_id']:row for row in r['rows'] if row['arm']=='original'}
    changes=[dict(qa_id=row['qa_id'],original=original[row['qa_id']]['prediction'],visible=row['prediction'],score_before=original[row['qa_id']]['score'],score_after=row['score']) for row in r['rows'] if row['arm']=='visible' and row['prediction']!=original[row['qa_id']]['prediction']]
    return dict(status='passed',questions=len(original),affected=r['plan']['affected'],baseline_score=r['plan']['baseline_score'],scores=r['scores'],live_calls=total,changes=changes,provider_errors=[c.get('error_type') for row in r['rows'] for c in row['live_calls'] if c['status']=='error'],caveat=r['plan']['caveat'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=asyncio.run(audit(a.replay));a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='caveat'}))
