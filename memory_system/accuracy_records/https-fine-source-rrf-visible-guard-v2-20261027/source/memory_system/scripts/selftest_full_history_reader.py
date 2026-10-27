"""Offline integration: raw-history preservation, failures, and tamper rejection."""
import asyncio,contextlib,copy,io,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import probe_full_history_reader as p
from summarize_full_history_reader import summarize

async def check():
    run=Path('runs/https-cycle-20261011-cohort24-control');prepared=Path('runs/https-cycle-20261011-cohort24-input.jsonl')
    originals=p.records(prepared);hist=p.histories(prepared)
    for conv in originals:
        flat=[m for session in conv['sessions'] for m in session]
        assert [(m['role'],m['content'],m.get('timestamp')) for m in flat]==[(m['role'],m['content'],m['timestamp']) for m in hist[conv['conversation_id']]]
    cohort=p.load_cohort(run,prepared)
    for r,q,packet,s,c in cohort:
        _,base=p.make_prompt(q,packet,s,c,[]);_,full=p.make_prompt(q,packet,s,c,hist[r['conversation_id']])
        b=json.loads(base);f=json.loads(full);assert b.pop('additional_original_history')==[];assert f.pop('additional_original_history')==hist[r['conversation_id']];assert b==f
    calls=0
    async def fake(prompt,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:raise TimeoutError('synthetic timeout')
        if calls==2:return {}
        return {'answer':kwargs['schema']['properties']['answer']['enum'][0]}
    old=p.llm.complete_json;oldfake=p.config.FAKE;p.llm.complete_json=fake;p.config.FAKE=False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'mock.jsonl'
            with contextlib.redirect_stdout(io.StringIO()):await p.main(SimpleNamespace(run=run,prepared=prepared,output=output))
            result=summarize(output);assert len(result['errors'])==2 and result['complete_pairs']==28 and calls==56
            rows=[json.loads(l) for l in output.read_text().splitlines()];plan=json.loads(output.with_suffix('.plan.json').read_text());rejected=[]
            for label in ['prompt','score','order','history','incomplete']:
                changed=copy.deepcopy(rows);changedplan=copy.deepcopy(plan)
                if label=='prompt':changed[2]['prompt_sha256']='bad'
                if label=='score':changed[2]['score']=1-changed[2]['score']
                if label=='order':changed[2],changed[3]=changed[3],changed[2]
                if label=='history':next(iter(changedplan['history'].values()))['sha256']='bad'
                if label=='incomplete':changed.pop()
                output.write_text(''.join(json.dumps(r)+'\n' for r in changed));output.with_suffix('.plan.json').write_text(json.dumps(changedplan))
                try:summarize(output)
                except ValueError:rejected.append(label)
                else:raise AssertionError(label)
            assert len(rejected)==5
            return dict(status='passed',full_population=28,raw_history_messages=230,mock_calls=calls,preserved_failure_rows=2,rejected_tampering=rejected)
    finally:p.llm.complete_json=old;p.config.FAKE=oldfake

if __name__=='__main__':print(json.dumps(asyncio.run(check())))
