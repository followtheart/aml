"""Replay the independent diagnostic through both production witness paths."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import choice_witness,config,llm
from probe_keyed_witness import decode_keyed


async def main(path,output):
    plan=json.loads(path.with_suffix('.plan.json').read_text());result=json.loads(path.read_text())
    jobs={j['qa_id']:j for j in plan['jobs']};rows=[]
    run=Path(plan['run'])
    source_results=[json.loads(l) for l in (run/'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines()]
    searches={r['qa_id']:r['search_id'] for r in source_results}
    traces={r['search_id']:r for r in map(json.loads,(run/'server-search-debug.jsonl').read_text().splitlines())}
    for row in result['rows']:
        job=jobs[row['qa_id']];enabled=row['arm']=='treatment';calls=[];diagnostics={}
        async def model(prompt,**kwargs):
            calls.append(dict(prompt=prompt,**kwargs));return copy.deepcopy(row.get('response',{}))
        with patch.multiple(config,FAKE=False,CHOICE_SEMANTIC_WITNESSES=True,
                CHOICE_WITNESS_SPAN_REFS=False,CHOICE_WITNESS_KEYED_OUTPUT=enabled),\
                patch.object(llm,'complete_json',model),patch('socket.socket.connect',side_effect=AssertionError('Offline')):
            # The exact question resides in the unchanged captured prompt; source
            # trace identity maps it without parsing untrusted quoted content.
            trace=traces[searches[job['qa_id']]]
            got=await choice_witness.select(trace['query'],job['options'],job['sources'],diagnostics)
        assert len(calls)==1
        request=calls[0];request.pop('stage');assert request==job['requests'][row['arm']]
        validation={}
        try:
            raw=(decode_keyed(row['response'],job['options'],job['sources'],validation) if enabled
                 else choice_witness.validate(row['response'],job['options'],job['sources'],validation))
            expected=choice_witness.supplement_first_party(job['options'],job['sources'],raw)
        except (ValueError,KeyError):expected=choice_witness.build(job['options'],job['sources'])
        assert got==expected
        rows.append(dict(qa_id=job['qa_id'],arm=row['arm'],status=diagnostics['answer_witness_retrieval']['status']))
    if len(rows)!=112:raise ValueError('Expected complete paired diagnostic')
    with output.open('x') as f:
        json.dump(dict(requests_verified=len(rows),responses_verified=len(rows),calls=0,rows=rows,
            diagnostic_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),f,indent=2);f.write('\n')
    print(json.dumps(dict(requests_verified=len(rows),responses_verified=len(rows),calls=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();asyncio.run(main(a.diagnostic,a.output))
