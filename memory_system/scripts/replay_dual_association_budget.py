"""Replay all four frozen arms, then the preregistered budget sensitivity.

No provider calls. Requires exact reproduction of the primary stage selections
before increasing only the diagnostic token allowance. Does not change the
primary gate, production configuration or any source/indexed content.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import config,store
import diagnose_dual_association as dual
from diagnose_association_views import cohort,identity,records,write


async def run(args):
    parent=args.input
    if not (parent/'complete.json').exists(): raise ValueError('Complete primary run required')
    manifest=json.loads((parent/'manifest.json').read_text())
    plan=json.loads((parent/'budget-sensitivity-plan.json').read_text())
    for name,sha in manifest['files'].items():
        if hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Primary code changed: '+name)
    for key,value in manifest['versions']['settings'].items(): setattr(config,key,value)
    base=config.SEARCH_MAX_TOKENS
    if base!=plan['primary_search_token_budget']: raise ValueError('Budget registration mismatch')
    args.baseline=Path(manifest['baseline'])
    jobs=cohort(args)
    rows={r['qa_id']:r for r in records(parent/'results.jsonl')}
    queries={r['qa_id']:r for r in records(parent/'queries.jsonl')}
    scores={r['qa_id']:r for r in records(parent/'ce-scores.jsonl')}
    if len(rows)!=len(jobs) or set(rows)!=set(queries) or set(rows)!=set(scores):
        raise ValueError('Incomplete cached inputs')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    write(args.output/'manifest.json',dict(primary=str(parent.resolve()),plan=plan,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),provider_calls=0))
    st=store.Store(str(parent/'memory.db'))
    original=identity(st);view_hash=dual.view_identity(st)
    try:
        with (args.output/'results.jsonl').open('x') as stream:
            for i,(result,trace) in enumerate(jobs):
                key=str(result['qa_id']);cached=queries[key];score=scores[key]
                base_vectors=np.asarray(cached['base_vectors'],dtype=np.float32)
                query_vectors=np.asarray(cached['query_vectors'],dtype=np.float32)
                values={d['text']:d['score'] for d in score['documents']}
                arms={}
                with st.snapshot(trace['user_id']) as snap:
                    for arm in dual.ARMS:
                        config.SEARCH_MAX_TOKENS=base
                        prepared=dual.build(snap,trace,base_vectors,query_vectors,arm)
                        cost=cached['preparation'] if dual.ARMS[arm][1] else {}
                        reproduced=await dual.rank(snap,prepared,score['query'],values,cost)
                        if reproduced['stages']!=rows[key]['arms'][arm]['stages']:
                            write(args.output/'validation-failure.json',dict(qa_id=key,arm=arm,reason='Primary stage mismatch'))
                            raise ValueError('Primary stage reproduction failed')
                        config.SEARCH_MAX_TOKENS=plan['sensitivity_search_token_budget']
                        prepared=dual.build(snap,trace,base_vectors,query_vectors,arm)
                        arms[arm]=await dual.rank(snap,prepared,score['query'],values,cost)
                row=dict(qa_id=key,conversation_id=result['conversation_id'],user_id=result['user_id'],
                    query_diagnostics=cached['diagnostics'],preparation=cached['preparation'],ce_errors=score['errors'],
                    arms=arms,primary_exactly_reproduced=True)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps(dict(completed=i+1,questions=len(jobs),qa_id=key,primary_exactly_reproduced=True)),flush=True)
        assert identity(st)==original and dual.view_identity(st)==view_hash
        write(args.output/'complete.json',dict(questions=len(jobs),original_state_unchanged=True,
            write_views_unchanged=True,provider_calls=0,primary_exactly_reproduced=True))
    finally:
        config.SEARCH_MAX_TOKENS=base
        st.conn.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    asyncio.run(run(args))


if __name__=='__main__': main()
