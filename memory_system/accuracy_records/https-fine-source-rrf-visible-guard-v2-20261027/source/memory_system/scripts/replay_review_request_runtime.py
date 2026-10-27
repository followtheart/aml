"""Paired full-reader evaluation of a bounded runtime reader intervention."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config, eval_scoring, evidence_packet, run_metadata
from replay_query_context import records
from replay_source_vector_route import source_route


async def main(args):
    if config.FAKE:raise ValueError('Real provider required')
    setting=getattr(args,'setting','CHOICE_REVIEW_REQUEST_TERMS')
    results=records(args.run/'completed/data/results/personamem-v2-32k.jsonl')
    traces={r['search_id']:r for r in records(args.run/'server-search-debug.jsonl')}
    questions={(d['conversation_id'],q['id']):q for d in records(args.prepared) for q in d['qa']}
    if setting == 'CHOICE_DIRECT_CATALOG_READER':
        if (args.run/'exit-code').read_text().strip() != '0':
            raise ValueError('Direct reader comparison requires successful complete archive')
        identities = [(r['conversation_id'], r['qa_id']) for r in results]
        if len(identities) != len(set(identities)) or set(identities) != set(questions):
            raise ValueError('Require complete prepared cohort without duplicate questions')
    probe=json.loads(args.probe.read_text()) if args.probe else None
    if probe and hashlib.sha256(json.dumps([s['content'] for s in probe['sources']],ensure_ascii=False).encode()).hexdigest()!=probe['source_text_sha256']:
        raise ValueError('Changed source universe')
    routes={q['qa_id']:q for q in probe['questions']} if probe else {}
    if probe and set(routes)!={r['qa_id'] for r in results}:raise ValueError('Require complete cohort, no score selection')
    plan=dict(run=str(args.run),probe=str(args.probe) if args.probe else None,questions=[r['qa_id'] for r in results],
        setting=setting,query_context=args.query_context,repeat=args.repeat,
        prepared=str(args.prepared),prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=run_metadata.versions(),caveat='Both arms have exactly the same packet and context-focus flag. All reader calls live; retrieval frozen.')
    args.output.with_suffix('.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with args.output.open('x') as out:
        for index,r in enumerate(results):
            trace=traces[r['search_id']];packet=trace['returned']
            if evidence_packet.digest(packet)!=r['packet_hash']:raise ValueError('Original packet mismatch')
            if probe:
                q=routes[r['qa_id']]
                if q['query_specs']!=trace['query_specs'] or trace['user_id']!=probe['user_id']:raise ValueError('Route identity mismatch')
                packet,_=source_route(probe,q,packet,trace['evidence_token_budget'])
            digest=evidence_packet.digest(packet);qa=questions[r['conversation_id'],r['qa_id']]
            for repetition in range(args.repeat):
                for enabled in ([False,True] if (index+repetition)%2==0 else [True,False]):
                    with patch.multiple(config,**{setting:enabled,'CHOICE_QUERY_CONTEXT':args.query_context}):
                        prediction,score,diagnostics=await eval_scoring.evaluate(qa,copy.deepcopy(packet))
                        metadata=copy.deepcopy(run_metadata.versions())
                        # versions() caches the startup settings alongside code hashes.
                        # Record actual settings while the intervention is active.
                        metadata['settings']={key:getattr(config,key) for key in metadata['settings']}
                    row=dict(qa_id=r['qa_id'],conversation_id=r['conversation_id'],enabled=enabled,repetition=repetition+1,
                        prediction=prediction,score=score,diagnostics=diagnostics,versions=metadata,
                        packet_hash=digest,original_packet_hash=r['packet_hash'],query_context=args.query_context,
                        caveat='All reader calls live; exact same frozen packet in both arms. Source/query generation omitted; not fresh HTTPS.')
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                    print(json.dumps({k:row[k] for k in ['qa_id','enabled','repetition','prediction','score']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--probe',type=Path)
    p.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--query-context',action='store_true');p.add_argument('--repeat',type=int,choices=(1,2,3),default=1)
    p.add_argument('--setting',choices=('CHOICE_REVIEW_REQUEST_TERMS','CHOICE_CONDITIONAL_LIST_SCOPE','CHOICE_TEMPLATE_PATH_SCOPE','CHOICE_EXISTING_STATE_GUARD','CHOICE_REPAIR_VALID_CLAIMS', 'CHOICE_CITED_SCOPE', 'CHOICE_RECOVERY_BEFORE_SCOPE', 'CHOICE_SOURCE_NOTES_SELECTION', 'CHOICE_DIRECT_CATALOG_READER'),default='CHOICE_REVIEW_REQUEST_TERMS')
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
