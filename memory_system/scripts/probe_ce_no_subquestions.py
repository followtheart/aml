"""Paired CE question plus original option premises, without planner subquestions.

Candidate documents, retrieval queries and fine/main policy stay fixed. Changed
queries get fresh paired CE calls; unchanged queries explicitly reuse historical
scores. No provider call receives gold. This is not HTTPS or reader accuracy.
"""
import argparse
import asyncio
import copy
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config,rerank_query
import probe_full_option_ce as probe


def queries_for(t):
    req=SimpleNamespace(query=t['query'],options=t['options'])
    plan=copy.deepcopy(t['plan']);plan['_query_specs']=copy.deepcopy(t['query_specs'])
    with patch.object(config,'CE_FULL_OPTION_TEXT',False):
        original=rerank_query.build(req,plan)
        candidate=rerank_query.build(req,dict(plan,_query_specs=[]))
    if original!=t['cascade']['cross_encoder']['query']:
        raise ValueError('Historical CE query mismatch')
    if not candidate.startswith(req.query) or len(candidate.encode())>len(original.encode()):
        raise ValueError('Question integrity or query budget changed')
    return dict(control=original,candidate=candidate)


def main(args):
    if args.audit:
        from audit_full_option_ce import audit
        plan=json.loads((args.output/'plan.json').read_text())
        if plan['query_policy']!='no_planner_subquestions':raise ValueError('Wrong experiment')
        with patch('socket.socket.connect',side_effect=AssertionError('Offline audit')):
            result=audit(args.output,args.gold_csv,query_builder=queries_for)
        result['limitation']=__doc__
        result['inputs'][str(Path(__file__).resolve())]=probe.sha(Path(__file__))
        with (args.output/'audit.json').open('x') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
        print(json.dumps(result,ensure_ascii=False))
        return
    if not args.run:raise ValueError('--run is required for evaluation')
    args.query_builder=queries_for
    args.additional_scripts=[Path(__file__),Path(__file__).with_name('audit_full_option_ce.py')]
    args.query_policy='no_planner_subquestions';args.limitation=__doc__
    asyncio.run(probe.main(args))


if __name__=='__main__':
    logger=logging.getLogger('aml.metrics');logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler());logger.propagate=False
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--audit',action='store_true')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
