"""Reproduce the independent source-first proposal with production code offline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice,config,evidence_packet,search_pipeline,source_first_residual
from evaluation_model_guard import check_runtime,check_versions
from probe_source_rank_residual import prepare,records


def main(args):
    check_runtime()
    if args.output.exists():raise FileExistsError(args.output)
    diagnostic=json.loads(args.diagnostic.read_text())
    helper=Path(__file__).with_name('probe_source_rank_residual.py')
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=diagnostic['script_sha256']:
        raise ValueError('Independent diagnostic code changed')
    expected={(r['run'],r['qa_id']):r for r in diagnostic['rows']}
    rows=[]
    for spec in diagnostic['inputs']:
        run=Path(spec['run']);trace_path=run/'server-search-debug.jsonl'
        if hashlib.sha256(trace_path.read_bytes()).hexdigest()!=spec['trace_sha256']:
            raise ValueError('Source trace changed')
        rs={r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        for t in records(trace_path):
            qa=rs[t['search_id']]['qa_id'];old=expected[str(run),qa]
            check_versions(t['versions'],'service trace')
            with patch.multiple(config,**t['versions']['settings']),patch('socket.socket.connect',side_effect=AssertionError('Offline')):
                proposed=prepare(t)['packets']['question_first']
                pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
                    if c.get('view_status')!='stale' and '_packet_item' in c}
                champions,nomination=source_first_residual.nominate(pool,routes=t['routes'],query_specs=t['query_specs'],
                    query=t['query'],coarse_ids=t['cascade']['coarse']['selected_ids'],
                    fine_ids=t['cascade']['fine']['selected_ids'],selected_ids=t['selection']['selected_ids'])
                packet,digest,manifest=source_first_residual.attach(t['returned'],t['packet_hash'],t['coverage_manifest'],
                    champions,nomination,user_id=t['user_id'],top_k=t['top_k'])
                assert packet==[dict(p,packet_hash=evidence_packet.digest(proposed)) for p in proposed]
                extra=manifest['conversation_context']['source_first']
                assert extra['appended_ids']==old['proposals']['question_first']['appended_ids']
                catalogs=answer_choice.build_catalog(packet)
                assert sum(len(c['text'].encode()) for cat in catalogs for c in cat.values())==old['catalog_bytes']['question_first']
                assert evidence_packet.digest(packet)==digest
            rows.append(dict(run=str(run),qa_id=qa,appended_ids=extra['appended_ids']))
    assert len(rows)==len(expected)
    result=dict(questions=len(rows),actual_baseline_packets_verified=len(rows),production_packets_verified=len(rows),
        catalogs_verified=len(rows),new_provider_calls=0,rows=rows,
        diagnostic_sha256=hashlib.sha256(args.diagnostic.read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
