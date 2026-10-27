"""Reproduce the independent marginal literal-query proposal with production code offline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice,config,conversation_context,evidence_packet,search_pipeline,marginal_query_context
from evaluation_model_guard import check_runtime,check_versions
from probe_marginal_query_context import prepare
from probe_source_rank_residual import records


def main(args):
    check_runtime()
    if args.output.exists():raise FileExistsError(args.output)
    diagnostic=json.loads(args.diagnostic.read_text())
    helper=Path(__file__).with_name('probe_marginal_query_context.py')
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
                proposed=prepare(t)['packets']['literal_marginal']
                pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
                    if c.get('view_status')!='stale' and '_packet_item' in c}
                ctx=t['cascade']['conversation_context']
                base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
                base_manifest=copy.deepcopy(t['coverage_manifest']);base_manifest.pop('conversation_context')
                base_manifest.update(included_ids=ctx['base_packet_ids'],token_upper_bound=ctx['base_token_upper_bound'],
                    evidence_count=base_manifest['evidence_count']-len(ctx['appended_ids']))
                fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids'] if mid in pool and mid not in t['selection']['selected_ids']]
                prefix,prefix_digest,prefix_manifest,_=conversation_context.attach(base,ctx['base_packet_hash'],base_manifest,fine,
                    user_id=t['user_id'],top_k=t['top_k'],enabled=False,reserve_enabled=True,reserve_extended=True,
                    lexical_enabled=True,query=t['query'],options=t['options'])
                champions,nomination=marginal_query_context.nominate(pool,prefix,routes=t['routes'],query_specs=t['query_specs'],
                    query=t['query'],options=t['options'],user_id=t['user_id'],coarse_ids=t['cascade']['coarse']['selected_ids'],
                    selected_ids=t['selection']['selected_ids'])
                packet,digest,manifest=marginal_query_context.attach(prefix,prefix_digest,prefix_manifest,
                    champions,nomination,user_id=t['user_id'],top_k=t['top_k'])
                assert packet==[dict(p,packet_hash=evidence_packet.digest(proposed)) for p in proposed]
                extra=manifest['conversation_context']['marginal_query']
                assert extra['appended_ids']==old['proposals']['literal_marginal']['appended_ids']
                by_id={v['candidate_id']:v for v in old['inventory']}
                for v in nomination['nominations']:
                    if v['candidate_id'] in by_id:
                        independent=by_id[v['candidate_id']]
                        assert v['score']==independent['scores']['literal_marginal']
                        assert v['gains']==independent['gains']['literal_marginal']
                catalogs=answer_choice.build_catalog(packet)
                assert sum(len(c['text'].encode()) for cat in catalogs for c in cat.values())==old['catalog_bytes']['literal_marginal']
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
