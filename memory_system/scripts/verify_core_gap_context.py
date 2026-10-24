"""Replay full production packet assembly against the fixed CE-core gap proposal."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice,config,evidence_packet,schemas,search_pipeline
from probe_core_gap_context import prepare


def records(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def replay(t,enabled):
    plan=copy.deepcopy(t['plan'])
    plan.update(_cascade=copy.deepcopy(t['cascade']),_selection=copy.deepcopy(t['selection']),
        _routes=copy.deepcopy(t['routes']),_query_specs=copy.deepcopy(t['query_specs']),
        _evidence_groups=copy.deepcopy(t['evidence_groups']),
        _coverage_requirements=copy.deepcopy(t['coverage_manifest']['coverage']['requirements']),
        _unit_omitted=copy.deepcopy(t['coverage_manifest'].get('unit_omitted',[])))
    by_id={c['id']:copy.deepcopy(c) for c in t['ranked']}
    selected=[by_id[mid] for mid in t['selection']['selected_ids']]
    req=schemas.SearchRequest(**{k:t[k] for k in ('user_id','query','options','top_k','reference_time','evidence_token_budget')})
    settings=dict(t['versions']['settings'],PACKET_CORE_GAP_CONTEXT=enabled)
    with patch.multiple(config,**settings),patch('socket.socket.connect',side_effect=AssertionError('Offline')):
        packed,digest,manifest=search_pipeline._pack_evidence(None,req,plan,selected,t['anchor_time'],list(by_id.values()))
        # The live endpoint serializes through SearchItem before returning.
        return [schemas.SearchItem.model_validate(p).model_dump() for p in packed],digest,manifest


def main(args):
    diagnostic=json.loads(args.diagnostic.read_text());rows=[]
    assert hashlib.sha256(Path(__file__).with_name('probe_core_gap_context.py').read_bytes()).hexdigest()==diagnostic['script_sha256']
    expected={(r['run'],r['qa_id']):r for r in diagnostic['rows']}
    for spec in diagnostic['inputs']:
        run=Path(spec['run']);path=run/'server-search-debug.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['trace_sha256']
        rs={r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        for t in records(path):
            q=rs[t['search_id']]['qa_id'];old=expected[str(run),q]
            baseline,baseline_digest,baseline_manifest=replay(t,False)
            assert baseline==t['returned'] and baseline_digest==t['packet_hash'],('baseline',q)
            for key,value in baseline_manifest.items():assert value==t['coverage_manifest'][key],(q,key)
            with patch.multiple(config,**t['versions']['settings']):proposed=prepare(t)['packets']['core_gap_allowance']
            result,digest,manifest=replay(t,True)
            assert result==[dict(p,packet_hash=evidence_packet.digest(proposed)) for p in proposed],('core_gap_allowance',q)
            assert digest==evidence_packet.digest(result)
            cats=answer_choice.build_catalog(result)
            assert sum(len(c['text'].encode()) for cat in cats for c in cat.values())==old['catalog_bytes']['core_gap_allowance']
            rows.append(dict(run=str(run),qa_id=q,changed=digest!=baseline_digest,
                context_bytes=manifest['conversation_context']['added_bytes']))
    assert len(rows)==len(expected)
    result=dict(questions=len(rows),baseline_packets_and_manifests_verified=len(rows),
        core_gap_packets_and_catalogs_verified=len(rows),provider_calls=0,rows=rows,
        diagnostic_sha256=hashlib.sha256(args.diagnostic.read_bytes()).hexdigest())
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
