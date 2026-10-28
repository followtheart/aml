"""Offline main-admission comparison of lossless graph metadata encodings.

Historical fine/main are first reproduced exactly. Compact JSON and column
tables run the same actual admission procedure, including candidate replacement
and all original byte/token/count bounds. No provider calls or answer claims.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import cascade_rerank, config, prompts, schemas
from probe_full_option_ce import prepare, base, sha
from evaluate_recall_context import reference_turns
import audit_https_funnel as funnel
import listwise_graph_tables as encoding


def main(args):
    original_render = prompts.render
    fixed=[]
    with patch('socket.socket.connect',side_effect=AssertionError('Offline')):
        jobs,manifest,conversations=prepare(args.run)
        for job in jobs:
            t=job['t'];scores=[c['_ce_score'] for c in job['coarse']]
            original=base.replay(job,scores)
            variants={'original':original}; savings={}
            plan=copy.deepcopy(t['plan'])
            plan['_coverage_requirements']=[{k:r[k] for k in ('id','text','origin') if k in r}
                for r in t['coverage_manifest']['coverage']['requirements']]
            plan['_fusion_edges']=t['fusion_edges'];plan['_fusion_triples']=t['fusion_triples']
            plan['_fine_path_groups']=t['cascade']['fine']['selection']['adaptive']['path_groups']
            req=schemas.SearchRequest(user_id=t['user_id'],query=t['query'],options=t['options'])
            for mode in ('compact','tables'):
                with patch.object(prompts,'render',encoding.renderer(original_render,mode)):
                    variants[mode]=base.replay(job,scores)
                    with patch.multiple(config,**t['versions']['settings']):
                        request=cascade_rerank.listwise_request(req,plan,original['main'])
                savings[mode]=original['main_bytes']-len(request['prompt'].encode())
                assert [c['id'] for c in variants[mode]['fine']]==[c['id'] for c in original['fine']]
            fixed.append((job,variants,savings))
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    raw={}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'],{})[row['request_id'],row['message_index']]=dict(row)
    convs={c['conversation_id']:c for c in conversations};rows=[]
    for job,variants,savings in fixed:
        conv=convs[job['conversation_id']];q=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(q,conv['source_history'],gold[int(job['qa_id'])])
        values={}
        for name,v in variants.items():
            cards=[dict(s,id=f'{i}:{j}',text=s.get('content','')) for i,c in enumerate(v['main'])
                   for j,s in enumerate(c['_packet_item'].get('sources',[]))]
            coverage=funnel.turn_coverage(turns,cards,raw[job['t']['user_id']])
            values[name]=dict(ids=[c['id'] for c in v['main']],main_bytes=v['main_bytes'],
                              user=[c['full'] for c in coverage])
        rows.append(dict(qa_id=job['qa_id'],conversation_id=job['conversation_id'],variants=values,fixed_savings=savings))
    summary=dict(questions=len(rows),reconstructed_fine_and_main=len(rows),provider_calls=0,variants={})
    for name in ('original','compact','tables'):
        summary['variants'][name]=dict(
            reference_questions=sum(any(r['variants'][name]['user']) for r in rows),
            user_turns=sum(sum(r['variants'][name]['user']) for r in rows),
            gained=[r['qa_id'] for r in rows if any(r['variants'][name]['user']) and not any(r['variants']['original']['user'])],
            lost=[r['qa_id'] for r in rows if any(r['variants']['original']['user']) and not any(r['variants'][name]['user'])],
            fewer_turns=[r['qa_id'] for r in rows if sum(r['variants'][name]['user'])<sum(r['variants']['original']['user'])],
            changed_members=sum(set(r['variants'][name]['ids'])!=set(r['variants']['original']['ids']) for r in rows),
            total_candidates=sum(len(r['variants'][name]['ids']) for r in rows),
            fixed_prompt_savings=sum(r['fixed_savings'].get(name,0) for r in rows))
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary,rows=rows,limitation=__doc__,inputs={str(p.resolve()):sha(p) for p in (
            Path(__file__),Path(encoding.__file__),args.run/'server-search-debug.jsonl',
            args.run/'completed/data/results/personamem-v2-32k.jsonl',Path(manifest['data']),args.gold_csv)}),stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(parser.parse_args())
