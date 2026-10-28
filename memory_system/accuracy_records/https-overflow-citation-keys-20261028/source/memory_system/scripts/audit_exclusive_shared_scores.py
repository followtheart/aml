"""Separate score changes from observed exclusive shared context additions in completed HTTPS.

Both arms make independent live Search/reader calls. An added source can affect
an answer, but its presence does not isolate that causal effect when upstream
packets or independent reader judgments differ. No new provider calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def main(args):
    root=args.run.resolve()
    if (root/'exit-code').read_text().strip() not in ('0','1'):
        raise ValueError('Trial has not finished')
    plan=json.loads((root/'plan.json').read_text())
    if plan['toggle']!='PACKET_EXCLUSIVE_SHARED_CONTEXT':
        raise ValueError('Wrong ablation')
    sys.path.insert(0,str(root/'source/memory_system'))
    from app import answer_choice, evidence_packet
    from evaluation_model_guard import check_versions
    arms={}
    for arm in ('control','treatment'):
        run=root/arm
        results={r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        if len(results)!=plan['questions']:
            raise ValueError('Incomplete cohort')
        rows={}
        for t in records(run/'server-search-debug.jsonl'):
            r=results[t['search_id']]
            check_versions(t['versions'],arm)
            if t['versions']['settings']['PACKET_EXCLUSIVE_SHARED_CONTEXT']!=(arm=='treatment'):
                raise ValueError('Wrong effective arm setting')
            ctx=t['cascade']['conversation_context']
            extra=ctx.get('exclusive_shared',{})
            count=len(ctx['prefix_packet_ids']) if arm=='treatment' else len(t['returned'])
            prefix=t['returned'][:count]
            prefix_digest=evidence_packet.digest(prefix)
            prefix=[dict(p,packet_hash=prefix_digest) for p in prefix]
            if arm=='treatment' and (prefix_digest!=ctx['prefix_packet_hash']
                    or [p['id'] for p in t['returned'][count:]]!=extra['appended_ids']):
                raise ValueError('Prefix or added inventory differs')
            actual=answer_choice.build_catalog(t['returned'])
            if not r.get('error_stage') and (list(actual[0].values())!=r['answer_source_catalog']
                    or list(actual[1].values())!=r['answer_constraint_catalog']):
                raise ValueError('Reader catalog identity differs')
            by_id={c['id']:c for c in t['ranked']}
            main_ids=t['cascade']['listwise']['candidate_ids']
            rows[r['qa_id']]=dict(score=r['score'],prediction=r.get('prediction'),
                terminal_error=r.get('error_type'),query=t['query'],options=t['options'],plan=t['plan'],
                fine_ids=t['cascade']['fine']['selected_ids'],
                main_input_sha256=digest([(mid,by_id[mid].get('_rank_text',by_id[mid].get('content'))) for mid in main_ids]),
                catalog_sha256=digest(actual),prefix_catalog_sha256=digest(answer_choice.build_catalog(prefix)),
                prefix_packet_hash=prefix_digest,appended_ids=extra.get('appended_ids',[]),
                added_bytes=extra.get('added_bytes',0),
                internal_reader_errors=[dict(stage=c['stage'],error_type=c.get('error_type'))
                    for c in r.get('answer_calls',[]) if c.get('status')=='error'])
        if len(rows)!=plan['questions']:
            raise ValueError('Trace cohort differs')
        arms[arm]=rows
    if arms['control'].keys()!=arms['treatment'].keys():
        raise ValueError('Question cohorts differ')
    rows=[]
    for q in sorted(arms['control'],key=int):
        c,t=arms['control'][q],arms['treatment'][q]
        if any(c[k]!=t[k] for k in ('query','options','plan')):
            raise ValueError('Frozen upstream inputs differ')
        rows.append(dict(qa_id=q,control_score=c['score'],treatment_score=t['score'],
            control_prediction=c['prediction'],treatment_prediction=t['prediction'],
            appended_ids=t['appended_ids'],added_bytes=t['added_bytes'],
            same_fine=c['fine_ids']==t['fine_ids'],same_main_input=c['main_input_sha256']==t['main_input_sha256'],
            same_prefix_packet=c['prefix_packet_hash']==t['prefix_packet_hash'],
            same_prefix_catalog=c['prefix_catalog_sha256']==t['prefix_catalog_sha256'],
            same_catalog=c['catalog_sha256']==t['catalog_sha256'],
            internal_errors={a:arms[a][q]['internal_reader_errors'] for a in arms},
            terminal_errors={a:arms[a][q]['terminal_error'] for a in arms}))
    changes=[r for r in rows if r['control_score']!=r['treatment_score']]
    summary=dict(questions=len(rows),scores={a:sum(r['score'] for r in arm.values()) for a,arm in arms.items()},
        wins=[r['qa_id'] for r in changes if r['treatment_score']>r['control_score']],
        losses=[r['qa_id'] for r in changes if r['treatment_score']<r['control_score']],
        activated=[r['qa_id'] for r in rows if r['appended_ids']],
        score_changes_with_addition=[r['qa_id'] for r in changes if r['appended_ids']],
        score_changes_without_addition=[r['qa_id'] for r in changes if not r['appended_ids']],
        same_catalog_score_changes=[r['qa_id'] for r in changes if r['same_catalog']],
        same_fine=sum(r['same_fine'] for r in rows),same_main_input=sum(r['same_main_input'] for r in rows),
        same_prefix_catalog=sum(r['same_prefix_catalog'] for r in rows),
        same_catalog=sum(r['same_catalog'] for r in rows))
    output=dict(summary=summary,rows=rows,limitation=__doc__,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (root/'score-attribution-audit.json').open('x') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args())
