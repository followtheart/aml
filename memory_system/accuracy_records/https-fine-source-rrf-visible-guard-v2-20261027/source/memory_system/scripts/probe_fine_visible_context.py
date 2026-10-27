"""Use already-visible source context for adaptive fine relevance only.

Preserve exact quote identities, core selection, redundancy rules, CE scores,
capacity and main admission. No provider calls or reference-driven selection.
"""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import evidence_selection as selection, fine_evidence
from probe_full_option_ce import prepare,base,coverage,sha
from evaluate_recall_context import reference_turns


def visible_context(row, pieces):
    """Expand a quote only when its original visible source is unambiguous.

    This does not add a source, change a quote identity or infer any fact.
    Keep ambiguous origins at their original quoted extent.
    """
    body=selection._text(row.get('_rank_text',row.get('content','')))
    originals=[]
    for source in row.get('_packet_item',{}).get('sources',[]):
        text=selection._text(source.get('content') or '')
        if not text or text not in body:
            continue
        sid=source.get('source_event_id')
        if not sid and source.get('request_id') is not None and source.get('message_index') is not None:
            sid=f"{source['request_id']}:{source['message_index']}"
        if sid:
            originals.append((sid,source.get('role'),text))
    out=[]
    for (sid,role),quote in pieces:
        matches={(origin,text) for origin,source_role,text in originals
                 if source_role==role and quote in text and
                 (sid==origin or sid==f"profile-quote:{row['id']}" and row.get('type')=='profile')}
        text=next(iter(matches))[1] if len(matches)==1 else quote
        out.append(((sid,role),text))
    assert [key for key,_ in out]==[key for key,_ in pieces]
    assert all(old in new for (_,old),(_,new) in zip(pieces,out))
    return out


def main(args):
    original_relevance=fine_evidence.relevance
    original_sparse=fine_evidence.sparse_matches
    fixed=[]
    with patch('socket.socket.connect',side_effect=AssertionError('Offline')):
        jobs,manifest,conversations=prepare(args.run)
        for job in jobs:
            scores=[c['_ce_score'] for c in job['coarse']]
            original=base.replay(job,scores)
            visible={c['id']:selection.fragments(c) for c in job['coarse']}
            expanded={c['id']:visible_context(c,visible[c['id']]) for c in job['coarse']}
            def relevance(row,requirements,pieces=None):
                return original_relevance(row,requirements,expanded[row['id']])
            def sparse(rows,requirements,pieces):
                return original_sparse(rows,requirements,{c['id']:expanded[c['id']] for c in rows})
            captured=[]
            original_propose=fine_evidence.propose
            def propose(*a,**kw):
                chosen,trace=original_propose(*a,**kw)
                captured.append(trace)
                return chosen,trace
            with patch.object(fine_evidence,'relevance',relevance),patch.object(fine_evidence,'sparse_matches',sparse),\
                 patch.object(fine_evidence,'propose',propose):
                proposed=base.replay(job,scores)
            assert len(captured)==1
            historical=job['t']['cascade']['fine']['selection']
            assert captured[0]['adaptive']['core_ids']==historical['adaptive']['core_ids']
            assert captured[0]['adaptive']['capacity']==historical['adaptive']['capacity']
            changes=[c['id'] for c in job['coarse'] if visible[c['id']]!=expanded[c['id']]]
            fixed.append((job,original,proposed,changes))
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    raw={}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'],{})[row['request_id'],row['message_index']]=dict(row)
    convs={c['conversation_id']:c for c in conversations};rows=[]
    for job,original,proposed,changes in fixed:
        conv=convs[job['conversation_id']];q=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(q,conv['source_history'],gold[int(job['qa_id'])])
        rows.append(dict(qa_id=job['qa_id'],conversation_id=job['conversation_id'],expanded_ids=changes,
            variants={name:coverage(job,value,turns,raw[job['t']['user_id']])
                      for name,value in (('original',original),('proposed',proposed))}))
    summary=dict(questions=len(rows),reproduced_fine_and_main=len(rows),core_preserved=len(rows),provider_calls=0,
        expanded_candidate_instances=sum(len(r['expanded_ids']) for r in rows),variants={})
    for name in ('original','proposed'):
        summary['variants'][name]={stage:dict(
            reference_questions=sum(any(r['variants'][name][stage]['user']) for r in rows),
            user_turns=sum(sum(r['variants'][name][stage]['user']) for r in rows),
            changed_order=sum(r['variants'][name][stage]['ids']!=r['variants']['original'][stage]['ids'] for r in rows),
            gained=[r['qa_id'] for r in rows if any(r['variants'][name][stage]['user']) and not any(r['variants']['original'][stage]['user'])],
            lost=[r['qa_id'] for r in rows if any(r['variants']['original'][stage]['user']) and not any(r['variants'][name][stage]['user'])],
            fewer_turns=[r['qa_id'] for r in rows if sum(r['variants'][name][stage]['user'])<sum(r['variants']['original'][stage]['user'])]) for stage in ('fine','main')}
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary,rows=rows,limitation=__doc__,inputs={str(p.resolve()):sha(p) for p in (
            Path(__file__),Path(fine_evidence.__file__),Path(selection.__file__),args.run/'server-search-debug.jsonl',
            args.run/'completed/data/results/personamem-v2-32k.jsonl',Path(manifest['data']),args.gold_csv)}),stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
