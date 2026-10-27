"""Offline 24 KB -> 32 KB main-prompt admission under unchanged Search budgets.

Uses original scores, fine selection and recorded remaining token budget.
Does not simulate model decisions, deadlines, overflow, packing or answers.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probe_full_option_ce import prepare, base, sha, coverage
from evaluate_recall_context import reference_turns


def main(args):
    selected=[]
    with patch('socket.socket.connect',side_effect=AssertionError('Offline')):
        jobs,manifest,conversations=prepare(args.run)
        for job in jobs:
            assert job['t']['versions']['settings']['RERANK_MAX_PROMPT_BYTES']==24000
            scores=[c['_ce_score'] for c in job['coarse']]
            original=base.replay(job,scores)
            changed=copy.deepcopy(job)
            changed['t']['versions']['settings']['RERANK_MAX_PROMPT_BYTES']=32000
            proposed=base.replay(changed,scores)
            assert [c['id'] for c in original['fine']]==[c['id'] for c in proposed['fine']]
            selected.append((job,original,proposed))
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    raw={}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'],{})[row['request_id'],row['message_index']]=dict(row)
    convs={c['conversation_id']:c for c in conversations};rows=[]
    for job,original,proposed in selected:
        conv=convs[job['conversation_id']];q=next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns=reference_turns(q,conv['source_history'],gold[int(job['qa_id'])])
        variants={name:coverage(job,v,turns,raw[job['t']['user_id']])
                  for name,v in (('original',original),('proposed',proposed))}
        rows.append(dict(qa_id=job['qa_id'],conversation_id=job['conversation_id'],variants=variants))
    summary=dict(questions=len(rows),original_limit=24000,proposed_limit=32000,
        reproduced_fine_and_main=len(rows),provider_calls=0,variants={})
    for name in ('original','proposed'):
        summary['variants'][name]=dict(
            reference_questions=sum(any(r['variants'][name]['main']['user']) for r in rows),
            user_turns=sum(sum(r['variants'][name]['main']['user']) for r in rows),
            gained=[r['qa_id'] for r in rows if any(r['variants'][name]['main']['user']) and not any(r['variants']['original']['main']['user'])],
            lost=[r['qa_id'] for r in rows if any(r['variants']['original']['main']['user']) and not any(r['variants'][name]['main']['user'])],
            fewer_turns=[r['qa_id'] for r in rows if sum(r['variants'][name]['main']['user'])<sum(r['variants']['original']['main']['user'])],
            total_candidates=sum(len(r['variants'][name]['main']['ids']) for r in rows),
            total_prompt_bytes=sum(r['variants'][name]['main_bytes'] for r in rows))
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary,rows=rows,limitation=__doc__,inputs={str(p.resolve()):sha(p) for p in (
            Path(__file__),args.run/'server-search-debug.jsonl',args.run/'completed/data/results/personamem-v2-32k.jsonl',
            Path(manifest['data']),args.gold_csv)}),stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
