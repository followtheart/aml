"""Extract every changed main pair from a terminal, exact offline swap replay.

Review inputs include question/options and the complete original displayed
candidate texts. They exclude gold, coverage and CE scores. This filtering does
not make review blinded if the reviewer has already seen aggregate results.
"""
import argparse
import json
from pathlib import Path

from audit_retained_porter_https import records, require, same, sha


def main(args):
    run = args.run.resolve()
    report = json.loads(args.replay.read_text())
    state = json.loads((args.replay.parent / 'state.json').read_text())
    same(state['status'], 'complete', 'Terminal replay')
    same((args.replay.with_suffix('.exit-code')).read_text().strip(), '0', 'Replay exit')
    trace_path = run / 'server-search-debug.jsonl'
    result_path = run / 'completed/data/results/personamem-v2-32k.jsonl'
    for path in (trace_path, result_path):
        same(sha(path), report['input_sha256'][str(path)], 'Replay source ' + str(path))
    results = {r['search_id']: {k:r[k] for k in ('qa_id','conversation_id','packet_hash')}
               for r in records(result_path)}
    rows = {r['qa_id']:r for r in report['rows']}
    same(len(rows), report['summary']['questions'], 'Full replay results')
    pairs, seen = [], set()
    for trace in records(trace_path):
        sid = trace['search_id']
        require(sid in results and sid not in seen, 'Unexpected/duplicate trace')
        seen.add(sid)
        result = results[sid]
        row = rows[result['qa_id']]
        same(row['packet_hash'], trace['packet_hash'], 'Packet identity')
        same(row['query'], trace['query'], 'Question')
        same(row['options'], trace['options'], 'Options')
        same(row['original_main_ids'], trace['cascade']['listwise']['candidate_ids'], 'Original main')
        swap = row['ordering']['swap']
        if not swap:
            same(row['original_main_ids'],row['proposed_main_ids'],'Unchanged main')
            continue
        before, after = set(row['original_main_ids']), set(row['proposed_main_ids'])
        same(before-after,{swap['removed_id']},'Only one removed')
        same(after-before,{swap['added_id']},'Only one added')
        pool = {c['id']:c for c in trace['ranked']}
        def view(mid):
            c = pool[mid]
            return dict(candidate_id=mid, displayed_text=c.get('_rank_text',c['content']),
                        context_diagnosis=c.get('_rank_context',{}))
        added = view(swap['added_id'])
        same(row['new_main'], [dict(candidate_id=added['candidate_id'],rank_text=added['displayed_text'])],
             'New main text from original trace')
        pairs.append(dict(qa_id=result['qa_id'],conversation_id=result['conversation_id'],
                          query=row['query'],options=row['options'],removed=view(swap['removed_id']),added=added))
    same(seen,set(results),'Complete traces')
    same(len(pairs),report['summary']['swapped_questions'],'Complete swap review population')
    output=dict(status='complete_main_swap_review_extraction',questions=len(rows),pairs=len(pairs),
                rows=pairs,provider_calls=0,limitation=__doc__,
                input_sha256={str(p):sha(p) for p in (Path(__file__),
                    Path(__file__).with_name('audit_retained_porter_https.py'),args.replay,trace_path,result_path)})
    with args.output.open('x') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(questions=len(rows),pairs=len(pairs))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
