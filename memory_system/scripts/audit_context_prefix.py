"""Compare actual controls with treatment prefixes before residual context.

Identical input with different answers reveals runtime variability. A single
paired outcome cannot establish an isolated causal effect of added context.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(root):
    root=root.resolve()
    if (root/'exit-code').read_text().strip()!='0':
        raise ValueError('Require complete successful trial')
    plan=json.loads((root/'plan.json').read_text())
    sys.path.insert(0,str(root/'source/memory_system'))
    from app import answer_choice,evidence_packet
    arms={}
    for arm in ('control','treatment'):
        run=root/arm
        traces={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
        results=records(run/'completed/data/results/personamem-v2-32k.jsonl')
        if len(results)!=plan['questions'] or any(r.get('error_stage') for r in results):
            raise ValueError('Incomplete arm')
        rows={}
        for r in results:
            t=traces[r['search_id']]
            if t['packet_hash']!=r['packet_hash'] or evidence_packet.digest(t['returned'])!=t['packet_hash']:
                raise ValueError('Result or trace hash differs')
            ctx=t['cascade']['conversation_context']
            if arm=='control':prefix=t['returned'];digest=t['packet_hash']
            else:
                ids=ctx['prefix_packet_ids'];digest=ctx['prefix_packet_hash']
                prefix=[dict(p,packet_hash=digest) for p in t['returned'][:len(ids)]]
                if [p['id'] for p in prefix]!=ids or evidence_packet.digest(prefix)!=digest:
                    raise ValueError('Original prefix hash differs')
            rows[r['qa_id']]=dict(result=r,trace=t,prefix=prefix,digest=digest,
                catalogs=answer_choice.build_catalog(prefix))
        if len(rows)!=plan['questions']:raise ValueError('Duplicate questions')
        arms[arm]=rows
    if arms['control'].keys()!=arms['treatment'].keys():raise ValueError('Different cohorts')
    pairs=[]
    for qid in sorted(arms['control'],key=int):
        a,b=arms['control'][qid],arms['treatment'][qid]
        if any(a['trace'][key]!=b['trace'][key] for key in ('user_id','query','options')):
            raise ValueError('Different question identity')
        pairs.append(dict(qa_id=qid,control_score=a['result']['score'],treatment_score=b['result']['score'],
            same_prefix_packet_hash=a['digest']==b['digest'],same_prefix_returned=a['prefix']==b['prefix'],
            same_prefix_reader_catalogs=a['catalogs']==b['catalogs'],
            same_final_reader_catalogs=(a['result']['answer_source_catalog']==b['result']['answer_source_catalog']
                and a['result']['answer_constraint_catalog']==b['result']['answer_constraint_catalog']),
            appended_ids=[p['id'] for p in b['trace']['returned'][len(b['prefix']):]]))
    summary=dict(questions=len(pairs),**{key:sum(r[key] for r in pairs) for key in
        ('same_prefix_packet_hash','same_prefix_returned','same_prefix_reader_catalogs','same_final_reader_catalogs')},
        score_changes=[r for r in pairs if r['control_score']!=r['treatment_score']])
    with (root/'context-prefix-comparison.json').open('x') as f:
        json.dump(dict(summary=summary,rows=pairs,limitation=__doc__,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
        f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
