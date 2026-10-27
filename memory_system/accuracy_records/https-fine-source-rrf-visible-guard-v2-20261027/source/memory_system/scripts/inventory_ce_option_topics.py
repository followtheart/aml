"""Offline lexical inventory of literal option topics missing from CE queries.

Uses every completed question, with no gold labels or snippet-based selection.
Missing literal terms do not establish semantic retrieval failure.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import personal_evidence, rerank_query, retrieval_queries


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def inventory(run):
    manifest=json.loads((run/'isolated-manifest.json').read_text())
    if manifest.get('client_exit_code') != 0 or manifest.get('audit_exit_code') != 0:
        raise ValueError('Require completed successful cohort')
    data=Path(manifest['data'])
    if hashlib.sha256(data.read_bytes()).hexdigest()!=manifest['data_sha256']:
        raise ValueError('Dataset hash changed')
    results=records(run/'completed/data/results/personamem-v2-32k.jsonl')
    prepared=records(data)
    expected=manifest.get('questions')
    if expected is None:
        if manifest['evaluation']['AML_EVAL_LIMIT']!=0:
            raise ValueError('Require complete selected histories')
        expected=sum(len(c['qa']) for c in prepared[:manifest['evaluation']['AML_EVAL_CONVS'] or None])
    if len(results)!=expected or len({(r['conversation_id'],r['qa_id']) for r in results})!=expected:
        raise ValueError('Incomplete or duplicate questions')
    questions={(c['conversation_id'],q['id']):q for c in prepared for q in c['qa']}
    traces={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
    rows=[]
    for r in results:
        t=traces[r['search_id']];q=questions[r['conversation_id'],r['qa_id']]
        if t['query']!=q['question'] or t['options']!=q['options']:
            raise ValueError('Question identity changed')
        query=t['cascade']['cross_encoder']['query']
        terms=personal_evidence.terms(query)
        topics=[]
        for i,option in enumerate(t['options'] or []):
            topic=retrieval_queries.topic_anchor(option)
            words=personal_evidence.terms(topic)
            topics.append(dict(index=i,literal_topic=topic,
                literal_premise=rerank_query._premise(option),
                absent_terms=sorted(words-terms),present_terms=sorted(words&terms)))
        byid={c['id']:c for c in t['ranked']}
        coarse=[byid[mid] for mid in t['cascade']['coarse']['selected_ids']]
        role_bytes=Counter()
        for c in coarse:
            for source in c.get('_packet_item',{}).get('sources',[]):
                role_bytes[source.get('role','unknown')]+=len(source.get('content','').encode())
        rows.append(dict(qa_id=r['qa_id'],conversation_id=r['conversation_id'],question=t['query'],
            ce_query=query,ce_query_bytes=len(query.encode()),topics=topics,
            coarse_source_bytes=dict(role_bytes)))
    summary=dict(run=str(run),questions=len(rows),
        questions_with_absent_option_terms=sum(any(o['absent_terms'] for o in r['topics']) for r in rows),
        questions_without_literal_premises=sum(not any(o['literal_premise'] for o in r['topics']) for r in rows),
        missing_term_options=sum(bool(o['absent_terms']) for r in rows for o in r['topics']),
        options=sum(len(r['topics']) for r in rows),
        source_trace_sha256=hashlib.sha256((run/'server-search-debug.jsonl').read_bytes()).hexdigest())
    return dict(summary=summary,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    reports=[inventory(run.resolve()) for run in args.run]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:
        json.dump(dict(cohorts=reports,provider_calls=0,limitation=__doc__),out,ensure_ascii=False,indent=2)
    print(json.dumps([r['summary'] for r in reports],indent=2))
