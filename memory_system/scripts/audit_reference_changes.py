"""Trace final reference gains/losses back to admission and review decisions.

Full source-turn transport is a diagnostic proxy, not semantic entailment.
Use after the full HTTPS funnel/identity audit; never infer causality from score.
"""
import argparse
import hashlib
import json
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def main(args):
    directory=args.run
    comparison=json.loads((directory/'funnel-comparison.json').read_text())
    if (directory/'exit-code').read_text().strip()!='0':
        raise ValueError('Require successful completed experiment')
    arms={};traces={}
    for arm in ('control','treatment'):
        run=directory/arm
        stage=json.loads((run/'funnel/stage-transport.json').read_text())
        if stage['summary']['progress_snapshot']:
            raise ValueError('Do not use a partial funnel')
        by_stage={(r['conversation_id'],r['qa_id']):r for r in stage['rows']}
        by_search={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
        rows={};traces[arm]={}
        for r in records(run/'completed/data/results/personamem-v2-32k.jsonl'):
            key=r['conversation_id'],r['qa_id'];t=by_search[r['search_id']];s=by_stage[key]
            if s['score']!=r['score'] or t['packet_hash']!=r['packet_hash']:
                raise ValueError('Result/trace/funnel identity differs')
            c=t['cascade'];main=c['listwise'];overflow=c.get('overflow',{})
            carriers=set().union(*(set(ids) for ids in s['carrier_ids'].values()))
            reviews=[('main',i,v) for i,v in enumerate(main.get('recovery',{}).get('batches',[]))]
            reviews.append(('overflow',0,overflow.get('review',{})))
            reference_reviews=[]
            for kind,index,v in reviews:
                relevant=[j for j in v.get('judgments',[]) if j['candidate_id'] in carriers]
                omitted=[o for o in v.get('omitted',[]) if o['candidate_id'] in carriers]
                if relevant or omitted:
                    reference_reviews.append(dict(stage=kind,batch=index,
                        prompt_sha256=v.get('prompt_sha256'),sources_sha256=fingerprint(v.get('sources',[])),
                        candidate_ids=v.get('candidate_ids',[]),judgments=relevant,omitted=omitted))
            omitted=c.get('omitted',[])+t['coverage_manifest'].get('omitted',[])+t['coverage_manifest'].get('unit_omitted',[])
            rows[r['qa_id']]=dict(conversation_id=r['conversation_id'],score=r['score'],
                reference={k:v['full_turns'] for k,v in s['visible'].items()},
                assistant_reference={k:v['full_turns'] for k,v in s['assistant_visible'].items()},
                complete_set=s['visible']['packet']['available_set_complete'],carrier_ids=s['carrier_ids'],
                primary_ref_ids=sorted(carriers&set(main.get('primary_selected_ids',[]))),
                unreviewed_ref_ids=sorted(carriers&set(main.get('recovery',{}).get('unreviewed_ids',[]))),
                context_ref_ids=sorted(carriers&set(c.get('conversation_context',{}).get('appended_ids',[]))),
                reference_reviews=reference_reviews,related_omissions=[o for o in omitted if o.get('id') in carriers],
                overflow_reference_decisions=[o for o in overflow.get('decisions',[]) if o.get('id') in carriers],
                packet_hash=t['packet_hash'],returned_sha256=fingerprint(t['returned']))
            traces[arm][r['qa_id']]=t
        if len(rows)!=comparison['summaries'][arm]['questions']:
            raise ValueError('Question coverage differs from full audit')
        arms[arm]=rows
    if arms['control'].keys()!=arms['treatment'].keys():
        raise ValueError('Arms have different question IDs')
    pairs=[]
    for qid in sorted(arms['control'],key=int):
        a,b=arms['control'][qid],arms['treatment'][qid]
        ta,tb=traces['control'][qid],traces['treatment'][qid]
        if a['conversation_id']!=b['conversation_id'] or ta['query']!=tb['query'] or ta['options']!=tb['options']:
            raise ValueError('Mismatched question identity')
        def main_input(t):
            by_id={v['id']:v for v in t['ranked']}
            return [(mid,by_id[mid].get('_rank_text',by_id[mid].get('content'))) for mid in t['cascade']['listwise']['candidate_ids']]
        ja={(v['stage'],j['candidate_id']):(j,v) for v in a['reference_reviews'] for j in v['judgments']}
        jb={(v['stage'],j['candidate_id']):(j,v) for v in b['reference_reviews'] for j in v['judgments']}
        transitions=[]
        for key in sorted(ja.keys()|jb.keys()):
            left,la=ja.get(key,(None,{}));right,rb=jb.get(key,(None,{}))
            def outcome(j):
                return 'not_reviewed' if j is None else 'invalid' if not j['valid'] else 'positive' if j['useful'] else 'negative'
            if left!=right:
                transitions.append(dict(stage=key[0],candidate_id=key[1],control=outcome(left),treatment=outcome(right),
                    same_prompt=bool(la.get('prompt_sha256')) and la.get('prompt_sha256')==rb.get('prompt_sha256'),
                    same_sources=bool(la.get('sources_sha256')) and la.get('sources_sha256')==rb.get('sources_sha256')))
        pairs.append(dict(qa_id=qid,control_score=a['score'],treatment_score=b['score'],
            control_packet=a['reference']['packet'],treatment_packet=b['reference']['packet'],
            same_main_input=main_input(ta)==main_input(tb),same_packet_hash=a['packet_hash']==b['packet_hash'],
            same_returned=a['returned_sha256']==b['returned_sha256'],reference_judgment_changes=transitions))
    gained=[r['qa_id'] for r in pairs if r['treatment_packet'] and not r['control_packet']]
    lost=[r['qa_id'] for r in pairs if r['control_packet'] and not r['treatment_packet']]
    if set(gained)!=set(comparison['gained']) or set(lost)!=set(comparison['lost']):
        raise ValueError('Reference change set differs from full funnel')
    summary=dict(questions=len(pairs),gained=gained,lost=lost,
        answer_changed_same_packet=[r['qa_id'] for r in pairs if r['control_score']!=r['treatment_score'] and r['same_packet_hash']],
        answer_changed_same_returned=[r['qa_id'] for r in pairs if r['control_score']!=r['treatment_score'] and r['same_returned']],
        same_main_input=sum(r['same_main_input'] for r in pairs),same_packet_hash=sum(r['same_packet_hash'] for r in pairs))
    review_comparison=None
    if all((directory/(arm+'-review-audit.json')).exists() for arm in arms):
        audits={arm:json.loads((directory/(arm+'-review-audit.json')).read_text()) for arm in arms}
        if any(v['summary']['partial'] or v['summary']['questions']!=len(pairs) for v in audits.values()):
            raise ValueError('Do not use partial request audits')
        by_sources={arm:{(r['qa_id'],r['stage'],r['sources_sha256']):r for r in v['rows']} for arm,v in audits.items()}
        if any(len(by_sources[arm])!=len(v['rows']) for arm,v in audits.items()):
            raise ValueError('Ambiguous repeated review input')
        matched=[]
        for key in sorted(by_sources['control'].keys()&by_sources['treatment'].keys()):
            a=by_sources['control'][key];b=by_sources['treatment'][key]
            old={j['candidate_id']:j for j in a['judgments']};new={j['candidate_id']:j for j in b['judgments']}
            def outcomes(js):
                return {mid:('invalid' if not j['valid'] else 'positive' if j['useful'] else 'negative') for mid,j in js.items()}
            before,after=outcomes(old),outcomes(new)
            matched.append(dict(qa_id=key[0],stage=key[1],same_prompt=a['prompt_sha256']==b['prompt_sha256'],
                same_request=(a['request_sha256']==b['request_sha256']
                              if a.get('request_sha256') and b.get('request_sha256') else None),
                same_system=(a['system_sha256']==b['system_sha256']
                             if a.get('system_sha256') and b.get('system_sha256') else None),
                affected_by_segmentation=a['affected_by_segmentation'] or b['affected_by_segmentation'],
                outcome_changes=[dict(candidate_id=mid,control=before.get(mid,'missing'),treatment=after.get(mid,'missing'))
                                 for mid in sorted(before.keys()|after.keys()) if before.get(mid)!=after.get(mid)]))
        review_comparison=dict(matched_sources=len(matched),same_prompt=sum(r['same_prompt'] for r in matched),
            request_comparable=sum(r['same_request'] is not None for r in matched),
            same_request=sum(r['same_request'] is True for r in matched),
            changed_request=sum(r['same_request'] is False for r in matched),
            prompt_hash_scope='Source prompt only; full request also includes system, schema, output limit and reservation.',
            changed_prompt=sum(not r['same_prompt'] for r in matched),
            changed_prompt_questions=sorted({r['qa_id'] for r in matched if not r['same_prompt']},key=int),
            unmatched_requests={arm:len(v['rows'])-len(matched) for arm,v in audits.items()},rows=matched)
        summary['review_comparison']={k:v for k,v in review_comparison.items() if k!='rows'}
    args.output.write_text(json.dumps(dict(summary=summary,pairs=pairs,arms=arms,review_comparison=review_comparison,limitation=__doc__),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
