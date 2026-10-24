"""Compare repeated complete source-query ablations on the same development set.

Repeated questions are not independent new examples. Compare actual paired
answers separately from deterministic packet transport with upstream fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def records(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    reference=None;trials=[]
    for root in args.runs:
        if (root/'exit-code').read_text().strip()!='0':raise ValueError('Incomplete trial')
        plan=read(root/'plan.json')
        if plan['toggle']!='PACKET_SOURCE_QUERY_CONTEXT':raise ValueError('Wrong intervention')
        identity={k:plan[k] for k in ('fixture_sha256','database_sha256','data_sha256','source_trace_sha256',
                                     'reader_versions','questions','concurrency_per_arm')}
        if reference is None:reference=identity
        elif identity!=reference:raise ValueError('Trial inputs or frozen reader differ')
        funnel=read(root/'funnel-comparison.json')
        effect=read(root/'source-query-effect-audit.json')
        runtime=read(root/'runtime-accounting-audit.json')
        if funnel['settings_differences']!={'PACKET_SOURCE_QUERY_CONTEXT':[False,True]}:
            raise ValueError('Mixed effective changes')
        questions={};arms={}
        for arm in ('control','treatment'):
            data=effect['arms'][arm]
            if data['summary']['actual_packets_verified']!=plan['questions'] or data['summary']['reader_catalogs_verified']!=plan['questions']:
                raise ValueError('Incomplete packet/catalog replay')
            results={r['qa_id']:r for r in records(root/arm/'completed/data/results/personamem-v2-32k.jsonl')}
            arms[arm]=dict(score=sum(r['score'] for r in results.values()),
                actual_reference_questions=funnel['summaries'][arm]['visible']['packet'],
                replay=data['summary'],provider_statuses=runtime['arms'][arm]['search_provider_statuses'])
            for row in data['rows']:
                q=row['qa_id'];r=results[q]
                questions.setdefault(q,{})[arm]=dict(score=r['score'],prediction=r['prediction'],
                    recovered=bool(r.get('reader_recovered') or r.get('transport_recovered')),
                    base_packet_hash=row['base_packet_hash'],packet_hashes=row['packet_hashes'],
                    source_seat_ids=row['source_seat']['appended_ids'],reader_references=row['reader_references'])
        trials.append(dict(run=str(root),arms=arms,questions=questions,
                           answer_wins=funnel['wins'],answer_losses=funnel['losses'],sensitivity=runtime['sensitivity']))
    if len(trials)<2:raise ValueError('Require at least two trials')
    comparisons=[]
    first=trials[0]
    for trial in trials[1:]:
        if first['questions'].keys()!=trial['questions'].keys():raise ValueError('Question inventory changed')
        rows=[]
        for q,b in trial['questions'].items():
            a=first['questions'][q]
            for arm in ('control','treatment'):
                mode='0' if arm=='control' else '1'
                rows.append(dict(qa_id=q,arm=arm,first_score=a[arm]['score'],repeat_score=b[arm]['score'],
                    same_base=a[arm]['base_packet_hash']==b[arm]['base_packet_hash'],
                    same_actual_packet=a[arm]['packet_hashes'][mode]==b[arm]['packet_hashes'][mode],
                    same_source_seat=a[arm]['source_seat_ids']==b[arm]['source_seat_ids']))
        comparisons.append(dict(run=trial['run'],rows=rows,
            answer_changes_same_packet=[r for r in rows if r['same_actual_packet'] and r['first_score']!=r['repeat_score']],
            source_seat_changed=[r for r in rows if not r['same_source_seat']]))
    result=dict(trials=trials,repeat_comparisons=comparisons,limitation=__doc__,
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(trials=[dict(run=t['run'],arms=t['arms']) for t in trials],
        repeat_comparisons=[{k:v for k,v in c.items() if k!='rows'} for c in comparisons]),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
