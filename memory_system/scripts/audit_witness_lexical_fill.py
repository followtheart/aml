"""Replay full hint/support-prompt construction and trace added hint usage.

Transport, valid quotations and observed citation usage are distinct from
personal entailment and answer causality. This is a single development cohort.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from audit_residual_reader import citations


def records(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class StopBeforeSupport(Exception):
    pass


async def main(args):
    root=args.run.resolve();plan=json.loads((root/'plan.json').read_text())
    audit=json.loads((root/'shared-reader-audit.json').read_text())
    if plan['reader_toggle']!='CHOICE_WITNESS_LEXICAL_FILL' or set(audit['settings_differences'])!={'CHOICE_WITNESS_LEXICAL_FILL'}:
        raise ValueError('Require audited lexical-fill trial')
    for name,value in plan['source_files'].items():
        if digest(root/'source'/name)!=value:raise ValueError('Frozen source changed')
    sys.path.insert(0,str(root/'source/memory_system'))
    from app import answer_choice,choice_witness,config,llm
    jobs={j['qa']['id']:j for j in records(root/'http-reader-inputs.jsonl')}
    results={arm:{r['qa_id']:r for r in records(root/arm/'completed/data/results/personamem-v2-32k.jsonl')}
             for arm in ['control','treatment']}
    rows=[]
    for arm in ['control','treatment']:
        settings=json.loads((root/arm/'reader-versions.json').read_text())['settings']
        stages={r['qa_id']:r for r in json.loads((root/arm/'funnel/stage-transport.json').read_text())['rows']}
        for q,record in results[arm].items():
            job=jobs[q];qa=job['qa'];packet=job['payload']['data']
            sources,constraints=answer_choice.build_catalog(packet)
            witness=next(c for c in record['answer_calls'] if c['stage']=='eval.choice_witness')
            support=next(c for c in record['answer_calls'] if c['stage']=='eval.choice_support')
            replay={}
            for enabled in [False,True]:
                captured=[];diag={}
                async def model(prompt,**kwargs):
                    stage=kwargs['stage']
                    if stage=='eval.choice_support':
                        captured.append(prompt);raise StopBeforeSupport
                    if stage!='eval.choice_witness' or hashlib.sha256(prompt.encode()).hexdigest()!=witness['prompt_sha256']:
                        raise ValueError('Unexpected witness request')
                    if 'response'not in witness:raise llm.LLMError('Recorded provider failure')
                    return copy.deepcopy(witness['response'])
                with patch.multiple(config,**dict(settings,CHOICE_WITNESS_LEXICAL_FILL=enabled)),\
                        patch.object(llm,'complete_json',model),\
                        patch('socket.socket.connect',side_effect=AssertionError('Offline')):
                    try:await answer_choice._answer(qa,packet,diag)
                    except StopBeforeSupport:pass
                if len(captured)!=1:raise ValueError('Support request not reached')
                replay[enabled]=dict(hints=diag['answer_witness_prefill'],prompt=captured[0],
                                    fill=diag.get('answer_witness_lexical_fill'))
            a,b=replay[False],replay[True]
            actual=b if arm=='treatment' else a
            if (actual['hints']!=record['answer_witness_prefill'] or
                    hashlib.sha256(actual['prompt'].encode()).hexdigest()!=support['prompt_sha256']):
                raise ValueError('Actual final hints or support request changed')
            if arm=='treatment' and b['fill']!=record['answer_witness_lexical_fill']:
                raise ValueError('Added-hint trace differs')
            if any(b['hints'][letter][:len(items)]!=items for letter,items in a['hints'].items()):
                raise ValueError('Original hint displaced')
            if any(len(items)>3 for items in b['hints'].values()):raise ValueError('Hint capacity exceeded')
            if b['prompt']!=a['prompt'].replace('\n<witnesses>\n'+choice_witness.render(a['hints'])+'\n</witnesses>',
                    '\n<witnesses>\n'+choice_witness.render(b['hints'])+'\n</witnesses>',1):
                raise ValueError('Non-hint support prompt changed')
            refs=set().union(*(set(v['source_ids']) for v in stages[q]['visible']['catalog']['turn_coverage']))
            label=qa['gold_labels'][0]
            added=b['fill']['added'];added_ids={w['source_id'] for w in added};added_sources={sid:sources[sid] for sid in added_ids}
            cited=[]
            for call in record['answer_calls']:
                if call['stage']=='eval.choice_witness':continue
                cited.extend(dict(stage=call['stage'],**v) for v in citations(call.get('response'),added_sources) if v['exact_quote'])
            alignment=next((e for e in record.get('choice_alignment',[]) if e['letter']==label),{})
            gold_valid_added=[c for claim in alignment.get('claims',[]) for c in claim.get('citations',[])
                if c.get('valid') and c['source_id']in added_ids]
            rows.append(dict(arm=arm,qa_id=q,score=record['score'],gold=label,
                original_hints=sum(map(len,a['hints'].values())),filled_hints=sum(map(len,b['hints'].values())),added=added,
                original_gold_reference=any(w['source_id']in refs for w in a['hints'].get(label,[])),
                filled_gold_reference=any(w['source_id']in refs for w in b['hints'].get(label,[])),
                support_prompt_bytes={key:len(value['prompt'].encode()) for key,value in [('baseline',a),('filled',b)]},
                downstream_citations=cited,gold_valid_added_citations=gold_valid_added,
                gold_status=alignment.get('status'),gold_eligible=label in record.get('answer_eligible_options',[])))
    arms={}
    for arm in ['control','treatment']:
        group=[r for r in rows if r['arm']==arm]
        arms[arm]=dict(questions=len(group),score=sum(r['score'] for r in group),
            hint_additions=sum(len(r['added']) for r in group),changed_questions=[r['qa_id'] for r in group if r['added']],
            before_gold_reference=sum(r['original_gold_reference'] for r in group),
            after_gold_reference=sum(r['filled_gold_reference'] for r in group),
            reference_gains=[r['qa_id'] for r in group if r['filled_gold_reference'] and not r['original_gold_reference']],
            reference_losses=[r['qa_id'] for r in group if r['original_gold_reference'] and not r['filled_gold_reference']],
            support_prompt_byte_growth=sum(r['support_prompt_bytes']['filled']-r['support_prompt_bytes']['baseline'] for r in group),
            added_source_cited_questions=[r['qa_id'] for r in group if r['added'] and r['downstream_citations']],
            gold_valid_added_citation_questions=[r['qa_id'] for r in group if r['gold_valid_added_citations']])
    pairs=[]
    for q in jobs:
        a,b=results['control'][q],results['treatment'][q]
        raw=lambda r:next(c.get('response') for c in r['answer_calls'] if c['stage']=='eval.choice_witness')
        pairs.append(dict(qa_id=q,control_score=a['score'],treatment_score=b['score'],same_witness_response=raw(a)==raw(b)))
    result=dict(arms=arms,rows=rows,pairs=pairs,production_paths_replayed=2*len(rows),
        same_witness_response_questions=sum(r['same_witness_response'] for r in pairs),provider_calls=0,
        caveat=__doc__+' Control fill results are counterfactual hint construction only; no counterfactual answers were run.',
        auditor_sha256=digest(Path(__file__)),shared_audit_sha256=digest(root/'shared-reader-audit.json'))
    with (root/'lexical-fill-effect-audit.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','pairs')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
