"""Replay both production hint paths up to the support prompt, without providers."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice,choice_witness,config,llm,evidence_packet
from evaluation_model_guard import check_versions
from selftest_witness_lexical_fill import StopBeforeSupport


def records(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def proposal(options,sources,original):
    out=copy.deepcopy(original)
    for letter,items in choice_witness.build(options,sources).items():
        current=out.get(letter,[]);ids={w['source_id'] for w in current}
        for item in items:
            if len(current)>=3:break
            source=sources[item['source_id']]
            if (source.get('declared') or source['role'])not in ('user','persona') or item['source_id']in ids:continue
            assert item['quote'] in source['text'] and 0<len(item['quote'])<=320
            current.append(dict(item,match_method='lexical_fill'));ids.add(item['source_id'])
        if current:out[letter]=current
    return out


async def main(args):
    prepared={(c['conversation_id'],q['id']):q for c in records(args.prepared) for q in c['qa']}
    rows=[]
    for run in args.runs:
        rs=records(run/'completed/data/results/personamem-v2-32k.jsonl')
        traces={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
        stages={r['qa_id']:r for r in json.loads((run/'funnel/stage-transport.json').read_text())['rows']}
        if len(rs)!=56:raise ValueError('Require full cohort')
        for r in rs:
            t=traces[r['search_id']];check_versions(t['versions'],'frozen trace')
            qa=prepared[r['conversation_id'],r['qa_id']]
            if qa['question']!=t['query'] or qa['options']!=t['options'] or evidence_packet.digest(t['returned'])!=r['packet_hash']:
                raise ValueError('Question or packet differs')
            sources,constraints=answer_choice.build_catalog(t['returned'])
            if list(sources.values())!=r['answer_source_catalog']:raise ValueError('Catalog differs')
            old=next(c for c in r['answer_calls'] if c['stage']=='eval.choice_witness')
            support=next(c for c in r['answer_calls'] if c['stage']=='eval.choice_support')
            observed={};prompts={}
            for enabled in [False,True]:
                diag={};captured=[]
                async def model(prompt,**kwargs):
                    if kwargs['stage']=='eval.choice_support':
                        captured.append(prompt);raise StopBeforeSupport
                    if kwargs['stage']!='eval.choice_witness':raise ValueError('Unexpected provider request')
                    if hashlib.sha256(prompt.encode()).hexdigest()!=old['prompt_sha256']:raise ValueError('Witness request changed')
                    if 'response'not in old:raise llm.LLMError('Recorded provider failure')
                    return copy.deepcopy(old['response'])
                settings=dict(t['versions']['settings'],CHOICE_WITNESS_LEXICAL_FILL=enabled)
                with patch.multiple(config,**settings),patch.object(llm,'complete_json',model),\
                        patch('socket.socket.connect',side_effect=AssertionError('Offline replay')):
                    try:await answer_choice._answer(qa,t['returned'],diag)
                    except StopBeforeSupport:pass
                if len(captured)!=1:raise ValueError('Support prompt not captured')
                observed[enabled]=diag;prompts[enabled]=captured[0]
            baseline=observed[False]['answer_witness_prefill'];filled=observed[True]['answer_witness_prefill']
            if baseline!=r['answer_witness_prefill'] or hashlib.sha256(prompts[False].encode()).hexdigest()!=support['prompt_sha256']:
                raise ValueError('Historical hints or support prompt differ')
            expected=proposal(qa['options'],sources,baseline)
            if filled!=expected:raise ValueError('Independent proposal differs')
            for letter,items in baseline.items():
                if filled[letter][:len(items)]!=items:raise ValueError('Original hint displaced')
            if any(len(items)>3 for items in filled.values()):raise ValueError('Hint cap exceeded')
            old_block=choice_witness.render(baseline);new_block=choice_witness.render(filled)
            if prompts[True]!=prompts[False].replace('\n<witnesses>\n'+old_block+'\n</witnesses>',
                    '\n<witnesses>\n'+new_block+'\n</witnesses>',1):
                raise ValueError('Non-witness support prompt changed')
            refs=set().union(*(set(c['source_ids']) for c in stages[r['qa_id']]['visible']['catalog']['turn_coverage']))
            gold=qa['gold_labels'][0]
            rows.append(dict(run=str(run),qa_id=r['qa_id'],gold=gold,
                before_reference=any(w['source_id']in refs for w in baseline.get(gold,[])),
                after_reference=any(w['source_id']in refs for w in filled.get(gold,[])),
                additions=observed[True]['answer_witness_lexical_fill']['added'],
                support_bytes_growth=len(prompts[True].encode())-len(prompts[False].encode()),
                baseline_prompt_sha256=hashlib.sha256(prompts[False].encode()).hexdigest(),
                treatment_prompt_sha256=hashlib.sha256(prompts[True].encode()).hexdigest()))
    summaries={str(run):dict(questions=sum(r['run']==str(run) for r in rows),
        before_reference=sum(r['before_reference'] for r in rows if r['run']==str(run)),
        after_reference=sum(r['after_reference'] for r in rows if r['run']==str(run)),
        gained=[r['qa_id'] for r in rows if r['run']==str(run) and r['after_reference'] and not r['before_reference']],
        lost=[r['qa_id'] for r in rows if r['run']==str(run) and r['before_reference'] and not r['after_reference']],
        added_hints=sum(len(r['additions']) for r in rows if r['run']==str(run)),
        support_bytes_growth=sum(r['support_bytes_growth'] for r in rows if r['run']==str(run))) for run in args.runs}
    with args.output.open('x') as f:
        json.dump(dict(summaries=summaries,rows=rows,production_paths_verified=2*len(rows),provider_calls=0,
            caveat='Repeated snapshots of the same 56 development questions; reference-source hints are not entailment or accuracy.',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(summaries))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    asyncio.run(main(p.parse_args()))
