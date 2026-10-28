"""Inspect delivered-reference answer gates and reproduce Q47 witness requests.

Uses recorded responses only, with network access blocked. Answer gates are
observations, not proof that the benchmark premise or model judgment is true.
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

from audit_https_funnel import reader_gate, reader_gate_detail, gold_forget_matches


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


async def main(args):
    root=args.run.resolve()
    if (root/'exit-code').read_text().strip()!='0':
        raise ValueError('Require completed successful trial')
    plan=json.loads((root/'plan.json').read_text())
    if hashlib.sha256(args.prepared.read_bytes()).hexdigest()!=plan['data_sha256']:
        raise ValueError('Prepared data differs')
    qa={(c['conversation_id'],q['id']):q for c in records(args.prepared) for q in c['qa']}
    sys.path.insert(0,str(root/'source/memory_system'))
    from app import budget, choice_witness, config
    effect=json.loads((root/'core-gap-effect-audit.json').read_text())
    q47_effect=next(r for r in effect['arms']['treatment']['rows'] if r['qa_id']=='47')
    novel=q47_effect['catalog_additions']['sources']
    arms, summaries, cases = {}, {}, {}
    for arm in ('control','treatment'):
        run=root/arm
        stage=json.loads((run/'funnel/stage-transport.json').read_text())
        stages={r['qa_id']:r for r in stage['rows']}
        output=[]
        q47=None
        for result in records(run/'completed/data/results/personamem-v2-32k.jsonl'):
            question=qa[result['conversation_id'],result['qa_id']]
            gold=question['gold_labels'][0]
            alignment=next((a for a in result.get('choice_alignment',[]) if a['letter']==gold),{})
            row=dict(qa_id=result['qa_id'],score=result['score'],gold=gold,prediction=result['prediction'],
                gold_alignment=alignment,eligible=result.get('answer_eligible_options',[]),
                gold_valid_citations=sum(c.get('valid',False) for claim in alignment.get('claims',[])
                    for c in claim.get('citations',[])),
                unresolved_support_options=result.get('answer_support_validation',{}).get('unresolved_options',[]),
                gold_forget_constraint_blocked=gold in result.get('answer_blocked_options',[]) and bool(gold_forget_matches(result,gold)),
                error_stage=result.get('error_stage'),answer_reader_mode=result.get('answer_reader_mode'),
                reference_turns=stages[result['qa_id']]['visible']['catalog']['full_turns'])
            row.update(reader_gate=reader_gate(row),reader_gate_detail=reader_gate_detail(row))
            output.append(row)
            if result['qa_id']=='47':
                q47=result
        if len(output)!=plan['questions'] or q47 is None:
            raise ValueError('Incomplete cohort')
        wrong=[r for r in output if r['score']<1 and r['reference_turns']]
        summaries[arm]=dict(questions=len(output),wrong_with_reference=len(wrong),
            gates=dict(Counter(r['reader_gate'] for r in wrong)),
            details=dict(Counter(r['reader_gate_detail'] for r in wrong)))
        arms[arm]=output
        trace=next(t for t in records(run/'server-search-debug.jsonl') if t['search_id']==q47['search_id'])
        question=qa[q47['conversation_id'],'47']
        if trace['query']!=question['question'] or trace['options']!=question['options']:
            raise ValueError('Q47 question identity differs')
        response=next(c['response'] for c in q47['answer_calls'] if c['stage']=='eval.choice_witness')
        captured=[]
        async def replay(prompt, **kwargs):
            captured.append(dict(prompt=prompt,**{k:kwargs[k] for k in ('schema','system','max_tokens')}))
            return copy.deepcopy(response)
        diagnostics={}
        with (patch.multiple(config,**trace['versions']['settings']),
                patch('socket.socket.connect',side_effect=AssertionError('Offline replay')),
                patch.object(choice_witness.llm,'complete_json',side_effect=replay),
                budget.scope(seconds=120,calls=10,tokens=500000)):
            selected=await choice_witness.select(question['question'],question['options'],
                {s['id']:s for s in q47['answer_source_catalog']},diagnostics)
        if len(captured)!=1 or diagnostics['answer_witness_retrieval']!=q47['answer_witness_retrieval']:
            raise ValueError('Witness request/validation did not reproduce')
        prompt=captured[0]['prompt']
        if hashlib.sha256(prompt.encode()).hexdigest()!=q47['answer_witness_retrieval']['prompt_sha256']:
            raise ValueError('Witness prompt hash differs')
        if arm=='treatment' and any(json.dumps(s['text'],ensure_ascii=False) not in prompt for s in novel.values()):
            raise ValueError('An appended source was omitted from witness prompt')
        cases[arm]=dict(score=q47['score'],prediction=q47['prediction'],witness_request=captured[0],
            witness_trace=q47['answer_witness_retrieval'],witness_selected=selected,
            witness_prefill=q47['answer_witness_prefill'],calls=q47['answer_calls'],
            alignment=q47['choice_alignment'],eligible=q47['answer_eligible_options'],
            scope=q47['answer_premise_scope'])
    identity=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name,value in [('reader-gate-audit.json',dict(summary=summaries,arms=arms)),
                       ('q47-reader-case.json',dict(arms=cases,new_sources=novel,
                        new_sources_in_treatment_witness_prompt=True,
                        explicit_citations_to_new_sources=q47_effect['reader_references'],provider_calls=0))]:
        with (root/name).open('x') as stream:
            json.dump(dict(value,auditor_sha256=identity,limitation=__doc__),stream,ensure_ascii=False,indent=2)
            stream.write('\n')
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,default=Path(__file__).resolve().parents[1]/'data/prepared/personamem-v2-32k.jsonl')
    asyncio.run(main(parser.parse_args()))
