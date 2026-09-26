"""Frozen support/witness replay with live candidate recovery and downstream gates.

Scores are diagnostic frozen-packet results, not a fresh retrieval experiment.
"""
import argparse,asyncio,copy,hashlib,importlib.util,json,sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,choice_witness,config,llm

async def main(args):
    if config.FAKE:raise RuntimeError('Requires real model')
    config.MEMORY_DEBUG_LOG=config.SEARCH_DEBUG_LOG=''
    spec=importlib.util.spec_from_file_location('app.answer_candidate',args.candidate)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    rs={r['qa_id']:r for r in map(json.loads,(args.archive/'data/results/personamem-v2-32k.jsonl').read_text().splitlines())}
    ts={t['search_id']:t for t in map(json.loads,(args.archive/'logs/search-debug.jsonl').read_text().splitlines())}
    qs={(d['conversation_id'],q['id']):q for d in map(json.loads,Path('data/prepared/personamem-v2-32k.jsonl').read_text().splitlines()) for q in d['qa']}
    real=llm.complete_json
    real_scope = budget.scope
    overrides = {}
    if args.support_overrides:
        for row in map(json.loads, args.support_overrides.read_text().splitlines()):
            if row['qa_id'] in overrides:
                raise ValueError('Only one support override per question is supported')
            if not row.get('structurally_valid'):
                raise ValueError('Support overrides must have passed structural validation')
            overrides[row['qa_id']] = row
    original_assess = gate.assess_support
    scopes = {}
    scope_targets = {}
    if args.premise_scopes:
        scope_probe = json.loads(args.premise_scopes.read_text())
        assert scope_probe['valid']
        scope_targets = {t['claim_id']: t for t in scope_probe['targets']}
        scopes = {c['claim_id']: c['personal_fact'] for c in scope_probe['response']['claims']}
        assert len(scopes) == len(scope_probe['response']['claims']) and set(scopes) == set(scope_targets)
    with args.output.open('x') as out:
        for qid in args.questions.split(','):
            r=rs[qid];q=qs[r['conversation_id'],qid]
            qa={k:q[k] for k in ['question','options','question_date','system_prompt'] if k in q}
            support=[c for c in r['answer_calls'] if c['stage'].startswith('eval.choice_support')]
            pending=[] if args.live_support else list(support);calls=[];diagnostics={}
            observed_budgets = []

            @contextmanager
            def tracked_scope(*scope_args, **scope_kwargs):
                with real_scope(*scope_args, **scope_kwargs) as usage:
                    observed_budgets.append(usage)
                    yield usage
            async def witness(question,options,sources,diag):
                diag['answer_witness_retrieval']={'status':'frozen_upstream'}
                return copy.deepcopy(r['answer_witness_prefill'])
            async def recorded(prompt,*a,**kw):
                if kw['stage'].startswith('eval.choice_support') and not args.live_support:
                    c=pending.pop(0);assert c['stage']==kw['stage']
                    span_repair = (args.live_span_repairs and kw['stage'].endswith('.repair') and
                                   'text' not in kw['schema']['$defs']['Claim']['properties'])
                    if not span_repair:
                        if 'response' not in c:raise ValueError('Recorded provider parse failure')
                        return copy.deepcopy(c['response'])
                call=dict(stage=kw['stage'],prompt=prompt);calls.append(call)
                result=await real(prompt,*a,**kw);call['response']=result;return result
            async def assess_with_override(prompt, schema, options, sources, diag):
                entries = await original_assess(prompt, schema, options, sources, diag)
                replacement = overrides.get(qid)
                if replacement is not None:
                    assert replacement['packet_hash'] == r['packet_hash']
                    letter = replacement['letter']
                    option = gate.option_map(options)[letter]
                    changed = gate.validate_assessments(replacement['response'], [option], sources,
                                                       expected={letter: option})[0]
                    assert not changed['validation_errors']
                    assert all(not c['validation_errors'] for c in changed['claims'])
                    entries = [changed if e['letter'] == letter else e for e in entries]
                    # Reflect the intervention in the final support diagnostic;
                    # preserve the original diagnostic for auditing.
                    diag['frozen_original_support_validation'] = copy.deepcopy(diag['answer_support_validation'])
                    validation = diag['answer_support_validation']
                    validation['unresolved_options'] = [x for x in validation['unresolved_options'] if x != letter]
                    if letter not in validation['accepted_options']:
                        validation['accepted_options'].append(letter)
                    validation.get('errors', {}).pop(letter, None)
                    validation['status'] = 'partial' if validation['unresolved_options'] else 'valid'
                    diag['support_intervention'] = dict(letter=letter, mode=replacement['mode'])
                if args.premise_scopes:
                    removed = []
                    consumed = set()
                    for entry in entries:
                        if entry.get('validation_status') != 'valid':
                            continue
                        retained = []
                        for index, claim in enumerate(entry['claims']):
                            cid = f"{qid}:{entry['letter']}:{index}"
                            assert cid in scopes, cid
                            assert scope_targets[cid]['claim'] == claim['text'], cid
                            assert scope_targets[cid]['option'] == entry['option'], cid
                            consumed.add(cid)
                            if scopes[cid]:
                                retained.append(claim)
                            else:
                                removed.append(dict(claim_id=cid, text=claim['text']))
                                entry.setdefault('removed_claims', []).append(dict(
                                    text=claim['text'], reason='frozen_scope_intervention', original_claim=copy.deepcopy(claim)))
                        entry['claims'] = retained
                        entry['primary_claim'] = gate._primary_index(retained, entry['option'])
                        if not retained:
                            entry['kind'] = 'generic'
                        entry['status'] = gate._option_status(entry['kind'], retained,
                                                              entry['validation_errors'], entry['option'])
                    assert consumed == {cid for cid in scopes if cid.startswith(qid + ':')}
                    diag['scope_intervention'] = dict(removed=removed, checked=len(consumed))
                return entries
            row=dict(qa_id=qid,packet_hash=r['packet_hash'],original_prediction=r['prediction'],original_score=r['score'],
                     candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                     limitation='Support and witnesses frozen; new downstream model outputs. Upstream provider costs not charged during replay.')
            if args.support_overrides:
                row['support_override_sha256'] = hashlib.sha256(args.support_overrides.read_bytes()).hexdigest()
                row['limitation'] += ' One option support may be replaced by a recorded probe; this is an intervention, not an integrated candidate.'
            if args.live_support:
                row['limitation'] = ('Only retrieval packet and witnesses frozen; support and all downstream stages live. '
                                     'Witness cost not charged. Comparison with historical outputs is not deterministic causal attribution.')
            elif args.live_span_repairs:
                row['limitation'] += ' Span-constrained repairs run live instead of replaying their original response.'
            if args.premise_scopes:
                row['scope_probe_sha256'] = hashlib.sha256(args.premise_scopes.read_bytes()).hexdigest()
                row['limitation'] += ' Recorded scope classification removes non-personal claims; its provider cost is not charged. Diagnostic intervention only.'
            try:
                with patch.object(choice_witness,'select',witness),patch.object(llm,'complete_json',recorded), \
                     patch.object(gate,'assess_support',assess_with_override), \
                     patch.object(budget,'scope',tracked_scope):
                    prediction=await gate.answer(qa,ts[r['search_id']]['returned'],diagnostics)
                assert not pending
                assert diagnostics['answer_source_catalog']==r['answer_source_catalog']
                row.update(prediction=prediction,score=float(prediction in q['gold_labels']))
            except Exception as exc:row.update(error_type=type(exc).__name__,error=str(exc)[:1200])
            row.update(diagnostics=diagnostics,live_calls=calls,
                       budget_usage=[dict(calls=b.calls, max_calls=b.max_calls, tokens=b.tokens,
                                          max_tokens=b.max_tokens, reserved_calls=b.reserved_calls,
                                          reserved_tokens=b.reserved_tokens) for b in observed_budgets])
            out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
            print(json.dumps({k:row.get(k) for k in ['qa_id','original_prediction','prediction','original_score','score','error']}
                |dict(recovery=diagnostics.get('answer_citation_recovery'))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--questions',default='15,25,4,18');ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--support-overrides',type=Path)
    ap.add_argument('--live-support',action='store_true')
    ap.add_argument('--live-span-repairs',action='store_true')
    ap.add_argument('--premise-scopes',type=Path)
    asyncio.run(main(ap.parse_args()))
