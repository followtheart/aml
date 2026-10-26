"""Summarize a terminal, fully audited new-coarse insurance HTTPS trial.

Count every original attempt. Per-turn preservation compares each reference
position, not only totals. This writes evidence for a decision, never defaults.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def summarize(root):
    paths=[root/name for name in ('plan.json','fixture.json','funnel-comparison.json',
        'runtime-accounting-audit.json','selection-admission-audit.json','exit-code')]
    plan=read(paths[0]);fixture=read(paths[1]);funnel=read(paths[2]);runtime=read(paths[3]);selection=read(paths[4])
    assert plan['toggle'] in ('FINE_NEW_COARSE_INSURANCE','FINE_NEW_COARSE_MISSING_ONLY')
    assert plan['setting_values']==dict(control=False,treatment=True)
    assert funnel['settings_differences']=={plan['toggle']:[False,True]}
    assert sha(paths[1])==plan['fixture_sha256']
    assert paths[-1].read_text().strip()=='0'
    assert selection['status']=='passed' and selection['progress_snapshot'] is False
    for stage in ('funnel','runtime','selection'):
        p=root/f'final-{stage}.exit-code';assert p.read_text().strip()=='0';paths.append(p)
    for name,digest in plan['source_files'].items():assert sha(root/'source'/name)==digest,name
    wanted={(r['conversation_id'],r['qa_id']) for r in fixture['requests']}
    assert len(wanted)==plan['questions']
    rows={};answers={};arms={}
    for arm in ('control','treatment'):
        run=root/arm;manifest=read(run/'isolated-manifest.json');cost=runtime['arms'][arm];s=funnel['summaries'][arm]
        assert manifest['client_exit_code']==manifest['audit_exit_code']==0
        assert manifest['server_exit_code'] in (0,-15)
        assert cost['calls_match'] and cost['tokens_match']
        assert cost['search_budget_calls']==s['calls'] and cost['search_budget_tokens']==s['tokens']
        assert not cost['client_errors'] and not runtime['recovered_questions']
        process=Path('/proc')/str(manifest['server_pid'])/'cmdline'
        assert not process.exists() or str(root.resolve()).encode() not in process.read_bytes()
        stage_path=run/'funnel/stage-transport.json';result_path=run/'completed/data/results/personamem-v2-32k.jsonl'
        items=read(stage_path)['rows'];results=records(result_path)
        assert len(items)==len(results)==plan['questions']
        assert {(r['conversation_id'],r['qa_id']) for r in items}==wanted
        assert {(r['conversation_id'],r['qa_id']) for r in results}==wanted
        assert all(not r.get('error_stage') and r['fake'] is False and r['transport']=='https' for r in results)
        rows[arm]={r['qa_id']:r for r in items};answers[arm]={r['qa_id']:r for r in results}
        assert sum(r['score'] for r in results)==s['score']
        assert sum(bool(r['visible']['catalog']['full_turns']) for r in items)==s['visible']['catalog']
        arms[arm]=dict(questions=s['questions'],score=s['score'],accuracy=s['score']/s['questions'],
            reference_available=s['reference_present'],stage_reference_questions=s['visible'],stage_reference_turns=s['turns'],
            final_complete_reference_sets=s['complete_sets']['catalog'],search_calls=s['calls'],search_tokens=s['tokens'],
            search_degraded=s['degraded'],search_median_seconds=s['median_search_seconds'],
            reader_internal_errors=cost['reader_error_types'],
            search_runtime_error_questions=sorted({r['qa_id'] for r in cost['search_runtime_errors']},key=int),
            terminal_errors=0,server_exit_code=manifest['server_exit_code'])
        paths.extend([run/'isolated-manifest.json',stage_path,result_path,run/'server-search-debug.jsonl'])
    gained=[];lost=[];turn_losses=[];wins=[];losses=[];answer_loss_details=[]
    for qa in rows['control']:
        a,b=rows['control'][qa],rows['treatment'][qa]
        ca,cb=a['visible']['catalog'],b['visible']['catalog']
        old=[v['full'] for v in ca['turn_coverage']];new=[v['full'] for v in cb['turn_coverage']]
        assert len(old)==len(new) and a['raw_full_turns']==b['raw_full_turns']
        if any(new) and not any(old):gained.append(qa)
        if any(old) and not any(new):lost.append(qa)
        if any(x and not y for x,y in zip(old,new)):turn_losses.append(qa)
        before=answers['control'][qa];after=answers['treatment'][qa]
        if after['score']>before['score']:wins.append(qa)
        if after['score']<before['score']:
            losses.append(qa);answer_loss_details.append(dict(qa_id=qa,control_turns=old,treatment_turns=new,
                same_reference_turns=old==new,same_source_catalog=before.get('answer_source_catalog')==after.get('answer_source_catalog')))
    residual=defaultdict(list);wrong_with_reference=[];wrong_without_available=[]
    for qa,row in rows['treatment'].items():
        visible=row['visible']
        if row['raw_full_turns'] and not visible['catalog']['full_turns']:
            first=next((stage for stage in ('fused','coarse','fine','listwise_input','listwise_selected','overflow_selected','packet','catalog')
                        if stage in visible and not visible[stage]['full_turns']),'unknown')
            residual[first].append(qa)
        if answers['treatment'][qa]['score']==0:
            if visible['catalog']['full_turns']:wrong_with_reference.append(qa)
            if not row['raw_full_turns']:wrong_without_available.append(qa)
    direct={}
    nomination_paths={}
    for arm in ('control','treatment'):
        selected=[r for r in selection['rows'] if r['arm']==arm]
        assert len(selected)==plan['questions'] and {r['qa_id'] for r in selected}==set(rows[arm])
        direct[arm]=dict(summary=selection['summary'][arm],individual_turn_losses={stage:[r['qa_id'] for r in selected
            if any(b and not a for b,a in zip(r['before'][stage]['user'],r['after'][stage]['user']))] for stage in ('fine','main')})
        before={r['qa_id']:r['before'] for r in selected}
        by_search={r['search_id']:qa for qa,r in answers[arm].items()}
        nominations=[];counts=Counter()
        for t in records(root/arm/'server-search-debug.jsonl'):
            qa=by_search[t['search_id']];cascade=t['cascade']
            added=[v['candidate_id'] for v in cascade['fine']['selection']['adaptive']['additions']
                   if v['reason']=='new_coarse_core_insurance']
            assert len(added)<=1
            if not added:continue
            mid=added[0];candidate=next(c for c in t['ranked'] if c['id']==mid)
            roles={s.get('role') for s in candidate['_packet_item'].get('sources',[])
                   if s.get('content') and ' '.join(s['content'].split()) in ' '.join(candidate['_rank_text'].split())}
            path=dict(qa_id=qa,candidate_id=mid,with_user_source='user' in roles,assistant_only=roles=={'assistant'},
                already_in_baseline_fine=mid in before[qa]['fine']['ids'],
                already_in_baseline_main=mid in before[qa]['main']['ids'],
                in_actual_main=mid in cascade['listwise']['candidate_ids'],
                in_main_selected=mid in cascade['listwise']['selected_ids'],
                in_overflow_selected=mid in cascade.get('overflow',{}).get('selected_ids',[]),
                in_packet_included_ids=mid in t['coverage_manifest']['included_ids'])
            nominations.append(path);counts['used_questions']+=1
            counts.update({k:int(v) for k,v in path.items() if type(v) is bool})
        nomination_paths[arm]=dict(counts=dict(counts),rows=nominations,
            limitation='Candidate-ID routing and visible source roles only; packet membership does not prove full source text or personal entailment.')
    score_nonregression=arms['treatment']['score']>=arms['control']['score']
    reference_gain=arms['treatment']['stage_reference_questions']['catalog']>arms['control']['stage_reference_questions']['catalog']
    result=dict(status='verified_complete_trial',toggle=plan['toggle'],arms=arms,reference_gained=gained,reference_lost=lost,
        individual_reference_turn_losses=turn_losses,wins=wins,losses=losses,answer_loss_details=answer_loss_details,
        treatment_residual_first_loss=dict(residual),treatment_wrong_with_reference=wrong_with_reference,
        treatment_wrong_without_available_reference=wrong_without_available,direct_selection=direct,
        nomination_paths=nomination_paths,
        decision_evidence=dict(answer_non_regression=score_nonregression,reference_preserved=not turn_losses,
            final_reference_question_gain=reference_gain,eligible_for_joint_review=score_nonregression and not turn_losses,
            default_changed=False,needs_both_cohorts=True),
        search_token_increase_fraction=arms['treatment']['search_tokens']/arms['control']['search_tokens']-1,
        same_precoarse=funnel['same_precoarse_questions'],same_coarse=funnel['same_coarse_input_questions'],
        inputs={str(p.resolve()):sha(p) for p in [*paths,Path(__file__)]},
        limitations='Isolated localhost HTTPS with frozen upstream and local reader. One stochastic paired trial; reference transport is not entailment. Selection replay does not identify counterfactual reader outcomes.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();result=summarize(a.run)
    with (a.run/'final-evaluation.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','direct_selection','answer_loss_details','nomination_paths')}))
