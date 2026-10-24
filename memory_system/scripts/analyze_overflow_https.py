"""Offline audit of a completed frozen-ingestion HTTPS overflow ablation."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from evaluation_model_guard import REQUIRED_MODEL, check_model, check_versions, check_provider_log


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provider_stage_costs(log):
    """Summarize recorded provider attempts; reconcile with client totals below."""
    stages = {}
    for line in log.splitlines():
        if 'provider_call ' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
        stage = re.sub(r'\.batch_\d+$', '', fields['stage'])
        entry = stages.setdefault(stage, dict(calls=0, tokens=0, retry_attempts=0, statuses={}))
        entry['calls'] += 1
        entry['tokens'] += int(fields['total_tokens'])
        entry['retry_attempts'] += int(fields['attempt']) > 1
        status = fields['status']
        entry['statuses'][status] = entry['statuses'].get(status, 0) + 1
    return stages


def audit_source_merge(t):
    """Verify allocation independently from recorded source sets and rank order."""
    cfg=t['versions']['settings']
    routes=[r for r in t['routes'] if r['channel']=='source_text']
    if not routes or not cfg['SOURCE_RECALL_DIVERSITY']:
        raise ValueError('Source novelty requires source routes and source diversity')
    for route in routes:
        m=route['source_merge']; ids=m['ordered_ids']; keys=m['source_keys']; limit=m['limit']
        if len(ids)!=len(set(ids)) or set(keys)!=set(ids) or limit!=cfg['RECALL_SOURCE_LIMIT']:
            raise ValueError('Source merge inventory or capacity mismatch')
        groups={}
        seen=set(); novel=[]; deferred=[]
        for mid in ids:
            sources={tuple(k) for k in keys[mid]}
            group=('sources',tuple(sorted(sources))) if sources else ('unlinked',mid)
            groups.setdefault(group,[]).append(mid)
            if not sources or sources-seen:
                novel.append(mid);seen.update(sources)
            else:
                deferred.append(mid)
        baseline=[group[i] for i in range(max(map(len,groups.values()),default=0))
                  for group in groups.values() if i<len(group)][:limit]
        expected=(novel+deferred)[:limit] if cfg['SOURCE_RECALL_NOVELTY'] else baseline
        policy='novel_sources_first_v1' if cfg['SOURCE_RECALL_NOVELTY'] else 'distinct_source_sets_v1'
        if (m['policy']!=policy or m['baseline_ids']!=baseline or m['selected_ids']!=expected
                or [c['id'] for c in route['candidates']]!=expected):
            raise ValueError('Source route did not follow recorded allocation policy')


def analyze(directory,include_reader_errors=False):
    plan=json.loads((directory/'plan.json').read_text())
    if plan['protocol']!='frozen-search-https-v1':
        raise ValueError('Unexpected experiment protocol')
    if 'required_llm_model' in plan:
        check_model(plan['required_llm_model'], 'plan constraint')
    check_versions(plan['reader_versions'], 'reader')
    model_audit = dict(required_model=REQUIRED_MODEL, arms={})
    for name,key in [('fixture.json','fixture_sha256'),('frozen-memory.db','database_sha256')]:
        if digest(directory/name)!=plan[key]:
            raise ValueError('Frozen input changed: '+name)
    for path,expected in plan['source_files'].items():
        if digest(directory/'source'/path)!=expected:
            raise ValueError('Source snapshot changed: '+path)
    fixture=json.loads((directory/'fixture.json').read_text())
    identities={(r['conversation_id'],r['qa_id']):r for r in fixture['requests']}
    summaries,arms,settings={}, {}, {}
    for arm in ('control','treatment'):
        run=directory/arm
        manifest=json.loads((run/'isolated-manifest.json').read_text())
        check_versions(manifest['reader_versions'], arm + ' reader manifest')
        model_audit['arms'][arm] = dict(llm_dispatches=check_provider_log(
            (run/'server.log').read_text(), arm))
        if not (run/'exit-code').exists() or 'server_exit_code' not in manifest or manifest.get('audit_exit_code')!=0:
            raise ValueError('Run not complete and audited: '+arm)
        if manifest.get('client_exit_code')!=0 and not include_reader_errors:
            raise ValueError('Reader failures require --include-reader-errors')
        if manifest['initial_database_sha256']!=plan['database_sha256'] or manifest['fixture_sha256']!=plan['fixture_sha256']:
            raise ValueError('Arms did not start from identical frozen inputs')
        if digest(Path(manifest['data']))!=plan['data_sha256']:
            raise ValueError('Prepared dataset changed')
        subprocess.run([sys.executable,str(Path(__file__).with_name('audit_https_stages.py')),
                        '--run',str(run),'--prepared',manifest['data']],check=True,stdout=subprocess.DEVNULL)
        rows=records(run/'completed/data/results/personamem-v2-32k.jsonl')
        traces=records(run/'server-search-debug.jsonl')
        for trace in traces:
            check_versions(trace['versions'], arm + ' service trace')
        if traces and 'PACKET_ADJACENT_CONTEXT' in traces[0]['versions']['settings']:
            subprocess.run([sys.executable,str(Path(__file__).with_name('audit_adjacent_context.py')),
                            '--run',str(run),*(['--include-reader-errors'] if include_reader_errors else [])],check=True,stdout=subprocess.DEVNULL)
        model_audit['arms'][arm]['service_traces'] = len(traces)
        by_search={t['search_id']:t for t in traces}
        if len(rows)!=plan['questions'] or len(traces)!=len(rows) or len(by_search)!=len(rows):
            raise ValueError('Missing or extra search')
        stage=json.loads((run/'funnel/stage-transport.json').read_text())
        stage_by={(r['conversation_id'],r['qa_id']):r for r in stage['rows']}
        joined={}
        observed_settings=set()
        for r in rows:
            key=r['conversation_id'],r['qa_id']
            if key in joined:
                raise ValueError('Duplicate question')
            expected=identities[key]
            t=by_search[r['search_id']]
            if any((t.get('requested_include_sensitive') if k=='include_sensitive' else t[k])!=v
                   for k,v in expected['request'].items()) or t['anchor_time']!=expected['anchor_time'] or t['plan']!=expected['plan']:
                raise ValueError('Frozen request/plan/anchor mismatch')
            if r['fake'] is not False or t['fake'] is not False or r['transport']!='https':
                raise ValueError('Not a real HTTPS experiment')
            for path,value in t['versions']['file_hashes'].items():
                if plan['source_files'].get(path)!=value:
                    raise ValueError('Service source fingerprint mismatch')
            observed_settings.add(json.dumps(t['versions']['settings'],sort_keys=True))
            c=t['cascade']; o=c.get('overflow',{})
            if plan['toggle'] in ('FINE_SEMANTIC_RESERVE','COARSE_EVIDENCE_EXTEND','FINE_COARSE_CORE_QUOTA','FINE_WEAK_PROMOTION_ONE','SOURCE_RECALL_NOVELTY'):
                coarse=c['coarse']; cfg=t['versions']['settings']
                extra_limit=8 if cfg['COARSE_EVIDENCE_EXTEND'] else 0
                if (coarse['selected_ids']!=coarse['baseline_ids']+coarse['extra_ids']
                        or len(coarse['baseline_ids'])>cfg['CASCADE_COARSE_LIMIT']
                        or len(coarse['extra_ids'])>extra_limit
                        or len(set(coarse['selected_ids']))!=len(coarse['selected_ids'])
                        or coarse['extra_limit']!=extra_limit
                        or coarse['limit']!=cfg['CASCADE_COARSE_LIMIT']+extra_limit):
                    raise ValueError('Coarse extension violated baseline prefix or count bound')
                if any(not set(batch['candidate_ids'])<=set(coarse['selected_ids'])
                       for batch in c['cross_encoder']['batches']):
                    raise ValueError('CE dispatched a candidate outside the coarse proposal')
            cfg=t['versions']['settings']
            if 'SENSITIVE_RECALL_ENABLED' in cfg:
                requested=t.get('requested_include_sensitive'); effective=t.get('include_sensitive')
                if (type(requested) is not bool or type(effective) is not bool
                        or effective != bool(cfg['SENSITIVE_RECALL_ENABLED'] and requested)):
                    raise ValueError('Recorded sensitive recall policy does not match request/settings')
            if plan['toggle']=='SOURCE_RECALL_NOVELTY':
                audit_source_merge(t)
            adaptive=c['fine']['selection'].get('adaptive',{})
            if adaptive:
                count=min(cfg['CASCADE_FINE_LIMIT'],len(c['coarse']['selected_ids']))
                expected_soft=min(1,count) if cfg.get('FINE_WEAK_PROMOTION_ONE') else max(1,count//8) if count else 0
                policy=c['fine']['selection']['soft_reservation_policy']
                if policy['limit']!=expected_soft:
                    raise ValueError('Fine weak promotion limit does not match effective setting')
                ranks={d['id']:d['rank'] for d in c['fine']['selection']['decisions']}
                promoted=[r['candidate_id'] for r in c['fine']['selection']['reservations']
                          if r['selected'] and r['candidate_id'] in adaptive['core_ids']
                          and set(r['reasons'])<={'option_topic_coverage','fused_head'}
                          and ranks[r['candidate_id']] is not None and ranks[r['candidate_id']]>count]
                if len(promoted)>expected_soft:
                    raise ValueError('Too many weak reservations promoted into fine core')
            preserve=bool(cfg.get('FINE_COARSE_CORE_QUOTA') and cfg.get('COARSE_EVIDENCE_EXTEND')
                          and cfg.get('FINE_EVIDENCE_ADAPTIVE') and c['cross_encoder']['scored_count'])
            if bool(adaptive.get('core_policy'))!=preserve:
                raise ValueError('Fine core policy does not match effective flags and CE state')
            if preserve:
                core=adaptive['core_ids']
                baseline=set(c['coarse']['baseline_ids'])
                by_id={v['id']:v for v in t['ranked']}
                new_scored=sorted([by_id[mid] for mid in c['coarse']['extra_ids']
                                   if by_id[mid].get('_ce_score') is not None],
                                  key=lambda v:(v['_ce_score'],v.get('_fused',0)),reverse=True)
                eligible=baseline | {v['id'] for v in new_scored[:1]}
                if (adaptive['core_policy']!='restricted_core_pool_v1'
                        or set(adaptive['core_pool_ids'])!=eligible
                        or len(set(core)-baseline)>1 or adaptive['new_coarse_core_limit']!=1
                        or not set(core)<=set(adaptive['core_pool_ids'])
                        or c['fine']['selected_ids'][:len(core)]!=core
                        or len(c['fine']['selected_ids'])>len(core)+cfg['FINE_EXTRA_LIMIT']
                        or {d['id'] for d in c['fine']['selection']['decisions']}!=set(c['coarse']['selected_ids'])):
                    raise ValueError('Fine core quota or total capacity invariant failed')
            review_batches=list(c['listwise'].get('recovery',{}).get('batches',[]))
            if o.get('review'):
                review_batches.append(o['review'])
            for review_batch in review_batches:
                expected_policy=('supplied_text_relevance_v1' if t['versions']['settings'].get('LISTWISE_TRANSFORM_CONTEXT') else None)
                expected_topic=('question_topic_separate_from_premise_v1' if cfg.get('LISTWISE_TOPIC_INTEREST') else None)
                if review_batch.get('topic_interest_policy')!=expected_topic:
                    raise ValueError('Topic interest policy does not match effective setting')
                if review_batch.get('context_policy')!=expected_policy:
                    raise ValueError('Review context policy does not match effective settings')
                if 'LISTWISE_ABBREVIATION_PARTS' in cfg and cfg['LISTWISE_SPAN_REFS']:
                    expected_parts='abbreviation_aware_v1' if cfg['LISTWISE_ABBREVIATION_PARTS'] else 'punctuation_whitespace_v1'
                    if review_batch.get('segmentation_policy')!=expected_parts:
                        raise ValueError('Review segmentation does not match effective setting')
            if t['versions']['settings']['LISTWISE_OVERFLOW_REVIEW']:
                main=c['listwise']['selected_ids']; combined=o['selected_ids']; added=o['appended_ids']
                if combined!=main+added or len(added)!=len(set(added)):
                    raise ValueError('Overflow did not preserve primary ordering')
                reviewed=o.get('review',{})
                valid={j['candidate_id'] for j in reviewed.get('judgments',[]) if j['valid'] and j['useful']}
                if not set(added)<=valid or set(o['proposed_ids'])&set(c['listwise']['candidate_ids']):
                    raise ValueError('Unvalidated or primary candidate admitted as overflow')
                extended=t['versions']['settings'].get('LISTWISE_OVERFLOW_EXTEND',False)
                if not set(reviewed.get('candidate_ids',[]))<=set(o['proposed_ids']) or len(o['proposed_ids'])>(6 if extended else 4):
                    raise ValueError('Overflow exceeded candidate bound')
                if extended:
                    baseline=o['baseline_ids']; extras=o['extra_ids']
                    if len(baseline)>4 or len(extras)>2 or o['proposed_ids']!=baseline+extras or set(baseline)&set(extras):
                        raise ValueError('Extended overflow displaced original proposal or exceeded extra bound')
                prompt_limit=min(16000 if extended else 12000,t['versions']['settings']['RERANK_MAX_PROMPT_BYTES'])
                if reviewed.get('input_bytes',0)>prompt_limit or reviewed.get('prompt_limit_bytes',prompt_limit)!=prompt_limit:
                    raise ValueError('Overflow exceeded per-call prompt bound')
                if any(b.get('prompt_limit_bytes',0)>12000 for b in c['listwise'].get('recovery',{}).get('batches',[])):
                    raise ValueError('Main deletion review prompt limit changed')
            elif o:
                raise ValueError('Disabled overflow ran')
            joined[key]=(r,t,stage_by[key])
        if set(joined)!=set(identities) or len(observed_settings)!=1:
            raise ValueError('Question identity/settings mismatch')
        settings[arm]=json.loads(next(iter(observed_settings)))
        arms[arm]=joined
        times=sorted(r['search_seconds'] for r in rows)
        s=stage['summary']
        server_log=(run/'server.log').read_text()
        provider_stages=provider_stage_costs(server_log)
        overflow_calls=[line for line in server_log.splitlines()
                        if 'provider_call kind=llm stage=search.listwise.overflow ' in line]
        if len(overflow_calls)>sum(bool(t['cascade'].get('overflow',{}).get('review',{}).get('request_bound')) for t in traces):
            raise ValueError('More overflow calls than dispatched review requests')
        if any('attempt=1 retries=0 ' not in line for line in overflow_calls):
            raise ValueError('Overflow performed a transport retry')
        summaries[arm]=dict(questions=len(rows),score=sum(r['score'] for r in rows),
            packet_context=dict(
                policies=dict(Counter(t['cascade'].get('conversation_context',{}).get('policy','none') for t in traces)),
                appended_items=sum(len(t['cascade'].get('conversation_context',{}).get('appended_ids',[])) for t in traces),
                added_bytes=sum(t['cascade'].get('conversation_context',{}).get('added_bytes',0) for t in traces),
                gained_reference_questions=[r['qa_id'] for r in stage['rows'] if 'packet_base' in r['visible']
                    and r['visible']['packet']['full_turns'] and not r['visible']['packet_base']['full_turns']],
                lost_reference_questions=[r['qa_id'] for r in stage['rows'] if 'packet_base' in r['visible']
                    and r['visible']['packet_base']['full_turns'] and not r['visible']['packet']['full_turns']]),
            reference_present=s['available_reference_question_count'],
            visible=s['visible_full_turn_question_counts'],
            assistant_transport=s['assistant_reference_transport'],
            complete_sets=s['available_set_complete_question_counts'],turns=s['delivered_available_turn_counts'],
            transitions=s['transport_transitions'],
            errors=dict(Counter(r['error_stage'] for r in rows if r.get('error_stage'))),
            degraded=sum(r.get('search_degraded',True) for r in rows),
            calls=sum(r['coverage_manifest']['provider_calls'] for r in rows),
            tokens=sum(r['coverage_manifest']['provider_tokens'] for r in rows),
            provider_stages=provider_stages,
            provider_accounting_matches=dict(
                calls=sum(v['calls'] for v in provider_stages.values())==sum(r['coverage_manifest']['provider_calls'] for r in rows),
                tokens=sum(v['tokens'] for v in provider_stages.values())==sum(r['coverage_manifest']['provider_tokens'] for r in rows)),
            median_search_seconds=statistics.median(times),p95_search_seconds=times[math.ceil(.95*len(times))-1],
            coarse_proposed=sum(len(t['cascade']['coarse']['selected_ids']) for t in traces),
            coarse_extra_proposed=sum(len(t['cascade']['coarse'].get('extra_ids',[])) for t in traces),
            ce_scored=sum(t['cascade']['cross_encoder']['scored_count'] for t in traces),
            coarse_extra_stages={stage:sum(len(set(t['cascade']['coarse'].get('extra_ids',[])) & set(
                [c['id'] for c in t['ranked'] if c.get('_ce_score') is not None] if stage=='ce_scored' else
                t['cascade']['fine']['selected_ids'] if stage=='fine' else
                t['cascade']['listwise']['candidate_ids'] if stage=='main_input' else
                t['cascade']['listwise']['selected_ids'] if stage=='main_selected' else
                t['cascade'].get('overflow',{}).get('selected_ids',[]) if stage=='overflow_selected' else
                t['coverage_manifest']['included_ids'])) for t in traces)
                for stage in ('ce_scored','fine','main_input','main_selected','overflow_selected','packet')},
            ce_status=dict(Counter(t['cascade']['cross_encoder']['status'] for t in traces)),
            listwise_status=dict(Counter(t['cascade']['listwise']['status'] for t in traces)),
            main_recovery_validation_errors=dict(Counter(e for t in traces
                for j in t['cascade']['listwise'].get('recovery',{}).get('judgments',[])
                for e in j.get('validation_errors',[]))),
            main_recovery_unreviewed=sum(len(t['cascade']['listwise'].get('recovery',{}).get('unreviewed_ids',[])) for t in traces),
            review_judgment_counts={kind:dict(
                total=sum(len(t['cascade']['listwise'].get('recovery',{}).get('judgments',[]) if kind=='main'
                              else t['cascade'].get('overflow',{}).get('review',{}).get('judgments',[])) for t in traces),
                invalid=sum(not j['valid'] for t in traces for j in (
                    t['cascade']['listwise'].get('recovery',{}).get('judgments',[]) if kind=='main'
                    else t['cascade'].get('overflow',{}).get('review',{}).get('judgments',[]))),
                role_invalid=sum(any(e in ('not_user_source','not_assistant_source') for e in j.get('validation_errors',[]))
                    for t in traces for j in (t['cascade']['listwise'].get('recovery',{}).get('judgments',[]) if kind=='main'
                    else t['cascade'].get('overflow',{}).get('review',{}).get('judgments',[])))) for kind in ('main','overflow')},
            main_unreviewed_retention_violations=[t['search_id'] for t in traces
                if not set(t['cascade']['listwise'].get('recovery',{}).get('unreviewed_ids',[]))
                <= set(t['cascade']['listwise']['selected_ids'])],
            overflow_status=dict(Counter(t['cascade'].get('overflow',{}).get('status','disabled') for t in traces)),
            overflow_provider_calls=len(overflow_calls),
            overflow_max_prompt_bytes=max(t['cascade'].get('overflow',{}).get('review',{}).get('input_bytes',0) for t in traces),
            overflow_extra_proposed=sum(len(t['cascade'].get('overflow',{}).get('extra_ids',[])) for t in traces),
            overflow_extra_added=sum(len(set(t['cascade'].get('overflow',{}).get('extra_ids',[]))
                                         & set(t['cascade'].get('overflow',{}).get('appended_ids',[]))) for t in traces),
            overflow_proposed=sum(len(t['cascade'].get('overflow',{}).get('proposed_ids',[])) for t in traces),
            overflow_reviewed=sum(len(t['cascade'].get('overflow',{}).get('review',{}).get('candidate_ids',[])) for t in traces),
            overflow_added=sum(len(t['cascade'].get('overflow',{}).get('appended_ids',[])) for t in traces),
            overflow_dependency_groups=sum(len(g)>1 for t in traces
                for g in t['cascade'].get('overflow',{}).get('groups',[])),
            overflow_validation_errors=dict(Counter(e for t in traces
                for j in t['cascade'].get('overflow',{}).get('review',{}).get('judgments',[])
                for e in j.get('validation_errors',[]))))
    differences={k:[settings['control'].get(k),settings['treatment'].get(k)] for k in settings['control'].keys()|settings['treatment'].keys()
                 if settings['control'].get(k)!=settings['treatment'].get(k)}
    if plan['toggle'] not in ('PACKET_UNMATCHED_NOUN_CONTEXT','PACKET_EXCLUSIVE_SHARED_CONTEXT','PACKET_SPARSE_SOURCE_CONTEXT','PACKET_CORE_GAP_CONTEXT','FINE_SEMANTIC_RESERVE','PACKET_QUERY_CONTEXT_COMBINED','PACKET_MARGINAL_QUERY_CONTEXT','PACKET_SOURCE_FIRST_RESIDUAL','PACKET_QUESTION_CONTEXT_CE','PACKET_SOURCE_QUERY_CONTEXT','LISTWISE_OVERFLOW_REVIEW','LISTWISE_ROLE_BOUND_REFS','LISTWISE_TRANSFORM_CONTEXT','LISTWISE_TOPIC_INTEREST','PACKET_ADJACENT_CONTEXT','PACKET_FINE_CONTEXT_RESERVE','PACKET_FINE_CONTEXT_5K','PACKET_FINE_CONTEXT_LEXICAL','PACKET_FINE_CONTEXT_FILL','LISTWISE_ABBREVIATION_PARTS','LISTWISE_OVERFLOW_EXTEND','COARSE_EVIDENCE_EXTEND','FINE_COARSE_CORE_QUOTA','FINE_WEAK_PROMOTION_ONE','SOURCE_RECALL_NOVELTY'):
        raise ValueError('Unknown experimental toggle')
    if differences!={plan['toggle']:[False,True]}:
        raise ValueError('Unexpected effective setting differences')
    if plan['toggle']=='PACKET_UNMATCHED_NOUN_CONTEXT':
        from app import unmatched_noun_context
        if not all(settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
                and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] and not any(settings[arm].get(k,False)
                    for k in unmatched_noun_context.INCOMPATIBLE) for arm in settings):
            raise ValueError('Unmatched noun context requires original lexical prefix')
        if plan['source_files'].get('memory_system/data/lexicons/wordnet.zip') != unmatched_noun_context.LEXICON_SHA256:
            raise ValueError('Pinned lexical resource differs')
    if plan['toggle']=='PACKET_EXCLUSIVE_SHARED_CONTEXT' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
            and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] and not any(settings[arm].get(k,False) for k in (
                'PACKET_FINE_CONTEXT_FILL','PACKET_SOURCE_QUERY_CONTEXT','PACKET_QUESTION_CONTEXT_CE',
                'PACKET_SOURCE_FIRST_RESIDUAL','PACKET_MARGINAL_QUERY_CONTEXT','PACKET_QUERY_CONTEXT_COMBINED',
                'PACKET_CORE_GAP_CONTEXT','PACKET_SPARSE_SOURCE_CONTEXT')) for arm in settings):
        raise ValueError('Exclusive shared context requires unchanged lexical prefix and other context experiments off')
    if plan['toggle']=='PACKET_SPARSE_SOURCE_CONTEXT' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
            and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] and not any(settings[arm].get(k,False) for k in (
                'PACKET_FINE_CONTEXT_FILL','PACKET_SOURCE_QUERY_CONTEXT','PACKET_QUESTION_CONTEXT_CE',
                'PACKET_SOURCE_FIRST_RESIDUAL','PACKET_MARGINAL_QUERY_CONTEXT','PACKET_QUERY_CONTEXT_COMBINED',
                'PACKET_CORE_GAP_CONTEXT')) for arm in settings):
        raise ValueError('Sparse source requires the unchanged lexical prefix and other context experiments off')
    if plan['toggle']=='PACKET_CORE_GAP_CONTEXT' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
            and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] and not any(settings[arm].get(k,False) for k in (
                'PACKET_FINE_CONTEXT_FILL','PACKET_SOURCE_QUERY_CONTEXT','PACKET_QUESTION_CONTEXT_CE',
                'PACKET_SOURCE_FIRST_RESIDUAL','PACKET_MARGINAL_QUERY_CONTEXT','PACKET_QUERY_CONTEXT_COMBINED'))
            for arm in settings):
        raise ValueError('Core gap requires unchanged lexical prefix and separate context experiments off')
    if plan['toggle']=='PACKET_QUERY_CONTEXT_COMBINED' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
            and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] and not any(settings[arm].get(k,False) for k in (
                'PACKET_FINE_CONTEXT_FILL','PACKET_SOURCE_QUERY_CONTEXT','PACKET_QUESTION_CONTEXT_CE',
                'PACKET_SOURCE_FIRST_RESIDUAL','PACKET_MARGINAL_QUERY_CONTEXT')) for arm in settings):
        raise ValueError('Combined query context requires original 5 KiB reserve and separate experiments off')
    if plan['toggle']=='PACKET_MARGINAL_QUERY_CONTEXT' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
            and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] and not settings[arm]['PACKET_FINE_CONTEXT_FILL']
            and not settings[arm]['PACKET_SOURCE_QUERY_CONTEXT'] and not settings[arm]['PACKET_QUESTION_CONTEXT_CE']
            and not settings[arm]['PACKET_SOURCE_FIRST_RESIDUAL'] for arm in settings):
        raise ValueError('Marginal query context requires the unchanged 5 KiB lexical reserve')
    if plan['toggle']=='PACKET_SOURCE_FIRST_RESIDUAL' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
            and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] and not settings[arm]['PACKET_FINE_CONTEXT_FILL']
            and not settings[arm]['PACKET_SOURCE_QUERY_CONTEXT'] and not settings[arm]['PACKET_QUESTION_CONTEXT_CE']
            for arm in settings):
        raise ValueError('Source first residual requires the unchanged 5 KiB lexical reserve')
    if plan['toggle']=='PACKET_QUESTION_CONTEXT_CE' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
            and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] and not settings[arm]['PACKET_FINE_CONTEXT_FILL']
            and not settings[arm]['PACKET_SOURCE_QUERY_CONTEXT'] for arm in settings):
        raise ValueError('Question CE ablation requires the unchanged 5 KiB lexical reserve')
    if plan['toggle']=='PACKET_SOURCE_QUERY_CONTEXT' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
            and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] and not settings[arm]['PACKET_FINE_CONTEXT_FILL']
            for arm in settings):
        raise ValueError('Source query ablation requires the original 5 KiB lexical reserve')
    if plan['toggle']=='PACKET_FINE_CONTEXT_FILL' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE']
            and settings[arm]['PACKET_FINE_CONTEXT_LEXICAL'] for arm in settings):
        raise ValueError('Residual capacity ablation requires the 5 KiB lexical reserve in both arms')
    if plan['toggle']=='PACKET_FINE_CONTEXT_LEXICAL' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_5K'] and settings[arm]['PACKET_FINE_CONTEXT_RESERVE'] for arm in settings):
        raise ValueError('Lexical ablation requires the 5 KiB reserve in both arms')
    if plan['toggle']=='PACKET_FINE_CONTEXT_5K' and not all(
            settings[arm]['PACKET_FINE_CONTEXT_RESERVE'] for arm in settings):
        raise ValueError('Capacity ablation requires reserve enabled in both arms')
    if plan['toggle'] in ('PACKET_FINE_CONTEXT_RESERVE','PACKET_FINE_CONTEXT_5K','PACKET_FINE_CONTEXT_LEXICAL','PACKET_FINE_CONTEXT_FILL') and not all(
            settings[arm]['LISTWISE_SPAN_REFS'] and settings[arm]['LISTWISE_OVERFLOW_REVIEW']
            and not any(settings[arm].get(k,False) for k in (
                'PACKET_ADJACENT_CONTEXT','LISTWISE_TOPIC_INTEREST','LISTWISE_TRANSFORM_CONTEXT',
                'LISTWISE_ABBREVIATION_PARTS','LISTWISE_ROLE_BOUND_REFS','LISTWISE_OVERFLOW_EXTEND')) for arm in settings):
        raise ValueError('Fine reserve requires span refs/overflow and other review/context experiments off')
    if plan['toggle']=='PACKET_ADJACENT_CONTEXT' and not all(
            settings[arm]['LISTWISE_SPAN_REFS'] and settings[arm]['LISTWISE_OVERFLOW_REVIEW']
            and not any(settings[arm].get(k,False) for k in (
                'LISTWISE_TOPIC_INTEREST','LISTWISE_TRANSFORM_CONTEXT','LISTWISE_ABBREVIATION_PARTS',
                'LISTWISE_ROLE_BOUND_REFS','LISTWISE_OVERFLOW_EXTEND')) for arm in settings):
        raise ValueError('Adjacent context requires span refs/overflow and other review experiments off')
    if plan['toggle']=='SOURCE_RECALL_NOVELTY' and not all(
            settings[arm]['SOURCE_RECALL_DIVERSITY'] for arm in settings):
        raise ValueError('Source novelty ablation requires source diversity in both arms')
    if plan['toggle']=='LISTWISE_ROLE_BOUND_REFS' and not all(
            settings[arm]['LISTWISE_SPAN_REFS'] and settings[arm]['LISTWISE_OVERFLOW_REVIEW'] for arm in settings):
        raise ValueError('Role constraint ablation requires span refs and overflow in both arms')
    if plan['toggle']=='LISTWISE_TRANSFORM_CONTEXT' and not all(
            settings[arm]['LISTWISE_SPAN_REFS'] and settings[arm]['LISTWISE_OVERFLOW_REVIEW']
            and not settings[arm]['LISTWISE_ROLE_BOUND_REFS'] for arm in settings):
        raise ValueError('Transform ablation requires span refs/overflow on and role constraint off')
    if plan['toggle']=='LISTWISE_TOPIC_INTEREST' and not all(
            settings[arm]['LISTWISE_SPAN_REFS'] and settings[arm]['LISTWISE_OVERFLOW_REVIEW']
            and not settings[arm]['LISTWISE_TRANSFORM_CONTEXT']
            and not settings[arm]['LISTWISE_ABBREVIATION_PARTS']
            and not settings[arm]['LISTWISE_ROLE_BOUND_REFS'] for arm in settings):
        raise ValueError('Topic interest ablation requires span refs/overflow and other review experiments off')
    if plan['toggle']=='LISTWISE_ABBREVIATION_PARTS' and not all(
            settings[arm]['LISTWISE_SPAN_REFS'] and settings[arm]['LISTWISE_OVERFLOW_REVIEW']
            and not settings[arm]['LISTWISE_TRANSFORM_CONTEXT'] and not settings[arm]['LISTWISE_ROLE_BOUND_REFS'] for arm in settings):
        raise ValueError('Abbreviation ablation requires span refs/overflow on and other review experiments off')
    if plan['toggle']=='LISTWISE_OVERFLOW_EXTEND' and not all(
            settings[arm]['LISTWISE_SPAN_REFS'] and settings[arm]['LISTWISE_OVERFLOW_REVIEW']
            and not settings[arm]['LISTWISE_TRANSFORM_CONTEXT'] and not settings[arm]['LISTWISE_ROLE_BOUND_REFS'] for arm in settings):
        raise ValueError('Extension ablation requires span refs/overflow on, other review experiments off')
    if plan['toggle'] in ('COARSE_EVIDENCE_EXTEND','FINE_COARSE_CORE_QUOTA','FINE_WEAK_PROMOTION_ONE') and any(
            settings[arm]['LISTWISE_OVERFLOW_EXTEND'] or settings[arm]['LISTWISE_TRANSFORM_CONTEXT']
            or settings[arm]['LISTWISE_ROLE_BOUND_REFS'] for arm in settings):
        raise ValueError('Coarse ablation requires other review experiments off')
    if plan['toggle']=='FINE_COARSE_CORE_QUOTA' and not all(
            settings[arm]['COARSE_EVIDENCE_EXTEND'] and settings[arm]['FINE_EVIDENCE_ADAPTIVE']
            and not settings[arm]['FINE_TOPIC_CONTEXT'] for arm in settings):
        raise ValueError('Fine core ablation requires coarse extension/adaptive fine, topic context off')
    if plan['toggle']=='FINE_WEAK_PROMOTION_ONE' and not all(
            settings[arm]['FINE_EVIDENCE_ADAPTIVE'] and not settings[arm]['FINE_TOPIC_CONTEXT'] for arm in settings):
        raise ValueError('Weak promotion ablation requires adaptive fine and topic context off')
    def precoarse(t):
        return sorted((c['id'], c.get('_fused'), c.get('_rank_text',c.get('content','')),
                       sorted(c.get('_supported_coverage_ids',[]))) for c in t['ranked'])
    def baseline_scores(t):
        ids=set(t['cascade']['coarse'].get('baseline_ids',t['cascade']['coarse']['selected_ids']))
        return sorted((c['id'],c.get('_ce_score')) for c in t['ranked'] if c['id'] in ids)
    def baseline_score_delta(a,b):
        left,right=dict(baseline_scores(a)),dict(baseline_scores(b))
        if left.keys()!=right.keys() or any(left[k] is None or right[k] is None for k in left):
            return None
        return max((abs(left[k]-right[k]) for k in left),default=0)
    def source_merge_inputs(t):
        first=next((r for r in t['routes'] if r['channel']=='source_text' and r['round']==1),{})
        merge=first.get('source_merge',{})
        return (first.get('query'),merge.get('ordered_ids'),merge.get('source_keys'))
    def initial_route_ids(t,channel):
        return [c['id'] for r in t['routes'] if r['channel']==channel and r['round']==1 for c in r['candidates']]
    pairs=[]
    for key in identities:
        a,ta,sa=arms['control'][key];b,tb,sb=arms['treatment'][key]
        pairs.append(dict(qa_id=key[1],conversation_id=key[0],control_score=a['score'],treatment_score=b['score'],
            control_recovered=bool(a.get('reader_recovered') or a.get('transport_recovered')),
            treatment_recovered=bool(b.get('reader_recovered') or b.get('transport_recovered')),
            control_error=a.get('error_stage'),treatment_error=b.get('error_stage'),
            control_packet=bool(sa['visible']['packet']['full_turns']),treatment_packet=bool(sb['visible']['packet']['full_turns']),
            same_precoarse=precoarse(ta)==precoarse(tb),
            same_baseline_ce_scores=baseline_scores(ta)==baseline_scores(tb),
            baseline_ce_max_abs_delta=baseline_score_delta(ta,tb),
            same_coarse_input=ta['cascade']['coarse']['selected_ids']==tb['cascade']['coarse']['selected_ids'],
            same_coarse_baseline=ta['cascade']['coarse'].get('baseline_ids')==tb['cascade']['coarse'].get('baseline_ids'),
            same_fine_order=ta['cascade']['fine']['selected_ids']==tb['cascade']['fine']['selected_ids'],
            same_main_input=ta['cascade']['listwise']['candidate_ids']==tb['cascade']['listwise']['candidate_ids'],
            same_source_merge_inputs=source_merge_inputs(ta)==source_merge_inputs(tb),
            same_initial_fts=initial_route_ids(ta,'full_text')==initial_route_ids(tb,'full_text'),
            same_initial_vector=initial_route_ids(ta,'vector')==initial_route_ids(tb,'vector'),
            control_ce_status=ta['cascade']['cross_encoder']['status'],
            treatment_ce_status=tb['cascade']['cross_encoder']['status'],
            treatment_main_turns=sb['visible']['listwise_selected']['full_turns'],
            treatment_overflow_turns=sb['visible']['overflow_selected']['full_turns'],
            treatment_packet_turns=sb['visible']['packet']['full_turns']))
    baseline_transitions=Counter()
    baseline_same_questions=0
    baseline_changes=[]
    if plan['toggle']=='LISTWISE_OVERFLOW_EXTEND':
        def outcomes(o):
            return {j['candidate_id']:('invalid' if not j['valid'] else 'positive' if j['useful'] else 'negative')
                    for j in o.get('review',{}).get('judgments',[])}
        for key in identities:
            a=arms['control'][key][1]['cascade']['overflow']
            b=arms['treatment'][key][1]['cascade']['overflow']
            if a['proposed_ids']!=b['baseline_ids']:
                continue
            baseline_same_questions+=1
            ja,jb=outcomes(a),outcomes(b)
            for mid in a['proposed_ids']:
                old,new=ja.get(mid,'missing'),jb.get(mid,'missing')
                baseline_transitions[old+' -> '+new]+=1
                if old!=new:
                    baseline_changes.append(dict(qa_id=key[1],candidate_id=mid,control=old,treatment=new))
    clean=[r for r in pairs if not r['control_error'] and not r['treatment_error']]
    def same_upstream(r):
        if plan['toggle']=='SOURCE_RECALL_NOVELTY':
            return r['same_source_merge_inputs'] and r['same_initial_fts'] and r['same_initial_vector']
        if plan['toggle'] in ('COARSE_EVIDENCE_EXTEND','FINE_COARSE_CORE_QUOTA','FINE_WEAK_PROMOTION_ONE'):
            return (r['same_precoarse'] and r['same_coarse_baseline']
                    and (plan['toggle']=='COARSE_EVIDENCE_EXTEND' or r['same_coarse_input']))
        return r['same_fine_order'] and r['same_main_input']
    matched=[r for r in clean if not r['control_recovered'] and not r['treatment_recovered'] and same_upstream(r)
             and r['control_ce_status']==r['treatment_ce_status']=='ok']
    matched_summary={}
    for arm in arms:
        retained=[arms[arm][r['conversation_id'],r['qa_id']] for r in matched]
        matched_summary[arm]=dict(score=sum(r['score'] for r,_,_ in retained),
            reference_present=sum(s['raw_full_turns']>0 for _,_,s in retained),
            packet_questions=sum(s['visible']['packet']['full_turns']>0 for _,_,s in retained),
            packet_turns=sum(s['visible']['packet']['delivered_available_turns'] for _,_,s in retained),
            packet_complete_sets=sum(s['visible']['packet']['available_set_complete'] for _,_,s in retained))
    report=dict(model_audit=model_audit,summaries=summaries,settings_differences=differences,pairs=pairs,
        wins=[r['qa_id'] for r in pairs if r['treatment_score']>r['control_score']],
        losses=[r['qa_id'] for r in pairs if r['treatment_score']<r['control_score']],
        gained=[r['qa_id'] for r in pairs if r['treatment_packet'] and not r['control_packet']],
        lost=[r['qa_id'] for r in pairs if r['control_packet'] and not r['treatment_packet']],
        same_baseline_questions=baseline_same_questions,baseline_judgment_transitions=dict(baseline_transitions),
        baseline_judgment_changes=baseline_changes,
        baseline_ce_max_abs_delta=max((r['baseline_ce_max_abs_delta'] for r in pairs if r['baseline_ce_max_abs_delta'] is not None),default=None),
        same_baseline_ce_score_questions=sum(r['same_baseline_ce_scores'] for r in pairs),
        same_precoarse_questions=sum(r['same_precoarse'] for r in pairs),
        same_coarse_input_questions=sum(r['same_coarse_input'] for r in pairs),
        same_coarse_baseline_questions=sum(r['same_coarse_baseline'] for r in pairs),
        same_fine_questions=sum(r['same_fine_order'] for r in pairs),same_main_input_questions=sum(r['same_main_input'] for r in pairs),
        same_source_merge_input_questions=sum(r['same_source_merge_inputs'] for r in pairs),
        same_initial_fts_questions=sum(r['same_initial_fts'] for r in pairs),
        same_initial_vector_questions=sum(r['same_initial_vector'] for r in pairs),
        matched_error_free=dict(questions=len(clean),control_score=sum(r['control_score'] for r in clean),
                               treatment_score=sum(r['treatment_score'] for r in clean)),
        matched_upstream=dict(questions=len(matched),summaries=matched_summary,
            excluded=[r['qa_id'] for r in pairs if r not in matched],
            caveat=('Descriptive frozen-request/plan/database, same initial source-merge inputs and FTS/vector IDs, complete-CE, runtime-error-free subset. Source-route output, graph/follow-up, coarse and fine differences are the intervention path; no matching on downstream candidate pools.'
                    if plan['toggle']=='SOURCE_RECALL_NOVELTY' else
                    'Descriptive same pre-coarse pool and baseline, complete-CE, runtime-error-free subset; downstream differences are the intervention path.'
                    if plan['toggle'] in ('COARSE_EVIDENCE_EXTEND','FINE_COARSE_CORE_QUOTA','FINE_WEAK_PROMOTION_ONE') else 'Descriptive original-attempt same-fine/main-input, complete-CE subset without top-level client errors. Internally caught Search/reader errors are reported separately; this is not an error-free or unbiased repaired-run estimate.')),
        caveat='Frozen ingestion/plans/usage reduce upstream confounding; live model judgments remain stochastic. Full reference turns measure transport, not entailment. Within-treatment additions are not a separately run no-overflow counterfactual packet.')
    (directory/'funnel-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','baseline_judgment_changes')},ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--include-reader-errors',action='store_true')
    args=p.parse_args();analyze(args.run,args.include_reader_errors)
