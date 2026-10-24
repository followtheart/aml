"""Reproduce context attachment using frozen code and verify actual provenance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
from evaluation_model_guard import check_versions


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(run,allow_reader_errors=False):
    run = run.resolve()
    if (run/'exit-code').read_text().strip() != '0':
        manifest=json.loads((run/'isolated-manifest.json').read_text())
        results=records(run/'completed/data/results/personamem-v2-32k.jsonl')
        failures=[r for r in results if r.get('error_stage')]
        if (not allow_reader_errors or manifest.get('client_exit_code')!=1
                or manifest.get('audit_exit_code')!=0 or 'server_exit_code' not in manifest
                or len(results)!=manifest['questions'] or not failures
                or any(r['error_stage']!='answer' or r['score']!=0 or not r.get('search_id') for r in failures)):
            raise ValueError('Require a completed successful arm or explicit completed reader-error audit')
    snapshot = run.parent/'source'
    sys.path.insert(0,str(snapshot/'memory_system'))
    from app import config, conversation_context, evidence_packet, schemas, search_pipeline
    rows = []
    for t in records(run/'server-search-debug.jsonl'):
        check_versions(t['versions'],'service trace')
        for path, expected in t['versions']['file_hashes'].items():
            if hashlib.sha256((snapshot/path).read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source changed')
        settings = t['versions']['settings']
        reserve_enabled = settings.get('PACKET_FINE_CONTEXT_RESERVE',False)
        reserve_extended = reserve_enabled and settings.get('PACKET_FINE_CONTEXT_5K',False)
        lexical_enabled = reserve_enabled and settings.get('PACKET_FINE_CONTEXT_LEXICAL',False)
        fill_enabled = lexical_enabled and settings.get('PACKET_FINE_CONTEXT_FILL',False)
        source_enabled = (lexical_enabled and reserve_extended and not fill_enabled
                          and settings.get('PACKET_SOURCE_QUERY_CONTEXT',False))
        question_enabled = (lexical_enabled and reserve_extended and not fill_enabled
                            and not settings.get('PACKET_SOURCE_QUERY_CONTEXT',False)
                            and settings.get('PACKET_QUESTION_CONTEXT_CE',False))
        first_enabled = (lexical_enabled and reserve_extended and not fill_enabled
                         and not source_enabled and not question_enabled
                         and settings.get('PACKET_SOURCE_FIRST_RESIDUAL',False))
        marginal_enabled = (lexical_enabled and reserve_extended and not fill_enabled
                            and not source_enabled and not question_enabled and not first_enabled
                            and settings.get('PACKET_MARGINAL_QUERY_CONTEXT',False))
        combined_enabled = (lexical_enabled and reserve_extended and not fill_enabled
                            and not source_enabled and not question_enabled and not first_enabled
                            and not marginal_enabled and settings.get('PACKET_QUERY_CONTEXT_COMBINED',False))
        core_gap_enabled = (lexical_enabled and reserve_extended and not any((fill_enabled,
                            source_enabled, question_enabled, first_enabled, marginal_enabled, combined_enabled))
                            and settings.get('PACKET_CORE_GAP_CONTEXT',False))
        sparse_source_enabled = (lexical_enabled and reserve_extended and not any((fill_enabled,
                            source_enabled, question_enabled, first_enabled, marginal_enabled, combined_enabled,
                            core_gap_enabled)) and settings.get('PACKET_SPARSE_SOURCE_CONTEXT',False))
        exclusive_shared_enabled = (lexical_enabled and reserve_extended and not any((fill_enabled,
                            source_enabled, question_enabled, first_enabled, marginal_enabled, combined_enabled,
                            core_gap_enabled, sparse_source_enabled)) and settings.get('PACKET_EXCLUSIVE_SHARED_CONTEXT',False))
        unmatched_noun_enabled = (lexical_enabled and reserve_extended and not any((fill_enabled,
                            source_enabled, question_enabled, first_enabled, marginal_enabled, combined_enabled,
                            core_gap_enabled, sparse_source_enabled, exclusive_shared_enabled))
                            and settings.get('PACKET_UNMATCHED_NOUN_CONTEXT',False))
        context = t['cascade']['conversation_context']
        manifest = t['coverage_manifest']
        if (context != manifest['conversation_context']
                or context['enabled'] != (settings['PACKET_ADJACENT_CONTEXT'] or reserve_enabled)
                or context['policy'] != ('unmatched_question_noun_context_v1' if unmatched_noun_enabled else
                                        'exclusive_shared_option_context_v1' if exclusive_shared_enabled else
                                        'sparse_source_representative_context_v1' if sparse_source_enabled else
                                        'ce_core_gap_context_v1' if core_gap_enabled else 'combined_query_context_v1' if combined_enabled else
                                        'marginal_literal_query_context_v1' if marginal_enabled else
                                        'source_first_residual_v1' if first_enabled else
                                        'question_ce_residual_context_v1' if question_enabled else
                                        'source_query_context_v1' if source_enabled else
                                        'fine_residual_context_v1' if fill_enabled else
                                        'fine_user_lexical_context_v1' if lexical_enabled else
                                        'fine_original_context_5k_v1' if reserve_extended else
                                        'fine_original_context_v1' if reserve_enabled else 'adjacent_user_context_v1')):
            raise ValueError('Effective context policy differs from trace/manifest')
        before_count = len(context['base_packet_ids'])
        returned = t['returned']
        if [p['id'] for p in returned[:before_count]] != context['base_packet_ids']:
            raise ValueError('Original packet prefix displaced')
        if [p['id'] for p in returned[before_count:]] != context['appended_ids'] or len(context['appended_ids']) > (max(0,t['top_k']-before_count) if fill_enabled else 5 if combined_enabled else 4 if unmatched_noun_enabled or exclusive_shared_enabled or sparse_source_enabled or core_gap_enabled or source_enabled or question_enabled or first_enabled or marginal_enabled else 3 if lexical_enabled else 2):
            raise ValueError('Context inventory or cap differs')
        base = [dict(item,packet_hash=context['base_packet_hash']) for item in returned[:before_count]]
        if evidence_packet.digest(base) != context['base_packet_hash']:
            raise ValueError('Original packet content changed')
        if evidence_packet.digest(returned) != t['packet_hash'] or any(p['packet_hash'] != t['packet_hash'] for p in returned):
            raise ValueError('Final packet digest differs')
        base_manifest = copy.deepcopy(manifest)
        base_manifest.pop('conversation_context')
        base_manifest.update(included_ids=context['base_packet_ids'], token_upper_bound=context['base_token_upper_bound'],
                             evidence_count=manifest['evidence_count']-len(context['appended_ids']))
        by_id = {r['id']:r for r in t['ranked']}
        with patch.multiple(config,**settings):
            available = {r['id']:r for r in search_pipeline._foresight_filter(t['ranked'],t['plan'])
                         if r.get('view_status')!='stale' and '_packet_item' in r}
            fine = [available[mid] for mid in t['cascade']['fine']['selected_ids']
                    if mid in available and mid not in t['selection']['selected_ids']] if context['enabled'] else []
            champions, nomination = [], None
            if source_enabled:
                from app import source_query_context
                champions, nomination = source_query_context.nominate(available,
                    routes=t['routes'], query_specs=t['query_specs'], query=t['query'], options=t['options'] or (),
                    coarse_ids=t['cascade']['coarse']['selected_ids'], fine_ids=t['cascade']['fine']['selected_ids'],
                    selected_ids=t['selection']['selected_ids'], fine_selection=t['cascade']['fine']['selection'])
            expected, digest, expected_manifest, expected_context = conversation_context.attach(
                base,context['base_packet_hash'],base_manifest,fine,user_id=t['user_id'],top_k=t['top_k'],
                enabled=settings['PACKET_ADJACENT_CONTEXT'],
                **({'reserve_enabled':reserve_enabled} if 'PACKET_FINE_CONTEXT_RESERVE' in settings else {}),
                **({'reserve_extended':reserve_extended} if 'PACKET_FINE_CONTEXT_5K' in settings else {}),
                **({'lexical_enabled':lexical_enabled,'query':t['query'],'options':t['options']}
                   if 'PACKET_FINE_CONTEXT_LEXICAL' in settings else {}),
                **({'fill_enabled':fill_enabled} if 'PACKET_FINE_CONTEXT_FILL' in settings else {}),
                **({'source_query_enabled':source_enabled,'source_champions':champions,'source_nomination':nomination}
                   if 'PACKET_SOURCE_QUERY_CONTEXT' in settings else {}))
            if combined_enabled:
                from app import combined_query_context
                expected,digest,expected_manifest,expected_context=combined_query_context.attach(
                    base,context['base_packet_hash'],base_manifest,fine,available,
                    user_id=t['user_id'],top_k=t['top_k'],routes=t['routes'],query_specs=t['query_specs'],
                    query=t['query'],options=t['options'],coarse_ids=t['cascade']['coarse']['selected_ids'],
                    fine_ids=t['cascade']['fine']['selected_ids'],selected_ids=t['selection']['selected_ids'],
                    fine_selection=t['cascade']['fine']['selection'])
                if (len(context['combined_residual']['appended_ids'])>1 or
                        context['combined_residual']['reserve_bytes']!=5120-context['prefix_context']['added_bytes']):
                    raise ValueError('Combined residual exceeded its shared budget or seat')
            if question_enabled:
                from app import question_context_ce
                if (context['prefix_context'] != expected_context or context['prefix_packet_hash'] != digest
                        or context['prefix_packet_ids'] != [p['id'] for p in expected]):
                    raise ValueError('Question CE displaced the unchanged lexical prefix')
                with patch('socket.socket.connect',side_effect=AssertionError('Replay offline')):
                    expected, digest, expected_manifest = question_context_ce.replay(expected, digest,
                        expected_manifest, fine, context['question_ce'], user_id=t['user_id'],
                        top_k=t['top_k'], query=t['query'])
                expected_context = expected_manifest['conversation_context']
            if marginal_enabled:
                from app import marginal_query_context
                if (context['prefix_context'] != expected_context or context['prefix_packet_hash'] != digest
                        or context['prefix_packet_ids'] != [p['id'] for p in expected]):
                    raise ValueError('Marginal query context displaced the unchanged lexical prefix')
                champions,nomination=marginal_query_context.nominate(available,expected,routes=t['routes'],
                    query_specs=t['query_specs'],query=t['query'],options=t['options'],user_id=t['user_id'],
                    coarse_ids=t['cascade']['coarse']['selected_ids'],selected_ids=t['selection']['selected_ids'])
                expected,digest,expected_manifest=marginal_query_context.attach(expected,digest,expected_manifest,
                    champions,nomination,user_id=t['user_id'],top_k=t['top_k'])
                expected_context=expected_manifest['conversation_context']
            if first_enabled:
                from app import source_first_residual
                if (context['prefix_context'] != expected_context or context['prefix_packet_hash'] != digest
                        or context['prefix_packet_ids'] != [p['id'] for p in expected]):
                    raise ValueError('Source first hit displaced the unchanged lexical prefix')
                champions,nomination=source_first_residual.nominate(available,routes=t['routes'],
                    query_specs=t['query_specs'],query=t['query'],coarse_ids=t['cascade']['coarse']['selected_ids'],
                    fine_ids=t['cascade']['fine']['selected_ids'],selected_ids=t['selection']['selected_ids'])
                expected,digest,expected_manifest=source_first_residual.attach(expected,digest,expected_manifest,
                    champions,nomination,user_id=t['user_id'],top_k=t['top_k'])
                expected_context=expected_manifest['conversation_context']
            if core_gap_enabled:
                from app import core_gap_context
                if (context['prefix_context'] != expected_context or context['prefix_packet_hash'] != digest
                        or context['prefix_packet_ids'] != [p['id'] for p in expected]
                        or context['core_gap']['reserve_bytes'] != 4096
                        or context['core_gap']['added_bytes'] > 4096
                        or len(context['core_gap']['appended_ids']) > 1):
                    raise ValueError('Core gap changed prefix or exceeded its separate bound')
                champions, nomination = core_gap_context.nominate(available,expected,
                    fine_ids=t['cascade']['fine']['selected_ids'], coarse_ids=t['cascade']['coarse']['selected_ids'],
                    selection=t['cascade']['fine']['selection'], count=settings['CASCADE_FINE_LIMIT'])
                expected,digest,expected_manifest = core_gap_context.attach(expected,digest,expected_manifest,
                    champions,nomination,user_id=t['user_id'],top_k=t['top_k'])
                expected_context=expected_manifest['conversation_context']
            if sparse_source_enabled:
                from app import sparse_source_context
                if (context['prefix_context'] != expected_context or context['prefix_packet_hash'] != digest
                        or context['prefix_packet_ids'] != [p['id'] for p in expected]
                        or len(context['sparse_source']['appended_ids']) > 1):
                    raise ValueError('Sparse source changed the original prefix or seat limit')
                champions, nomination = sparse_source_context.nominate(available, expected,
                    coarse_ids=t['cascade']['coarse']['selected_ids'], fine_ids=t['cascade']['fine']['selected_ids'],
                    requirements=t['coverage_manifest']['coverage']['requirements'], user_id=t['user_id'],
                    top_k=t['top_k'], byte_budget=manifest['token_budget'],
                    remaining_bytes=5120-expected_context['added_bytes'])
                expected,digest,expected_manifest = sparse_source_context.attach(expected,digest,expected_manifest,
                    champions,nomination,user_id=t['user_id'],top_k=t['top_k'])
                expected_context=expected_manifest['conversation_context']
            if exclusive_shared_enabled:
                from app import exclusive_shared_context
                if (context['prefix_context'] != expected_context or context['prefix_packet_hash'] != digest
                        or context['prefix_packet_ids'] != [p['id'] for p in expected]
                        or len(context['exclusive_shared']['appended_ids']) > 1):
                    raise ValueError('Shared option context changed the original prefix or seat limit')
                champions, nomination = exclusive_shared_context.nominate(available, expected,
                    coarse_ids=t['cascade']['coarse']['selected_ids'], options=t['options'],
                    user_id=t['user_id'], top_k=t['top_k'], byte_budget=manifest['token_budget'],
                    remaining_bytes=5120-expected_context['added_bytes'])
                expected,digest,expected_manifest = exclusive_shared_context.attach(expected,digest,expected_manifest,
                    champions,nomination,user_id=t['user_id'],top_k=t['top_k'])
                expected_context=expected_manifest['conversation_context']
            if unmatched_noun_enabled:
                from app import unmatched_noun_context
                if (context['prefix_context'] != expected_context or context['prefix_packet_hash'] != digest
                        or context['prefix_packet_ids'] != [p['id'] for p in expected]
                        or len(context['unmatched_noun']['appended_ids']) > 1):
                    raise ValueError('Unmatched noun context changed original prefix or seat limit')
                champions, nomination = unmatched_noun_context.nominate(available, expected,
                    coarse_ids=t['cascade']['coarse']['selected_ids'], query=t['query'],
                    user_id=t['user_id'], top_k=t['top_k'], byte_budget=manifest['token_budget'],
                    remaining_bytes=5120-expected_context['added_bytes'])
                expected,digest,expected_manifest = unmatched_noun_context.attach(expected,digest,expected_manifest,
                    champions,nomination,user_id=t['user_id'],top_k=t['top_k'])
                expected_context=expected_manifest['conversation_context']
        expected = [schemas.SearchItem.model_validate(p).model_dump() for p in expected]
        if expected != returned or expected_context != context or expected_manifest != manifest or digest != t['packet_hash']:
            raise ValueError('Recorded attachment does not reproduce')
        admitted = {p['candidate_id']:p for p in context['proposed'] if p['reason']=='admitted'}
        known = {conversation_context.source_key(s) for item in base for s in item['sources']
                 if conversation_context.usable(s,t['user_id'])}
        if lexical_enabled and not source_enabled and not combined_enabled:
            from app.fine_context_reserve import propose as original_propose
            old,old_trace=original_propose(base,fine,user_id=t['user_id'],top_k=t['top_k'],
                byte_budget=manifest['token_budget'],reserve_bytes=5120 if reserve_extended else 4096)
            if ([dict(p,packet_hash=context['base_packet_hash']) for p in returned[:len(base)]]!=base
                    or [dict(p,packet_hash=None) for p in returned[:len(old)]]!=[dict(p,packet_hash=None) for p in old]
                    or context['baseline_reserve_ids']!=old_trace['appended_ids']
                    or len(context['lexical']['appended_ids'])>1):
                raise ValueError('Lexical context displaced original reserve or exceeded one extra seat')
        if source_enabled or combined_enabled:
            from app import retrieval_queries
            seat = context['source_seat']
            if (len(seat['appended_ids']) > 1 or
                    context['remainder']['reserve_bytes'] != context['reserve_bytes']-seat['added_bytes']):
                raise ValueError('Source seat exceeded count or shared byte allowance')
            nominations = {p['candidate_id']:p for p in context['source_nomination']['nominations']}
            ranks = {d['id']:d['rank'] for d in t['cascade']['fine']['selection']['decisions']}
            window = t['cascade']['fine']['selection']['soft_reservation_policy']['max_ce_rank']
            specs = [s for s in t['query_specs'] if s['origin']!='grounded_followup']
            route = next(r for r in t['routes'] if r['channel']=='source_text' and r['round']==1)
            routed = {c['id']:c for c in route['candidates']}
            for mid in seat['appended_ids']:
                if (mid not in available or mid not in t['cascade']['coarse']['selected_ids']
                        or mid in t['cascade']['fine']['selected_ids'] or mid in t['selection']['selected_ids']
                        or type(ranks.get(mid)) is not int or not 1 <= ranks[mid] <= window):
                    raise ValueError('Source seat escaped eligible coarse outsider/window')
                for key in nominations[mid]['query_indices']:
                    spec = specs[int(key)]
                    exact_question = spec['origin']=='question' and spec['text']==t['query']
                    exact_topic = (spec['origin']=='option_topic' and type(spec.get('option_index')) is int
                        and 0 <= spec['option_index'] < len(t['options'])
                        and spec['text']==retrieval_queries.topic_anchor(t['options'][spec['option_index']]))
                    if routed[mid]['_query_ranks'].get(key)!=1 or not (exact_question or exact_topic):
                        raise ValueError('Source seat is not an exact question/topic first hit')
        if fill_enabled:
            from app.fine_context_lexical import propose as lexical_propose
            prefix,prefix_trace=lexical_propose(base,fine,user_id=t['user_id'],top_k=t['top_k'],
                byte_budget=manifest['token_budget'],reserve_bytes=5120 if reserve_extended else 4096,
                query=t['query'],options=t['options'])
            if ([dict(p,packet_hash=None) for p in returned[:len(prefix)]]!=[dict(p,packet_hash=None) for p in prefix]
                    or context['prefix_reserve_ids']!=prefix_trace['appended_ids']
                    or context['prefix_reserve_bytes']!=prefix_trace['added_bytes']
                    or context['residual']['reserve_bytes']!=context['reserve_bytes']-prefix_trace['added_bytes']):
                raise ValueError('Residual capacity displaced lexical prefix or changed byte allowance')
        for item in returned[before_count:]:
            if (item['personal_evidence'] is not False or item['coverage_ids'] or item['equivalent_ids']
                    or item['memory_type']!='episode' or item['is_constraint']):
                raise ValueError('Context upgraded to personal/coverage evidence')
            if reserve_enabled:
                from app import fine_context_reserve
                supplier=by_id[item['id']]
                originals=[s for s in supplier['_packet_item']['sources'] if s.get('content')]
                if (item['sources']!=originals or item['source_count']!=len(originals)
                        or item['score_kind']!=('source_query_context' if (source_enabled or combined_enabled) and
                            item['id'] in context['source_seat']['appended_ids'] else 'fine_recall_context')
                        or item['score']!=supplier['_ce_score']
                        or not all(conversation_context.usable(s,t['user_id']) for s in originals)
                        or not any(fine_context_reserve.full_user(s) and conversation_context.source_key(s) not in known for s in originals)):
                    raise ValueError('Reserve differs from eligible scored source or adds no complete user context')
                known.update(conversation_context.source_key(s) for s in originals)
                continue
            if (item.get('score') is not None or item['score_kind']!='conversation_context'
                    or item['source_count']!=1 or len(item['sources'])!=1):
                raise ValueError('Adjacent context upgraded or source inventory differs')
            parent = item['sources'][0]
            anchor = next(p for p in base if p['id']==admitted[item['id']]['anchor_id'])
            supplier = by_id[item['id']]['_packet_item']['sources']
            if parent not in supplier or parent['role']!='user' or parent['user_id']!=t['user_id']:
                raise ValueError('Context does not belong to the eligible user source')
            if not all(s['role']=='assistant' and s['request_id']==parent['request_id']
                       and s['message_index']==parent['message_index']+1 and s['user_id']==parent['user_id']
                       for s in anchor['sources']):
                raise ValueError('Reply is not adjacent to the original user turn')
        if reserve_enabled and (context['added_bytes']>(9216 if core_gap_enabled else 5120 if reserve_extended else 4096)
                                or context['reserve_bytes']!=(9216 if core_gap_enabled else 5120 if reserve_extended else 4096)):
            raise ValueError('Recall context reserve exceeded its extra byte cap')
        used = sum(len(p['content'].encode()) for p in returned)+max(0,len(returned)-1)
        if (used!=manifest['token_upper_bound'] or used>manifest['token_budget'] or len(returned)>t['top_k']
                or manifest['included_ids']!=[p['id'] for p in returned]
                or manifest['evidence_count']!=sum(not p['is_constraint'] for p in returned)):
            raise ValueError('Final accounting or packet bounds differ')
        rows.append(dict(search_id=t['search_id'], appended_ids=context['appended_ids'],
                         added_bytes=context['added_bytes'], original_count=before_count))
    result = dict(summary=dict(questions=len(rows), appended_items=sum(len(r['appended_ids']) for r in rows),
                              added_bytes=sum(r['added_bytes'] for r in rows)), rows=rows)
    (run/'adjacent-context-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['summary']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--include-reader-errors',action='store_true')
    args=parser.parse_args();audit(args.run,args.include_reader_errors)
