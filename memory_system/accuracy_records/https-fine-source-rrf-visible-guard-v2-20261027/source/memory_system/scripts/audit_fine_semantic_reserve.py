"""Audit the live semantic-seat ablation and replay both policies per CE result.

Counterfactual fine/main coverage isolates deterministic selection on each
arm's own CE scores. It does not replay changed downstream model judgments.
Reference text transport is not semantic support or answer accuracy.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_trace(value):
    """Only trace enumeration is unordered; candidate and admission order is not."""
    value = copy.deepcopy(value)
    value['reservations'].sort(key=lambda r: r['candidate_id'])
    if 'adaptive' in value:
        value['adaptive']['path_groups'].sort()
    return value


def replay(t, enabled):
    from app import config, evidence_selection, fine_evidence
    from probe_coarse_fine_core import admit
    c = t['cascade']
    by_id = {r['id']: r for r in t['ranked']}
    coarse = [by_id[mid] for mid in c['coarse']['selected_ids']]
    before = copy.deepcopy(coarse)
    with patch.multiple(config, **t['versions']['settings']), patch('socket.socket.connect', side_effect=AssertionError('Offline audit')):
        if not config.FINE_EVIDENCE_ADAPTIVE or not c['cross_encoder']['scored_count']:
            raise ValueError('Requires a scored adaptive fine stage')
        failed = [r for r in coarse if r.get('_ce_score') is None]
        count = min(config.CASCADE_FINE_LIMIT, len(coarse))
        reserve_count = min(max(1, min(config.CASCADE_FINE_LIMIT, config.CASCADE_LLM_LIMIT)//4), len(failed))
        failed_ids = {r['id'] for r in evidence_selection.rescue(failed, reserve_count, t['query'], t['query_specs'])}
        pool = coarse
        allowed = None
        if config.COARSE_EVIDENCE_EXTEND and config.FINE_COARSE_CORE_QUOTA:
            original = set(c['coarse']['baseline_ids'])
            pool = [r for r in coarse if r['id'] in original]
            extras = sorted([r for r in coarse if r['id'] not in original and r.get('_ce_score') is not None],
                            key=fine_evidence.key, reverse=True)
            allowed = original | {r['id'] for r in extras[:1]}
            failed = [r for r in pool if r.get('_ce_score') is None]
            reserve_count = min(max(1, min(config.CASCADE_FINE_LIMIT, config.CASCADE_LLM_LIMIT)//4), len(failed))
            failed_ids = {r['id'] for r in evidence_selection.rescue(failed, reserve_count, t['query'], t['query_specs'])}
        fused = {r['id'] for r in sorted(pool, key=lambda r: -(r.get('_fused') or 0))[:max(0, config.CASCADE_FUSED_RESERVE)]} - failed_ids
        fine, selection = fine_evidence.propose(coarse, count, t['coverage_manifest']['coverage']['requirements'],
            rescue_ids=failed_ids, fused_ids=fused, extra_limit=config.FINE_EXTRA_LIMIT, core_pool_ids=allowed,
            context_queries=t['query_specs'] if config.FINE_TOPIC_CONTEXT else (),
            soft_promotion_limit=1 if config.FINE_WEAK_PROMOTION_ONE else None, semantic_reserve=enabled)
        if allowed is not None:
            selection['adaptive']['new_coarse_core_limit'] = 1
        main, size = admit(t, fine, selection)
    assert coarse == before
    return dict(fine=fine, main=main, main_bytes=size, selection=selection)


def main(args):
    root = args.run.resolve()
    plan = json.loads((root/'plan.json').read_text())
    if plan['toggle'] != 'FINE_SEMANTIC_RESERVE' or (root/'exit-code').read_text().strip() not in ('0','1'):
        raise ValueError('Require completed semantic-seat trial')
    for name, expected in plan['source_files'].items():
        if sha(root/'source'/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    sys.path.insert(0, str(root/'source/memory_system'))
    sys.path.insert(0, str(root/'source/memory_system/scripts'))
    from app import config, evidence_packet
    import audit_https_funnel as audit
    from evaluation_model_guard import check_provider_log, check_versions
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    arms = {}
    settings = {}
    for arm in ('control', 'treatment'):
        run = root/arm
        results = {r['search_id']: r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        manifest = json.loads((run/'isolated-manifest.json').read_text())
        if len(results) != plan['questions'] or manifest.get('audit_exit_code') != 0 or 'server_exit_code' not in manifest:
            raise ValueError('Incomplete arm')
        raw = {}
        for event in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'], {}).update({audit.identity(s):s for s in event['source_messages']})
        rows = []
        for t in records(run/'server-search-debug.jsonl'):
            r = results[t['search_id']]
            check_versions(t['versions'], arm)
            cfg = t['versions']['settings']
            if (cfg['FINE_SEMANTIC_RESERVE'] != (arm == 'treatment')
                    or t['cascade']['cross_encoder']['model'] != 'qwen3-rerank'
                    or t['packet_hash'] != r['packet_hash'] or evidence_packet.digest(t['returned']) != r['packet_hash']):
                raise ValueError('Wrong configuration or packet identity')
            if arm in settings and cfg != settings[arm]:
                raise ValueError('Mixed settings')
            settings[arm] = cfg
            variants = {name:replay(t, enabled) for name, enabled in [('baseline', False), ('semantic', True)]}
            active = variants['semantic' if arm == 'treatment' else 'baseline']
            if (normalized_trace(active['selection']) != normalized_trace(t['cascade']['fine']['selection']) or
                    [v['id'] for v in active['fine']] != t['cascade']['fine']['selected_ids'] or
                    [v['id'] for v in active['main']] != t['cascade']['listwise']['candidate_ids'] or
                    active['main_bytes'] != t['cascade']['listwise']['input_bytes']):
                raise ValueError('Fine/main production replay mismatch: '+r['qa_id'])
            old, new = variants['baseline'], variants['semantic']
            assert old['selection']['adaptive']['core_ids'] == new['selection']['adaptive']['core_ids']
            assert old['selection']['reservations'] == new['selection']['reservations']
            assert len(new['fine']) <= old['selection']['adaptive']['capacity']
            additions = [a for a in new['selection']['adaptive']['additions'] if a['reason']=='ce_core_insurance']
            assert len(additions) <= 1 and all(len(a['ids']) == 1 for a in additions)
            # References are used only after both selections and requests are fixed.
            snippet = gold[int(r['qa_id'])]['related_conversation_snippet']
            try:
                snippet = json.loads(snippet)
            except ValueError:
                snippet = ast.literal_eval(snippet)
            scored = {}
            for name, value in variants.items():
                scored[name] = dict(main_bytes=value['main_bytes'])
                for stage in ('fine','main'):
                    cards = [dict(s, text=s.get('content',''), id=f'{i}:{j}')
                             for i,c in enumerate(value[stage]) for j,s in enumerate(c['_packet_item'].get('sources',[]))]
                    scored[name][stage] = dict(ids=[c['id'] for c in value[stage]],
                        user=[v['full'] for v in audit.turn_coverage(
                            [audit.norm(s['content']) for s in snippet if s['role']=='user'], cards, raw[t['user_id']])])
            rows.append(dict(qa_id=r['qa_id'], score=r['score'], variants=scored,
                semantic_additions=additions, ce_status=t['cascade']['cross_encoder']['status'],
                actual_main_changed=[c['id'] for c in old['main']] != [c['id'] for c in new['main']],
                actual_packet_ids=[p['id'] for p in t['returned']]))
        if len(rows) != plan['questions'] or len({r['qa_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate or missing questions')
        unknown = []
        log = (run/'server.log').read_text()
        for line in log.splitlines():
            if 'provider_call ' in line:
                fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
                if fields.get('usage_known') == 'false':
                    unknown.append(fields)
        summary = dict(questions=len(rows), score=sum(r['score'] for r in rows),
            fine_main_reproduced=len(rows), activated=[r['qa_id'] for r in rows if r['semantic_additions']],
            partial_ce=[r['qa_id'] for r in rows if r['ce_status'] != 'ok'], variants={})
        for name in variants:
            summary['variants'][name] = {stage:dict(
                reference_questions=sum(any(r['variants'][name][stage]['user']) for r in rows),
                turns=sum(sum(r['variants'][name][stage]['user']) for r in rows),
                gained=[r['qa_id'] for r in rows if any(r['variants'][name][stage]['user']) and not any(r['variants']['baseline'][stage]['user'])],
                lost=[r['qa_id'] for r in rows if any(r['variants']['baseline'][stage]['user']) and not any(r['variants'][name][stage]['user'])],
                turn_losses=[r['qa_id'] for r in rows if any(a and not b for a,b in zip(r['variants']['baseline'][stage]['user'],r['variants'][name][stage]['user']))])
                for stage in ('fine','main')}
        arms[arm] = dict(summary=summary, rows=rows, unknown_search_usage=unknown,
                        llm_dispatches=check_provider_log(log, arm))
    differences = {k:[settings['control'].get(k),settings['treatment'].get(k)]
                   for k in settings['control'].keys() | settings['treatment'].keys()
                   if settings['control'].get(k) != settings['treatment'].get(k)}
    assert differences == {'FINE_SEMANTIC_RESERVE':[False,True]}
    output = dict(arms=arms, settings_differences=differences, source_files_verified=len(plan['source_files']),
                  auditor_sha256=sha(Path(__file__)), limitation=__doc__)
    with (root/'semantic-reserve-effect-audit.json').open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({a:v['summary'] for a,v in arms.items()}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
