"""Offline supplemental recall retaining the original coarse pool and core eligibility.

All old coarse candidates survive. The existing one-extra core eligibility rule
chooses among old and new extra candidates; no extra core seat is introduced.
New CE pairs are reused from the completed Porter experiment. Main headroom
is tested both unchanged and charged for all incremental CE usage.
"""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, fine_evidence
from evaluate_porter_upstream_ce import identity, select, selection_code, sha
from audit_porter_upstream_ce import comparable
import audit_https_funnel as audit


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def augmented(original, revised, extra):
    result = copy.deepcopy(original)
    result['coarse'].extend(copy.deepcopy(extra))
    ids = {c['id'] for c in extra}
    for name in ('fusion_edges', 'fusion_triples'):
        old = {json.dumps(v, sort_keys=True) for v in result[name]}
        for relation in revised[name]:
            involved = {relation.get(k) for k in ('source', 'target', 'amu_id')}
            key = json.dumps(relation, sort_keys=True)
            if involved & ids and key not in old:
                result[name].append(copy.deepcopy(relation))
                old.add(key)
    meta = result['coarse_metadata']
    meta.update(selected_ids=[c['id'] for c in result['coarse']], output_count=len(result['coarse']),
                supplemental_ids=[c['id'] for c in extra], supplemental_limit=20,
                limit=meta['limit'] + 20)
    # baseline_ids and original extra_ids retain their original meanings.
    return result


def propose(t, variant, scores, original_core_pool, code):
    # Keep the original 50-member coarse baseline, then use the unchanged
    # production rule to choose one highest-scoring extra for core eligibility.
    return select(t, variant, scores, code)


def score(args):
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    raw = {}
    with sqlite3.connect((args.run / 'memory.db').resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for s in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(s['user_id'], {})[audit.identity(dict(s))] = dict(s)
    refs = {r['qa_id']: r for r in json.loads((args.run / 'funnel/stage-transport.json').read_text())['rows']}
    rows = []
    for j in map(json.loads, (args.output / 'selections.jsonl').open()):
        snippet = ast.literal_eval(gold[int(j['qa_id'])]['related_conversation_snippet'])
        coverage = {}
        for mode, variant in j['variants'].items():
            coverage[mode] = {}
            for stage in ('fine', 'main'):
                cards = [dict(s, text=s.get('content', ''), id=f'{i}:{k}') for i, c in enumerate(variant[stage])
                         for k, s in enumerate(c['_packet_item']['sources'])]
                coverage[mode][stage] = {role: [x['full'] for x in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role'] == role],
                    cards, raw[j['user_id']], role=role)] for role in ('user', 'assistant')}
        for ours, old in (('fine', 'fine'), ('main', 'listwise_input')):
            if coverage['baseline'][ours]['user'] != [x['full'] for x in refs[j['qa_id']]['visible'][old]['turn_coverage']]:
                raise ValueError('Original reference coverage differs')
        rows.append(dict(qa_id=j['qa_id'], coverage=coverage, supplemental_ids=j['supplemental_ids'],
            charged_tokens=j['charged_tokens'], counts={m: {s: len(v[s]) for s in ('fine', 'main')}
                for m, v in j['variants'].items()}))
    summary = {}
    for mode in ('baseline', 'retained_coarse', 'retained_coarse_charged'):
        summary[mode] = {}
        for stage in ('fine', 'main'):
            values = [r['coverage'][mode][stage]['user'] for r in rows]
            summary[mode][stage] = dict(reference_questions=sum(map(any, values)), user_turns=sum(map(sum, values)),
                complete_sets=sum(bool(v) and all(v) for v in values),
                gained=[r['qa_id'] for r in rows if any(r['coverage'][mode][stage]['user']) and not any(r['coverage']['baseline'][stage]['user'])],
                lost=[r['qa_id'] for r in rows if not any(r['coverage'][mode][stage]['user']) and any(r['coverage']['baseline'][stage]['user'])],
                turn_losses=[r['qa_id'] for r in rows if any(a and not b for a, b in zip(
                    r['coverage']['baseline'][stage]['user'], r['coverage'][mode][stage]['user'], strict=True))])
    return dict(summary=summary, rows=rows)


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    upstream = json.loads((args.upstream / 'result.json').read_text())
    ce = json.loads((args.ce / 'result.json').read_text())
    ce_audit = json.loads((args.ce / 'audit.json').read_text())
    ce_plan = json.loads((args.ce / 'plan.json').read_text())
    if (ce_audit['status'] != 'passed' or ce_audit['result_sha256'] != sha(args.ce / 'result.json') or
            ce['selections_sha256'] != sha(args.ce / 'selections.jsonl') or
            ce['plan_sha256'] != sha(args.ce / 'plan.json') or
            ce_plan['upstream_sha256'] != sha(args.upstream / 'selections.jsonl') or
            ce_plan['trace_sha256'] != sha(args.run / 'server-search-debug.jsonl') or
            ce_plan['database_sha256'] != sha(args.run / 'memory.db')):
        raise ValueError('Prior audited inputs changed')
    if args.ce.with_suffix('.audit-membership.exit-code').read_text().strip() != '0':
        raise ValueError('Prior audit not terminal')
    if ce['questions'] != args.expected_count or upstream['questions'] != args.expected_count:
        raise ValueError('Incomplete input cohort')
    for hashes in (upstream['source_hashes'], ce_plan['source_hashes']):
        if any(sha(Path(p)) != value for p, value in hashes.items()):
            raise ValueError('Production/replay source changed')
    traces = {t['search_id']: t for t in records(args.run / 'server-search-debug.jsonl')}
    prior_outputs = {j['qa_id']: j for j in records(args.ce / 'selections.jsonl')}
    events = {}
    for event in records(args.ce / 'live.jsonl'):
        events.setdefault(event['qa_id'], []).append(event)
    code, fragment_sha = selection_code()
    args.output.mkdir(parents=True)
    source_files = [Path(__file__), Path('scripts/evaluate_porter_upstream_ce.py'),
        Path('scripts/probe_coarse_fine_core.py'), Path('scripts/audit_porter_upstream_ce.py'),
        Path('scripts/audit_https_funnel.py'), *sorted(Path('app').glob('*.py'))]
    hashes = {str(p): sha(p) for p in source_files}
    (args.output / 'plan.json').write_text(json.dumps(dict(source_hashes=hashes, selection_ast_sha256=fragment_sha,
        run=str(args.run), upstream=str(args.upstream), ce=str(args.ce),
        upstream_sha256=sha(args.upstream / 'selections.jsonl'), ce_result_sha256=sha(args.ce / 'result.json'),
        questions=args.expected_count, limitation=__doc__), indent=2) + '\n')
    count = 0
    with patch('socket.socket.connect', side_effect=AssertionError('Offline retained coarse diagnostic')):
        for j in map(json.loads, (args.upstream / 'selections.jsonl').open()):
            t = traces[j['search_id']]
            prior = prior_outputs[j['qa_id']]
            original = j['variants']['baseline']
            revised = j['variants']['porter_supplement']
            for v in (original, revised):
                v.pop('fused')
                v.pop('prepared')
            old_ids = set(t['cascade']['coarse']['selected_ids'])
            scores = {identity(c): c.get('_ce_score') for c in t['ranked'] if c['id'] in old_ids}
            charged = 0
            for event in events.get(j['qa_id'], []):
                scores.update(zip(event['keys'], event['scores'], strict=True))
                charged += event['tokens']
            new_ids = set(j['nominated_ids'])
            historical_ids = {c['id'] for r in t['routes'] for c in r['candidates']}
            if new_ids & historical_ids or len(new_ids) > 20:
                raise ValueError('Supplement is not a bounded new route')
            extra = [c for c in revised['coarse'] if c['id'] in new_ids]
            combined = augmented(original, revised, extra)
            pool = prior['variants']['baseline']['fine_selection']['adaptive']['core_pool_ids']
            with patch.multiple(config, **t['versions']['settings']):
                base = select(t, original, scores, code)
                zero = propose(t, augmented(original, revised, []), scores, pool, code)
                if comparable(base) != comparable(prior['variants']['baseline']) or comparable(base) != comparable(zero):
                    raise ValueError('Baseline or zero-addition does not reproduce')
                proposed = propose(t, combined, scores, pool, code)
                charged_t = copy.deepcopy(t)
                headroom = charged_t['cascade']['admission_budget']['token_headroom']
                if headroom is not None:
                    charged_t['cascade']['admission_budget']['token_headroom'] = max(0, headroom - charged)
                charged_result = propose(charged_t, combined, scores, pool, code)
                core = base['fine_selection']['adaptive']['core_ids']
                for result in (proposed, charged_result):
                    adaptive = result['fine_selection']['adaptive']
                    baseline_ids = set(original['coarse_metadata']['baseline_ids'])
                    allowed = set(adaptive['core_pool_ids'])
                    if not baseline_ids <= allowed or len(allowed - baseline_ids) > 1:
                        raise ValueError('Original coarse core eligibility or one-extra rule changed')
                    if len(result['fine']) > len(core) + config.FINE_EXTRA_LIMIT:
                        raise ValueError('Fine capacity changed')
                    if len(result['main']) > config.CASCADE_LLM_LIMIT + config.FINE_EXTRA_LIMIT or result['main_bytes'] > config.RERANK_MAX_PROMPT_BYTES:
                        raise ValueError('Main capacity exceeded')
                    fine_by_id = {c['id']: c for c in result['fine']}
                    for c in result['fine']:
                        if c['id'] in new_ids:
                            group, missing = fine_evidence.closure(c, fine_by_id)
                            if missing or not group <= set(fine_by_id):
                                raise ValueError('New evidence has an incomplete dependency group')
            row = dict(qa_id=j['qa_id'], user_id=j['user_id'], search_id=j['search_id'],
                supplemental_ids=[c['id'] for c in extra], original_coarse_count=len(original['coarse']),
                combined_coarse_count=len(combined['coarse']), charged_tokens=charged,
                variants=dict(baseline=base, retained_coarse=proposed, retained_coarse_charged=charged_result))
            with (args.output / 'selections.jsonl').open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
            if count % 10 == 0:
                print(json.dumps(dict(completed=count, expected=args.expected_count)), flush=True)
    if count != args.expected_count:
        raise ValueError('Incomplete selections')
    final = score(args)
    if any(sha(Path(p)) != value for p, value in hashes.items()):
        raise ValueError('Source changed during replay')
    final.update(questions=count, reproduced_original_and_zero_addition=count, provider_calls=0,
        selection_sha256=sha(args.output / 'selections.jsonl'), plan_sha256=sha(args.output / 'plan.json'),
        gold_sha256=sha(args.gold_csv), limitation=__doc__)
    (args.output / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(final['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--ce', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-count', type=int, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    main(p.parse_args())
