"""Replay complete real CE results; compare bounded weak core reservations.

No provider calls. Reference text is used only after selection for offline
transport scoring; it is not a ranking signal or proof of semantic support.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, fine_evidence
import audit_https_funnel as audit
from evaluate_recall_context import CSV, records, reference_turns
from probe_coarse_fine_core import admit

VARIANTS = ('soft_0', 'soft_1', 'fused_in_score_core', 'main_strong_first', 'soft_1_main_strong_first')

def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.run/'exit-code').read_text().strip() != '0':
        raise ValueError('Requires a completed successful arm')
    traces = {t['search_id']: t for t in records(args.run/'server-search-debug.jsonl')}
    conversations = {r['conversation_id']: r for r in records(args.prepared)}
    csv.field_size_limit(32*1024*1024)
    with CSV.open() as source:
        gold = list(csv.DictReader(source))
    raw = {}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'], {})[row['request_id'], row['message_index']] = dict(row)
    output = []
    for result in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
        t = traces[result['search_id']]; c = t['cascade']
        by_id = {v['id']: v for v in t['ranked']}
        coarse = [by_id[mid] for mid in c['coarse']['selected_ids']]
        if c['cross_encoder']['status'] != 'ok' or any(v.get('_ce_score') is None for v in coarse):
            raise ValueError('Requires complete real CE scores')
        with patch.multiple(config, **t['versions']['settings']), patch('socket.socket.connect', side_effect=AssertionError('Offline')):
            if not config.FINE_EVIDENCE_ADAPTIVE or config.FINE_TOPIC_CONTEXT:
                raise ValueError('Requires adaptive fine without topic-context experiment')
            baseline = set(c['coarse'].get('baseline_ids', c['coarse']['selected_ids']))
            allowed = None
            pool = coarse
            if config.COARSE_EVIDENCE_EXTEND and config.FINE_COARSE_CORE_QUOTA:
                extras = sorted([v for v in coarse if v['id'] not in baseline], key=fine_evidence.key, reverse=True)
                allowed = baseline | {v['id'] for v in extras[:1]}
                pool = [v for v in coarse if v['id'] in baseline]
            fused = {v['id'] for v in sorted(pool, key=lambda v: -(v.get('_fused') or 0))[:config.CASCADE_FUSED_RESERVE]}
            kw = dict(fused_ids=fused, extra_limit=config.FINE_EXTRA_LIMIT, core_pool_ids=allowed,
                      soft_promotion_limit=1 if config.FINE_WEAK_PROMOTION_ONE else None)
            count = min(config.CASCADE_FINE_LIMIT, len(coarse))
            requirements = t['coverage_manifest']['coverage']['requirements']
            fine, selection = fine_evidence.propose(coarse, count, requirements, **kw)
            if [v['id'] for v in fine] != c['fine']['selected_ids']:
                raise ValueError('Historical fine did not reproduce: '+result['qa_id'])
            main_input, main_bytes = admit(t, fine, selection)
            if ([v['id'] for v in main_input] != c['listwise']['candidate_ids']
                    or main_bytes != c['listwise']['input_bytes']):
                raise ValueError('Historical main admission did not reproduce: '+result['qa_id'])
            candidates = dict(baseline=dict(fine=fine, main=main_input, main_bytes=main_bytes,
                core_ids=selection['adaptive']['core_ids']))
            core_pool = [v for v in coarse if allowed is None or v['id'] in allowed]
            score_core = {v['id'] for v in sorted(core_pool, key=fine_evidence.key, reverse=True)[:count]}
            for name in VARIANTS:
                new_kw = dict(kw)
                if name.startswith('soft_'):
                    new_kw['soft_promotion_limit'] = int(name.split('_')[1])
                elif name == 'fused_in_score_core':
                    new_kw['fused_ids'] = fused & score_core
                proposed, trace = fine_evidence.propose(coarse, count, requirements, **new_kw)
                submitted, size = admit(t, proposed, trace, prioritize_soft=not name.endswith('main_strong_first'))
                assert len(proposed) <= selection['adaptive']['capacity']
                candidates[name] = dict(fine=proposed, main=submitted, main_bytes=size,
                    core_ids=trace['adaptive']['core_ids'])
        conv = conversations[result['conversation_id']]
        qa = next(q for q in conv['qa'] if q['id'] == result['qa_id'])
        assert t['query'] == qa['question'] and t['options'] == qa['options']
        user_turns = reference_turns(qa, conv['source_history'], gold[int(result['qa_id'])])
        snippet = gold[int(result['qa_id'])]['related_conversation_snippet']
        try: snippet = json.loads(snippet)
        except ValueError: snippet = ast.literal_eval(snippet)
        turns = dict(user=user_turns, assistant=[audit.norm(s['content']) for s in snippet if s['role']=='assistant'])
        scored = {}
        for name, candidate in candidates.items():
            scored[name] = {k: candidate[k] for k in ('main_bytes', 'core_ids')}
            for stage in ('fine', 'main'):
                rows = candidate[stage]
                cards = [dict(s, id=f'{i}:{j}', text=s.get('content', '')) for i, v in enumerate(rows)
                         for j, s in enumerate(v['_packet_item'].get('sources', []))]
                scored[name][stage] = dict(ids=[v['id'] for v in rows])
                for role, text in turns.items():
                    coverage = audit.turn_coverage(text, cards, raw[t['user_id']], role=role)
                    scored[name][stage][role] = dict(visible=any(v['full'] for v in coverage),
                        full_turns=sum(v['full'] for v in coverage), covered_chars=sum(v['covered_chars'] for v in coverage))
        output.append(dict(qa_id=result['qa_id'], variants=scored))
    summary = dict(questions=len(output), reproduced_fine=len(output), reproduced_main=len(output), variants={})
    for name in ('baseline', *VARIANTS):
        summary['variants'][name] = {}
        for stage in ('fine', 'main'):
            summary['variants'][name][stage] = dict(candidates=sum(len(r['variants'][name][stage]['ids']) for r in output))
            for role in ('user', 'assistant'):
                values = [r['variants'][name][stage][role] for r in output]
                summary['variants'][name][stage][role] = dict(
                    **{key:sum(v[key] for v in values) for key in ('visible','full_turns','covered_chars')},
                    gained=[r['qa_id'] for r in output if r['variants'][name][stage][role]['visible'] and not r['variants']['baseline'][stage][role]['visible']],
                    lost=[r['qa_id'] for r in output if r['variants']['baseline'][stage][role]['visible'] and not r['variants'][name][stage][role]['visible']])
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary, rows=output, limitation=__doc__,
            source_trace_sha256=hashlib.sha256((args.run/'server-search-debug.jsonl').read_bytes()).hexdigest()), stream, indent=2)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('memory_system/data/prepared/personamem-v2-32k.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
