"""Offline fine/main admission probe for independently rare source terms.

Only remove the query-length ratio gate from the existing sparse matcher.
Keep source identity deduplication, rarity, requirement specificity, the core,
extra capacity, source views and request budgets. References score completed
selections only. This is not a final-packet or answer evaluation.
"""
import argparse
import ast
from collections import Counter
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, fine_evidence
from app import evidence_selection as selection
import audit_https_funnel as audit
from probe_source_ce import records, select


def sparse_without_ratio(rows, requirements, visible):
    corpus = {}
    for pieces in visible.values():
        for source, text in pieces:
            corpus.setdefault(source, set()).update(selection._topic_concepts(text))
    frequency = Counter(term for terms in corpus.values() for term in terms)
    wanted = {r['id']: selection._topic_concepts(r['text']) for r in requirements}
    requirement_frequency = Counter(term for terms in wanted.values() for term in terms)
    rare_limit = max(1, len(corpus) // 10)
    out = {}
    for row in rows:
        matches = {}
        for rid, terms in wanted.items():
            for (_, role), text in visible[row['id']]:
                hits = terms & selection._topic_concepts(text)
                distinctive = {t for t in hits if len(t) >= 4 and not t.startswith('topic_family:')
                               and requirement_frequency[t] == 1 and frequency[t] <= rare_limit}
                if distinctive:
                    value = (role == 'user', False, len(hits) / len(terms))
                    matches[rid] = max(matches.get(rid, (False, False, 0)), value)
        out[row['id']] = matches
    return out


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    jobs = []
    raw = {}
    prototype = fine_evidence
    if args.bounded:
        if args.prototype is None:
            raise ValueError('--bounded requires the archived --prototype implementation')
        spec = importlib.util.spec_from_file_location('app._sparse_fine_prototype', args.prototype)
        prototype = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prototype)
    for run in args.runs:
        if (run/'exit-code').read_text().strip() != '0':
            raise ValueError('Require completed successful arm')
        results = {r['search_id']: r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        for event in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'], {}).update({audit.identity(s): s for s in event['source_messages']})
        count = 0
        for t in records(run/'server-search-debug.jsonl'):
            r = results[t['search_id']]
            if t['cascade']['cross_encoder']['status'] != 'ok':
                raise ValueError('Require complete recorded CE')
            pool = {c['id']: c for c in t['ranked']}
            coarse = [pool[mid] for mid in t['cascade']['coarse']['selected_ids']]
            before = copy.deepcopy(coarse)
            variants = {}
            traces = {}
            original = prototype.propose
            with patch.multiple(config, **t['versions']['settings']), patch(
                    'socket.socket.connect', side_effect=AssertionError('Offline')):
                for name in ('baseline', 'rare'):
                    def propose(*a, **kw):
                        if args.bounded:
                            kw['rare_term_reserve'] = name == 'rare'
                        value = original(*a, **kw)
                        traces[name] = value[1]
                        return value
                    matcher = fine_evidence.sparse_matches if name == 'baseline' or args.bounded else sparse_without_ratio
                    with patch.object(fine_evidence, 'propose', side_effect=propose), patch.object(
                            fine_evidence, 'sparse_matches', side_effect=matcher):
                        variants[name] = select(t, coarse)
                base = variants['baseline']
                assert [c['id'] for c in base['fine']] == t['cascade']['fine']['selected_ids']
                assert [c['id'] for c in base['main']] == t['cascade']['listwise']['candidate_ids']
                assert base['main_bytes'] == t['cascade']['listwise']['input_bytes']
                assert traces['rare']['adaptive']['core_ids'] == traces['baseline']['adaptive']['core_ids']
                assert len(variants['rare']['fine']) <= traces['baseline']['adaptive']['capacity']
                assert variants['rare']['main_bytes'] <= config.RERANK_MAX_PROMPT_BYTES
            assert coarse == before
            jobs.append(dict(run=str(run), qa_id=r['qa_id'], user_id=t['user_id'], query=t['query'],
                             variants=variants, traces=traces))
            count += 1
        assert count == len(results)
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    rows = []
    for job in jobs:
        meta = gold[int(job['qa_id'])]
        assert audit.norm(ast.literal_eval(meta['user_query'])['content']) == audit.norm(job['query'])
        snippet = ast.literal_eval(meta['related_conversation_snippet'])
        variants = {}
        for name, value in job['variants'].items():
            entry = dict(main_bytes=value['main_bytes'], selection=job['traces'][name])
            for stage in ('fine', 'main'):
                cards = [dict(s, text=s.get('content', ''), id=f'{i}:{j}')
                         for i, c in enumerate(value[stage]) for j, s in enumerate(c['_packet_item']['sources'])]
                entry[stage] = dict(ids=[c['id'] for c in value[stage]], **{
                    role: [v['full'] for v in audit.turn_coverage(
                        [audit.norm(s['content']) for s in snippet if s['role'] == role],
                        cards, raw[job['user_id']], role=role)] for role in ('user', 'assistant')})
            variants[name] = entry
        rows.append(dict(run=job['run'], qa_id=job['qa_id'], variants=variants))
    summary = {}
    for run in map(str, args.runs):
        subset = [r for r in rows if r['run'] == run]
        summary[run] = dict(questions=len(subset), variants={})
        for name in ('baseline', 'rare'):
            summary[run]['variants'][name] = {stage: dict(
                reference_questions=sum(any(r['variants'][name][stage]['user']) for r in subset),
                user_turns=sum(sum(r['variants'][name][stage]['user']) for r in subset),
                gained=[r['qa_id'] for r in subset if any(r['variants'][name][stage]['user']) and not any(r['variants']['baseline'][stage]['user'])],
                lost=[r['qa_id'] for r in subset if any(r['variants']['baseline'][stage]['user']) and not any(r['variants'][name][stage]['user'])],
                turn_losses=[r['qa_id'] for r in subset if any(a and not b for a, b in zip(r['variants']['baseline'][stage]['user'], r['variants'][name][stage]['user']))],
                changed=[r['qa_id'] for r in subset if r['variants'][name][stage]['ids'] != r['variants']['baseline'][stage]['ids']]) for stage in ('fine', 'main')}
    args.output.write_text(json.dumps(dict(summary=summary, rows=rows, limitation=__doc__,
        bounded_rare_term_reserve=args.bounded,
        implementation_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(prototype.__file__), Path(__file__).resolve().parents[1]/'app/fine_sparse_reserve.py')},
        source_trace_hashes={str(run): hashlib.sha256((run/'server-search-debug.jsonl').read_bytes()).hexdigest() for run in args.runs},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bounded', action='store_true', help='Use one novel-user-term seat; retain original sparse matcher')
    p.add_argument('--prototype', type=Path, help='Archived app/fine_evidence prototype for rejected bounded mode')
    p.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
