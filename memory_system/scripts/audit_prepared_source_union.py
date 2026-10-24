"""Can complementary prepared fragments recover a missing complete user source?

Inventory only: no provider calls, hidden-source reads during reconstruction,
candidate selection, runtime changes or answer estimates. Exact source identity,
declared offsets and matching overlaps are required. Raw observations and gold
are opened only after the complete inventory has been fixed, for verification
and transport measurement. A positive upper bound would still require an
admission policy and evaluation under actual budgets.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, evidence_packet, search_pipeline
from app.conversation_context import usable
from evaluation_model_guard import check_versions
import audit_https_funnel as transport


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity(source):
    return tuple(source.get(k) for k in
                 ('user_id', 'request_id', 'message_index', 'source_event_id', 'role'))


def reconstruct(sources):
    if not sources or len({identity(s) for s in sources}) != 1:
        raise ValueError('Require one exact original observation')
    pieces = []
    for source in sources:
        text = source['content']
        span = source.get('content_span')
        if span is None:
            start, end, size = 0, len(text), len(text)
        else:
            start, end, size = (span.get(k) for k in ('start', 'end', 'original_length'))
        if (any(type(x) is not int for x in (start, end, size))
                or not 0 <= start < end <= size or end-start != len(text)):
            raise ValueError('Invalid span')
        pieces.append((start, end, size, text))
    if len({p[2] for p in pieces}) != 1:
        raise ValueError('Conflicting original lengths')
    merged, cursor, gaps = '', 0, []
    for start, end, size, text in sorted(set(pieces)):
        if start > cursor:
            gaps.append([cursor, start])
            merged += '\0' * (start-cursor)
        overlap = max(0, min(cursor, end)-start)
        if overlap and merged[start:start+overlap] != text[:overlap]:
            raise ValueError('Conflicting original characters')
        if end > cursor:
            merged += text[max(0, cursor-start):]
            cursor = end
    size = pieces[0][2]
    if cursor < size:
        gaps.append([cursor, size])
    return dict(complete=not gaps, text=merged if not gaps else None,
                original_length=size, gaps=gaps, unique_fragments=len(set(pieces)))


def inventory(t):
    check_versions(t['versions'], 'source union trace')
    if evidence_packet.digest(t['returned']) != t['packet_hash']:
        raise ValueError('Actual packet changed')
    with patch.multiple(config, **t['versions']['settings']):
        pool = search_pipeline._foresight_filter(copy.deepcopy(t['ranked']), copy.deepcopy(t['plan']))
    coarse = set(t['cascade']['coarse']['selected_ids'])
    grouped = defaultdict(list)
    excluded = Counter()
    for candidate in pool:
        item = candidate.get('_packet_item', {})
        score = candidate.get('_ce_score')
        if (candidate['id'] not in coarse or candidate.get('view_status') == 'stale'
                or item.get('is_constraint') or type(score) not in (int, float)
                or not math.isfinite(score)):
            continue
        for source in item.get('sources', []):
            if source.get('role') != 'user':
                continue
            if not usable(source, t['user_id']):
                excluded['source_identity_or_owner'] += 1
                continue
            text = source['content']
            if text not in item.get('content', '') or text not in candidate.get('_rank_text', ''):
                excluded['not_visible'] += 1
                continue
            grouped[identity(source)].append((candidate['id'], source))
    rows = []
    for key, observations in grouped.items():
        by_candidate = defaultdict(list)
        for mid, source in observations:
            by_candidate[mid].append(source)
        combined = reconstruct([s for _, s in observations])
        single_complete = [mid for mid, sources in by_candidate.items() if reconstruct(sources)['complete']]
        delivered = [s for p in t['returned'] for s in p.get('sources', [])
                     if identity(s) == key and usable(s, t['user_id'])]
        packet_full = bool(delivered) and reconstruct(delivered)['complete']
        union_only = combined['complete'] and not single_complete
        row = dict(identity=list(key), candidate_ids=sorted(by_candidate),
                   single_complete_candidate_ids=sorted(single_complete), packet_full=packet_full,
                   union_only=union_only, missing_union_only=union_only and not packet_full,
                   **{k:v for k,v in combined.items() if k != 'text'})
        if combined['complete']:
            source = copy.deepcopy(observations[0][1])
            source.update(content=combined['text'], content_span=None)
            row['reconstructed_source'] = source
            row['text_sha256'] = hashlib.sha256(combined['text'].encode()).hexdigest()
        rows.append(row)
    return dict(user_id=t['user_id'], search_id=t['search_id'], packet_hash=t['packet_hash'],
                sources=rows, excluded=dict(excluded), actual_packet=t['returned'])


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    jobs, inputs = [], []
    with patch('socket.socket.connect', side_effect=AssertionError('Offline audit')):
        for run in args.runs:
            if (run/'exit-code').read_text().strip() != '0':
                raise ValueError('Require complete successful run')
            results = records(run/'completed/data/results/personamem-v2-32k.jsonl')
            by_search = {r['search_id']: r for r in results}
            traces = records(run/'server-search-debug.jsonl')
            if len(results) != 56 or len(traces) != 56 or {t['search_id'] for t in traces} != by_search.keys():
                raise ValueError('Require complete 56-question cohort')
            inputs.append(dict(run=str(run), trace_sha256=sha(run/'server-search-debug.jsonl'),
                               results_sha256=sha(run/'completed/data/results/personamem-v2-32k.jsonl')))
            for t in traces:
                result = by_search[t['search_id']]
                if result['packet_hash'] != t['packet_hash'] or result['user_id'] != t['user_id']:
                    raise ValueError('Search/result mismatch')
                jobs.append(dict(run=str(run), qa_id=result['qa_id'], **inventory(t)))
    # No hidden source text or labels were used above. Validate against originals now.
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    for run in args.runs:
        raw = {}
        for event in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'], {}).update({transport.identity(s):s for s in event['source_messages']})
        for job in jobs:
            if job['run'] != str(run):
                continue
            sources = raw[job['user_id']]
            for row in job['sources']:
                if row['complete']:
                    s = row['reconstructed_source']; original = sources[transport.identity(s)]
                    if s['content'] != original['content'] or s['role'] != original['role']:
                        raise ValueError('Reconstruction differs from original observation')
            snippet = gold[int(job['qa_id'])]['related_conversation_snippet']
            try:
                snippet = json.loads(snippet)
            except ValueError:
                snippet = ast.literal_eval(snippet)
            turns = [transport.norm(s['content']) for s in snippet if s['role'] == 'user']
            cards = [dict(s, text=s.get('content', ''), id=f'{i}:{j}') for i,p in enumerate(job.pop('actual_packet'))
                     for j,s in enumerate(p.get('sources', []))]
            added = [dict(s['reconstructed_source'], text=s['reconstructed_source']['content'], id=f'union:{i}')
                     for i,s in enumerate(job['sources']) if s['missing_union_only']]
            job['coverage'] = {name:[v['full'] for v in transport.turn_coverage(turns, values, sources)]
                               for name,values in [('actual',cards),('union_upper_bound',cards+added)]}
            for row in job['sources']:
                row.pop('reconstructed_source', None)
    summaries = []
    for run in args.runs:
        rows = [r for r in jobs if r['run'] == str(run)]
        sources = [s for r in rows for s in r['sources']]
        summaries.append(dict(run=str(run), questions=len(rows), source_request_pairs=len(sources),
            complete_sources=sum(s['complete'] for s in sources),
            complementary_only_sources=sum(s['union_only'] for s in sources),
            missing_complementary_only_sources=sum(s['missing_union_only'] for s in sources),
            affected_questions=[r['qa_id'] for r in rows if any(s['missing_union_only'] for s in r['sources'])],
            reference_questions={m:sum(any(r['coverage'][m]) for r in rows) for m in ('actual','union_upper_bound')},
            user_turns={m:sum(sum(r['coverage'][m]) for r in rows) for m in ('actual','union_upper_bound')},
            gained=[r['qa_id'] for r in rows if any(r['coverage']['union_upper_bound']) and not any(r['coverage']['actual'])]))
    output=dict(summary=summaries, rows=jobs, inputs=inputs, limitation=__doc__,
                script_sha256=sha(Path(__file__)), provider_calls=0)
    with args.output.open('x') as f:
        json.dump(output,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    main(parser.parse_args())
