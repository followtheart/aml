"""Use original observations as IDF documents for option-topic reservations.

Only the option_champions corpus changes. The production function's remaining
source is reused verbatim: thresholds, role preference, profile-field rules,
option order, CE scores and capacities remain fixed. This differs from the
previous sparse_matches source-identity probe. Offline stage transport only;
no final answer or packet improvement is inferred.
"""
import argparse
import ast
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, evidence_selection as selection
from evaluation_model_guard import check_versions
from probe_source_ce import records, select
import audit_https_funnel as audit


def source_corpus(ordered, visible):
    corpus = {}
    for row in ordered:
        for (key, role), text in visible[row['id']]:
            if key.startswith('profile-quote:'):
                candidates = [s for s in row.get('_packet_item', {}).get('sources', [])
                    if s.get('role') == role and text in selection._text(s.get('content', ''))]
                # A quote needs exact provenance; ambiguous text alone cannot
                # identify one observation. Keep its separate identity otherwise.
                candidates = [s for s in candidates if s.get('source_event_id')
                    or s.get('request_id') is not None and s.get('message_index') is not None]
                ids = {s.get('source_event_id') or f"{s['request_id']}:{s['message_index']}" for s in candidates}
                if len(ids) == 1:
                    key = ids.pop()
            corpus.setdefault((key, role), set()).update(selection._topic_concepts(text))
    return corpus


def variant():
    original = inspect.getsource(selection.option_champions)
    old = 'corpus = {mid: set().union(*(terms for _, terms in pieces)) for mid, pieces in passages.items()}'
    if original.count(old) != 1:
        raise ValueError('Production corpus expression changed')
    changed = original.replace(old, 'corpus = source_corpus(ordered, visible)')
    scope = dict(selection.__dict__, source_corpus=source_corpus)
    exec(compile(changed, '<option-original-source-frequency>', 'exec'), scope)
    return scope['option_champions'], changed


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    changed, changed_source = variant()
    jobs = []
    inputs = []
    for run in args.runs:
        if (run/'exit-code').read_text().strip() != '0':
            raise ValueError('Incomplete run')
        results = {r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        traces = records(run/'server-search-debug.jsonl')
        if len(results) != 56 or len(traces) != 56 or {t['search_id'] for t in traces} != results.keys():
            raise ValueError('Require full cohort')
        inputs.append(dict(run=str(run), trace_sha256=sha(run/'server-search-debug.jsonl')))
        for t in traces:
            check_versions(t['versions'], 'option frequency trace')
            if t['cascade']['cross_encoder']['status'] != 'ok':
                raise ValueError('Partial original CE')
            pool = {c['id']:c for c in t['ranked']}
            coarse = [pool[mid] for mid in t['cascade']['coarse']['selected_ids']]
            original = copy.deepcopy(coarse)
            with patch.multiple(config, **t['versions']['settings']), patch(
                    'socket.socket.connect', side_effect=AssertionError('Offline only')):
                baseline = select(t, coarse)
                if ([c['id'] for c in baseline['fine']] != t['cascade']['fine']['selected_ids']
                        or [c['id'] for c in baseline['main']] != t['cascade']['listwise']['candidate_ids']
                        or baseline['main_bytes'] != t['cascade']['listwise']['input_bytes']):
                    raise ValueError('Original fine/main admission did not reproduce')
                with patch.object(selection, 'option_champions', changed):
                    treatment = select(t, coarse)
                if len(treatment['fine']) > config.CASCADE_FINE_LIMIT + config.FINE_EXTRA_LIMIT:
                    raise ValueError('Fine capacity exceeded')
                if treatment['main_bytes'] > config.RERANK_MAX_PROMPT_BYTES:
                    raise ValueError('Main prompt exceeds original cap')
            if coarse != original:
                raise ValueError('Original sources or scores changed')
            jobs.append(dict(run=str(run),qa_id=results[t['search_id']]['qa_id'], user_id=t['user_id'],
                query=t['query'], variants=dict(baseline=baseline,treatment=treatment)))
    # Complete deterministic selections precede all gold reads.
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:
        gold=list(csv.DictReader(f))
    rows=[]
    for run in args.runs:
        raw={}
        for event in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'],{}).update({audit.identity(s):s for s in event['source_messages']})
        for job in jobs:
            if job['run'] != str(run):
                continue
            meta=gold[int(job['qa_id'])]
            if audit.norm(ast.literal_eval(meta['user_query'])['content']) != audit.norm(job['query']):
                raise ValueError('Question identity mismatch')
            snippet=ast.literal_eval(meta['related_conversation_snippet'])
            variants={}
            for name,value in job['variants'].items():
                v=dict(main_bytes=value['main_bytes'])
                for stage in ('fine','main'):
                    cards=[dict(s,text=s.get('content',''),id=f'{i}:{j}')
                        for i,c in enumerate(value[stage]) for j,s in enumerate(c['_packet_item'].get('sources',[]))]
                    v[stage]=dict(ids=[c['id'] for c in value[stage]],**{
                        role:[x['full'] for x in audit.turn_coverage(
                            [audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[job['user_id']],role=role)]
                        for role in ('user','assistant')})
                variants[name]=v
            rows.append(dict(run=str(run),qa_id=job['qa_id'],variants=variants))
    summaries=[]
    for run in args.runs:
        rs=[r for r in rows if r['run']==str(run)]
        summary=dict(run=str(run),questions=len(rs),stages={})
        for stage in ('fine','main'):
            summary['stages'][stage]=dict(
                reference_questions={a:sum(any(r['variants'][a][stage]['user']) for r in rs) for a in ('baseline','treatment')},
                user_turns={a:sum(sum(r['variants'][a][stage]['user']) for r in rs) for a in ('baseline','treatment')},
                gained=[r['qa_id'] for r in rs if any(r['variants']['treatment'][stage]['user']) and not any(r['variants']['baseline'][stage]['user'])],
                lost=[r['qa_id'] for r in rs if any(r['variants']['baseline'][stage]['user']) and not any(r['variants']['treatment'][stage]['user'])],
                turn_losses=[r['qa_id'] for r in rs if any(a and not b for role in ('user','assistant')
                    for a,b in zip(r['variants']['baseline'][stage][role],r['variants']['treatment'][stage][role],strict=True))],
                changed=[r['qa_id'] for r in rs if r['variants']['baseline'][stage]['ids']!=r['variants']['treatment'][stage]['ids']])
        summaries.append(summary)
    output=dict(summary=summaries,rows=rows,inputs=inputs,provider_calls=0,limitation=__doc__,
        variant_source=changed_source,script_sha256=sha(Path(__file__)),
        dependencies_sha256={str(p):sha(p) for p in (Path(selection.__file__),Path('scripts/probe_source_ce.py'),Path('scripts/probe_coarse_fine_core.py'))})
    with args.output.open('x') as f:
        json.dump(output,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    main(p.parse_args())
