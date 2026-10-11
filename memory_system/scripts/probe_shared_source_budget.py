"""Offline lossless shared-source rendering and listwise admission diagnostic."""
import argparse
import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import cascade_rerank
from app.listwise_sources import normalized, shared_candidates, prompt_with_shared
from probe_ranking_capacity import replay








def preserved_admission(search, control, original):
    """Retain all original admissions, then append affordable rejected rows."""
    by_id = {r['id']: r for r in search['ranked']}
    positions = {mid: i for i, mid in enumerate(control['fine_ids'])}
    plan = copy.deepcopy(search['plan'])
    plan.update(_coverage_requirements=[{k: r[k] for k in ('id', 'text', 'origin')}
                                      for r in search['coverage_manifest']['coverage']['requirements']],
                _fusion_triples=search['fusion_triples'], _fusion_edges=search['fusion_edges'])
    req = SimpleNamespace(query=search['query'], options=search['options'])
    shared = prompt_with_shared(original)
    limit = search['versions']['settings']['RERANK_MAX_PROMPT_BYTES']

    def render(ids):
        rows = [by_id[mid] for mid in sorted(ids, key=positions.get)]
        plain = original(req, plan, rows)
        compact = shared(req, plan, rows)
        if len(compact.encode()) < len(plain.encode()):
            return compact, 'shared'
        return plain, 'original'

    admitted = list(control['input_ids'])
    if not admitted:
        raise ValueError('Empty baseline admission needs separate replay')
    prompt, mode = render(admitted)
    if len(prompt.encode()) > limit:
        raise ValueError('Baseline exceeds archived budget')
    rejected = []
    for mid in control['prompt_rejected']:
        trial, trial_mode = render(admitted + [mid])
        if len(trial.encode()) <= limit:
            admitted.append(mid)
            prompt, mode = trial, trial_mode
        else:
            rejected.append(mid)
    admitted.sort(key=positions.get)
    if not set(control['input_ids']) <= set(admitted):
        raise ValueError('Baseline candidate lost')
    return dict(fine_ids=control['fine_ids'], input_ids=admitted, prompt_rejected=rejected,
                prompt_bytes=len(prompt.encode()), rendering=mode)


def main(args):
    original = cascade_rerank.listwise_prompt
    records = []
    with args.search_log.open() as stream:
        for line in stream:
            search = json.loads(line)
            if not search.get('cascade'):
                continue
            settings = search['versions']['settings']
            try:
                control = replay(search, settings['CASCADE_FINE_LIMIT'], settings['CASCADE_LLM_LIMIT'])
                if control['input_ids'] != search['cascade']['listwise']['candidate_ids']:
                    raise ValueError('Original admission not reproduced')
                if args.preserve_admission:
                    shared = preserved_admission(search, control, original)
                else:
                    with patch.object(cascade_rerank, 'listwise_prompt', prompt_with_shared(original)):
                        shared = replay(search, settings['CASCADE_FINE_LIMIT'], settings['CASCADE_LLM_LIMIT'])
                records.append(dict(search_id=search['search_id'], control=control, shared=shared,
                                    gained=sorted(set(shared['input_ids']) - set(control['input_ids'])),
                                    lost=sorted(set(control['input_ids']) - set(shared['input_ids']))))
            except (ValueError, AssertionError, KeyError) as exc:
                records.append(dict(search_id=search.get('search_id'), error=str(exc)))
    report = dict(rows=records, caveat='Offline admission only, same configured byte budget. '
                  'Normalized candidate text reconstructs exactly; no ranking/reader model calls or score claim.')
    with args.output.open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(dict(count=len(records), errors=sum('error' in r for r in records),
                         gained=sum(len(r.get('gained', [])) for r in records),
                         lost=sum(len(r.get('lost', [])) for r in records))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-log', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preserve-admission', action='store_true')
    main(p.parse_args())
