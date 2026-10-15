"""Summarize a completed audited PERMA run and its paired empty-memory control."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(run):
    manifest = json.loads((run / 'isolated-manifest.json').read_text())
    if manifest.get('client_exit_code') != 0 or manifest.get('audit_exit_code') != 0:
        raise ValueError('Memory run must finish and pass the temporal audit')
    audit = json.loads((run / 'funnel/https-summary.json').read_text())
    paths = list((run / 'completed/data/results').glob('*.jsonl'))
    if len(paths) != 1:
        raise ValueError('Ambiguous memory results')
    memory = [json.loads(line) for line in paths[0].read_text().splitlines()]
    control = [json.loads(line) for line in (run / 'no-memory.jsonl').read_text().splitlines()]
    digest = hashlib.sha256(Path(manifest['data']).read_bytes()).hexdigest()
    if digest != manifest['data_sha256'] or not memory or len(memory) != len(control):
        raise ValueError('Input or population mismatch')
    pairs = []
    for m, c in zip(memory, control):
        for key in ('conversation_id', 'qa_id', 'category', 'model', 'protocol', 'versions'):
            if m[key] != c[key]:
                raise ValueError(f'Unpaired {key}')
        if c['data_sha256'] != digest or any(r.get('fake') or r.get('error_stage') for r in (m, c)):
            raise ValueError('Fake, failed, or mismatched control')
        pairs.append(dict(conversation_id=m['conversation_id'], category=m['category'],
                          memory_prediction=m['prediction'], no_memory_prediction=c['prediction'],
                          memory_score=m['score'], no_memory_score=c['score'],
                          search_degraded=m.get('search_degraded', False)))
    def aggregate(rows):
        n = len(rows)
        a = sum(r['memory_score'] for r in rows)
        b = sum(r['no_memory_score'] for r in rows)
        return dict(count=n, memory_correct=a, no_memory_correct=b,
                    memory_accuracy=a/n, no_memory_accuracy=b/n, delta=(a-b)/n)
    report = dict(status='completed', model=memory[0]['model'], protocol=memory[0]['protocol'],
                  data_sha256=digest, audit=audit, overall=aggregate(pairs),
                  by_category={cat: aggregate([r for r in pairs if r['category'] == cat])
                               for cat in sorted({r['category'] for r in pairs})},
                  memory_enabled=aggregate([r for r in pairs if not r['category'].endswith('checkpoint-1')]),
                  questions=pairs,
                  caveat='One user, first item per domain/checkpoint stratum; not a random sample or official PERMA score. Checkpoint 1 has no history in both arms. Small-sample differences do not establish general effectiveness.')
    (run / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    print(json.dumps(summarize(p.parse_args().run), ensure_ascii=False))
