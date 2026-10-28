"""Audit fixed-prefix and full-cohort scope diagnostics, never HTTPS scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluation_model_guard import check_provider_log
from replay_query_context import records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger(path):
    content = path.read_text()
    check_provider_log(content, str(path))
    events = [dict(part.split('=', 1) for part in line.split() if '=' in part)
              for line in content.splitlines() if 'provider_call ' in line]
    if any(e.get('usage_known') == 'false' for e in events):
        raise ValueError('Unknown usage must be audited separately')
    return dict(calls=len(events), tokens=sum(int(e.get('total_tokens', 0)) for e in events),
                statuses=dict(Counter(e['status'] for e in events)))


def main(root):
    fixed = records(root / 'q11-scope-fixed-prefix.jsonl')
    plan = fixed[0]
    if digest(root / 'q11-scope-fixed-prefix-script.py') != plan['source_sha256']:
        raise ValueError('Original Q11 runner hash differs')
    rows = [r for r in fixed if r['event'] == 'result']
    if len(rows) != 12 or len({(r['archive_arm'], r['enabled'], r['repetition']) for r in rows}) != 12:
        raise ValueError('Incomplete Q11 repeats')
    fixed_summary = {}
    for arm in ('control', 'treatment'):
        subset = [r for r in rows if r['archive_arm'] == arm]
        if len({json.dumps(r['scope_inputs'], sort_keys=True) for r in subset}) != 1:
            raise ValueError('Pre-scope inputs varied within fixed stratum')
        if len({json.dumps(r['fixed_calls'], sort_keys=True) for r in subset}) != 1:
            raise ValueError('Prefix varied within fixed stratum')
        if len({r['packet_hash'] for r in subset}) != 1:
            raise ValueError('Packet varied within fixed stratum')
        a, b = [next(r['actual_settings'] for r in subset if r['enabled'] == mode) for mode in (False, True)]
        if {k: [a[k], b[k]] for k in a if a[k] != b[k]} != {'CHOICE_CITED_SCOPE': [False, True]}:
            raise ValueError('Unexpected runtime setting differences')
        fixed_summary[arm] = {str(mode): dict(repeats=3,
            predictions=[r['prediction'] for r in subset if r['enabled'] == mode],
            score=sum(r['score'] for r in subset if r['enabled'] == mode)) for mode in (False, True)}
    preflight = records(root / 'all-scope-preflight-verified.jsonl')
    if len(preflight) != 113:
        raise ValueError('Expected 112 offline preflights')
    inventory = records(root / 'cited-scope-inventory.jsonl')
    if digest(root / 'all-scope-preflight-verified.jsonl') != inventory[0]['preflight_sha256']:
        raise ValueError('Preflight input changed')
    live = inventory[1:]
    if len(live) != inventory[0]['expected_rows'] or len({(r['archive_arm'], r['qa_id'], r['enabled']) for r in live}) != len(live):
        raise ValueError('Incomplete inventory')
    removed = []
    for r in live:
        if not r['enabled']:
            continue
        for entry in r['entries']:
            cited = [c for c in entry.get('removed_claims', []) if c.get('original_claim', {}).get('citations')]
            if cited:
                removed.append(dict(archive_arm=r['archive_arm'], qa_id=r['qa_id'], letter=entry['letter'],
                    removed_cited_claims=[c['text'] for c in cited],
                    retained_claims=[c['text'] for c in entry['claims']]))
    costs = {name: ledger(root / (name + '.provider.log'))
             for name in ('q11-scope-fixed-prefix', 'cited-scope-inventory')}
    if costs['q11-scope-fixed-prefix']['calls'] != sum(len(r['live_requests']) for r in rows):
        raise ValueError('Q11 usage ledger differs')
    if costs['cited-scope-inventory']['calls'] != sum(r['calls'] for r in live) or costs['cited-scope-inventory']['tokens'] != sum(r['tokens'] for r in live):
        raise ValueError('Inventory usage ledger differs')
    summary = dict(q11_fixed_prefix=fixed_summary, offline_reproductions=112,
        inventory=dict(affected=inventory[0]['affected'], paired_rows=len(live),
            statuses=dict(Counter(r['diagnostics'].get('answer_premise_scope', {}).get('status', 'no_targets') for r in live)),
            removed_cited=removed), recorded_provider_usage=costs,
        limitation='Selected Q11 repeats and scope-only full-input inventory; not final answer accuracy, not HTTPS. Frozen prefix token/latency not simulated. No blanket safety or cohort benefit claim.',
        next_step='Fresh shared-Search HTTPS comparison of existing CHOICE_CITED_SCOPE; production default remains false.',
        inputs_sha256={name: digest(root / name) for name in ('q11-scope-fixed-prefix.jsonl',
            'all-scope-preflight-verified.jsonl', 'cited-scope-inventory.jsonl', 'fine-source-identity-proposal.json')},
        auditor_sha256=digest(Path(__file__)))
    with (root / 'scope-diagnostic-audit.json').open('x') as out:
        json.dump(summary, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('inventory', 'inputs_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
