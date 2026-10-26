"""Offline bounded review proposals for unreviewed coarse units.

Keep fine/main and every original overflow candidate. Try source-novel coarse
outsiders in CE order within the same four-candidate/12 KB request. References
are read only after proposals are fixed. No model judgments or answer gains.
The explicit separate-batch mode instead preserves the completed original
reviews and spends one remaining call on up to four coarse outsiders.
"""
import argparse
import contextlib
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root = args.run.resolve().parent
    plan = json.loads((root / 'plan.json').read_text())
    final = json.loads((root / 'final-evaluation.json').read_text())
    assert final['status'] == 'verified_complete_trial'
    for name, digest in plan['source_files'].items():
        assert sha(root / 'source' / name) == digest, name
    for relative in ('server-search-debug.jsonl', 'completed/data/results/personamem-v2-32k.jsonl'):
        path = args.run.resolve() / relative
        assert sha(path) == final['inputs'][str(path)]
    snapshot = root / 'source/memory_system'
    sys.path.insert(0, str(snapshot))
    sys.path.insert(0, str(snapshot / 'scripts'))
    import probe_overflow_second_batch as base
    from app import budget, config, fine_evidence, listwise_overflow, listwise_recovery, schemas
    manifest = json.loads((args.run / 'isolated-manifest.json').read_text())
    assert manifest['client_exit_code'] == manifest['audit_exit_code'] == 0
    assert manifest['server_exit_code'] in (0, -15)
    prepared = Path(manifest['data'])

    def propose(t, result):
        base.check_versions(t['versions'], 'source Search')
        cascade = t['cascade']
        previous = cascade['overflow']
        lookup = {c['id']: c for c in t['ranked']}
        coarse = sorted([lookup[mid] for mid in cascade['coarse']['selected_ids']],
                        key=fine_evidence.key, reverse=True)
        fine = [lookup[mid] for mid in cascade['fine']['selected_ids']]
        fine_ids = {c['id'] for c in fine}
        submitted = [lookup[mid] for mid in cascade['listwise']['candidate_ids']]
        selected = [lookup[mid] for mid in cascade['listwise']['selected_ids']]
        req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'])
        omissions = list(cascade['omitted'])
        original_ids = set(previous['proposed_ids']) | {d['id'] for d in previous['decisions']}
        old_omitted = {o['id'] for o in omissions if o['stage'] == 'listwise' and o['reason'] == 'prompt_budget'}
        omissions.extend(dict(id=mid, stage='listwise', reason='prompt_budget')
                         for mid in sorted(original_ids - old_omitted))
        headroom = cascade['admission_budget']
        with patch.multiple(config, **t['versions']['settings']):
            assert not config.LISTWISE_OVERFLOW_EXTEND and not config.LISTWISE_OVERFLOW_BATCHED
            assert config.LISTWISE_SPAN_REFS
            with budget.scope(seconds=max(.01, headroom['deadline_seconds']),
                              calls=headroom['call_headroom'], tokens=headroom['token_headroom']):
                baseline, groups, _ = listwise_overflow._propose(req, fine, submitted, selected, omissions)
            assert [c['id'] for c in baseline] == previous['proposed_ids']
            assert [c['id'] for c in selected] + previous['appended_ids'] == previous['selected_ids']
            elapsed = (datetime.fromisoformat(t['finished_at']) - datetime.fromisoformat(t['started_at'])).total_seconds()
            assert elapsed >= 0
            remainder = dict(calls=config.SEARCH_MAX_CALLS-result['coverage_manifest']['provider_calls'],
                             tokens=config.SEARCH_MAX_TOKENS-result['coverage_manifest']['provider_tokens'],
                             conservative_seconds_proxy=max(0, min(config.RERANK_DEADLINE_SECONDS-elapsed,
                                                                   config.SEARCH_DEADLINE_SECONDS-elapsed-2)))
            eligible = [c for c in coarse if c['id'] not in fine_ids
                        and type(c.get('_ce_score')) in (int, float) and math.isfinite(c['_ce_score'])
                        and c.get('view_status') != 'stale']
            # Adapt the existing planner's eligibility interface only inside this
            # diagnostic. These are fine omissions, not real prompt-byte omissions.
            mapped = [dict(id=c['id'], stage='listwise', reason='prompt_budget') for c in eligible]
            separate = args.mode == 'separate-batch'
            seed = [] if separate else baseline
            kept = (selected + [lookup[mid] for mid in previous['appended_ids']]
                    if separate else selected + baseline)
            seed_groups = [] if separate else groups
            pending, decisions = seed, []
            if (remainder['calls'] >= (1 if separate or not baseline else 0) and remainder['tokens'] > 0
                    and remainder['conservative_seconds_proxy'] > 0):
                with budget.scope(seconds=remainder['conservative_seconds_proxy'],
                                  calls=max(1, remainder['calls']), tokens=remainder['tokens']):
                    pending, groups, decisions = listwise_overflow._propose(
                        req, coarse, fine, kept, mapped, seed=seed, seed_groups=seed_groups)
            baseline_ids = [c['id'] for c in baseline]
            if not separate:
                assert [c['id'] for c in pending[:len(baseline)]] == baseline_ids
            extra = pending if separate else pending[len(baseline):]
            assert not {c['id'] for c in extra} & fine_ids
            request = None
            if extra:
                cards = [dict(candidate_id=c['id'], sources=listwise_recovery.sources_for(c)) for c in pending]
                request = listwise_recovery.request_for(req, cards)
                assert len(pending) <= 4 and len(request['prompt'].encode()) <= 12000
                assert request['reservation'] <= remainder['tokens']
        return dict(qa_id=result['qa_id'], conversation_id=result['conversation_id'], search_id=t['search_id'],
            baseline_ids=baseline_ids, original_selected_ids=previous['selected_ids'],
            proposed_ids=[c['id'] for c in extra], combined_ids=[c['id'] for c in pending],
            groups=groups, decisions=decisions, remainder=remainder,
            new_request=bool(extra) and (separate or not baseline), mode=args.mode,
            input_bytes=len(request['prompt'].encode()) if request else 0,
            request_reservation=request['reservation'] if request else 0,
            request_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest() if request else None)

    # Reuse the frozen complete-cohort, identity and source-attribution checks.
    with tempfile.TemporaryDirectory(prefix='coarse-overflow-fill-') as temporary:
        intermediate = Path(temporary) / 'report.json'
        options = SimpleNamespace(run=args.run, prepared=prepared, output=intermediate, allow_partial=False)
        with patch.object(base, 'propose', propose), patch.object(base, 'CSV', args.gold_csv), \
             patch('socket.socket.connect', side_effect=AssertionError('Offline proposal')), \
             contextlib.redirect_stdout(io.StringIO()):
            base.main(options)
        report = json.loads(intermediate.read_text())
    summary = report['summary']
    summary['planned_requests'] = summary.pop('additional_batches')
    summary['additional_requests'] = sum(r['new_request'] for r in report['rows'])
    summary['mode'] = args.mode
    summary['provider_calls'] = 0
    report['limitation'] = __doc__
    report['budget_limitation'] = (
        ('Preserves completed original reviews. ' if args.mode == 'separate-batch' else
         'Preserves original overflow input, not its model judgments. ') +
        'Uses end-of-Search token remainder '
        'to fund the entire changed request as a conservative proxy; monotonic timing is not replayed. '
        'Upper bounds append only new proposals without judgments or final packet limits. Synthetic '
        'prompt-budget omissions are an offline adapter, never claimed as recorded admission reasons.')
    report['script_sha256'] = sha(Path(__file__))
    report['dependencies_sha256'] = {str(p.resolve()): sha(p) for p in (
        Path(base.__file__), Path(listwise_overflow.__file__), Path(listwise_recovery.__file__),
        root/'plan.json', root/'final-evaluation.json', args.gold_csv, args.run/'memory.db')}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('fill', 'separate-batch'), default='fill')
    parser.add_argument('--gold-csv', type=Path,
                        default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(parser.parse_args())
