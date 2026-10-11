"""Audit supplementary extraction from committed Add traces, without QA labels.

Source coverage measures transport, not the usefulness or truth of every claim.
Counts describe committed requests only; a running request is absent from this log.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def committed_matches(row, facts):
    """Conservative trace-to-storage links; shared sources are not entailment."""
    details = []
    for fact in facts:
        wanted = {(row['request_id'], i) for i in fact.get('_sources', [])}
        linked, exact = [], []
        for memory in row.get('memories', []):
            if memory.get('type') == 'episode' or memory.get('user_id') != row.get('user_id'):
                continue
            sources = {(s.get('request_id'), s.get('message_index'))
                       for s in memory.get('sources', [])}
            if not wanted or not wanted.intersection(sources):
                continue
            snapshot = {k: memory.get(k) for k in
                        ('id', 'content', 'resolution_status', 'view_status', 'superseded_by')}
            linked.append(snapshot)
            if wanted <= sources and fact.get('content') and memory.get('content') == fact['content']:
                exact.append(snapshot)
        status = ('exact_at_commit' if exact else
                  'same_source_only' if linked else 'unmatched_unknown')
        details.append(dict(content=fact.get('content'), source_indices=sorted(i for _, i in wanted),
                            status=status, exact_matches=exact, source_linked_memories=linked))
    return details


def audit(rows):
    requests = []
    seen = set()
    for row in rows:
        request_id = row['request_id']
        if request_id in seen:
            raise ValueError('Duplicate committed request: ' + request_id)
        seen.add(request_id)
        source_roles = {m['message_index']: m['role'] for m in row['source_messages']}
        primary, added, retries, events = [], [], [], []
        for trace in row.get('extraction_trace', []):
            stages = {e['stage'] for e in trace['events']}
            events.extend(trace['events'])
            if 'coverage_retry' in stages:
                retries.append(trace['message_indices'])
                added.extend(f for e in trace['events'] if e['stage'] == 'coverage_output'
                             for f in e['facts'])
            elif 'segment_output' in stages:
                primary.extend(f for e in trace['events'] if e['stage'] == 'segment_output'
                               for f in e['facts'] if f.get('type') != 'episode')
        covered_before = {i for f in primary for i in f.get('_sources', [])
                          if source_roles.get(i) == 'user'}
        deterministic_sources = {i for e in events if e['stage'] == 'coverage_known_sources'
                                 for i in e['source_indices'] if source_roles.get(i) == 'user'}
        covered_before.update(deterministic_sources)
        covered_added = {i for f in added for i in f.get('_sources', [])
                         if source_roles.get(i) == 'user'}
        # Trace rows are sorted by parent segment, so the last row need not
        # contain the final budget counters. Counters are cumulative per Add.
        stats = [e for e in events if e['stage'] in ('coverage_output', 'coverage_stopped')]
        retry_events = [e for e in events if e['stage'] == 'coverage_retry']
        persistence = committed_matches(row, added)
        requests.append(dict(
            request_id=request_id, retries=len(retries), retry_indices=retries,
            candidates=max([e.get('candidate_count', 0) for e in retry_events], default=0),
            primary_atomic_facts=len(primary), added_atomic_facts=len(added),
            persistence_counts=dict(Counter(f['status'] for f in persistence)),
            persistence=persistence,
            added_types=dict(Counter(f.get('type') for f in added)),
            added_assistant_only=sum(bool(f.get('_sources')) and all(
                source_roles.get(i) == 'assistant' for i in f['_sources']) for f in added),
            ignored_unanchored_facts=sum(len(e['facts']) for e in events
                                        if e['stage'] == 'coverage_unanchored'),
            omitted_before_verification=sum(e['stage'] == 'coverage_preverification_omitted'
                                            for e in events),
            user_sources_before=len(covered_before),
            deterministic_source_indices=sorted(deterministic_sources),
            newly_covered_user_sources=sorted(covered_added - covered_before),
            provider_calls=max([e.get('calls', 0) for e in stats], default=0),
            provider_tokens=max([e.get('tokens', 0) for e in stats], default=0),
            stops=[e.get('reason') for e in events if e['stage'] == 'coverage_stopped'],
            skips=[e.get('reason') for e in events if e['stage'] == 'coverage_skipped']))
    total_fields = ('retries', 'candidates', 'primary_atomic_facts', 'added_atomic_facts',
                    'added_assistant_only', 'ignored_unanchored_facts', 'omitted_before_verification',
                    'provider_calls', 'provider_tokens')
    return dict(committed_requests=len(requests),
                totals={k: sum(r[k] for r in requests) for k in total_fields},
                newly_covered_user_sources=sum(len(r['newly_covered_user_sources']) for r in requests),
                persistence_counts=dict(sum((Counter(r['persistence_counts']) for r in requests), Counter())),
                requests=requests,
                caveat='Persistence links describe the commit snapshot, not current eligibility. '
                       'Same-source matches do not prove the fact survived; unmatched is unknown '
                       '(deduplication or merging can alter links). Retrieval and answer benefit '
                       'require separate checks. No counts for uncommitted/in-flight Add requests.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-log', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with args.memory_log.open() as stream:
        result = audit(json.loads(line) for line in stream if line.strip())
    result['memory_log'] = str(args.memory_log)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'requests'}))


if __name__ == '__main__':
    main()
