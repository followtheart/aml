"""Archive one HTTPS run and trace gold snippet transport using server logs.

Offline diagnostic only: gold metadata never enters retrieval or answering.
Snippet presence is a transport proxy, not proof that an option is entailed.
Use after the run exits; --allow-partial permits progress snapshots.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def norm(text):
    return ' '.join((text or '').split()).casefold()


def matches(text, turns):
    text = norm(text)
    return bool(text) and any(turn and (turn in text or text in turn) for turn in turns)


def identity(source):
    return source.get('request_id'), source.get('message_index')


def turn_coverage(turns, cards, raw, role='user'):
    """Measure literal coverage, without treating a shared introduction as a turn.

    Join fragments only from the original message that contains the CSV turn.
    Full matches can stand alone; coverage is still not semantic entailment.
    """
    result = []
    for index, turn in enumerate(turns):
        origins = {key for key, source in raw.items()
                   if source.get('role') == role and turn and turn in norm(source.get('content'))}
        intervals, ids = [], []
        for card in cards:
            if card.get('role') != role:
                continue
            text = norm(card.get('text'))
            if not text or not turn:
                continue
            if turn in text:
                intervals.append((0, len(turn))); ids.append(card['id'])
            elif text in turn and identity(card) in origins:
                # Ambiguous repeated text cannot prove multiple occurrences.
                start = turn.find(text)
                intervals.append((start, start + len(text)))
                ids.append(card['id'])
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        covered = sum(end - start for start, end in merged)
        # Whitespace between adjacent excerpts carries no missing content.
        missing = ''.join(turn[a:b] for a, b in zip(
            [0] + [end for _, end in merged], [start for start, _ in merged] + [len(turn)]))
        result.append(dict(turn_index=index, normalized_chars=len(turn),
                           covered_chars=covered, fraction=covered / len(turn) if turn else 0,
                           full=bool(turn) and not missing.strip(), source_ids=list(dict.fromkeys(ids)),
                           provenance_origins=len(origins)))
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def archive(source, destination, users):
    # Read shared logs as streams. Never mix experiments or overwrite live logs.
    rows = [r for r in records(source) if r.get('user_id') in users]
    if source.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
                               encoding='utf-8')
    return rows


def classify(row):
    if row['error_stage']:
        return 'runtime:' + row['error_stage']
    if row.get('answer_reader_mode') == 'direct_catalog':
        return 'S6:direct_selection' if row['score'] < 1 else 'OK'
    if not row['observed_add'] or not row['observed_search']:
        return 'unobserved:server_logs'
    # Eligibility identifies the last observed gate, not the causal failure.
    # An option may be misclassified as generic, losing its personal premise;
    # a competing option may also survive an omitted constraint check.
    # A missing CSV snippet cannot overrule observed eligibility.
    if row['gold'] in row['eligible']:
        return 'S6:selection' if row['score'] < 1 else 'OK'
    if not row['snippet_turns']:
        return 'unobserved:gold_snippet'
    if not row['prepared_hits']:
        return 'S0:snippet_absent_from_dataset'
    if not row['raw_hits']:
        return 'S0:ingestion'
    # A source may survive through an episode or dependency expansion even
    # when there is no atomic fact. Later observed transport takes precedence.
    if not row['catalog_hits']:
        if not row['memory_hits']:
            return 'S1:memory'
        if not row['pool_hits'] and not row['packet_hits']:
            return 'S2:recall'
        if not row['packet_hits']:
            return 'S3:packet'
        return 'S3:source_catalog'
    if row.get('gold_kind') != 'generic' and not row['gold_valid_citations']:
        return 'S4:support'
    if row['gold'] not in row['eligible']:
        return 'S5:eligibility'
    return 'S6:selection' if row['score'] < 1 else 'OK'


def snippet_transport_stage(row):
    """Track source delivery independently of answer correctness or eligibility.

    A delivered CSV snippet is a transport observation, not proof of the gold
    option. Later observed delivery takes precedence over missing earlier links.
    """
    if not row.get('observed_add') or not row.get('observed_search'):
        return 'unobserved:server_logs'
    if not row.get('snippet_turns'):
        return 'unobserved:gold_snippet'
    coverage = row.get('catalog_turn_coverage')
    constraints = row.get('constraint_turn_coverage')
    if row.get('catalog_hits') and (coverage is None or any(c['full'] for c in coverage)):
        return 'delivered:source_catalog'
    if row.get('constraint_catalog_hits') and (constraints is None or any(c['full'] for c in constraints)):
        return 'delivered:constraint_catalog'
    if row.get('catalog_hits'):
        return ('partial:source_catalog' if any(c['covered_chars'] for c in coverage)
                else 'fragment:unattributed_source_catalog')
    if row.get('constraint_catalog_hits'):
        return ('partial:constraint_catalog' if any(c['covered_chars'] for c in constraints)
                else 'fragment:unattributed_constraint_catalog')
    if row.get('packet_hits'):
        return 'S3:source_catalog'
    if row.get('pool_hits'):
        return 'S3:packet'
    if row.get('memory_hits'):
        return 'S2:recall'
    if row.get('raw_hits'):
        return 'S1:memory'
    if row.get('prepared_hits'):
        return 'S0:ingestion'
    return 'S0:snippet_absent_from_dataset'


def reader_gate(row):
    """Observed answer gate, independent of CSV snippet transport.

An entailment rejection does not establish that the rejected claim is true
or that the verifier is wrong; it identifies the recorded decision only.
"""
    if row.get('error_stage'):
        return 'runtime:' + row['error_stage']
    if row.get('answer_reader_mode') == 'direct_catalog':
        return 'S6:direct_selection' if row['score'] < 1 else 'OK'
    if row['gold'] in row.get('eligible', []):
        return 'S6:selection' if row['score'] < 1 else 'OK'
    alignment = row.get('gold_alignment') or {}
    errors = set(alignment.get('validation_errors', []))
    errors.update(e for claim in alignment.get('claims', [])
                  for e in claim.get('validation_errors', []))
    if 'not_entailed' in errors:
        return 'S5:independent_entailment'
    if not alignment:
        return 'unobserved:answer_alignment'
    if row.get('gold_valid_citations') or alignment.get('kind') == 'generic':
        return 'S5:eligibility'
    return 'S4:support'


def reader_gate_detail(row):
    """Detail observed gates without claiming the recorded judgment is correct."""
    if row.get('error_stage'):
        return reader_gate(row)
    alignment = row.get('gold_alignment') or {}
    if ('support_unresolved' in alignment.get('validation_errors', [])
            or row.get('gold') in row.get('unresolved_support_options', [])):
        return 'S4:support_unresolved'
    gate = reader_gate(row)
    if gate == 'S5:eligibility' and row.get('gold_forget_constraint_blocked'):
        return 'S5:forget_constraint'
    return gate


def gold_forget_matches(result, label):
    """Retain only recorded, validated positive matches for this answer."""
    return [match for match in result.get('answer_constraints', [])
            if match.get('letter') == label and match.get('valid') is True
            and match.get('violates', True) is True]


def analyze(results, memory, searches, prepared, gold):
    by_search = {r['search_id']: r for r in searches}
    if len(by_search) != len(searches):
        raise ValueError('Duplicate search IDs in server log')
    rows = []
    for result in results:
        conv = prepared[result['conversation_id']]
        qa = next(q for q in conv['qa'] if str(q['id']) == str(result['qa_id']))
        # The adapter stores the CSV row index as qa_id. Verify the history,
        # question and answer as well; a repeated question is not an identity.
        metadata = gold[int(result['qa_id'])]
        query = ast.literal_eval(metadata['user_query'])['content']
        if (norm(query) != norm(qa['question'])
                or metadata['correct_answer'] != qa['answer']
                or conv['source_history'] not in
                (metadata['chat_history_32k_link'], metadata['chat_history_128k_link'])):
            raise ValueError(f'Gold identity mismatch: {result["qa_id"]}')
        try:
            snippet = json.loads(metadata['related_conversation_snippet'])
        except ValueError:
            snippet = ast.literal_eval(metadata['related_conversation_snippet'])
        turns = [norm(m['content']) for m in snippet if m.get('role') == 'user']
        assistant_turns = [norm(m['content']) for m in snippet if m.get('role') == 'assistant']
        user = result['user_id']
        events = [e for e in memory if e['user_id'] == user]
        raw = {identity(s): s for e in events for s in e.get('source_messages', [])}
        raw_hits = {key for key, s in raw.items()
                    if s['role'] == 'user' and matches(s['content'], turns)}
        memories = {m['id']: m for e in events for m in e.get('memories', [])}
        memory_hits = {mid for mid, m in memories.items()
                       if any(identity(s) in raw_hits for s in m.get('sources', []))}
        search = by_search.get(result.get('search_id'))
        if search:
            if (search['user_id'] != user or search['query'] != qa['question']
                    or search['options'] != qa['options']
                    or search['packet_hash'] != result['packet_hash']):
                raise ValueError(f'Search identity/packet mismatch: {result["qa_id"]}')
            if search.get('fake') or any(e.get('fake') for e in events):
                raise ValueError('Fake server data cannot establish a scored experiment')
        search = search or {}
        catalog = result.get('answer_source_catalog', [])
        catalog_hits = [c['id'] for c in catalog if c.get('role') == 'user'
                        and matches(c.get('text'), turns)]
        # Forget requests are deliberately routed to constraint cards, never
        # affirmative fact cards. Their delivery is not a catalog omission.
        constraint_catalog_hits = [c['id'] for c in result.get('answer_constraint_catalog', [])
                                   if c.get('role') == 'user' and matches(c.get('text'), turns)]
        label = qa['gold_labels'][0]
        alignment = next((a for a in result.get('choice_alignment', [])
                          if a['letter'] == label), {})
        citations = [c for claim in alignment.get('claims', [])
                     for c in claim.get('citations', []) if c.get('valid')]
        # Inspect actual packet sources as well as direct memory carriers.
        packet_hits = {m['id'] for m in search.get('returned', [])
                       if m['id'] in memory_hits or any(
                           s.get('role') == 'user' and matches(s.get('content'), turns)
                           for s in m.get('sources', []))}
        row = dict(conversation_id=result['conversation_id'], qa_id=result['qa_id'],
                   user_id=user, search_id=result.get('search_id'), score=result['score'],
                   gold=label, prediction=result['prediction'], question=qa['question'],
                   gold_option=qa['options'][ord(label) - 65],
                   preference=metadata['preference'], observed_add=len(events),
                   observed_search=bool(search), snippet_turns=len(turns),
                   prepared_hits=sum(s['role'] == 'user' and matches(s['content'], turns)
                                     for session in conv['sessions'] for s in session),
                   raw_hits=len(raw_hits), memory_hits=len(memory_hits),
                   fact_hits=sum(memories[m].get('type') != 'episode' for m in memory_hits),
                   pool_hits=len(memory_hits & {m['id'] for m in search.get('fused', [])}),
                   packet_hits=len(packet_hits), catalog_hits=catalog_hits,
                   constraint_catalog_hits=constraint_catalog_hits,
                   catalog_turn_coverage=turn_coverage(turns, catalog, raw),
                   catalog_assistant_turn_coverage=turn_coverage(assistant_turns, catalog, raw, role='assistant'),
                   snippet_role_chars={role: sum(len(norm(m.get('content'))) for m in snippet if m.get('role') == role)
                                       for role in ('user', 'assistant')},
                   constraint_turn_coverage=turn_coverage(turns, result.get('answer_constraint_catalog', []), raw),
                   gold_kind=alignment.get('kind'), gold_valid_citations=len(citations),
                   gold_verified_personal_claims=sum(
                       c.get('entailment_verified') is True for c in alignment.get('claims', [])),
                   constraint_catalog_size=len(result.get('answer_constraint_catalog', [])),
                   constraint_decisions=len(result.get('answer_constraints', [])),
                   gold_recorded_blocked=label in result.get('answer_blocked_options', []),
                   gold_forget_constraint_matches=gold_forget_matches(result, label),
                   gold_forget_constraint_blocked=(label in result.get('answer_blocked_options', [])
                                                   and bool(gold_forget_matches(result, label))),
                   gold_snippet_citations=sum(c.get('source_id') in catalog_hits for c in citations),
                   eligible=result.get('answer_eligible_options', []),
                   unresolved_support_options=result.get('answer_support_validation', {}).get('unresolved_options', []),
                   support_validation_errors=result.get('answer_support_validation', {}).get('errors', {}),
                   support_repair_calls=sum(c.get('stage', '').startswith('eval.choice_support.repair')
                                            for c in result.get('answer_calls', [])),
                   gold_alignment=alignment, error_stage=result.get('error_stage'),
                   answer_reader_mode=result.get('answer_reader_mode'),
                   search_degraded=result.get('search_degraded'),
                   server_versions=search.get('versions'))
        row['stage'] = classify(row)
        row['reader_gate'] = reader_gate(row)
        row['reader_gate_detail'] = reader_gate_detail(row)
        row['snippet_transport_stage'] = snippet_transport_stage(row)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--memory-log', type=Path, default=ROOT / 'logs/memory-debug.jsonl')
    parser.add_argument('--search-log', type=Path, default=ROOT / 'logs/search-debug.jsonl')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/prepared/personamem-v2-32k.jsonl')
    parser.add_argument('--gold-csv', type=Path, default=ROOT / 'data/raw/personamem-v2.csv')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if not args.allow_partial and not (args.run / 'exit-code').exists():
        parser.error('Run is still active; use --allow-partial for a progress snapshot')
    base = args.run / 'completed'
    result_path = base / 'data/results/personamem-v2-32k.jsonl'
    results = list(records(result_path))
    if not results:
        parser.error('No scored results yet')
    if any(r.get('transport') != 'https' or r.get('fake') for r in results):
        parser.error('Expected non-fake HTTPS results')
    if len({r['run_id'] for r in results}) != 1:
        parser.error('Results contain multiple runs')
    if len({(r['conversation_id'], r['qa_id']) for r in results}) != len(results):
        parser.error('Results contain duplicate question identities')
    users = {r['user_id'] for r in results}
    memory = archive(args.memory_log, base / 'logs/memory-debug.jsonl', users)
    searches = archive(args.search_log, base / 'logs/search-debug.jsonl', users)
    prepared = {r['conversation_id']: r for r in records(args.prepared)}
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open(encoding='utf-8') as stream:
        gold = list(csv.DictReader(stream))
    rows = analyze(results, memory, searches, prepared, gold)
    exit_path = args.run / 'exit-code'
    exit_code = int(exit_path.read_text().strip()) if exit_path.exists() else None
    planned = re.search(r'^protocol=.*\bqa=(\d+)', (args.run / 'output.log').read_text(), re.M)
    expected = int(planned[1]) if planned else None
    summary = dict(run_id=results[0]['run_id'], exit_code=exit_code, expected_qa=expected,
                   partial=exit_code is None or expected is None or len(rows) != expected,
                   count=len(rows), score=sum(r['score'] for r in rows),
                   errors=dict(Counter(r['error_stage'] for r in rows if r['error_stage'])),
                   wrong_stages=dict(Counter(r['stage'] for r in rows if r['score'] < 1)),
                   wrong_reader_gates=dict(Counter(r['reader_gate'] for r in rows if r['score'] < 1)),
                   wrong_reader_gate_details=dict(Counter(r['reader_gate_detail'] for r in rows if r['score'] < 1)),
                   unresolved_support_rows=sum(bool(r['unresolved_support_options']) for r in rows),
                   support_repair_calls=sum(r['support_repair_calls'] for r in rows),
                   wrong_snippet_transport_stages=dict(Counter(
                       r['snippet_transport_stage'] for r in rows if r['score'] < 1)),
                   all_snippet_transport_stages=dict(Counter(r['snippet_transport_stage'] for r in rows)),
                   all_stages=dict(Counter(r['stage'] for r in rows)),
                   add_events=len(memory), searches=len(searches),
                   prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
                   snippet_transport_policy='full_user_turn_coverage_v2',
                   caveat='Gold snippet transport is a proxy; it does not prove option entailment. '
                          'Delivered requires at least one complete user turn; partial and unattributed fragments are separate. '
                          'Coverage is normalized literal text, not semantic support; legacy hit counters still include fragments. '
                          'Assistant-turn coverage is reported separately and cannot establish user facts. '
                          'snippet_transport_stage ignores eligibility and score; stage retains its legacy combined meaning. '
                          'reader_gate separately records the observed answer decision, not whether '
                          'the verifier is correct or which intervention would repair it. '
                          'Forget-constraint detail requires both a recorded block and a validated positive match; '
                          'it does not establish semantic correctness and does not override another observed gate. '
                          'Eligible gold may have lost its personal premise through a generic label; '
                          'S6 is not proof that only selection needs repair.')
    out = args.run / 'funnel'
    out.mkdir(exist_ok=True)
    write_json(out / 'https-summary.json', summary)
    write_json(out / 'https-rows.json', rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
