"""Offline Q1 semantic-invariance diagnostic; known mismatches stay visible.

Run from the repository root:
  memory_system/.venv/Scripts/python.exe memory_system/scripts/probe_q1_paraphrases.py
Add --strict to return exit 1 when an expectation fails. The default exit 0
means the diagnostic completed, not that semantic invariance passed.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# These two modules are pure text logic: no config, credentials or model clients.
from app import search_coverage as coverage


ARCHIVE_SHA256 = '46c970273a3066cb534ecdead0d3722c704208a8bd55fe8cda8a9853e933397f'
FIXTURE_SHA256 = '3c80c5efb6682857d76bfc9604a80aa45e1f28feb5d8bbfbb27a4d538d75d9a3'
CANDIDATE_ID = 'amu_4c00a5d3611b4e41'
CONVERSATION_ID = '911d1d0140349ab4dd01'
SOURCE_REQUEST = f'local:personamem-v2:{CONVERSATION_ID}:session:0:chunk:1'
BASELINE = 'experience with being present for students sharing troubling matters'
PARAPHRASES = [
    ('observed_old_requirement', BASELINE),
    ('observed_rerun_requirement',
     'ways to be present without having all the answers when a student shares something troubling'),
    ('same_event_plain',
     'being present when a student shares troubling personal circumstances'),
    ('same_event_synonyms',
     'experience with being present for pupils sharing distressing personal situations'),
    ('same_event_disclosed',
     'being present for a student who disclosed troubling personal circumstances'),
    ('same_event_listening',
     'listening to a student sharing deeply troubling personal circumstances'),
]


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def load_fixture(archive):
    archive_hash = sha256(archive.read_bytes())
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError(f'Frozen archive hash mismatch: {archive_hash}')
    with ZipFile(archive) as zipped:
        with zipped.open('memory_system/data/results/personamem-v2-32k.jsonl') as rows:
            result = next(row for line in rows if line.strip()
                          if (row := json.loads(line))['conversation_id'] == CONVERSATION_ID
                          and row['qa_id'] == '1')
        with zipped.open('memory_system/logs/search-debug.jsonl') as rows:
            trace = next(row for line in rows if line.strip()
                         if (row := json.loads(line))['search_id'] == result['search_id'])
    candidate = next(row for row in trace['ranked'] if row['id'] == CANDIDATE_ID)
    source = next(row for row in candidate['_packet_item']['sources']
                  if row['request_id'] == SOURCE_REQUEST and row['message_index'] == 12)
    fixture = dict(query=trace['query'], candidate_id=candidate['id'],
                   rank_text=candidate['_rank_text'], source=source)
    fixture_hash = sha256(canonical(fixture))
    if fixture_hash != FIXTURE_SHA256:
        raise ValueError(f'Frozen fixture hash mismatch: {fixture_hash}')
    if source['content'] not in candidate['_rank_text']:
        raise ValueError('Frozen source is not visible in the actual rank text')
    return fixture, trace, archive_hash


def scope_conditions():
    """Label actual False-return lines without duplicating the scope rules."""
    module = ast.parse(Path(coverage.__file__).read_text(encoding='utf-8'))
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_scope_matches')
    conditions = {}
    for node in ast.walk(function):
        if isinstance(node, ast.If):
            for statement in node.body:
                if (isinstance(statement, ast.Return)
                        and isinstance(statement.value, ast.Constant)
                        and statement.value.value is False):
                    conditions[statement.lineno] = ast.unparse(node.test)
    return conditions


def scope_probe(passage, source_text, requirement, conditions):
    """Run the real scope function independently, recording its return branch."""
    returned = {}
    target = coverage._scope_matches.__code__
    def tracer(frame, event, value):
        if frame.f_code is target:
            if event == 'return' and value is False:
                returned['line'] = frame.f_lineno
            return tracer
        return None
    previous = sys.gettrace()
    try:
        sys.settrace(tracer)
        accepted = coverage._scope_matches(passage, source_text, requirement)
    finally:
        sys.settrace(previous)
    return dict(accepted=accepted, reject_line=returned.get('line'),
                reject_condition=conditions.get(returned.get('line')))


def passage_diagnostics(source, requirement, conditions):
    wanted = coverage._concepts(requirement['text'])
    implementation = inspect.getsource(coverage._witnesses)
    threshold = re.search(r'len\(hits\) / len\(wanted\) < ([\d.]+)', implementation)
    if threshold is None:
        raise RuntimeError('Lexical gate changed; update diagnostic before interpreting results')
    minimum_ratio = float(threshold[1])
    rows = []
    for start, end, passage in coverage._local_passages(source['content']):
        hits = wanted & coverage._concepts(passage)
        lexical_ok = bool(wanted and len(hits) >= min(2, len(wanted))
                          and len(hits) / len(wanted) >= minimum_ratio)
        scope = scope_probe(passage, source['content'], requirement, conditions)
        # Scope replay is independent: annotate short-circuits before scope when
        # the lexical gate fails. Do not imply both gates executed in that call.
        rows.append(dict(span=dict(start=start, end=end), content=passage,
            matched_terms=sorted(hits), wanted_terms=sorted(wanted),
            hit_count=len(hits), wanted_count=len(wanted),
            overlap_ratio=len(hits) / len(wanted) if wanted else 0,
            minimum_ratio=minimum_ratio, lexical_pass=lexical_ok,
            first_blocker='lexical' if not lexical_ok else 'scope' if not scope['accepted'] else None,
            independent_scope_replay=scope))
    return rows


def make_cases(fixture):
    source = fixture['source']
    cases = [dict(id=name, kind='frozen_source_paraphrase', requirement=text,
                  expected_support=True,
                  expectation_basis='Manual event-level expectation: the same disclosed student incident and listening/support, not complete answer sufficiency.',
                  source=copy.deepcopy(source), rank_text=fixture['rank_text'])
             for name, text in PARAPHRASES]
    # Fixed verbatim part of the pinned source; negatives are explicitly synthetic.
    core = source['content'][109:392]
    controls = [
        ('third_party_quote', 'Karen wrote: ' + core,
         'The first-person disclosure belongs to Karen, not the user.'),
        ('hypothetical_frame', 'Imagine the following fictional scenario:\n\n' + core,
         'A fictional event must not establish the user experienced it.'),
        ('explicit_never', 'I have never been present for students sharing troubling personal circumstances.',
         'The user explicitly denies the event.'),
        ('negative_no_student', 'No student ever shared troubling personal circumstances with me.',
         'A negative quantifier denies the event despite matching topic terms.'),
    ]
    for name, text, basis in controls:
        synthetic = dict(request_id=f'probe:q1:{name}', message_index=0,
                         role='user', content=text, source_event_id=f'probe:q1:{name}')
        cases.append(dict(id=name, kind='synthetic_negative_control', requirement=BASELINE,
                          expected_support=False, expectation_basis=basis,
                          source=synthetic, rank_text=text))
    return cases


def evaluate(case, conditions):
    source = copy.deepcopy(case['source'])
    before = copy.deepcopy(source)
    requirement = dict(id='q1:student_disclosure', text=case['requirement'])
    item = dict(id='probe:' + case['id'], content=case['rank_text'], _rank_text=case['rank_text'])
    # Authoritative measured result uses the unchanged public production entry.
    coverage.annotate(item, [requirement], [source])
    if source != before:
        raise AssertionError('Production coverage modified the source fixture')
    actual = requirement['id'] in item['_supported_coverage_ids']
    witnesses = item['_coverage_witnesses'].get(requirement['id'], [])
    for witness in witnesses:
        start, end = witness['span']['start'], witness['span']['end']
        if source['content'][start:end] != witness['content']:
            raise AssertionError('Witness did not preserve the supplied source span')
    return dict(case, source_sha256=sha256(source['content'].encode('utf-8')),
                actual_support=actual, expectation_met=actual == case['expected_support'],
                soft_coverage_ids=item['_coverage_ids'],
                supported_coverage_ids=item['_supported_coverage_ids'], witnesses=witnesses,
                passage_gates=passage_diagnostics(source, requirement, conditions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path,
        default=ROOT / 'runs/personamem-v9-rerun-audit/inputs.zip')
    parser.add_argument('--output', type=Path,
        default=ROOT / 'runs/personamem-controlled-validation/q1-paraphrases.json')
    parser.add_argument('--strict', action='store_true',
        help='Exit 1 for unmet semantic expectations; default only reports them')
    args = parser.parse_args()
    fixture, trace, archive_hash = load_fixture(args.archive)
    conditions = scope_conditions()
    cases = make_cases(fixture)
    with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden in offline probe')):
        results = [evaluate(case, conditions) for case in cases]
    positives = [row for row in results if row['expected_support']]
    negatives = [row for row in results if not row['expected_support']]
    frozen_source_hash = sha256(fixture['source']['content'].encode('utf-8'))
    controls = dict(
        all_positive_source_hashes_match_frozen=all(row['source_sha256'] == frozen_source_hash for row in positives),
        all_positive_rank_texts_identical=all(row['rank_text'] == fixture['rank_text'] for row in positives),
        all_positive_and_negative_source_roles_user=all(row['source']['role'] == 'user' for row in results),
        all_negative_contents_differ_from_frozen=all(row['source_sha256'] != frozen_source_hash for row in negatives))
    if not all(controls.values()):
        raise AssertionError('Source/role control invariants failed')
    failed = [row['id'] for row in results if not row['expectation_met']]
    code_hash = sha256(Path(coverage.__file__).read_bytes())
    record = dict(probe='q1_paraphrase_invariance', diagnostic_completed=True,
        probe_sha256=sha256(Path(__file__).read_bytes()),
        all_expectations_met=not failed, mismatches=failed, control_invariants=controls,
        summary=dict(cases=len(results), positive_cases=len(positives),
            positives_supported=sum(row['actual_support'] for row in positives),
            positive_support_invariant=len({row['actual_support'] for row in positives}) == 1,
            negative_controls=len(negatives), negatives_rejected=sum(not row['actual_support'] for row in negatives)),
        fixture=dict(archive=str(args.archive), archive_sha256=archive_hash,
            fixture_sha256=FIXTURE_SHA256, source_sha256=sha256(fixture['source']['content'].encode('utf-8')),
            cases_sha256=sha256(canonical(cases)),
            search_id=trace['search_id'], candidate_id=fixture['candidate_id'],
            source_request_id=SOURCE_REQUEST, source_message_index=12, source_reference='S1:12'),
        production=dict(entrypoint='app.search_coverage.annotate', file=str(Path(coverage.__file__)),
            sha256=code_hash, frozen_sha256=trace['versions']['file_hashes']['memory_system/app/search_coverage.py'],
            matches_frozen=code_hash == trace['versions']['file_hashes']['memory_system/app/search_coverage.py'],
            personal_evidence_sha256=sha256(Path(coverage.personal_evidence.__file__).read_bytes()),
            frozen_personal_evidence_sha256=trace['versions']['file_hashes']['memory_system/app/personal_evidence.py']),
        methodology=dict(expected_labels='Manual event-level labels, not provider output or official dataset gold.',
            gate_order='Lexical first; scope replay reports independent blockers even when lexical short-circuits.',
            completion_exit='Default exit 0 records completion, not passing expectations; --strict exits 1 for mismatches.',
            model_calls=0, network_connections_allowed=False),
        reproduction=dict(default='memory_system/.venv/Scripts/python.exe memory_system/scripts/probe_q1_paraphrases.py',
            strict='memory_system/.venv/Scripts/python.exe memory_system/scripts/probe_q1_paraphrases.py --strict'),
        cases=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'DIAGNOSTIC COMPLETE: {len(results)-len(failed)}/{len(results)} expectations met')
    print('MISMATCHES: ' + (', '.join(failed) or '(none)'))
    print('FIXTURE_SHA256: ' + FIXTURE_SHA256)
    print('OUTPUT: ' + str(args.output))
    return int(args.strict and bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
