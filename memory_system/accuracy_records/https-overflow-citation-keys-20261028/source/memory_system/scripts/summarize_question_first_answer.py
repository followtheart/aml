"""Verify question-only drafting, full-source selection and complete paired outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, direct_catalog_reader, eval_scoring
from probe_question_first_answer import INSTRUCTION, SUFFIX, draft_request, final_prompt, validate_draft
from probe_question_source_filter import load_cohort, sha
from summarize_reader_replay import read_complete


def summarize(path, allow_partial=False):
    plan = json.loads(path.with_suffix('.plan.json').read_text())
    prepared = Path(plan['prepared'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest() != plan['prepared_sha256']:
        raise ValueError('Prepared input changed')
    if plan['instruction'] != INSTRUCTION or plan['suffix'] != SUFFIX or plan['arms'] != ['direct', 'draft']:
        raise ValueError('Policy changed')
    cohort = load_cohort(Path(plan['run']), prepared)
    if [r['qa_id'] for r, *_ in cohort] != plan['questions']: raise ValueError('Cohort changed')
    expected = [(data, arm) for i, data in enumerate(cohort)
                for arm in (['direct', 'draft'] if i % 2 == 0 else ['draft', 'direct'])]
    rows = read_complete(path, allow_partial)
    if len(rows) > len(expected) or (not allow_partial and len(rows) != len(expected)):
        raise ValueError('Incomplete or extra rows')
    errors = []
    for row, (data, arm) in zip(rows, expected):
        r, qa, packet, sources, constraints = data
        if (row['qa_id'], row['conversation_id'], row['arm'], row['packet_hash']) != (r['qa_id'], r['conversation_id'], arm, r['packet_hash']):
            raise ValueError('Identity/order changed')
        rebuilt_sources, rebuilt_constraints = answer_choice.build_catalog(packet)
        if rebuilt_sources != sources or list(rebuilt_constraints.values()) != constraints:
            raise ValueError('Original catalog changed')
        base, context = direct_catalog_reader.make_prompt(qa, packet, sources, constraints)
        if row['source_sha256'] != sha(context): raise ValueError('Sources or constraints changed')
        calls = row['calls']; offset = 0; failure = False; prompt = base
        if arm == 'draft':
            request, _ = draft_request(qa['question'], sources, constraints)
            if not calls or calls[0]['stage'] != 'diagnostic.question_first.draft' or calls[0]['prompt_sha256'] != sha(request):
                raise ValueError('Question-only draft prompt changed')
            offset = 1
            if calls[0]['status'] == 'error': failure = True
            elif calls[0]['status'] == 'ok':
                if row.get('draft_response') != calls[0]['response']: raise ValueError('Draft response changed')
                try: validate_draft(row['draft_response'], sources)
                except ValueError: failure = True
                if not failure: prompt = final_prompt(base, row['draft_response'])
            else: raise ValueError('Unknown call status')
        if failure:
            if len(calls) != offset: raise ValueError('Failure incorrectly followed by answer call')
        else:
            if len(calls) != offset + 1: raise ValueError('Missing/extra selection call')
            call = calls[offset]
            if call['stage'] != 'diagnostic.question_first.select.' + arm or call['prompt_sha256'] != sha(prompt) or row['prompt_sha256'] != sha(prompt):
                raise ValueError('Selection prompt changed')
            if call['status'] == 'error': failure = True
            elif call['status'] == 'ok':
                response = call['response']
                if row.get('response') != response: raise ValueError('Answer response changed')
                letters = answer_choice.option_map(qa['options'])
                valid = isinstance(response, dict) and set(response) == {'answer'} and isinstance(response['answer'], str) and response['answer'] in letters
                failure = not valid
                if valid and (row.get('prediction') != response['answer'] or row['score'] != eval_scoring.choice_score(response['answer'], qa['gold_labels'], qa['qa_type'])):
                    raise ValueError('Prediction/score changed')
            else: raise ValueError('Unknown call status')
        if failure:
            if row['status'] != 'error' or row['score'] != 0 or not row.get('error_type'):
                raise ValueError('Failure must remain scored zero')
            errors.append(dict(qa_id=r['qa_id'], arm=arm, error_type=row['error_type']))
        elif row['status'] != 'ok': raise ValueError('Successful calls cannot become error')
    pairs = []
    for i in range(0, len(rows) - 1, 2):
        by_arm = {r['arm']: r for r in rows[i:i+2]}
        pairs.append(dict(qa_id=rows[i]['qa_id'], **{a: {k: by_arm[a].get(k) for k in ['prediction', 'score', 'status']} for a in ['direct', 'draft']}))
    return dict(partial=len(rows) != len(expected), expected_pairs=len(cohort), complete_pairs=len(pairs),
        rows=len(rows), errors=errors, direct=sum(p['direct']['score'] for p in pairs), draft=sum(p['draft']['score'] for p in pairs),
        wins=[p['qa_id'] for p in pairs if p['draft']['score'] > p['direct']['score']],
        losses=[p['qa_id'] for p in pairs if p['draft']['score'] < p['direct']['score']],
        recorded_calls={a: sum(c['calls'] for r in rows if r['arm'] == a for c in r['calls']) for a in ['direct', 'draft']},
        pairs=pairs, caveat=plan['caveat'])

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true'); args = p.parse_args()
    result = summarize(args.replay, args.allow_partial)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['pairs', 'caveat']}))
