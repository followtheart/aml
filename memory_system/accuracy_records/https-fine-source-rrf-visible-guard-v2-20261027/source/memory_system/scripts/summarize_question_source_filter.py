"""Audit original-text retention, question-only selection, and complete paired scores."""
import argparse
import hashlib
import json
from pathlib import Path
from probe_question_source_filter import (INSTRUCTION, MAX_SOURCES, load_cohort, selected_catalog,
    selection_request, sha, direct_catalog_reader, answer_choice)
from summarize_reader_replay import read_complete


def summarize(path, allow_partial=False):
    plan = json.loads(path.with_suffix('.plan.json').read_text())
    prepared = Path(plan['prepared'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest() != plan['prepared_sha256']:
        raise ValueError('Prepared input changed')
    if plan['instruction'] != INSTRUCTION or plan['max_sources'] != MAX_SOURCES:
        raise ValueError('Selection policy changed')
    mode = plan.get('selection_mode', 'semantic')
    if mode not in ['semantic', 'first_party']:
        raise ValueError('Unknown selection mode')
    cohort = load_cohort(Path(plan['run']), prepared)
    if [r['qa_id'] for r, *_ in cohort] != plan['questions']:
        raise ValueError('Cohort changed')
    expected = {(r['conversation_id'], r['qa_id']): (r, qa, packet, sources, constraints)
                for r, qa, packet, sources, constraints in cohort}
    groups = {}; errors = []; retention = []
    for row in read_complete(path, allow_partial):
        key = row['conversation_id'], row['qa_id']; arm = row['arm']
        if key not in expected or arm not in ['direct', 'filtered']:
            raise ValueError('Unexpected question/arm')
        arms = groups.setdefault(key, {})
        if arm in arms:
            raise ValueError('Duplicate arm')
        arms[arm] = row
        r, qa, packet, sources, constraints = expected[key]
        _, context = direct_catalog_reader.make_prompt(qa, packet, sources, constraints)
        if row['packet_hash'] != r['packet_hash'] or row['full_source_sha256'] != sha(context):
            raise ValueError('Original packet/catalog changed')
        calls = row['calls']; offset = 0; selected = sources; selection_invalid = False
        if arm == 'filtered' and mode == 'first_party':
            selected = {sid: s for sid, s in sources.items()
                        if (s.get('declared') or s['role']) in ('user', 'persona')}
        if arm == 'filtered' and mode == 'semantic':
            prompt, _ = selection_request(qa['question'], sources, constraints)
            if not calls or calls[0]['stage'] != 'diagnostic.question_source_filter.select' or calls[0]['prompt_sha256'] != sha(prompt):
                raise ValueError('Selection prompt changed/exposed options')
            offset = 1
            try:
                selected = selected_catalog(row.get('selection_response'), sources)
            except ValueError:
                selection_invalid = True
                if len(calls) != 1 or row['status'] != 'error':
                    raise ValueError('Invalid selection must stop and fail')
            if not selection_invalid and calls[0].get('response') != row['selection_response']:
                raise ValueError('Selection response differs from trace')
        if not selection_invalid:
            prompt, context = direct_catalog_reader.make_prompt(qa, packet, selected, constraints)
            if (row['selected_ids'] != list(selected) or row['source_sha256'] != sha(context)
                    or row['prompt_sha256'] != sha(prompt)):
                raise ValueError('Filtered source text/order/constraints changed')
            if (len(calls) != offset + 1 or calls[offset]['prompt_sha256'] != sha(prompt)
                    or calls[offset]['stage'] != 'diagnostic.question_source_filter.' + arm):
                raise ValueError('Unexpected answer call')
            if row.get('response') != calls[offset].get('response'):
                raise ValueError('Answer differs from call trace')
            if arm == 'filtered':
                retention.append(dict(qa_id=row['qa_id'], original=len(sources), retained=len(selected)))
        if row['status'] == 'ok':
            response = row['response']; letters = answer_choice.option_map(qa['options'])
            if (set(response) != {'answer'} or response['answer'] not in letters
                    or row['prediction'] != response['answer']
                    or row['score'] != float(response['answer'] == qa['gold_labels'][0])):
                raise ValueError('Invalid answer/score')
        elif row['status'] == 'error':
            if row['score'] != 0:
                raise ValueError('Error must score zero')
            errors.append(dict(qa_id=row['qa_id'], arm=arm, error_type=row.get('error_type')))
        else:
            raise ValueError('Unknown status')
    missing = [key for key in expected if set(groups.get(key, {})) != {'direct', 'filtered'}]
    if missing and not allow_partial:
        raise ValueError('Incomplete comparison')
    pairs = [dict(conversation_id=key[0], qa_id=key[1], **{
        arm: dict(prediction=arms[arm].get('prediction'), score=arms[arm]['score'], status=arms[arm]['status'])
        for arm in ['direct', 'filtered']}) for key, arms in groups.items() if key not in missing]
    return dict(expected_pairs=len(expected), complete_pairs=len(pairs), partial=bool(missing), missing=missing,
        direct=sum(p['direct']['score'] for p in pairs), filtered=sum(p['filtered']['score'] for p in pairs),
        wins=[p['qa_id'] for p in pairs if p['filtered']['score'] > p['direct']['score']],
        losses=[p['qa_id'] for p in pairs if p['filtered']['score'] < p['direct']['score']],
        errors=errors, retention=retention, pairs=pairs,
        caveat=plan['caveat'] + ' Prompt hashes reconstructed, source text/order and full constraints verified; score does not establish grounding.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args(); result = summarize(args.replay, args.allow_partial)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['pairs', 'retention', 'missing', 'caveat']}))
