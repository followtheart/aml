"""Paired grammatical-scope probe; labels remain offline, no runtime changes."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, budget, config, eval_scoring, evidence_packet, llm
from probe_verification_scope import scope_prefix
from replay_query_context import recorded_response


GUIDANCE = (
    ' Distinguish a hypothetical comparison from a report about the user. '
    'A comparison describing how one would treat an indefinite person is not '
    'evidence that the user has that relationship or has experienced that event. '
    'Still mark a claim personal when it includes an actual user-specific fact, '
    'including a named relationship, a past event, an existing possession, or '
    'a stated preference. A metaphor does not erase factual clauses embedded in it.'
)


async def original_prompt(run, prepared):
    def records(path):
        return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
                if line.endswith(b'\n')]
    row = next(r for r in records(run / 'completed/data/results/personamem-v2-32k.jsonl')
               if str(r['qa_id']) == '257')
    trace = next(t for t in records(run / 'server-search-debug.jsonl')
                 if t['search_id'] == row['search_id'])
    packet = trace['returned']
    if evidence_packet.digest(packet) != row['packet_hash']:
        raise ValueError('Packet identity changed')
    conv = next(c for c in records(prepared) if c['conversation_id'] == row['conversation_id'])
    qa = next(q for q in conv['qa'] if str(q['id']) == '257')
    pending = copy.deepcopy(row['answer_calls'])
    captured, mismatches = [], []

    async def provider(prompt, *args, **kwargs):
        if not pending:
            mismatches.append('unexpected_call')
            raise ValueError('Live calls forbidden during reconstruction')
        old = pending.pop(0)
        if old['stage'] != kwargs['stage'] or old['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest():
            mismatches.append(kwargs['stage'])
            raise ValueError('Original prompt mismatch')
        if old['stage'] == 'eval.choice_premise_scope':
            captured.append(prompt)
        budget.current.get().before_call()
        return recorded_response(old)

    with patch.object(llm, 'complete_json', provider):
        prediction, score, _ = await eval_scoring.evaluate(qa, copy.deepcopy(packet))
    if pending or mismatches or len(captured) != 1 or (prediction, score) != (row['prediction'], row['score']):
        raise ValueError('Could not reproduce original trajectory exactly')
    return captured[0]


def cases(run):
    path = run / 'completed/data/results/personamem-v2-32k.jsonl'
    rows = [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
            if line.endswith(b'\n')]
    row = next(r for r in rows if str(r['qa_id']) == '257')
    analogy = 'the heart needs as much tenderness as we’d give to an old friend in crisis'
    result = []
    for entry in row['choice_alignment']:
        claim = next(c['text'] for c in entry['claims'] if c['text'] == analogy)
        result.append(dict(claim_id='real_' + entry['letter'], option=entry['option'],
                           claim=claim, expected=False, origin='Q257'))
    examples = [
        ('Treat yourself as gently as you would a stranger having a difficult day.', False),
        ('Give yourself the patience you would offer a beginner learning a skill.', False),
        ('You deserve the same kindness one might offer a traveler who is lost.', False),
        ('Treat yourself as kindly as you treated your friend after her divorce last year.', True),
        ('Treat yourself with the care you give your dog every morning.', True),
        ('Like your friend Ana, you have experienced a painful divorce.', True),
        ('Be as patient as you were when you learned the violin you already own.', True),
        ('Since you do not have a close friend nearby, try a community group.', True),
        ('It may feel like a storm, but your friendship ended last month.', True),
        ('Imagine how you would comfort a friend if they were feeling sad.', False),
    ]
    result.extend(dict(claim_id=f'control_{i}', option=text, claim=text,
                       expected=expected, origin='synthetic')
                  for i, (text, expected) in enumerate(examples))
    return result


async def main(args):
    if config.FAKE:
        raise ValueError('Requires a real provider')
    controls = cases(args.run)
    captured = await original_prompt(args.run, args.prepared) if args.full_original else None
    if captured:
        targets = json.loads(captured.split('\n', 1)[1])
        analogy = controls[0]['claim']
        controls = [dict(t, expected=False if t['claim'] == analogy else None,
                         origin='original_full_scope') for t in targets]
    plan = dict(run=str(args.run), controls=controls, guidance=GUIDANCE,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                model=config.LLM_MODEL, repeats=2, full_original=args.full_original,
                original_prompt_sha256=hashlib.sha256(captured.encode()).hexdigest() if captured else None,
                caveat='Selected scope diagnostic, not answer accuracy. Labels never sent.')
    args.output.with_suffix('.plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    with args.output.open('x') as out:
        for repetition in range(2):
            ordered = controls if repetition == 0 or captured else list(reversed(controls))
            for enabled in ([False, True] if repetition == 0 else [True, False]):
                prompt = scope_prefix() + (GUIDANCE if enabled else '') + '\n' + json.dumps(
                    [{k: c[k] for k in ('claim_id', 'option', 'claim')} for c in ordered],
                    ensure_ascii=False)
                if captured:
                    prefix, data = captured.split('\n', 1)
                    prompt = prefix + (GUIDANCE if enabled else '') + '\n' + data
                with budget.scope(seconds=65, calls=1, tokens=24000):
                    response = await llm.complete_json(
                        prompt, schema=answer_choice.Scopes.model_json_schema(),
                        stage='diagnostic.analogy_scope', max_tokens=1536,
                        timeout=60, attempts=1)
                parsed = answer_choice.Scopes.model_validate(response)
                predictions = {c.claim_id: c.personal_fact for c in parsed.claims}
                if len(parsed.claims) != len(controls) or set(predictions) != {c['claim_id'] for c in controls}:
                    raise ValueError('Incomplete scope decisions')
                mismatches = [c['claim_id'] for c in controls if c['expected'] is not None
                              and predictions[c['claim_id']] != c['expected']]
                row = dict(repetition=repetition + 1, enabled=enabled, response=response,
                           mismatches=mismatches, prompt=prompt,
                           prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                out.flush()
                print(dict(repetition=repetition + 1, enabled=enabled, mismatches=mismatches), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-original', action='store_true')
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/personamem-v2-32k.jsonl'))
    asyncio.run(main(parser.parse_args()))
