"""Matched grammatical-scope controls; no evidence, question or answer labels sent."""
import argparse
import ast
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, budget, config, llm


def scope_prefix():
    tree = ast.parse(Path(answer_choice.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                    and n.name == 'reclassify_uncited')
    assignment = next(n for n in function.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets))
    return ast.literal_eval(assignment.value.left).rstrip('\n')


GUIDANCE = answer_choice.VERIFICATION_SCOPE_GUIDANCE


def cases(path):
    inventory = json.loads(path.read_text())['cases']
    result = []
    for cid in ('A:0', 'B:0', 'D:0'):
        case = next(c for c in inventory if c['qa_id'] == '63' and c['claim_id'] == cid)
        result.append(dict(claim_id='real_' + cid, option=case['option'], claim=case['claim'],
                           expected=cid != 'A:0', origin='Q63'))
    examples = [
        ('Check whether the account has been charged before retrying the payment.',
         'the account has been charged', False),
        ('Verify that your account number (e.g., [ACCOUNT_NUMBER]) matches the form.',
         'your account number (e.g., [ACCOUNT_NUMBER]) matches the form', False),
        ('Verify that your account number 123456 matches the form.',
         'your account number 123456', True),
        ('Check that the violin you already own is stored safely.',
         'the violin you already own', True),
        ('Ensure that your diagnosed asthma is monitored by your clinician.',
         'your diagnosed asthma', True),
        ('Since you do not own a bicycle, consider renting one.',
         'you do not own a bicycle', True),
        ('Check whether a rental bicycle is available tomorrow.',
         'a rental bicycle is available tomorrow', False),
    ]
    result.extend(dict(claim_id=f'control_{i}', option=o, claim=c, expected=e, origin='synthetic')
                  for i, (o, c, e) in enumerate(examples))
    return result


async def main(args):
    if config.FAKE:
        raise ValueError('Requires real provider')
    controls = cases(args.inventory)
    with args.output.open('x') as output:
        for repetition in range(2):
            ordered = controls if repetition == 0 else list(reversed(controls))
            for enabled in ([False, True] if repetition == 0 else [True, False]):
                prompt = scope_prefix() + (GUIDANCE if enabled else '') + '\n' + json.dumps(
                    [{k: c[k] for k in ('claim_id', 'option', 'claim')} for c in ordered],
                    ensure_ascii=False)
                with budget.scope(seconds=65, calls=1, tokens=24000):
                    response = await llm.complete_json(prompt, schema=answer_choice.Scopes.model_json_schema(),
                        stage='diagnostic.verification_scope', max_tokens=1024, timeout=60, attempts=1)
                parsed = answer_choice.Scopes.model_validate(response)
                predictions = {c.claim_id: c.personal_fact for c in parsed.claims}
                if len(parsed.claims) != len(controls) or set(predictions) != {c['claim_id'] for c in controls}:
                    raise ValueError('Incomplete scope decisions')
                mismatches = [c['claim_id'] for c in controls if predictions[c['claim_id']] != c['expected']]
                row = dict(repetition=repetition + 1, enabled=enabled, controls=controls,
                    response=response, mismatches=mismatches, prompt=prompt, model=config.LLM_MODEL,
                    caveat='Selected real and synthetic grammatical controls; expected labels never sent. '
                           'Not benchmark accuracy or a production change.')
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                print(dict(repetition=repetition + 1, enabled=enabled, mismatches=mismatches), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
