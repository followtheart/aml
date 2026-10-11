"""Disambiguate actual personal identifiers from supplied objects and examples.

Diagnostic only. Expected labels stay outside prompts and production behavior
is unchanged. Includes existing verification controls and contrasting advice.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, budget, config, llm
from probe_verification_scope import cases, scope_prefix


GUIDANCE = (
    ' Distinguish requesting an unspecified value from asserting its actual value. '
    '"Enter your employee number" merely requests input, whereas '
    '"enter your employee number ZX-314159" assigns a specific existing identifier '
    'to the user, even though entering it is a proposed action. The assigned value '
    'need not be valid or supported to be an assertion. Explicit examples, placeholders, '
    'newly chosen identifiers, third-party identifiers and conditional possibilities '
    'do not establish an existing identifier of the current user. Preserve the original '
    'subject and modality; do not infer ownership from an identifier string alone.')


async def main(args):
    if config.FAKE:
        raise ValueError('Requires real provider')
    controls = cases(args.inventory)
    examples = [
        ('Enter your employee number AB-24680 in the form.', True),
        ('Enter your employee number in the form.', False),
        ('Enter a sample employee number such as AB-24680.', False),
        ('If your employee number is AB-24680, contact support.', False),
        ('Enter your colleague’s employee number AB-24680.', False),
        ('Check whether your account ending in 4051 has been charged.', True),
        ('Check whether an account has been charged.', False),
        ('Choose a new username such as brightleaf for this service.', False),
        ('Use your existing username brightleaf to sign in.', True),
        ('Use the example account number [ACCOUNT_NUMBER] to test the form.', False),
        ('Upload a scan of your license through the portal.', False),
        ('Include your license number INVALID-24680 in the upload.', True),
    ]
    controls.extend(dict(claim_id=f'identifier_{i}', option=text, claim=text,
                         expected=expected, origin='synthetic')
                    for i, (text, expected) in enumerate(examples))
    with args.output.open('x') as output:
        for repetition in range(2):
            ordered = controls if repetition == 0 else list(reversed(controls))
            batch_size = args.batch_size or len(ordered)
            for start in range(0, len(ordered), batch_size):
                batch = ordered[start:start + batch_size]
                for enabled in ([False, True] if repetition == 0 else [True, False]):
                    if args.arms != "both" and enabled != (args.arms == "treatment"):
                        continue
                    prompt = scope_prefix() + answer_choice.VERIFICATION_SCOPE_GUIDANCE
                    if enabled:
                        prompt += GUIDANCE
                    prompt += '\n' + json.dumps(
                        [{k: c[k] for k in ('claim_id', 'option', 'claim')} for c in batch],
                        ensure_ascii=False)
                    with budget.scope(seconds=65, calls=1, tokens=24000):
                        response = await llm.complete_json(
                            prompt, schema=answer_choice.Scopes.model_json_schema(),
                            stage='diagnostic.identifier_scope', attempts=1, timeout=60,
                            max_tokens=2000)
                    parsed = answer_choice.Scopes.model_validate(response)
                    predictions = {c.claim_id: c.personal_fact for c in parsed.claims}
                    if len(parsed.claims) != len(batch) or set(predictions) != {c['claim_id'] for c in batch}:
                        raise ValueError('Incomplete decisions')
                    mismatches = [c['claim_id'] for c in batch if predictions[c['claim_id']] != c['expected']]
                    row = dict(repetition=repetition + 1, enabled=enabled, controls=batch, batch_size=batch_size, batch_start=start,
                               response=response, mismatches=mismatches, prompt=prompt,
                               caveat='Selected grammatical controls, not benchmark accuracy.')
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    output.flush()
                    print(dict(repetition=repetition + 1, enabled=enabled,
                               mismatches=mismatches), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=range(1, 101), default=None)
    parser.add_argument('--arms', choices=['control', 'treatment', 'both'], default='both')
    asyncio.run(main(parser.parse_args()))
