"""Synthetic OpenRouter Decisions smoke. No evaluation/user data is sent."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, jev


STATE = {'role': 'user', 'evidence': 'I own a red bicycle. I do not own a car.'}
EXPECTED = {'bicycle': 'supported', 'car': 'contradicted', 'boat': 'insufficient'}
QUESTIONS = {key: dict(type='choice',
    instructions='Classify whether the evidence supports this claim about the speaker: ' + claim,
    criteria={'supported': 'Evidence establishes the claim.',
              'contradicted': 'Evidence explicitly denies the claim.',
              'insufficient': 'Evidence neither establishes nor denies the claim.'})
    for key, claim in {'bicycle': 'The speaker owns a bicycle.',
                       'car': 'The speaker owns a car.', 'boat': 'The speaker owns a boat.'}.items()}


async def main(live):
    if not live:
        print(json.dumps(dict(status='dry_run', model=config.JEV_MODEL,
                              state=STATE, questions=QUESTIONS), ensure_ascii=False))
        return 0
    if config.FAKE or not config.JEV_API_KEY.strip():
        print(json.dumps(dict(status='not_run', reason='fake' if config.FAKE else 'missing_key')))
        return 2
    try:
        with budget.scope(seconds=30, calls=1, tokens=16000) as usage:
            result = await jev.decide(STATE, QUESTIONS, stage='smoke.jev')
        actual = {key: value['choice'] for key, value in result['answers'].items()}
        ok = actual == EXPECTED
        print(json.dumps(dict(status='passed' if ok else 'unexpected_judgments',
                              result=result, expected=EXPECTED, provider_calls=usage.calls), ensure_ascii=False))
        return 0 if ok else 1
    except Exception as exc:
        # Provider messages may include request details; report only the class.
        print(json.dumps(dict(status='error', error_type=type(exc).__name__)))
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Send one billed request containing synthetic facts only')
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.live)))
