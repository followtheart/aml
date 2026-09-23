"""Replay the actual answer entry point against frozen packets, never retrieval."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import controlled_validation as frozen
from app import answer_choice, budget, config, eval_scoring, llm, run_metadata


async def main(args):
    data = frozen.read_sealed(args.fixtures)
    original = llm.complete_json
    questions = args.questions.split(',')
    if args.output.resolve() == args.fixtures.resolve():
        raise ValueError('Output must not overwrite frozen input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    versions = run_metadata.versions()
    with args.output.open('x', encoding='utf-8') as output:
        for repetition in range(1, args.repeat + 1):
            for qid in questions:
                fixture = data['fixtures'][qid]
                qa = dict(fixture['qa'], gold_labels=fixture['oracle'])
                requests = []
                async def recorded(prompt, *positional, **kwargs):
                    request = dict(prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                   kwargs=kwargs)
                    requests.append(request)
                    try:
                        result = await original(prompt, *positional, **kwargs)
                        request['response'] = result
                        return result
                    except Exception as exc:
                        request['error_type'] = type(exc).__name__
                        raise
                calls = 8 + int(config.CHOICE_SEMANTIC_WITNESSES and not config.FAKE)
                with budget.scope(seconds=240, calls=calls, tokens=128000) as usage:
                    with patch.object(llm, 'complete_json', recorded):
                        prediction, score, diagnostics = await eval_scoring.evaluate(qa, fixture['packet'])
                row = dict(qid=qid, repetition=repetition, fixture_sha256=frozen.digest(data),
                           packet_hash=fixture['packet_hash'], prediction=prediction, score=score,
                           policy=answer_choice.VERSION, versions=versions, fake=config.FAKE,
                           provider_calls=usage.calls, tokens=usage.tokens,
                           diagnostics=diagnostics, requests=requests)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                print(json.dumps({k: row[k] for k in ('qid', 'repetition', 'prediction', 'score', 'provider_calls', 'tokens')}
                                 | dict(error=diagnostics.get('error_type')), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=ROOT / 'runs/personamem-controlled-validation/fixtures.json')
    parser.add_argument('--questions', default='3,9,11,12,15')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 3:
        parser.error('--repeat must be between 1 and 3')
    asyncio.run(main(args))
