"""Compare semantic-check payloads on fixed synthetic examples, without retrieval."""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import add_pipeline, config, llm


CASES = [
    ('inquiry', 'How can I plan a small herb garden?',
     'The user asked how to plan a small herb garden.', 'asked_about', 'planning a small herb garden', True),
    ('self_report', 'I own a telescope.', 'The user owns a telescope.', 'owns', 'a telescope', True),
    ('question_not_ownership', 'What telescope should I buy?',
     'The user owns a telescope.', 'owns', 'a telescope', False),
    ('third_party', 'My colleague said: "I own a telescope."',
     'The user owns a telescope.', 'owns', 'a telescope', False),
    ('negation', 'I do not own a telescope.',
     'The user owns a telescope.', 'owns', 'a telescope', False),
]


async def main(output):
    if config.FAKE:
        raise RuntimeError('This probe requires real model judgments')
    original = llm.complete_json
    with output.open('x', encoding='utf-8') as out:
        for case, source, content, relation, obj, expected in CASES:
            fact = dict(content=content, type='fact', state=None, time_expression=None,
                        epistemic_status='asserted', sensitivity='normal', _sources=[0], _segment=0,
                        event_time=None, entities=['user'], keywords=[], retrieval_key=content,
                        evidence=[dict(message_index=0, quote=source)],
                        triples=[dict(subject='user', relation=relation, object=obj)])
            messages = [dict(message_index=0, role='user', content=source)]
            for mode in ('current', 'semantic_fields_only'):
                row = dict(case=case, mode=mode, expected=expected, model=config.LLM_MODEL)

                async def recorded(prompt, *args, **kwargs):
                    if mode == 'semantic_fields_only':
                        header, payload = prompt.rsplit('\n', 1)
                        payload = json.loads(payload)
                        keys = ('content', 'state', 'time_expression', 'evidence', 'triples')
                        payload['facts'] = [{k: f[k] for k in keys if k in f} for f in payload['facts']]
                        prompt = header + '\n' + json.dumps(payload, ensure_ascii=False)
                    row['prompt'] = prompt
                    kwargs.update(stage='probe.semantic.' + mode, max_tokens=768, timeout=30, attempts=1)
                    result = await original(prompt, *args, **kwargs)
                    row['response'] = result
                    return result

                try:
                    with patch.object(llm, 'complete_json', recorded):
                        await add_pipeline._verify_semantics([copy.deepcopy(fact)], messages)
                except Exception as exc:
                    row['error_type'] = type(exc).__name__
                    row['error_detail'] = str(exc)[:1000]
                row['matches_expected'] = row.get('response', {}).get('valid') is expected
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                out.flush()
                print(json.dumps({k: row[k] for k in ('case', 'mode', 'expected', 'matches_expected')}
                                 | dict(response=row.get('response')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.output))
