"""Audit omitted constraint pairs without changing eligibility or choosing answers."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, budget, config, llm, prompts
from replay_query_context import records


class WitnessDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    pair_id: str
    forgotten_span: str
    option_span: str
    same_predicate_and_context: bool
    violates: bool


class WitnessDecisions(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    decisions: list[WitnessDecision]


async def main(args):
    if config.FAKE:
        raise ValueError('Real semantic probe required')
    questions = {(d['conversation_id'], q['id']): q
                 for d in records(args.prepared) for q in d['qa']}
    with args.output.open('x') as output:
        for result in records(args.results):
            qa = questions[result['conversation_id'], result['qa_id']]
            # Selection by lexical omissions only, independent of score/gold.
            pairs = []
            for letter, option in answer_choice.option_map(qa['options']).items():
                for constraint in result['answer_constraint_catalog']:
                    text = constraint['text']
                    shared = answer_choice._topic_terms(text) & answer_choice._topic_terms(option)
                    if shared and not answer_choice._scope_overlap(text, option):
                        pairs.append(dict(pair_id=f"{letter}:{constraint['id']}", letter=letter,
                            constraint_id=constraint['id'], constraint_quote=text,
                            option_quote=option, shared_terms=sorted(shared)))
            if not pairs:
                continue
            if len(pairs) > 24:
                raise ValueError('Probe exceeds bounded pair count')
            prompt = prompts.render('13_choice_constraints.txt', question=qa['question'],
                pairs=json.dumps([{k: v for k, v in p.items() if k != 'shared_terms'}
                                  for p in pairs], ensure_ascii=False))
            schema_class = answer_choice.ConstraintDecisions
            if args.witness:
                schema_class = WitnessDecisions
                prompt += ('\nBefore deciding, identify the specific forgotten predicate/object AND '
                    'its qualifying context. Return forgotten_span as an exact substring of the '
                    'constraint and option_span as an exact substring of the option describing '
                    'that same act or experience. Use empty spans when absent. Shared topic, '
                    'emotion, occupation, object, or a word alone is insufficient. Do not '
                    'generalize a context-specific experience to every occurrence of its emotion. '
                    'Set same_predicate_and_context=true only when both actually match; '
                    'violates=true requires that match. Include these fields for every pair_id.')
            with budget.scope(seconds=65, calls=1, tokens=32000):
                response = await llm.complete_json(prompt,
                    schema=schema_class.model_json_schema(),
                    stage='diagnostic.constraint_recall', attempts=1, timeout=60)
            parsed = schema_class.model_validate(response)
            expected = {p['pair_id'] for p in pairs}
            if len(parsed.decisions) != len(expected) or {d.pair_id for d in parsed.decisions} != expected:
                raise ValueError('Incomplete constraint decisions')
            invalid_witnesses = []
            if args.witness:
                by_id = {p['pair_id']: p for p in pairs}
                for decision in parsed.decisions:
                    pair = by_id[decision.pair_id]
                    if decision.violates and not (
                        decision.same_predicate_and_context and decision.forgotten_span
                        and decision.forgotten_span in pair['constraint_quote']
                        and decision.option_span and decision.option_span in pair['option_quote']):
                        invalid_witnesses.append(decision.pair_id)
            row = dict(qa_id=result['qa_id'], pairs=pairs, response=response,
                       witness_mode=args.witness, invalid_witnesses=invalid_witnesses,
                       model=config.LLM_MODEL, fake=config.FAKE,
                       prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                       caveat='Omitted lexical-overlap pairs only. No gold supplied, no eligibility change. '
                              'Semantic judgments require independent scope review before any gate change.')
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            print(dict(qa_id=row['qa_id'], pairs=len(pairs),
                       positives=[d.pair_id for d in parsed.decisions if d.violates]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--witness', action='store_true')
    asyncio.run(main(parser.parse_args()))
