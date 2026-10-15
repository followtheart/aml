"""Experimental single-call reader over the same original source catalog."""
from contextlib import nullcontext
import hashlib
import json

from . import answer_choice, answer_context, budget, llm

VERSION = 'direct-source-catalog-v1'


def make_prompt(qa, memories, sources, constraints):
    # Import locally: eval_scoring dispatches to this module.
    from .eval_scoring import answer_prompt
    context = json.dumps(dict(original_sources=answer_choice._cards(sources),
                              forget_constraints=constraints), ensure_ascii=False)
    original = answer_prompt(qa, memories)
    marker = '<memories>\n' + answer_context.build(memories) + '\n</memories>'
    if original.count(marker) != 1:
        raise ValueError('Ambiguous context boundary')
    prompt = original.replace(marker, '<memories>\n' + context + '\n</memories>')
    return prompt + '\nReturn the chosen letter in the structured answer field.', context


async def answer(qa, memories, diagnostics):
    sources, constraints = answer_choice.build_catalog(memories)
    constraints = list(constraints.values())
    prompt, context = make_prompt(qa, memories, sources, constraints)
    letters = list(answer_choice.option_map(qa['options']))
    schema = dict(type='object', additionalProperties=False, required=['answer'],
                  properties=dict(answer=dict(type='string', enum=letters)))
    diagnostics.update(answer_policy=VERSION, answer_reader_mode='direct_catalog',
                       answer_source_catalog=list(sources.values()),
                       answer_constraint_catalog=constraints,
                       answer_validation='selection_only_no_entailment_check',
                       answer_source_sha256=hashlib.sha256(context.encode()).hexdigest())
    call = dict(stage='eval.direct_catalog', attempt=1,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    diagnostics.setdefault('answer_calls', []).append(call)
    scope = nullcontext() if budget.current.get() else budget.scope(seconds=95, calls=1, tokens=64000)
    try:
        with scope:
            response = await llm.complete_json(prompt, schema=schema,
                stage=call['stage'], attempts=1, timeout=90, max_tokens=128)
        call['response'] = response
        if (not isinstance(response, dict) or set(response) != {'answer'}
                or not isinstance(response['answer'], str) or response['answer'] not in letters):
            raise ValueError('Invalid direct catalog answer')
        call['status'] = 'ok'
        diagnostics.update(answer_status='ok', answer_selected=response['answer'])
        return response['answer']
    except Exception as exc:
        call.update(status='error', error_type=type(exc).__name__, error_detail=str(exc)[:500])
        raise
