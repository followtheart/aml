"""Experimental option-local verification without changing evidence objects."""
import json
import time

from . import budget


async def verify(prompt, schema, checks, independent, judge, validate, diagnostics):
    trace = diagnostics['answer_entailment_groups'] = dict(status='fallback')

    async def batch(reason):
        trace['reason'] = reason
        return await judge(prompt, schema, lambda value: validate(value, checks),
                           'eval.choice_entailment', diagnostics)

    try:
        before, rest = prompt.split('<checks>\n', 1)
        raw, after = rest.split('\n</checks>', 1)
        if json.loads(raw) != independent:
            raise ValueError('check block mismatch')
        ids = [c['claim_id'] for c in checks]
        if len(set(ids)) != len(ids) or ids != [c['claim_id'] for c in independent]:
            raise ValueError('check identity mismatch')
        groups = {}
        for item in independent:
            letter, separator, suffix = item['claim_id'].partition(':')
            if not separator or not suffix or len(letter) != 1 or not letter.isupper():
                raise ValueError('invalid option identity')
            groups.setdefault(letter, []).append(item)
        if not 2 <= len(groups) <= 4:
            raise ValueError('unsupported group count')
    except (ValueError, KeyError, TypeError):
        return await batch('invalid_grouping')
    requests = [(letter, before + '<checks>\n' + json.dumps(items, ensure_ascii=False)
                 + '\n</checks>' + after, items) for letter, items in groups.items()]
    # UTF-8 bytes deliberately overestimate usual input token counts. Include
    # schema/system overhead and the same output limit used by the judge.
    estimate = sum(len(p.encode()) + len(json.dumps(schema).encode()) + 4096
                   for _, p, _ in requests)
    limits = budget.current.get()
    trace.update(groups={k: [c['claim_id'] for c in v] for k, v in groups.items()},
                 estimated_tokens=estimate)
    if limits is None:
        return await batch('missing_budget')
    if limits.max_calls - limits.calls - limits.reserved_calls < len(groups) + 4:
        return await batch('call_headroom')
    if limits.max_tokens - limits.tokens - limits.reserved_tokens < estimate + 16384:
        return await batch('token_headroom')
    if limits.deadline - time.monotonic() < 20 * len(groups) + 30:
        return await batch('time_headroom')
    limits.reserve_calls(4)
    trace.update(status='grouped', failures=[])
    combined = []
    try:
        for letter, subprompt, items in requests:
            group_ids = {c['claim_id'] for c in items}
            original_checks = [c for c in checks if c['claim_id'] in group_ids]
            try:
                result = await judge(subprompt, schema,
                    lambda value: validate(value, original_checks),
                    'eval.choice_entailment.group.' + letter, diagnostics,
                    attempts=1, timeout=20)
            except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError) as exc:
                trace['failures'].append(dict(group=letter, error_type=type(exc).__name__))
                # Every failed identity remains present and fails closed. Use
                # the same validator to populate optional typed-scope metadata.
                result = validate(dict(checks=[dict(claim_id=c['claim_id'],
                    verdict='unsupported_personal') for c in original_checks]), original_checks)
            combined.extend(result['checks'])
    finally:
        limits.release_calls(4)
    by_id = {c['claim_id']: c for c in combined}
    return dict(checks=[by_id[c['claim_id']] for c in checks])
