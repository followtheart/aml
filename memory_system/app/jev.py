"""Bounded OpenRouter Decisions transport for independent Choice judgments.

This endpoint has a different contract from chat completions. No prompt text,
credentials or provider error bodies are emitted to metrics or returned as debug
metadata. The caller owns any decision to recover from an unavailable judge.
"""
import asyncio
import json
import math
import time

import httpx

from . import budget, config, metrics


ENDPOINT = 'https://openrouter.ai/api/alpha/decisions'


class JevUnavailable(RuntimeError):
    """JEV is deliberately offline or its configuration is incomplete."""


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _request(state, questions):
    if not isinstance(state, (str, dict, list)) or not state or (
            isinstance(state, str) and not state.strip()):
        raise ValueError('JEV requires nonempty JSON state')
    if not isinstance(questions, dict) or not questions:
        raise ValueError('JEV requires a nonempty question map')
    for question_id, question in questions.items():
        if not _nonempty_string(question_id) or not isinstance(question, dict):
            raise ValueError('JEV requires nonempty string question identities')
        if question.get('type') != 'choice':
            raise ValueError('JEV adapter only accepts Choice questions')
        instructions, criteria = question.get('instructions'), question.get('criteria')
        if not isinstance(instructions, (str, dict, list)) or not instructions or (
                isinstance(instructions, str) and not instructions.strip()):
            raise ValueError('JEV Choice requires nonempty instructions')
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255 or any(
                not _nonempty_string(label) for label in criteria):
            raise ValueError('JEV Choice requires 2 to 255 nonempty option identities')
    body = {'model': config.JEV_MODEL, 'state': state, 'questions': questions}
    try:
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False,
                             separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('JEV request must contain finite JSON values') from None
    if len(encoded) > config.JEV_MAX_REQUEST_BYTES:
        raise ValueError('JEV request exceeds configured bound; evidence is not truncated')
    return body, len(encoded)


def _response_allowance(questions):
    # One UTF-8 byte per token is a conservative reservation. Choice responses
    # have fixed keys and finite numeric values, without generated free text.
    # Include identities verbatim, punctuation, numeric widths and metadata.
    size = 512
    for question_id, question in questions.items():
        labels = question['criteria']
        size += len(json.dumps(question_id).encode('utf-8')) + 160
        size += max(len(json.dumps(label).encode('utf-8')) for label in labels)
        size += sum(len(json.dumps(label).encode('utf-8')) + 32 for label in labels)
    return size


def _token_count(value):
    return value if type(value) is int and value >= 0 else None


def _usage(payload, input_bound, output_bound):
    raw = payload.get('usage') if isinstance(payload, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    prompt = _token_count(raw.get('input_tokens'))
    if prompt is None:
        prompt = _token_count(raw.get('prompt_tokens'))
    completion = _token_count(raw.get('output_tokens'))
    if completion is None:
        completion = _token_count(raw.get('completion_tokens'))
    # Missing/malformed provider usage cannot make billed requests free.
    prompt = input_bound if prompt is None else prompt
    completion = output_bound if completion is None else completion
    reported_total = _token_count(raw.get('total_tokens')) or 0
    result = {'input_tokens': prompt, 'output_tokens': completion,
        'prompt_tokens': prompt, 'completion_tokens': completion,
        'total_tokens': max(prompt + completion, reported_total)}
    cost = raw.get('cost')
    if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
        result['cost'] = cost
    return result


def _probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _parse(payload, questions, usage):
    if not isinstance(payload, dict) or not isinstance(payload.get('answers'), dict):
        raise ValueError('JEV response requires an answer map')
    answers = payload['answers']
    if answers.keys() != questions.keys():
        raise ValueError('JEV must answer every requested identity exactly once')
    clean = {}
    for question_id, question in questions.items():
        answer = answers[question_id]
        if not isinstance(answer, dict) or answer.get('type') != 'choice':
            raise ValueError('JEV returned a non-Choice answer')
        selected = answer.get('choice')
        probabilities = answer.get('probabilities')
        if not isinstance(selected, str) or selected not in question['criteria']:
            raise ValueError('JEV selected an unknown option')
        if (not isinstance(probabilities, dict)
                or probabilities.keys() != question['criteria'].keys()
                or not all(_probability(value) for value in probabilities.values())
                or not _probability(answer.get('confidence'))):
            raise ValueError('JEV Choice requires complete finite probabilities and confidence')
        # Official examples round each probability to two decimals. Accommodate
        # that rounding only, without rescaling or repairing returned values.
        tolerance = .005 * len(probabilities) + 1e-9
        if abs(sum(probabilities.values()) - 1.) > tolerance:
            raise ValueError('JEV probabilities do not sum to one')
        if probabilities[selected] < max(probabilities.values()):
            raise ValueError('JEV selected option is not a probability maximizer')
        clean[question_id] = {'type': 'choice', 'choice': selected,
            'confidence': float(answer['confidence']),
            'probabilities': {key: float(value) for key, value in probabilities.items()}}
    resolved_model = payload.get('model')
    if resolved_model is not None and not _nonempty_string(resolved_model):
        raise ValueError('JEV returned an invalid model identity')
    result = {'answers': clean, 'model': resolved_model or config.JEV_MODEL,
        'requested_model': config.JEV_MODEL, 'resolved_model': resolved_model,
        'usage': usage}
    if 'id' in payload:
        if not _nonempty_string(payload['id']):
            raise ValueError('JEV returned an invalid response identity')
        result['id'] = payload['id']
    return result


async def decide(state, questions, *, stage='eval.choice_jev'):
    """Return strictly validated Choice answers; never synthesize judge output.

    Retries share the existing provider budget/pacer. A single stage deadline
    includes semaphore waits, throttling, HTTP and retry backoff. Each dispatched
    attempt reserves a conservative input/output bound and releases it even when
    cancelled. Provider usage is charged for invalid structured responses too.
    """
    if config.FAKE:
        raise JevUnavailable('JEV is unavailable in offline mode')
    if not _nonempty_string(config.JEV_API_KEY) or not _nonempty_string(config.JEV_MODEL):
        raise JevUnavailable('JEV model or OpenRouter credential is not configured')
    body, input_bound = _request(state, questions)
    output_bound = _response_allowance(questions)
    reservation = input_bound + output_bound
    budget.check()
    limits = budget.current.get()
    timeout = config.JEV_TIMEOUT_SECONDS
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise JevUnavailable('JEV requires a finite positive timeout')
    if limits:
        timeout = min(timeout, max(0., limits.deadline - time.monotonic()))

    async def call(_attempt):
        if limits:
            limits.reserve_tokens(reservation)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                response = await client.post(ENDPOINT, json=body,
                    headers={'Authorization': 'Bearer ' + config.JEV_API_KEY})
                response.raise_for_status()
                payload, duplicates = None, False

                def unique_object(pairs):
                    nonlocal duplicates
                    result = {}
                    for key, value in pairs:
                        duplicates = duplicates or key in result
                        result[key] = value
                    return result

                try:
                    payload = response.json(object_pairs_hook=unique_object)
                    usage = _usage(payload, input_bound, output_bound)
                    if duplicates:
                        raise ValueError('JEV returned duplicate JSON identities')
                    return _parse(payload, questions, usage)
                except (ValueError, TypeError, OverflowError):
                    raise metrics.ResponseParseError({'usage':
                        _usage(payload, input_bound, output_bound)}) from None
        finally:
            if limits:
                limits.release_tokens(reservation)

    async with asyncio.timeout(timeout):
        return await metrics.measured_call(kind='decision', stage=stage,
            model=config.JEV_MODEL, call=call, attempts=1,
            input_count=len(questions))
