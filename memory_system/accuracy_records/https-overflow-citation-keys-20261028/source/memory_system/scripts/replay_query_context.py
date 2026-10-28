"""Paired reader interventions with identical recorded upstream responses.

Query-context mode reruns context and selection; repair-premises mode freezes
witness/support responses and reruns scope, citations, entailment and selection.
Citation-context mode freezes witness/support and adds question-only original
source hints at citation recovery, keeping the late selection context disabled.
Current-question mode freezes only the retrieval packet and reruns the entire
reader because adding a source necessarily changes upstream prompts.
Verification-scope mode freezes witness/support and adds grammatical guidance
only to live premise-scope classification; all later evidence gates remain live.
Frozen responses must match original prompt hashes. This is a downstream
diagnostic, not end-to-end accuracy; validate winners with the HTTPS experiment.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import budget, choice_context, config, eval_scoring, llm, metrics, run_metadata


def records(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def recorded_response(call):
    """Replay recorded structured-output failure without inventing a payload."""
    if 'response' in call:
        return copy.deepcopy(call['response'])
    if call.get('failure_phase') == 'provider_parse':
        detail = call.get('error_detail', 'Recorded structured-output failure')
        if call.get('error_type') == 'LLMError':
            raise llm.LLMError(detail) from metrics.ResponseParseError(detail)
        if call.get('error_type') == 'ResponseParseError':
            raise metrics.ResponseParseError(detail)
        if call.get('error_type') == 'ValueError':
            raise ValueError(detail)
    raise LookupError('Recorded call has neither a response nor a reproducible parse failure')


async def main(args):
    if config.FAKE:
        raise ValueError('Live comparison requires non-fake configuration')
    results = records(args.results)
    if args.questions:
        selected = set(args.questions.split(','))
        results = [r for r in results if r['qa_id'] in selected]
    search_ids = {r['search_id'] for r in results}
    searches = {r['search_id']: r for r in records(args.search_log) if r['search_id'] in search_ids}
    qa = {(d['conversation_id'], q['id']): q for d in records(args.prepared) for q in d['qa']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original = llm.complete_json
    metadata = run_metadata.versions()
    support_stages = {'eval.choice_witness', 'eval.choice_support', 'eval.choice_support.repair'}
    flag = {'query-context': 'CHOICE_QUERY_CONTEXT', 'citation-context': 'CHOICE_QUERY_CONTEXT',
            'verification-scope': 'CHOICE_VERIFICATION_SCOPE',
            'scope-roles': 'CHOICE_SCOPE_ROLES',
            'grouped-entailment': 'CHOICE_GROUPED_ENTAILMENT',
            'conditional-list': 'CHOICE_CONDITIONAL_LIST_SCOPE',
            'template-path': 'CHOICE_TEMPLATE_PATH_SCOPE',
            'repair-premises': 'CHOICE_PRESERVE_REPAIR_PREMISES',
            'current-question': 'CHOICE_CURRENT_QUESTION_SOURCE'}[args.intervention]
    with args.output.open('x', encoding='utf-8') as output:
        for repetition in range(args.repeat):
            for index, result in enumerate(results):
                # Alternate arm order to avoid coupling an arm to provider time.
                for enabled in ([False, True] if (index + repetition) % 2 == 0 else [True, False]):
                    upstream = [c for c in result['answer_calls'] if (
                        c['stage'] in support_stages if args.intervention != 'query-context' else
                        c['stage'] not in ('eval.choice_select', 'eval.choice_select_review', 'eval.choice_context'))]
                    if args.intervention == 'current-question':
                        upstream = []
                    if args.intervention == 'grouped-entailment':
                        calls = result['answer_calls']
                        boundary = next((i for i, c in enumerate(calls)
                                         if c['stage'] == 'eval.choice_entailment'), len(calls))
                        upstream = calls[:boundary]
                    pending = list(upstream)
                    live_calls, mismatches, context_diagnostics = [], [], {}

                    async def replay(prompt, *positional, **kwargs):
                        stage = kwargs['stage']
                        live = (stage not in support_stages if args.intervention != 'query-context' else
                                stage in ('eval.choice_select', 'eval.choice_select_review', 'eval.choice_context'))
                        if args.intervention == 'current-question':
                            live = True
                        if args.intervention == 'grouped-entailment':
                            live = not pending
                        if live:
                            if args.intervention == 'citation-context' and enabled and stage == 'eval.choice_citation_recovery':
                                # Retrieve by question only; preserve every original card and
                                # all later citation/provenance/entailment checks.
                                with patch.object(config, 'CHOICE_QUERY_CONTEXT', True):
                                    focused = await choice_context.select(
                                        qa[(result['conversation_id'], result['qa_id'])]['question'],
                                        {s['id']: s for s in result['answer_source_catalog']},
                                        context_diagnostics)
                                prompt += ('\nQuestion-focused original sources for evidence lookup:\n'
                                           + json.dumps(focused, ensure_ascii=False)
                                           + '\nRelevance labels are retrieval hints, not proof of a premise. '
                                             'Read original wording, retain scope and attribution, and use '
                                             'any stronger evidence in the full catalog. Do not choose an answer.')
                            live_calls.append(dict(stage=stage, prompt=prompt))
                            value = await original(prompt, *positional, **kwargs)
                            live_calls[-1]['response'] = value
                            return value
                        call = pending.pop(0) if pending else {}
                        if (call.get('stage') != stage
                                or call.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest()):
                            mismatches.append(dict(stage=stage, expected=call.get('stage')))
                            raise ValueError('Frozen upstream prompt mismatch')
                        # Preserve call headroom; original token/time use cannot
                        # be reconstructed from this diagnostic log.
                        if budget.current.get():
                            budget.current.get().before_call()
                        try:
                            return recorded_response(call)
                        except LookupError:
                            mismatches.append(dict(stage=stage, reason='unreplayable_recorded_call'))
                            raise

                    search = searches[result['search_id']]
                    if search['packet_hash'] != result['packet_hash']:
                        raise ValueError('Frozen packet mismatch')
                    flag_value = False if args.intervention == 'citation-context' else enabled
                    with patch.object(config, flag, flag_value), patch.object(llm, 'complete_json', replay):
                        prediction, score, diagnostics = await eval_scoring.evaluate(
                            qa[(result['conversation_id'], result['qa_id'])], copy.deepcopy(search['returned']))
                    if mismatches or pending:
                        raise ValueError(f'Upstream replay diverged for QA {result["qa_id"]}: {mismatches}, pending={len(pending)}')
                    row = dict(conversation_id=result['conversation_id'], qa_id=result['qa_id'],
                               repetition=repetition + 1, intervention=args.intervention, enabled=enabled,
                               context_enabled=flag_value if flag == 'CHOICE_QUERY_CONTEXT' else config.CHOICE_QUERY_CONTEXT,
                               citation_context=context_diagnostics,
                               prediction=prediction, score=score, packet_hash=search['packet_hash'],
                               versions=dict(metadata, settings=dict(metadata['settings'], **{flag: flag_value})),
                               diagnostics=diagnostics, live_calls=live_calls,
                               caveat=('Frozen retrieval packet, all reader calls live; upstream model responses may differ.'
                                       if args.intervention == 'current-question' else
                                       'Frozen upstream responses; upstream token/time budget not replayed.'))
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    output.flush()
                    print(json.dumps({k: row[k] for k in ('qa_id', 'repetition', 'intervention', 'enabled',
                                                        'prediction', 'score')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--search-log', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/prepared/personamem-v2-32k.jsonl')
    parser.add_argument('--questions')
    parser.add_argument('--intervention', choices=['query-context', 'repair-premises', 'citation-context', 'current-question', 'verification-scope', 'scope-roles', 'grouped-entailment', 'conditional-list', 'template-path'], default='query-context')
    parser.add_argument('--repeat', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
