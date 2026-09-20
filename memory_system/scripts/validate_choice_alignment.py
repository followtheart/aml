"""Match recorded packets by user/question/options and replay cited alignment.

--inspect validates pairings and hashes without model calls. By default replay
calls the configured model; --answer additionally evaluates final answers.
No Add ingestion is performed. Original traces/results are never overwritten.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, choice_alignment as alignment, config, eval_scoring as scoring, run_metadata


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pair_traces(traces, conversations, allow_repacked=False):
    index = {}
    for trace in traces:
        repacked = trace.get('status') == 'offline_repacked_not_verified'
        if trace.get('fake') and not (allow_repacked and repacked):
            continue
        if trace.get('status') == 'error' or 'returned' not in trace:
            continue
        if not trace.get('user_id', '').startswith('local:'):
            continue
        key = (trace['user_id'], trace['query'], tuple(trace.get('options') or []))
        if key in index:
            raise ValueError('Multiple recorded searches match the same question; select one run first')
        index[key] = trace
    pairs = []
    for conv in conversations:
        uid = f"local:{conv['dataset']}:{conv['conversation_id']}"
        for qa in conv['qa']:
            key = (uid, qa['question'], tuple(qa.get('options') or []))
            if key in index:
                pairs.append((conv, qa, index.pop(key)))
    if index:
        raise ValueError('Some real traces do not match the supplied dataset')
    if not pairs:
        raise ValueError('No matching real evaluation traces')
    return pairs


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traces', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output')
    ap.add_argument('--inspect', action='store_true', help='No provider calls')
    ap.add_argument('--answer', action='store_true', help='Also regenerate and score answers')
    ap.add_argument('--allow-repacked', action='store_true', help='Explicitly permit offline repacked, unverified experimental packets')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    pairs = pair_traces(read(args.traces), read(args.data), args.allow_repacked)
    if args.limit:
        pairs = pairs[:args.limit]
    for _, _, trace in pairs:
        answer_context.build(trace['returned'])
    if args.inspect:
        print(f'{len(pairs)} packets matched by user, question and options; hashes verified; provider_calls=0')
        return
    if args.output and Path(args.output).resolve() in {Path(args.traces).resolve(), Path(args.data).resolve()}:
        ap.error('Output cannot overwrite input data')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = Path(args.output).open('x', encoding='utf-8') if args.output else None
    correct = 0
    try:
        for conv, qa, trace in pairs:
            if qa.get('scoring') != 'choice' or qa.get('qa_type') != 'single_choice':
                continue
            row = dict(qa_id=qa['id'], conversation_id=conv['conversation_id'],
                       search_id=trace.get('search_id'), packet_hash=trace.get('packet_hash') or
                       next((m.get('packet_hash') for m in trace['returned']), None),
                       gold=qa['gold_labels'][0], fake=config.FAKE, model=config.LLM_MODEL,
                       input_status=trace.get('status'), versions=run_metadata.versions())
            try:
                # Keep every field, including hash version and provenance roles.
                memories = trace['returned']
                if args.answer:
                    prediction, score, diagnostics = await scoring.evaluate(qa, memories)
                    row.update(prediction=prediction, score=score, **diagnostics)
                    entries = diagnostics.get('choice_alignment') or []
                    correct += score
                else:
                    _, entries = await alignment.align(qa, memories, row)
                    row['choice_alignment'] = entries
                if args.answer and 'answer_eligible_options' in row:
                    row['top_tier'] = row['answer_eligible_options']
                    row['unique_supported_choice'] = (row['top_tier'][0] if len(row['top_tier']) == 1
                        and any(e['letter'] == row['top_tier'][0] and e.get('status') == 'supported' for e in entries) else None)
                    row['invalid_citations'] = sum(not c['valid'] for e in entries for claim in e.get('claims', [])
                                                   for c in claim.get('citations', []))
                else:
                    ranked = alignment.rank_alignment(entries or [])
                    best = alignment.unique_supported_choice(entries or [])
                    row['top_tier'] = [e['letter'] for e in ranked if alignment.rank_key(e) == alignment.rank_key(ranked[0])]
                    row['unique_supported_choice'] = best['letter'] if best else None
                    row['invalid_citations'] = sum('unsupported_citation' in e.get('validation_errors', []) for e in entries or [])
            except Exception as exc:
                row.update(error_stage='alignment_replay', error_type=type(exc).__name__)
            if output:
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
            print(f"qa{qa['id']} tier={row.get('top_tier')} score={row.get('score', 'not_evaluated')} error={row.get('error_type', '')}")
    finally:
        if output:
            output.close()
    if args.answer:
        print(f'answer correct: {correct}/{len(pairs)}')


if __name__ == '__main__':
    asyncio.run(main())
