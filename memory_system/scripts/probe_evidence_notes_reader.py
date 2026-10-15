"""Paired direct-source versus structured evidence-note reader, with no gold in prompts.

Both arms use the same archived source catalog and forget constraints. The notes
are a bounded evidence summary, not a new retrieval or an eligibility filter.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import BaseModel, ConfigDict
from app import answer_choice, answer_context, budget, config, eval_scoring, evidence_packet, llm, run_metadata
from replay_query_context import records


class Reference(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    source_id: str
    quote: str


class Note(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    letter: str
    core_personal_premise: str
    evidence: list[Reference]
    unsupported_details: str
    relevance: str


class NotedAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    notes: list[Note]
    answer: str


class ReferenceNote(Note):
    evidence: list[str]


class ReferenceAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    notes: list[ReferenceNote]
    answer: str


NOTE_GUIDANCE = (
    '\nBefore selecting, record one concise evidence note for EACH option. '
    'core_personal_premise describes a pre-existing user fact asserted by the option; '
    'leave it empty for generic advice or world facts. evidence lists exact quotes with '
    'source_id from the supplied source/constraint cards, or [] when there is no relevant evidence. '
    'Keep notes short: at most two quotes per option. unsupported_details identifies missing, '
    'contradicted, outdated or prohibited personal details without treating them as facts. '
    'relevance briefly describes how the option addresses the request given the user context. '
    'A recommendation and its intended benefit are not prior personal history. '
    'Apply explicit forget constraints, preserve speaker attribution and temporal updates, '
    'and do not infer diagnoses or ownership merely from topical questions. '
    'Compare all options using these notes: prefer a relevant supported core with fewer '
    'unsupported details over unrelated personalization. A secondary unsupported detail is '
    'a limitation, not automatic rejection of an independently supported core. '
    'When personalization lacks support, compare the relevance of generic alternatives. '
    'Return the notes and chosen answer in the requested JSON schema.')


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    results = records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')
    if (args.run / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require completed HTTPS archive')
    traces = {r['search_id']: r for r in records(args.run / 'server-search-debug.jsonl')}
    questions = {(c['conversation_id'], q['id']): q for c in records(args.prepared) for q in c['qa']}
    if {(r['conversation_id'], r['qa_id']) for r in results} != set(questions):
        raise ValueError('Require complete prepared cohort, no score filtering')
    reference_only = getattr(args, 'reference_only', False)
    note_model = ReferenceAnswer if reference_only else NotedAnswer
    guidance = NOTE_GUIDANCE
    if reference_only:
        guidance = guidance.replace(
            'evidence lists exact quotes with source_id from the supplied source/constraint cards, or [] when there is no relevant evidence.',
            'evidence lists source_id strings from the supplied source/constraint cards, or [] when there is no relevant evidence. '
            'Return IDs only, never reconstruct quotations; the caller will resolve each ID to its original text.')
        guidance = guidance.replace('at most two quotes per option', 'at most two source IDs per option')
    plan = dict(run=str(args.run), prepared=str(args.prepared),
        prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        questions=[r['qa_id'] for r in results], arms=['direct', 'notes'],
        note_format='source_ids' if reference_only else 'literal_quotes', versions=run_metadata.versions(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grounding='llm-memory-survey/notes/13-longmemeval.md: structured evidence notes at reading stage',
        caveat='Same frozen sources; all provider calls live. No upstream retrieval or end-to-end score claim.')
    args.output.with_suffix('.plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    with args.output.open('x') as stream:
        for index, archived in enumerate(results):
            qa = questions[archived['conversation_id'], archived['qa_id']]
            if qa['qa_type'] != 'single_choice':
                raise ValueError('Single choice only')
            packet = traces[archived['search_id']]['returned']
            if evidence_packet.digest(packet) != archived['packet_hash']:
                raise ValueError('Packet identity mismatch')
            sources = {s['id']: s for s in archived['answer_source_catalog']}
            constraints = archived['answer_constraint_catalog']
            catalog = dict(sources, **{s['id']: s for s in constraints})
            context = json.dumps(dict(original_sources=answer_choice._cards(sources),
                                     forget_constraints=constraints), ensure_ascii=False)
            original = eval_scoring.answer_prompt(qa, packet)
            marker = '<memories>\n'+answer_context.build(packet)+'\n</memories>'
            if original.count(marker) != 1:
                raise ValueError('Ambiguous context boundary')
            base = original.replace(marker, '<memories>\n'+context+'\n</memories>')
            letters = list(answer_choice.option_map(qa['options']))
            for arm in (['direct', 'notes'] if index % 2 == 0 else ['notes', 'direct']):
                prompt = base + (guidance if arm == 'notes' else
                                 '\nReturn the chosen letter in the structured answer field.')
                schema = (note_model.model_json_schema() if arm == 'notes' else dict(
                    type='object', additionalProperties=False, required=['answer'],
                    properties=dict(answer=dict(type='string', enum=letters))))
                row = dict(qa_id=qa['id'], conversation_id=archived['conversation_id'], arm=arm,
                    packet_hash=archived['packet_hash'], source_sha256=hashlib.sha256(context.encode()).hexdigest(),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), prompt=prompt,
                    archived_score=archived['score'], archived_prediction=archived['prediction'])
                try:
                    with budget.scope(seconds=95, calls=1, tokens=64000) as usage:
                        response = await llm.complete_json(prompt, schema=schema,
                            stage='diagnostic.evidence_notes.'+arm, attempts=1, timeout=90,
                            max_tokens=2200 if arm == 'notes' else 128)
                    row.update(response=response, calls=usage.calls, tokens=usage.tokens)
                    if arm == 'notes':
                        parsed = note_model.model_validate(response)
                        if len(parsed.notes) != len(letters) or {n.letter for n in parsed.notes} != set(letters):
                            raise ValueError('Need one evidence note per option')
                        for note in parsed.notes:
                            if len(note.evidence) > 2:
                                raise ValueError('Evidence note exceeds quote bound')
                            for ref in note.evidence:
                                if reference_only:
                                    if ref not in catalog:
                                        raise ValueError('Unknown evidence source ID')
                                elif (ref.source_id not in catalog or not ref.quote.strip()
                                      or ref.quote not in catalog[ref.source_id]['text']):
                                    raise ValueError('Evidence quote is not from the cited source')
                        if reference_only:
                            row['resolved_evidence'] = {note.letter: [catalog[sid] for sid in note.evidence]
                                                        for note in parsed.notes}
                    elif set(response) != {'answer'}:
                        raise ValueError('Unexpected direct answer fields')
                    prediction = response['answer']
                    if prediction not in letters:
                        raise ValueError('Unknown answer')
                    row.update(prediction=prediction, score=eval_scoring.choice_score(
                        prediction, qa['gold_labels'], qa['qa_type']), status='ok')
                except Exception as exc:
                    row.update(status='error', error_type=type(exc).__name__, error=str(exc)[:500], score=0)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
                print(json.dumps({k:row.get(k) for k in ('qa_id','arm','prediction','score','status','error_type')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'prepared', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true')
    asyncio.run(main(parser.parse_args()))
