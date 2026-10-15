"""Compare all archived coarse candidates using whole versus focused CE views.

The focused view retains the memory body and one exact user-source window
selected by embedding similarity to the actual CE query. Packets are untouched.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, budget, config, embeddings
from app.personal_evidence import _sentences
from probe_ce_source_units import records, score_documents


async def main(args):
    if config.FAKE:
        raise ValueError('Requires real providers')
    result = next(r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')
                  if r['qa_id'] == args.question)
    trace = next(r for r in records(args.run / 'server-search-debug.jsonl')
                 if r.get('search_id') == result['search_id'])
    by_id = {r['id']: r for r in trace['ranked']}
    candidates = [by_id[mid] for mid in trace['cascade']['coarse']['selected_ids']]
    query = trace['cascade']['cross_encoder']['query']
    windows, lookup, refs = [], {}, []
    whole = [c.get('_rank_text', c['content']) for c in candidates]
    if args.output.exists():
        raise FileExistsError(args.output)
    for candidate, original in zip(candidates, whole):
        choices = []
        for source in (candidate.get('_packet_item') or {}).get('sources', []):
            text = source.get('content')
            if source.get('role') != 'user' or not text:
                continue
            if text not in original:
                raise ValueError('Source absent from original ranking unit')
            sentences = _sentences(text)
            spans = [(0, len(text))] if len(sentences) <= 2 else [
                (sentences[i].start(), sentences[i+1].end()) for i in range(len(sentences)-1)]
            for start, end in spans:
                window = text[start:end]
                if window not in lookup:
                    lookup[window] = len(windows)
                    windows.append(window)
                choices.append(dict(window_index=lookup[window], source=source,
                                    local_start=start, local_end=end))
        refs.append(choices)
    with budget.scope(seconds=240, calls=100, tokens=500000) as usage:
        vectors = await embeddings.embed(windows + [query], stage='probe.ce_window_selection')
        scores = vectors[:-1] @ vectors[-1]
        focused, selected = [], []
        for original, choices in zip(whole, refs):
            if not choices:
                focused.append(original)
                selected.append(None)
                continue
            best = max(choices, key=lambda r: float(scores[r['window_index']]))
            source = dict(best['source'], content=windows[best['window_index']])
            view = answer_context.with_evidence(original.split(answer_context.SOURCE_MARKER, 1)[0], [source])
            if len(view.encode()) > len(original.encode()):
                raise ValueError('Focused view grew instead of preserving the byte bound')
            focused.append(view)
            selected.append(best | {'similarity': float(scores[best['window_index']])})
        baseline_scores, baseline_batches = await score_documents(query, whole)
        focused_scores, focused_batches = await score_documents(query, focused)
    rows = [dict(id=c['id'], whole_score=float(a), focused_score=float(b),
                 selected=s, whole_document=w, focused_document=f)
            for c,a,b,s,w,f in zip(candidates,baseline_scores,focused_scores,selected,whole,focused)]
    report = dict(qa_id=args.question, search_id=trace['search_id'], query=query,
                  candidates=rows, windows=windows, calls=usage.calls, tokens=usage.tokens,
                  baseline_batches=baseline_batches, focused_batches=focused_batches,
                  orders={mode: [r['id'] for r in sorted(rows, key=lambda r: -r[mode])]
                          for mode in ('whole_score','focused_score')},
                  caveat='All original coarse candidates; actual archived CE query includes every option. '
                  'No gold or target ID used for selecting windows. User-source-only focus with whole '
                  'fallback; may discard relevant assistant or additional user evidence. Exact spans '
                  'relative to visible sources, preserved role/time. Expanded diagnostic budget. '
                  'No listwise, packet mutation, reader or HTTPS accuracy claim.')
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(candidates=len(rows), windows=len(windows), calls=usage.calls,
                          tokens=usage.tokens)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--output', type=Path, required=True)
    asyncio.run(main(p.parse_args()))
