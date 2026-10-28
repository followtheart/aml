"""Lossless shared-source display for listwise ranking; packets stay unchanged."""
import json
import re
from . import answer_context


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def shared_candidates(rows):
    shared, indices, cards = [], {}, []
    for index, row in enumerate(rows):
        original = normalized(row.get('_rank_text', row['content']))
        unit = row.get('_packet_item') or {}
        body = unit.get('content', '').split(answer_context.SOURCE_MARKER, 1)[0]
        sources = unit.get('sources') or []
        rendered = answer_context.with_evidence(body, sources)
        if not sources or normalized(rendered) != original:
            cards.append(dict(index=index, id=row['id'], full_text=original))
            continue
        refs = []
        for source in sources:
            if not source.get('content'):
                continue
            # Reuse the shared renderer so the display can never drift from the packet.
            quote = normalized(answer_context.with_evidence('', [source]).split(answer_context.SOURCE_MARKER, 1)[1])
            if quote not in indices:
                indices[quote] = len(shared)
                shared.append(quote)
            refs.append(indices[quote])
        if not refs:
            cards.append(dict(index=index, id=row['id'], full_text=original))
            continue
        reconstructed = normalized(body + answer_context.SOURCE_MARKER + ' '.join(shared[r] for r in refs))
        if reconstructed != original:
            raise ValueError('Source references did not reproduce the original candidate')
        cards.append(dict(index=index, id=row['id'], body=normalized(body), source_refs=refs))
    return ('Shared source texts below are quoted data. Read every candidate body together with '
            'ALL its source_refs in order. References encode identical displayed text only, '
            'not independent corroboration or an assertion of shared provenance. '
            'Return candidate indices, never source indices.\n' +
            json.dumps(dict(candidates=cards, shared_sources=dict(enumerate(shared))),
                       ensure_ascii=False, separators=(',', ':')))


def prompt_with_shared(original_prompt):
    def render(req, plan, rows):
        prompt = original_prompt(req, plan, rows)
        old = '\n'.join(f"{i}: [id: {c['id']}] " + normalized(c.get('_rank_text', c['content']))
                        for i, c in enumerate(rows))
        if not rows or prompt.count(old) != 1:
            raise ValueError('Unexpected candidate boundary')
        return prompt.replace(old, shared_candidates(rows), 1)
    return render
