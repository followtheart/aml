"""Nominate one whole affordable candidate from complete novel-source scores.

Scores only order nominees. Original candidate CE, content, fine/main membership
and factual-support labels are never overwritten. Missing/invalid scores fail
closed to the original main input; partial batches cannot nominate a winner.
"""
import math


def wire_body(model, api_format, query, documents):
    if api_format == 'cohere':
        return dict(model=model, query=query, documents=documents, top_n=len(documents))
    if api_format == 'dashscope':
        return dict(model=model, input=dict(query=query, documents=documents),
                    parameters=dict(top_n=len(documents), return_documents=False))
    raise ValueError('Unsupported CE API format')


def nominate(job, scores):
    if not job['scheduling']['eligible']:
        return dict(status='skipped', reason=job['scheduling']['reason'], addition=None)
    if set(scores) != set(job['views']):
        return dict(status='fallback', reason='incomplete_or_unknown_scores', addition=None)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in scores.values()):
        return dict(status='fallback', reason='invalid_scores', addition=None)
    ranked = []
    for candidate in job['nominees']:
        keys = candidate['source_view_ids']
        if not keys or any(key not in scores for key in keys):
            return dict(status='fallback', reason='invalid_membership', addition=None)
        ranked.append(dict(id=candidate['id'], score=max(scores[key] for key in keys),
                           ce_rank=candidate['ce_rank']))
    ranked.sort(key=lambda row: (-row['score'], row['ce_rank']))
    if not ranked:
        return dict(status='fallback', reason='no_nominees', addition=None)
    return dict(status='nominated', addition=ranked[0]['id'], ranking=ranked,
                policy='max_novel_complete_user_source_ce_then_original_ce_order',
                factual_support_inferred=False)
