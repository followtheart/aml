"""Use residual context bytes after preserving the CE and lexical reserve.

Keep complete immutable source units in their existing fine order. The caller's
item/byte caps and original extra-context byte allowance remain authoritative.
No new relevance judgments, source reads, or personal-fact labels are added.
"""
from .fine_context_lexical import propose as lexical_propose
from .fine_context_reserve import propose as reserve_propose

POLICY = 'fine_residual_context_v1'


def propose(packed, fine, *, user_id, top_k, byte_budget, query, options, reserve_bytes=4096):
    first, prefix = lexical_propose(packed, fine, user_id=user_id, top_k=top_k,
        byte_budget=byte_budget, query=query, options=options, reserve_bytes=reserve_bytes)
    result, tail = reserve_propose(first, fine, user_id=user_id, top_k=top_k,
        byte_budget=byte_budget, reserve_bytes=reserve_bytes-prefix['added_bytes'],
        limit=max(0, top_k-len(first)))
    if result[:len(first)] != first:
        raise ValueError('Existing reserve prefix changed')
    trace = dict(prefix, policy=POLICY, limit=max(0, top_k-len(packed)),
        result_count=len(result), total_bytes=tail['total_bytes'],
        proposed=prefix['proposed']+tail['proposed'],
        appended_ids=prefix['appended_ids']+tail['appended_ids'],
        added_bytes=prefix['added_bytes']+tail['added_bytes'],
        prefix_reserve_ids=prefix['appended_ids'], prefix_reserve_bytes=prefix['added_bytes'],
        residual=tail)
    return result, trace
