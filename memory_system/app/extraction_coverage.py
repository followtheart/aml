"""Optional bounded retries for user exchanges buried in larger segments.

Coverage is a retry heuristic, not proof that every fact was extracted.
Original episodes and the ordinary grounding/semantic checks remain authoritative.
"""
import asyncio

from . import budget, memory_debug


async def supplement_facts(req, summary, segments, facts, extract):
    """Run after deterministic persona and forget-rule extraction.

Count only facts actually produced, rather than skipping messages by syntax.
Malformed/disabled persona parsing therefore still permits a coverage retry.
Keep original fact order and append only validated supplementary facts.
"""
    groups = [[f for f in facts if f.get('_segment') == index]
              for index in range(len(segments))]
    lengths = [len(group) for group in groups]
    with memory_debug.segment_scope(-1, []):
        memory_debug.extraction_event('coverage_known_sources', source_indices=sorted({
            i for fact in facts if fact.get('_deterministic')
            for i in fact.get('_sources', [])}))
    await supplement(req, summary, segments, groups, extract)
    for length, group in zip(lengths, groups):
        facts.extend(group[length:])


def uncovered_exchanges(req, segments, groups):
    for segment_index, indices in enumerate(segments):
        # The extractor's local-to-global mapping requires contiguous indices.
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            continue
        users = [i for i in indices if req.messages[i].role == 'user']
        if len(users) < 2:
            continue
        covered = {i for fact in groups[segment_index]
                   if fact.get('type') != 'episode' for i in fact.get('_sources', [])}
        for offset, start in enumerate(users):
            if start in covered:
                continue
            stop = users[offset + 1] if offset + 1 < len(users) else indices[-1] + 1
            yield segment_index, list(range(start, stop))


async def supplement(req, summary, segments, groups, extract, *, max_exchanges=4,
                     seconds=60, calls=8, tokens=64000):
    candidates = list(uncovered_exchanges(req, segments, groups))
    if not candidates:
        return
    # Never replace or bypass an enclosing provider budget. Add currently has
    # no such budget; callers that install one keep the normal extraction only.
    if budget.current.get() is not None:
        with memory_debug.segment_scope(-1, []):
            memory_debug.extraction_event('coverage_skipped', reason='enclosing_budget')
        return
    with budget.scope(seconds=seconds, calls=calls, tokens=tokens) as limits:
        try:
            async with asyncio.timeout(seconds):
                for segment_index, indices in candidates[:max_exchanges]:
                    limits.check()
                    if limits.calls >= limits.max_calls or limits.tokens >= limits.max_tokens:
                        break
                    with memory_debug.segment_scope(segment_index, indices):
                        memory_debug.extraction_event('coverage_retry',
                            parent_indices=segments[segment_index],
                            candidate_count=len(candidates), max_exchanges=max_exchanges)
                        result = await extract(req, summary, segment_index, indices)
                        # Retry episodes duplicate the original raw segment.
                        # Supplement user-source coverage only. Assistant-only
                        # facts cannot repair that gap and can rephrase a user
                        # trait without carrying its original user evidence.
                        user_indices = {i for i in indices if req.messages[i].role == 'user'}
                        atomic = [fact for fact in result if fact.get('type') != 'episode']
                        added = [fact for fact in atomic
                                 if user_indices.intersection(fact.get('_sources', []))]
                        ignored = [fact for fact in atomic
                                   if not user_indices.intersection(fact.get('_sources', []))]
                        if ignored:
                            memory_debug.extraction_event('coverage_unanchored', facts=ignored,
                                reason='no_original_user_source_in_retry_exchange')
                        groups[segment_index].extend(added)
                        memory_debug.extraction_event('coverage_output', facts=added,
                                                      calls=limits.calls, tokens=limits.tokens)
        except (TimeoutError, budget.BudgetExceeded) as exc:
            with memory_debug.segment_scope(-1, []):
                memory_debug.extraction_event('coverage_stopped', reason=str(exc) or type(exc).__name__,
                                              calls=limits.calls, tokens=limits.tokens)
