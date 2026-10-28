"""Constraints from full source texts visible in ranking-stage inputs.

This is not the final answer catalog and must not bypass runtime packet checks.
Source visibility is checked against the actual normalized candidate rank text.
"""
import copy
from . import answer_choice, answer_context, listwise_sources


def catalog(rows):
    memories=[];display=[]
    for row in rows:
        unit=copy.deepcopy(row['_packet_item'])
        actual=listwise_sources.normalized(row.get('_rank_text',row.get('content','')))
        body=unit.get('content','').split(answer_context.SOURCE_MARKER,1)[0]
        if listwise_sources.normalized(body) not in actual:body=''
        kept=[];hidden=[]
        for source in unit.get('sources',[]):
            if not source.get('content'):
                hidden.append(source);continue
            quote=answer_context.with_evidence('',[source]).split(answer_context.SOURCE_MARKER,1)[1]
            if listwise_sources.normalized(quote) in actual:kept.append(source)
            else:hidden.append(dict(source,content=''))
        # A summary with hidden source references must not become a legacy
        # source-less rule merely because those references are not displayed.
        unit['sources']=kept+hidden
        unit['content']=answer_context.with_evidence(body,kept)
        memories.append(unit);display.append(unit['content'])
    # Reuse the exact scope/role/forget parser, but supply verified rank-stage
    # visibility rather than the legacy, character-capped answer renderer.
    return list(answer_choice._catalog_from_visible(memories, '\n'.join(display))[1].values())


def missing(before, after):
    def identity(card):
        return tuple(card.get(key) for key in ('request_id','message_index','source_event_id','role'))
    return [card for card in before if not any(identity(card)==identity(other)
            and card['text'] in other['text'] for other in after)]
