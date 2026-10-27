"""Offline constraints from full source texts visible in ranking-stage inputs.

This is not the final answer catalog and must not bypass runtime packet checks.
Source visibility is checked against the actual normalized candidate rank text.
"""
import copy
from unittest.mock import patch


def catalog(rows):
    from app import answer_choice, answer_context, listwise_sources
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
    with patch.object(answer_context,'build',return_value='\n'.join(display)):
        return list(answer_choice.build_catalog(memories)[1].values())
