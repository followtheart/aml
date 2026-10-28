"""Diagnostic CE premise boundary for impersonal help-to advice clauses."""
import copy
import re
from types import SimpleNamespace
from unittest.mock import patch
from app import rerank_query


def extended_pattern():
    return re.compile(rerank_query._ADVICE.pattern.replace(
        r'(?=(?:', r'(?=(?:it\s+(?:might|may|can|could)\s+help(?:\s+you)?\s+to\b|', 1), re.I)


def queries_for(trace):
    req = SimpleNamespace(query=trace['query'], options=trace['options'])
    plan = copy.deepcopy(trace['plan'])
    plan['_query_specs'] = copy.deepcopy(trace['query_specs'])
    control = rerank_query.build(req, plan)
    if control != trace['cascade']['cross_encoder']['query']:
        raise ValueError('Original CE query not reconstructed exactly')
    with patch.object(rerank_query, '_ADVICE', extended_pattern()):
        candidate = rerank_query.build(req, plan)
    return {'control': control, 'candidate': candidate}
