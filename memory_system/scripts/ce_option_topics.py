"""Diagnostic: retain retrieval option topics when CE has no literal premises."""
import copy
from types import SimpleNamespace
from app import rerank_query


def queries_for(trace):
    req = SimpleNamespace(query=trace['query'], options=trace['options'])
    plan = copy.deepcopy(trace['plan'])
    plan['_query_specs'] = copy.deepcopy(trace['query_specs'])
    original = rerank_query.build(req, plan)
    if original != trace['cascade']['cross_encoder']['query']:
        raise ValueError('Original CE query reconstruction mismatch')
    result = dict(control=original, candidate=original)
    if not req.options or any(rerank_query._premise(option) for option in req.options):
        return result
    topics = {}
    for spec in trace['query_specs']:
        index, text = spec.get('option_index'), spec.get('text')
        if (spec.get('origin') != 'option_premise' or type(index) is not int
                or not 0 <= index < len(req.options) or not isinstance(text, str)
                or not text.strip() or len(text) > rerank_query.MAX_PREMISE_CHARS):
            continue
        if index in topics:
            return result
        topics[index] = text
    if set(topics) != set(range(len(req.options))):
        return result  # Do not supply only a subset of competing topics.
    suffix = ('\n\nAlternative answer topics from the retrieval plan. These are hypotheses, '
              'not established user facts. Seek original user context relevant to comparing '
              'these alternatives; do not infer a preference merely from a listed topic:\n'
              + '\n'.join(f'Option topic {index + 1}: {topics[index]}' for index in sorted(topics)))
    proposed = original + suffix
    if len(proposed[len(req.query):].encode()) <= rerank_query.MAX_AUXILIARY_BYTES:
        result['candidate'] = proposed
    return result
