"""Strict listwise output contracts and repairs isolated to one field.

Candidate relevance is a retrieval judgment, never a factual support label.
"""
import copy

RELEVANCE = ('useful', 'context', 'uncertain', 'irrelevant')


def schema(count, *, keyed=False, field=None, candidate_keys=None, group_keys=None,
           context_label=False):
    index = dict(type='integer', minimum=0, maximum=count - 1)
    keys = [f'c{i}' for i in range(count)]
    item = dict(type='object', additionalProperties=False, required=['priority', 'relevance'],
                properties=dict(priority=dict(type='integer', minimum=0, maximum=100),
                                relevance=dict(type='string', enum=list(RELEVANCE) if context_label else
                                               ['useful', 'uncertain', 'irrelevant'])))
    properties = (dict(candidates=dict(type='object', additionalProperties=False,
                                      required=keys if candidate_keys is None else candidate_keys,
                                      properties={k: copy.deepcopy(item) for k in (keys if candidate_keys is None else candidate_keys)}))
                  if keyed else dict(
                      ranking=dict(type='array', minItems=count, maxItems=count,
                                   uniqueItems=True, items=index),
                      irrelevant=dict(type='array', maxItems=count, uniqueItems=True, items=index)))
    group_index = dict(type='string', enum=keys if group_keys is None else (group_keys or keys)) if keyed else index
    properties['groups'] = dict(type='array', maxItems=(count if group_keys is None else len(group_keys)) // 2,
        items=dict(type='array', minItems=2, maxItems=4, uniqueItems=True, items=group_index))
    if field is not None:
        properties = {field: properties[field]}
    return dict(type='object', additionalProperties=False, required=list(properties), properties=properties)


def indices(value, count):
    return (isinstance(value, list) and all(type(i) is int and 0 <= i < count for i in value)
            and len(set(value)) == len(value))


class State:
    def __init__(self, count, *, keyed=False, tie_order=None):
        self.count, self.keyed = count, keyed
        self.fields = ['candidates', 'groups'] if keyed else ['ranking', 'irrelevant', 'groups']
        self.payload = {}
        self.accepted_candidates = {}
        self.ignored_candidate_keys = set()
        self.tie_order = list(range(count)) if tie_order is None else list(tie_order)
        if sorted(self.tie_order) != list(range(count)):
            raise ValueError('Tie order must be a complete permutation')
        self.errors, self.ranking, self.irrelevant, self.groups = {}, None, None, None

    def accept(self, result, field=None):
        expected = {field} if field else set(self.fields)
        if (not isinstance(result, dict) or set(result) - expected
                or field is not None and set(result) != expected):
            # An out-of-scope repair cannot overwrite any accepted field.
            error = ValueError('repair_fields_invalid' if field else
                               'response_not_object' if not isinstance(result, dict) else 'response_fields_invalid')
            error.validation_code = str(error)
            error.validation_details = dict(expected_fields=sorted(expected))
            raise error
        # Missing primary fields are diagnosed independently, so a missing
        # groups field cannot discard a complete candidate ordering.
        result = copy.deepcopy(result)
        if self.keyed and 'candidates' in result:
            cards = result['candidates']
            if isinstance(cards, dict):
                expected = {f'c{i}' for i in range(self.count)}
                self.ignored_candidate_keys.update(set(cards) - expected)
                for key, value in cards.items():
                    if key in expected and key not in self.accepted_candidates and self.valid_card(value):
                        self.accepted_candidates[key] = copy.deepcopy(value)
            # Accepted records are immutable across partial repair responses.
            result['candidates'] = copy.deepcopy(self.accepted_candidates)
        self.payload.update(result)
        self.evaluate()

    @staticmethod
    def valid_card(value):
        return (isinstance(value, dict) and set(value) == {'priority', 'relevance'}
                and type(value['priority']) is int and 0 <= value['priority'] <= 100
                and value['relevance'] in RELEVANCE)

    def pending_candidate_keys(self):
        return [f'c{i}' for i in range(self.count) if f'c{i}' not in self.accepted_candidates]

    def evaluate(self):
        self.errors, self.ranking, self.irrelevant, self.groups = {}, None, None, None
        p, n = self.payload, self.count
        if self.keyed:
            cards = p.get('candidates')
            expected = {f'c{i}' for i in range(n)}
            if not isinstance(cards, dict) or set(cards) != expected:
                self.errors['candidates'] = dict(code='candidate_keys_invalid',
                    missing=sorted(expected - set(cards)) if isinstance(cards, dict) else sorted(expected),
                    extra_count=len(set(cards) - expected) if isinstance(cards, dict) else 0)
            else:
                bad = [k for k, v in cards.items() if not isinstance(v, dict)
                       or set(v) != {'priority', 'relevance'}
                       or type(v['priority']) is not int or not 0 <= v['priority'] <= 100
                       or v['relevance'] not in RELEVANCE]
                if bad:
                    self.errors['candidates'] = dict(code='candidate_record_invalid', invalid_keys=sorted(bad))
                else:
                    # Lower is better. Ties preserve the CE order, not source IDs
                    # or the presentation order used by the prompt.
                    self.ranking = sorted(self.tie_order, key=lambda i: cards[f'c{i}']['priority'])
                    self.irrelevant = {i for i in range(n) if cards[f'c{i}']['relevance'] == 'irrelevant'}
        else:
            ranking, irrelevant = p.get('ranking'), p.get('irrelevant')
            if not indices(ranking, n) or len(ranking) != n:
                valid = [i for i in ranking if type(i) is int and 0 <= i < n] if isinstance(ranking, list) else []
                self.errors['ranking'] = dict(
                    code='ranking_incomplete' if indices(ranking, n) else 'ranking_invalid_indices',
                    missing=sorted(set(range(n)) - set(valid)),
                    repeated=sorted({i for i in valid if valid.count(i) > 1}),
                    invalid_entries=len(ranking) - len(valid) if isinstance(ranking, list) else 0)
            else:
                self.ranking = list(ranking)
            if not indices(irrelevant, n):
                self.errors['irrelevant'] = dict(code='irrelevant_invalid_indices')
            else:
                self.irrelevant = set(irrelevant)
        raw_groups = p.get('groups')
        groups, used = [], set()
        if not isinstance(raw_groups, list):
            self.errors['groups'] = dict(code='groups_not_array')
        elif len(raw_groups) > n // 2:
            self.errors['groups'] = dict(code='groups_count_exceeded')
        else:
            for group in raw_groups:
                if self.keyed:
                    keyset = {f'c{i}': i for i in range(n)}
                    group = ([keyset[k] for k in group] if isinstance(group, list)
                             and all(isinstance(k, str) and k in keyset for k in group) else None)
                code = None
                if not indices(group, n) or not 2 <= len(group) <= 4:
                    code = 'group_invalid_indices_or_size'
                elif self.irrelevant is not None and set(group) & self.irrelevant:
                    code = 'group_contains_irrelevant'
                elif set(group) & used:
                    code = 'groups_overlap'
                if code:
                    self.errors['groups'] = dict(code=code, group_index=len(groups))
                    break
                groups.append(list(group))
                used.update(group)
            if 'groups' not in self.errors and self.irrelevant is not None:
                self.groups = groups

    def next_field(self):
        return next((f for f in self.fields if f in self.errors), None)

    def require_valid(self):
        field = self.next_field()
        if field:
            detail = self.errors[field]
            error = ValueError(detail['code'])
            error.validation_code = detail['code']
            error.validation_details = dict(detail, repair_field=field)
            if self.ranking is not None:
                error.valid_ranking = list(self.ranking)
            raise error
        if self.ranking is None or self.irrelevant is None or self.groups is None:
            raise ValueError('Incomplete listwise contract')
        return self.ranking, self.irrelevant, self.groups
