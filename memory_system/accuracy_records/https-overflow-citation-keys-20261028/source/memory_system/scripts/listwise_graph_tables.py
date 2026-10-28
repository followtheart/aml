"""Experimental lossless column encoding of listwise graph metadata only."""
import json

GUIDANCE = ('Column tables preserve the original records: map each row value to the column '
            'at the same position. Missing lists column indexes absent from that row; '
            'other null values remain explicit nulls. Row order is unchanged.\n')


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def table(records):
    if not records or not all(isinstance(row, dict) for row in records):
        return records
    columns = list(dict.fromkeys(key for row in records for key in row))
    value = dict(format='column_table_v1', columns=columns,
                 rows=[[row.get(key) for key in columns] for row in records])
    missing = {str(i): [j for j, key in enumerate(columns) if key not in row]
               for i, row in enumerate(records)}
    missing = {i: positions for i, positions in missing.items() if positions}
    if missing:
        value['missing'] = missing
    return value if len(dumps(value).encode()) < len(dumps(records).encode()) else records


def decode(metadata):
    restored = dict(metadata)
    for key in ('triples', 'associations'):
        value = restored[key]
        if isinstance(value, list):
            continue
        assert value['format'] == 'column_table_v1'
        columns = value['columns']
        assert len(columns) == len(set(columns))
        assert all(len(row) == len(columns) for row in value['rows'])
        restored[key] = [{k: v for j, (k, v) in enumerate(zip(columns, row))
                          if j not in value.get('missing', {}).get(str(i), [])}
                         for i, row in enumerate(value['rows'])]
    return restored


def encode(metadata):
    compact = dumps(metadata)
    transformed = dict(metadata)
    for key in ('triples', 'associations'):
        transformed[key] = table(metadata[key])
    assert decode(transformed) == metadata
    proposed = GUIDANCE + dumps(transformed)
    return proposed if len(proposed.encode()) < len(compact.encode()) else compact


def renderer(original, mode):
    def render(name, **kwargs):
        if name == '05c_listwise_rerank.txt':
            metadata = json.loads(kwargs['relations'])
            if mode == 'compact':
                kwargs['relations'] = dumps(metadata)
            elif mode == 'tables':
                kwargs['relations'] = encode(metadata)
            else:
                raise ValueError('Unknown graph encoding')
        return original(name, **kwargs)
    return render
