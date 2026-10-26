"""Remove only JSON separator spaces from canonical listwise graph metadata."""
import json


def renderer(original):
    def render(name, **kwargs):
        if name == '05c_listwise_rerank.txt':
            encoded = kwargs['relations']
            decoded = json.loads(encoded)
            # Production uses this exact encoder. Unexpected representations
            # remain untouched, including duplicate keys and alternate escapes.
            if json.dumps(decoded, ensure_ascii=False) == encoded:
                compact = json.dumps(decoded, ensure_ascii=False, separators=(',', ':'))
                if json.loads(compact) != decoded:
                    raise ValueError('Graph metadata roundtrip changed')
                if len(compact.encode('utf-8')) < len(encoded.encode('utf-8')):
                    kwargs['relations'] = compact
        return original(name, **kwargs)
    return render
