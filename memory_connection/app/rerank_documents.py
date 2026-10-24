"""Lossless, source-attributed fragments for bounded rerank APIs.

Fragments affect ranking only. Verification always receives original events.
"""
import json
from .temporal import source_time


def encoded(value):
    return json.dumps(value, ensure_ascii=False)


def documents(event, limit):
    original = encoded({'primary': event['text'], 'sources': [
        {'speaker': s.get('speaker_id'), 'role': s.get('role'), 'text': s['content'], **source_time(s)}
        for s in event['sources']]})
    if len(original.encode()) <= limit:
        return [original]
    result = []
    for source in event['sources']:
        text = source['content']
        header = {'role': source.get('role'), 'speaker': source.get('speaker_id'),
                  'primary': source['source_id'] == event['primary_source_id'],
                  'timestamp': source.get('timestamp')}
        start = 0
        while start < len(text):
            def render(end):
                return encoded({**header, 'start': start, 'end': end, 'text': text[start:end]})
            low, high = start, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if len(render(middle).encode()) <= limit:
                    low = middle
                else:
                    high = middle - 1
            if low == start:
                raise ValueError('rerank document limit cannot hold attributed text')
            end = low
            if end < len(text):
                # Prefer a sentence boundary without dropping any characters.
                boundary = max(text.rfind(mark, start, end) + len(mark)
                               for mark in ('\n', '. ', '。', '! ', '? '))
                if start + (end-start)//2 < boundary <= end:
                    end = boundary
            result.append(render(end))
            start = end
    return result
