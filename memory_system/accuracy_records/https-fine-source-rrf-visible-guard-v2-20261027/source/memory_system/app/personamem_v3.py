"""Time-isolated PersonaMem-v3 chatbot text proxy, not the official evaluator."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

APPS = ('instagram', 'facebook', 'threads', 'chatbot', 'ai_studio')
PROTOCOL = 'personamem-v3-chatbot-text-proxy-v1'
CONTENT_FIELDS = ('title', 'caption', 'overall_description', 'audio_transcript', 'text')


def event_messages(app, event):
    """Allowlist observations; never turn inferred preference annotations into history."""
    ts = event['source_timestamp']
    if type(ts) is not int or ts <= 0:
        raise ValueError('Invalid event timestamp')
    meta = {k: event[k] for k in ('source_object_id', 'source_timestamp', 'source_hashtags',
            'author_id', 'recipient_id', 'relationship', 'is_self_authored', 'is_dm') if k in event}
    interaction = event.get('interaction_format') or {}
    meta['interaction'] = {k: interaction[k] for k in ('action', 'action_label', 'user_message') if k in interaction}
    meta['app'] = app
    content = event.get('content')
    if isinstance(content, str):
        meta['content'] = content
    elif isinstance(content, dict):
        meta['content'] = {k: content[k] for k in CONTENT_FIELDS if k in content}
    messages = [{'role': 'user', 'content': 'Observed platform event. Content may be authored by another person; '
                 'engagement alone does not establish a personal fact.\n' + json.dumps(meta, ensure_ascii=False),
                 'timestamp': ts * 1000}]
    for turn in event.get('conversation') or []:
        if turn.get('role') not in ('user', 'assistant') or not isinstance(turn.get('content'), str):
            raise ValueError('Invalid conversation turn')
        if turn['content'].strip():
            messages.append({'role': turn['role'], 'content': turn['content'], 'timestamp': ts * 1000})
    return messages


def conversations(index_path):
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    if index.get('dataset') != 'personamem-v3':
        raise ValueError('Expected downloaded PersonaMem-v3 index')
    for uid in index['users']:
        if not str(uid).isdigit():
            raise ValueError('Invalid user ID')
        root = index_path.parent / 'backend' / str(uid)
        # Verify every downloaded byte before interpreting it.
        for name in [*(app + '.json' for app in APPS), 'test.json']:
            relative = f'backend/{uid}/{name}'
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != index['files'][relative]['sha256']:
                raise ValueError('Source hash mismatch: ' + relative)
        events = [(e['source_timestamp'], app, str(e['source_object_id']), e)
                  for app in APPS for e in json.loads((root / (app + '.json')).read_text())]
        events.sort(key=lambda x: x[:3])
        seen = set()
        for q in json.loads((root / 'test.json').read_text()):
            if q['query_id'] in seen:
                raise ValueError('Duplicate query ID')
            seen.add(q['query_id'])
            if q['task_type'] != 'chatbot_personalized_response':
                continue
            if q.get('prior_conversation'):
                raise ValueError('Prior-conversation variant requires a dedicated adapter')
            cutoff = q['ts']
            if type(cutoff) is not int or cutoff <= 0:
                raise ValueError('Invalid query cutoff')
            preference = q.get('groundtruth_preference')
            if not isinstance(preference, str) or not preference.strip():
                raise ValueError('Missing chatbot evaluation preference')
            heldout = str((q.get('reference_example') or {}).get('source_object_id', ''))
            selected = [(t, app, oid, e) for t, app, oid, e in events if t < cutoff and oid != heldout]
            if not selected:
                raise ValueError('No pre-query history')
            messages = [m for _, app, _, e in selected for m in event_messages(app, e)]
            yield {'conversation_id': str(uid) + ':' + q['query_id'], 'dataset': 'personamem-v3',
                   'protocol': PROTOCOL, 'source_revision': index['revision'], 'source_user_id': str(uid),
                   'cutoff_seconds': cutoff, 'source_event_count': len(selected),
                   'sessions': [messages], 'session_ids': ['time-masked-events'],
                   'qa': [{'id': q['query_id'], 'question': q['user_query'],
                           'question_date': datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
                           'category': q['task_type'], 'scoring': 'rubric_mean',
                           'rubrics': ['Give useful, coherent advice that directly answers this question; '
                                       'avoid unsupported factual claims about the user.',
                                       'Appropriately tailor the answer to this evaluation-only preference: ' + preference +
                                       '. Equivalent suggestions are acceptable; do not require a particular example answer.'],
                           'evaluation_reference': q.get('example_response'),
                           'evaluation_policy': PROTOCOL}]}
