"""Full-fidelity, local JSONL snapshots of committed Add requests."""
import copy
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger('aml.memory_debug')
_lock = threading.Lock()
_extraction = ContextVar('memory_debug_extraction', default=None)
_segment = ContextVar('memory_debug_segment', default=None)


@contextmanager
def extraction_scope():
    """Keep diagnostics private to this publish attempt, including child tasks."""
    token = _extraction.set([] if config.MEMORY_DEBUG_LOG else None)
    try:
        yield
    finally:
        _extraction.reset(token)


@contextmanager
def segment_scope(index, message_indices):
    trace = _extraction.get()
    row = None
    if trace is not None:
        row = dict(segment_index=index, message_indices=list(message_indices), events=[])
        trace.append(row)
    token = _segment.set(row)
    try:
        yield
    finally:
        _segment.reset(token)


def extraction_event(stage, **fields):
    """Snapshot proposals before mutation; diagnostics never decide acceptance."""
    row = _segment.get()
    if row is None:
        return
    try:
        row['events'].append(dict(stage=stage, **copy.deepcopy(fields)))
    except Exception:
        log.warning('Could not capture extraction diagnostic', exc_info=True)


def capture(st, req):
    """Read actual stored representations from the final private snapshot."""
    ids = {r[0] for r in st.conn.execute(
        'SELECT amu_id FROM amu_sources WHERE request_id=?', (req.request_id,))}
    # Superseded predecessors and dependent views can both change in this Add.
    pending = list(ids)
    while pending:
        source_id = pending.pop()
        related = [r[0] for r in st.conn.execute(
            "SELECT a.id FROM amu current JOIN amu a ON a.id=current.supersedes "
            "WHERE current.id=? AND a.user_id=?", (source_id, req.user_id))]
        related.extend(r[0] for r in st.conn.execute(
            "SELECT view_id FROM view_dependencies WHERE view_kind='amu' "
            "AND source_id=? AND user_id=?", (source_id, req.user_id)))
        for aid in related:
            if aid not in ids:
                ids.add(aid)
                pending.append(aid)
    memories = []
    for aid in sorted(ids):
        row = st.conn.execute('SELECT * FROM amu WHERE id=?', (aid,)).fetchone()
        if row is None:
            continue
        memory = st._row_to_dict(row)
        vec = memory.pop('embedding', None)
        memory['embedding'] = {
            'dtype': 'float32',
            'dimensions': len(vec) if vec is not None else 0,
            'norm': float(np.linalg.norm(vec)) if vec is not None else None,
            'values': vec.tolist() if vec is not None else None,
        }
        memory['full_text_index'] = [dict(r) for r in st.conn.execute(
            'SELECT amu_id,user_id,content,retrieval_key FROM amu_fts WHERE amu_id=?', (aid,))]
        memory['triples'] = [dict(r) for r in st.conn.execute(
            'SELECT * FROM triples WHERE amu_id=? ORDER BY id', (aid,))]
        memory['sources'] = st.sources_for_amu(aid)
        memory['dependencies'] = st.dependencies_for(aid)
        memories.append(memory)
    scene_ids = sorted({m.get('scene_id') for m in memories if m.get('scene_id')})
    scenes = []
    for scene in st.get_scenes_by_ids(scene_ids):
        scene.pop('centroid', None)
        scenes.append(scene)
    return {
        'event': 'memory.add.committed', 'schema_version': 4,
        'request_id': req.request_id, 'user_id': req.user_id, 'session_id': req.session_id,
        'fake': config.FAKE, 'configured_embedding_model': config.EMBED_MODEL,
        'configured_embedding_space': config.EMBEDDING_SPACE,
        'source_messages': [dict(r) for r in st.conn.execute(
            'SELECT * FROM source_messages WHERE request_id=? ORDER BY message_index',
            (req.request_id,))],
        'memories': memories,
        'scenes': scenes,
        'session_summary': st.get_summary(req.user_id, req.session_id),
        'extraction_trace': copy.deepcopy(sorted(_extraction.get() or [],
                                                key=lambda row: row['segment_index'])),
    }


def append(record):
    """Write only after commit; logging failures must not change Add success."""
    try:
        record['committed_at'] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
        path = Path(config.MEMORY_DEBUG_LOG)
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as output:
                output.write(line)
    except Exception:
        log.warning('Could not append committed memory debug log', exc_info=True)


def purge_user(user_id):
    """Best-effort removal from an explicitly enabled local JSONL log."""
    if not config.MEMORY_DEBUG_LOG:
        return
    path = Path(config.MEMORY_DEBUG_LOG)
    if not path.exists():
        return
    with _lock:
        kept = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                if json.loads(line).get('user_id') == user_id:
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            kept.append(line)
        path.write_text(('\n'.join(kept) + '\n') if kept else '', encoding='utf-8')
