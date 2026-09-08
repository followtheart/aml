"""Detailed local JSONL diagnostics for search routes and reranking."""
import json
import logging
import threading
from pathlib import Path
from . import config

_lock = threading.Lock()


def append(record):
    if not config.SEARCH_DEBUG_LOG:
        return
    try:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
        path = Path(config.SEARCH_DEBUG_LOG)
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as out:
                out.write(line)
    except Exception:
        logging.getLogger('aml.search_debug').warning('Could not write search debug log', exc_info=True)
