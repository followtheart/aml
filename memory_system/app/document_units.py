"""Recognise pasted documents and instruction blocks inside platform messages.

The leaderboard contract sends only ``{role, content, timestamp}``, so the
``source_kind`` field never arrives. Context-learning tasks (CLBench) store
rules, specifications and reference texts as ordinary user messages; atomic
fact extraction fragments them, and the platform answer model then cannot
follow the full procedure. Messages that look like documents therefore keep
their whole segment as a retrievable episode and are packed verbatim.
"""
import re

from . import config

# retrieval_key of whole-segment episodes that preserve a pasted document.
RETRIEVAL_KEY = "Document or instruction text"

_FENCE = re.compile(r"```|~~~")
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)]|[a-zA-Z][.)]|（?\d{1,3}）|[一二三四五六七八九十]+[、.])\s*\S", re.M)
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+\S|[A-Z][A-Za-z0-9 /&-]{2,48}:\s*$|[\u4e00-\u9fff]{2,12}[：:]\s*$)", re.M)
_TABLE = re.compile(r"^\s*\|.*\|\s*$", re.M)
_INSTRUCTION = re.compile(
    r"\b(?:must|should|always|never|do not|don't|required|rules?|steps?|format|output|return|"
    r"respond|answer with|follow|instructions?|guidelines?|policy|procedure|requirements?|constraints?)\b"
    r"|必须|不得|禁止|应当|应该|规则|步骤|格式|输出|遵循|说明|要求|约束", re.I)


def is_document_like(text) -> bool:
    """Structural markup, or a long text dense with directives, marks a document."""
    if not config.DOCUMENT_UNITS_ENABLED or not isinstance(text, str):
        return False
    size = len(text)
    if size < config.DOCUMENT_MIN_CHARS:
        return False
    structural = (bool(_FENCE.search(text)) or len(_LIST_LINE.findall(text)) >= 3
                  or len(_HEADING.findall(text)) >= 2 or len(_TABLE.findall(text)) >= 2)
    if structural:
        return True
    if size >= config.DOCUMENT_LONG_CHARS:
        return len(_INSTRUCTION.findall(text)) >= 3 or text.count("\n") >= 8
    return False


def any_document(messages) -> bool:
    return any(is_document_like(getattr(m, "content", None) if not isinstance(m, dict) else m.get("content"))
               for m in messages)
