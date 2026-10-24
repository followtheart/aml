from .schemas import Expansion, View, Span
from .event_builder import valid_spans, resolve_spans


def original_view(event):
    return View(kind="original", text=event["text"], anchor_spans=[Span(
        source_id=event["primary_source_id"], start=0, end=len(event["text"]), quote=event["text"])])


def clean_expansion(raw, limit, event=None, original=""):
    parsed = Expansion.model_validate(raw)
    result, seen = [], {" ".join(original.casefold().split())}
    for view in parsed.views:
        key = " ".join(view.text.casefold().split())
        if not key or key in seen or view.kind == "original" or len(view.text) > 600:
            continue
        if event:
            resolved = resolve_spans(view.anchor_spans, event)
            if not resolved or not valid_spans(resolved, event):
                continue
            view = view.model_copy(update={"anchor_spans": resolved})
        if not event and view.anchor_spans:
            continue
        seen.add(key)
        result.append(view)
        if len(result) >= limit:
            break
    return result if limit else []
