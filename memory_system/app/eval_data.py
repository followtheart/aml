"""Streaming public dataset adapters. Gold/rubrics never enter source histories."""
import ast
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

DATASETS = ("normalized", "locomo-refined", "longmemeval-s",
            "longmemeval-refined", "scriptmem", "clbench", "personamem-v2", "beam")
CHOICE_TYPES = {"single_choice", "multi_select", "ordering"}


def read_records(path):
    """Stream JSON arrays, JSONL, CSV or Parquet, one record at a time."""
    path = Path(path)
    if path.suffix == ".csv":
        csv.field_size_limit(32 * 1024 * 1024)
        with path.open(encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
            yield from batch.to_pylist()
    elif path.suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8-sig") as f:
            for number, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{number}: invalid JSON") from exc
    else:
        import ijson
        with path.open("rb") as f:
            first = f.read(3)
            f.seek(3 if first == b"\xef\xbb\xbf" else 0)
            start = f.tell()
            while (ch := f.read(1)) and ch.isspace():
                pass
            if ch != b"[":
                raise ValueError(f"{path}: expected JSON array, or use .jsonl")
            f.seek(start)
            yield from ijson.items(f, "item", use_float=True)


def timestamp(value):
    if value is None or value == "" or value == "Unknown":
        return None
    if isinstance(value, (int, float)):
        return int(value)  # contract specifies milliseconds; do not guess units
    value = str(value).strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        dt = None
        for fmt in ("%Y/%m/%d (%a) %H:%M", "%B-%d-%Y", "%B %d, %Y",
                    "%I:%M %p on %d %B, %Y"):
            try:
                dt = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            raise ValueError(f"Unsupported source timestamp: {value!r}")
    return int(dt.replace(tzinfo=dt.tzinfo or timezone.utc).timestamp() * 1000)


def message(raw, date=None, speaker=False):
    content = raw.get("content", raw.get("text"))
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Source message must be nonempty text; multimodal input unsupported")
    role = raw.get("role", "user")
    if speaker and raw.get("speaker"):
        content = f"{raw['speaker']}: {content}"
    if role not in {"user", "assistant"}:
        content = f"[{role}] {content}"
        role = "user"
    date = raw.get("timestamp") if raw.get("timestamp") is not None else date
    return {"role": role, "content": content, "timestamp": timestamp(date)}


def chunks(messages, max_messages=20, max_words=2000):
    """Preserve order and text, splitting at message/sentence/word boundaries."""
    if max_messages < 1 or max_words < 1:
        raise ValueError("Chunk limits must be positive")
    batch, count = [], 0
    for msg in messages:
        text = msg["content"]
        words = list(re.finditer(r"\S+", text))
        start = 0
        while len(words) > max_words:
            end_word = max_words
            for i in range(max_words - 1, max(max_words // 2 - 1, 0), -1):
                if re.search(r"[.!?。！？][\"'）)]?$", words[i].group()):
                    end_word = i + 1
                    break
            cut = words[end_word].start()
            if batch:
                yield batch
                batch, count = [], 0
            yield [dict(msg, content=text[start:cut])]
            start = cut
            words = words[end_word:]
        part = dict(msg, content=text[start:])
        if batch and (len(batch) >= max_messages or count + len(words) > max_words):
            yield batch
            batch, count = [], 0
        batch.append(part)
        count += len(words)
    if batch:
        yield batch


def required_text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing/non-text {key}")
    return value


def rubrics(value):
    if not isinstance(value, list) or not value:
        raise ValueError("Rubric-scored question requires a nonempty rubric list")
    return [required_text(v, "rubric_criteria") if isinstance(v, dict)
            else required_text({"text": v}, "text") for v in value]


def literal(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return ast.literal_eval(value)


def longmemeval(row, dataset):
    sessions = row["haystack_sessions"]
    dates, ids = row["haystack_dates"], row["haystack_session_ids"]
    if not len(sessions) == len(dates) == len(ids):
        raise ValueError("LongMemEval session/date/id lengths differ")
    ident = str(row["question_id"])
    blank_count = 0
    cleaned, cleaned_ids = [], []
    for sid, sess, date in zip(ids, sessions, dates):
        msgs = []
        for m in sess:
            if isinstance(m.get("content"), str) and not m["content"].strip():
                blank_count += 1
                continue
            msgs.append(message(m, date))
        if msgs:
            cleaned.append(msgs)
            cleaned_ids.append(sid)
    # Some official haystacks repeat filler-session IDs. Keep every occurrence
    # and distinguish ingestion IDs so idempotency cannot silently drop history.
    ingestion_ids = (cleaned_ids if len(set(cleaned_ids)) == len(cleaned_ids)
                     else [f"{i}:{sid}" for i, sid in enumerate(cleaned_ids)])
    return {"conversation_id": ident, "dataset": dataset, "session_ids": ingestion_ids,
            "source_session_ids": cleaned_ids,
            "preprocessing": {"empty_messages_removed": blank_count,
                              "empty_sessions_removed": len(sessions) - len(cleaned)},
            "sessions": cleaned,
            "qa": [{"id": ident, "question": row["question"],
                    "answer": str(row["answer"]), "category": row["question_type"],
                    "question_date": row.get("question_date"),
                    "evidence": row.get("answer_session_ids", []),
                    "unanswerable": ident.endswith("_abs"), "scoring": "binary"}]}


def locomo_refined(row):
    if "sessions" in row:
        return dict(row, dataset="locomo-refined")
    conv = row["conversation"]
    ident = required_text(row, "sample_id")
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                  key=lambda k: int(k.split("_")[1]))
    sessions, dialogue_ids = [], []
    for k in keys:
        sessions.append([message(dict(m, role="user" if m.get("speaker") == conv["speaker_a"]
                                      else "assistant"), conv.get(k + "_date_time"), speaker=True)
                         for m in conv[k]])
        dialogue_ids.append([m.get("dia_id") for m in conv[k]])
    qas = []
    for i, q in enumerate(row["qa"]):
        answers = q.get("answer")
        if not isinstance(answers, list) or not answers or any(
                type(a) not in {str, int, float} or not str(a).strip() for a in answers):
            raise ValueError("LoCoMo-Refined answer must be a nonempty list of complete candidates")
        qas.append({"id": f"{ident}#q{i:04d}", "question": q["question"],
                    "answer": [str(a) for a in answers], "category": str(q["category"]),
                    "evidence": q.get("evidence", []),
                    "is_multi_modality": q.get("is_multi_modality", False),
                    "scoring": "refined_binary"})
    return {"conversation_id": ident, "dataset": "locomo-refined",
            "speaker_a": conv["speaker_a"], "speaker_b": conv["speaker_b"],
            "sessions": sessions, "session_ids": keys, "dialogue_ids": dialogue_ids,
            "qa": qas, "modality": "text-only",
            "source": "https://github.com/mem-eval-suite/LoCoMo_refined"}


def scriptmem(row, source, history_dir):
    conv = row.get("conversation") or {}
    ident = str(row.get("sample_id", "conv-0"))
    if history_dir:
        supplied = json.loads((Path(history_dir) / f"{source}.json").read_text(encoding="utf-8"))
        conv = supplied[ident]
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                  key=lambda k: int(k.split("_")[1]))
    if not keys:
        raise ValueError("ScriptMem omits source conversations. Supply --history-dir with "
                         "real source sessions; format_example is not evaluation history.")
    qas = []
    for i, q in enumerate(row["qa"]):
        answers = q["answer"] if isinstance(q["answer"], list) else [q["answer"]]
        labels = []
        for a in answers:
            match = re.match(r"\s*([A-Z])\.", a)
            if not match:
                raise ValueError("ScriptMem gold answer must have an option label")
            labels.append(match[1])
        qas.append({"id": f"{source}:{ident}#q{i:04d}", "question": q["question"],
                    "answer": q["answer"], "options": q["option"],
                    "gold_labels": labels, "qa_type": q["qa_type"],
                    "category": q["qa_type"], "scoring": "choice"})
    return {"conversation_id": f"{source}:{ident}", "dataset": "scriptmem",
            "sessions": [[message(m, conv.get(k + "_date_time"), speaker=True)
                          for m in conv[k]] for k in keys], "session_ids": keys, "qa": qas}


def clbench(row):
    msgs = row["messages"]
    if not msgs or msgs[-1]["role"] != "user":
        raise ValueError("CLBench source must end with the unanswered user task")
    meta = row["metadata"]
    # The public file combines reference document and task in the user message.
    # Preserve it verbatim rather than guessing a document/task split.
    history = [message(m) for m in msgs if m["role"] != "system"]
    return {"conversation_id": str(meta["task_id"]), "dataset": "clbench",
            "sessions": [history], "metadata": meta,
            "qa": [{"id": str(meta["task_id"]), "question": msgs[-1]["content"],
                    "system_prompt": "\n".join(m["content"] for m in msgs if m["role"] == "system"),
                    "rubrics": rubrics(row["rubrics"]),
                    "category": meta.get("context_category", "unknown"),
                    "scoring": "rubric_all"}]}


def beam(row):
    sessions = []
    for batch in literal(row["chat"]):
        date, msgs = None, []
        for m in batch:
            date = m.get("time_anchor") or date
            msgs.append(message(m, date))
        sessions.append(msgs)
    qas = []
    for category, questions in literal(row["probing_questions"]).items():
        for i, q in enumerate(questions or []):
            qas.append({"id": f"{category}:{i}", "question": q["question"],
                        "answer": q.get("ideal_answer", q.get("ideal_response", "")),
                        "category": category, "rubrics": rubrics(q["rubric"]),
                        "scoring": "rubric_mean"})
    return {"conversation_id": str(row["conversation_id"]), "dataset": "beam",
            "sessions": sessions, "qa": qas}


def safe_history_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("History path escapes --history-dir")
    return path


def persona_groups(path, history_dir, size):
    if not history_dir:
        raise ValueError("PersonaMem-v2 requires --history-dir with official chat histories")
    groups = {}
    for i, row in enumerate(read_records(path)):
        rel = row[f"chat_history_{size}_link"]
        groups.setdefault(rel, []).append((i, row))
    for rel, rows in groups.items():
        data = json.loads(safe_history_path(history_dir, rel).read_text(encoding="utf-8"))
        history = data if isinstance(data, list) else data.get("chat_history", data.get("conversations"))
        if not isinstance(history, list) or not history:
            raise ValueError(f"No chat history in {rel}")
        cleaned = [m for m in history if not (isinstance(m.get("content"), str)
                                             and not m["content"].strip())]
        qas = []
        for i, row in rows:
            query = literal(row["user_query"])
            question = query["content"] if isinstance(query, dict) else str(query)
            opts = [row["correct_answer"]] + list(literal(row["incorrect_answers"]))
            if len(opts) < 2 or len(opts) != len(set(opts)):
                raise ValueError("PersonaMem choices must be distinct and include distractors")
            # Unlike Python hash(), this permutation is reproducible across processes.
            key = f"{row['persona_id']}:{i}:{question}"
            order = sorted(range(len(opts)), key=lambda j: hashlib.sha256(f"{key}:{j}".encode()).digest())
            qas.append({"id": str(i), "question": question,
                        "options": [f"{chr(65+j)}. {opts[k]}" for j, k in enumerate(order)],
                        "answer": row["correct_answer"], "gold_labels": [chr(65 + order.index(0))],
                        "qa_type": "single_choice", "scoring": "choice",
                        "category": row.get("topic_query", "unknown")})
        yield {"conversation_id": hashlib.sha256(rel.encode()).hexdigest()[:20],
               "dataset": "personamem-v2", "source_history": rel,
               "preprocessing": {"empty_messages_removed": len(history) - len(cleaned)},
               "sessions": [[message(m) for m in cleaned]], "qa": qas}


def validate(conv):
    required_text(conv, "conversation_id")
    if not isinstance(conv.get("sessions"), list) or not conv["sessions"]:
        raise ValueError("Conversation requires nonempty sessions")
    ids = conv.get("session_ids")
    if ids is not None and (len(ids) != len(conv["sessions"]) or len(set(map(str, ids))) != len(ids)):
        raise ValueError("session_ids must be unique and aligned with sessions")
    for session in conv["sessions"]:
        if not session:
            raise ValueError("Empty source session")
        for m in session:
            message(m)
    qas = conv.get("qa")
    if not isinstance(qas, list) or not qas:
        raise ValueError("Conversation requires QA records")
    seen = set()
    for i, q in enumerate(qas):
        required_text(q, "question")
        ident = str(q.get("id", i))
        if ident in seen:
            raise ValueError(f"Duplicate QA ID: {ident}")
        seen.add(ident)
        scoring = q.get("scoring", "binary")
        if scoring == "binary":
            if "answer" not in q or q["answer"] is None:
                raise ValueError("Binary QA requires an answer")
        elif scoring == "refined_binary":
            if (not isinstance(q.get("answer"), list) or not q["answer"]
                    or any(not isinstance(a, str) or not a.strip() for a in q["answer"])):
                raise ValueError("Refined QA requires nonempty answer candidates")
        elif scoring == "choice":
            if q.get("qa_type") not in CHOICE_TYPES or not q.get("options") or not q.get("gold_labels"):
                raise ValueError("Choice QA requires options, qa_type and gold_labels")
            valid = {chr(65+i) for i in range(len(q["options"]))}
            gold = q["gold_labels"]
            if len(set(gold)) != len(gold) or not set(gold) <= valid:
                raise ValueError("Invalid gold option labels")
            if q["qa_type"] == "single_choice" and len(gold) != 1:
                raise ValueError("Single choice requires exactly one gold label")
        elif scoring in {"rubric_all", "rubric_mean"}:
            rubrics(q.get("rubrics"))
        else:
            raise ValueError(f"Unsupported scoring: {scoring}")
    return conv


def convert_records(path, dataset, history_dir=None, size="32k"):
    if dataset == "personamem-v2":
        records = persona_groups(path, history_dir, size)
    else:
        def converted():
            for r in read_records(path):
                if dataset == "normalized":
                    yield r
                elif dataset in {"longmemeval-s", "longmemeval-refined"}:
                    yield longmemeval(r, dataset)
                elif dataset == "beam":
                    yield beam(r)
                elif dataset == "clbench":
                    yield clbench(r)
                elif dataset == "scriptmem":
                    yield scriptmem(r, Path(path).stem, history_dir)
                elif dataset == "locomo-refined":
                    yield locomo_refined(r)
                else:
                    raise ValueError(f"Unsupported dataset: {dataset}")
        records = converted()
    seen = set()
    for conv in records:
        validate(conv)
        ident = (conv.get("dataset", dataset), conv["conversation_id"])
        if ident in seen:
            raise ValueError(f"Duplicate conversation ID: {ident}")
        seen.add(ident)
        yield conv


def iter_plan(data, limit=0, convs=0):
    if limit < 0 or convs < 0:
        raise ValueError("--limit and --convs must be nonnegative")
    remaining = limit
    for conv in islice(data, convs or None):
        qas = list(conv.get("qa", []))
        if limit:
            qas = qas[:remaining]
            remaining -= len(qas)
        if qas:
            yield conv, qas
        if limit and remaining <= 0:
            break


def ingestion_chunks(conv, max_messages=20, max_words=2000):
    ids = conv.get("session_ids") or [str(i) for i in range(len(conv["sessions"]))]
    for sid, session in zip(ids, conv["sessions"]):
        for ci, batch in enumerate(chunks(session, max_messages, max_words)):
            yield str(sid), ci, batch
