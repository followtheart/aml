"""Convert snap-research/locomo locomo10.json into the harness format.

LoCoMo QA categories (per repo/paper): 1=multi-hop, 2=temporal,
3=single-hop(?), 4=open-domain, 5=adversarial/unanswerable.
We keep the numeric id and a best-effort name; verify against the paper
before publishing numbers.
"""
import json
import re
import sys
from datetime import datetime

CAT = {1: "multi-hop", 2: "temporal", 3: "single-hop",
       4: "open-domain", 5: "adversarial"}


def parse_dt(s: str):
    # e.g. "1:56 pm on 8 May, 2023"
    m = re.match(r"(.+?)\s+on\s+(.+)", s.strip())
    if not m:
        return None
    t, d = m.groups()
    for fmt in ("%I:%M %p %d %B, %Y",):
        try:
            return int(datetime.strptime(f"{t} {d}", fmt).timestamp() * 1000)
        except ValueError:
            return None
    return None


def convert(src: str, dst: str):
    data = json.load(open(src, encoding="utf-8"))
    out = []
    for sample in data:
        conv = sample["conversation"]
        sessions = []
        i = 1
        while f"session_{i}" in conv:
            ts = parse_dt(conv.get(f"session_{i}_date_time", ""))
            msgs = [{
                "role": "user" if m["speaker"] == conv["speaker_a"]
                        else "assistant",
                "content": f'{m["speaker"]}: {m["text"]}',
                "timestamp": ts,
            } for m in conv[f"session_{i}"]]
            if msgs:
                sessions.append(msgs)
            i += 1
        qa = [{
            "question": q["question"],
            "answer": str(q.get("answer", "")),
            "category": CAT.get(q.get("category"), f"cat-{q.get('category')}"),
            "evidence": q.get("evidence", []),
        } for q in sample.get("qa", []) if q.get("answer") is not None]
        out.append({"conversation_id": sample.get("sample_id", "unknown"),
                    "speaker_a": conv["speaker_a"],
                    "speaker_b": conv["speaker_b"],
                    "sessions": sessions, "qa": qa})
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"converted {len(out)} conversations, "
          f"{sum(len(c['qa']) for c in out)} QA pairs -> {dst}")


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "data/locomo10.json",
            sys.argv[2] if len(sys.argv) > 2 else "data/locomo_eval.json")
