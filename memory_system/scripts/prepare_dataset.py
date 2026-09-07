"""Convert a public benchmark to validated, streaming local-eval JSONL."""
import argparse
import hashlib
import json
import os
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.eval_data import DATASETS, convert_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="JSONL output")
    p.add_argument("--history-dir")
    p.add_argument("--size", choices=["32k", "128k"], default="32k")
    p.add_argument("--convs", type=int, default=0, help="0 means all source conversations")
    args = p.parse_args()
    if args.convs < 0:
        p.error("--convs must be nonnegative")
    dst = Path(args.output)
    if dst.suffix != ".jsonl" or dst.resolve() == Path(args.input).resolve():
        p.error("Output must be a separate .jsonl file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_suffix(".jsonl.partial")
    count = qas = sessions = empty_messages = 0
    digest = hashlib.sha256()
    try:
        with temp.open("wb") as f:
            source = convert_records(args.input, args.dataset, args.history_dir, args.size)
            for conv in islice(source, args.convs or None):
                conv["protocol"] = "local-text-proxy-v1"
                b = (json.dumps(conv, ensure_ascii=False) + "\n").encode()
                f.write(b)
                digest.update(b)
                count += 1
                qas += len(conv["qa"])
                sessions += len(conv["sessions"])
                empty_messages += conv.get("preprocessing", {}).get("empty_messages_removed", 0)
        if not count:
            raise ValueError("Dataset contains no conversations")
        os.replace(temp, dst)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    manifest = {"dataset": args.dataset, "input": str(Path(args.input).resolve()),
                "conversations": count, "sessions": sessions, "qa": qas,
                "conversation_limit": args.convs, "size": args.size,
                "empty_messages_removed": empty_messages,
                "sha256": digest.hexdigest(), "protocol": "local-text-proxy-v1",
                "official_aml_score": False}
    dst.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
