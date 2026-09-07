"""Download public upstream text datasets; never private AML evaluation bundles."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.eval_data import read_records, safe_history_path

HF = "https://huggingface.co/datasets/"
SOURCES = {
    "locomo-refined": {
        "url": "https://raw.githubusercontent.com/mem-eval-suite/LoCoMo_refined/main/data/raw/locomo_refined.json",
        "file": "locomo-refined.json", "license": "CC-BY-NC-4.0",
        "source": "https://github.com/mem-eval-suite/LoCoMo_refined"},
    "longmemeval-s": {
        "url": HF + "xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        "file": "longmemeval-s.json", "license": "MIT (see upstream dataset/repository)",
        "source": "https://github.com/xiaowu0162/LongMemEval"},
    "clbench": {
        "url": HF + "tencent/CL-bench/resolve/main/CL-bench.jsonl",
        "file": "clbench.jsonl", "license": "See tencent/CL-bench/LICENSE.txt",
        "source": "https://github.com/Tencent-Hunyuan/CL-bench"},
    "personamem-v2": {
        "url": HF + "bowen-upenn/PersonaMem-v2/resolve/main/benchmark/text/benchmark.csv",
        "file": "personamem-v2.csv", "license": "CC-BY-4.0",
        "source": "https://github.com/bowen-upenn/PersonaMem-v2"},
    "beam": {
        "url": HF + "Mohammadta/BEAM/resolve/main/data/100K-00000-of-00001.parquet",
        "file": "beam-100k.parquet", "license": "CC-BY-SA-4.0",
        "source": "https://github.com/mohammadtavakoli78/BEAM"},
    "scriptmem": {
        "source": "https://github.com/memorax-ai/ScriptMem",
        "license": "CC-BY-NC-4.0; original script histories NOT included"},
}


def digest_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + ".download.json")
    if destination.exists() and receipt.exists():
        meta = json.loads(receipt.read_text(encoding="utf-8"))
        if meta["url"] == url and meta["sha256"] == digest_file(destination):
            return meta
    partial = destination.with_name(destination.name + ".partial")
    sha = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AML-local-eval/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            size = int(response.headers.get("Content-Length", 0))
            if size and shutil.disk_usage(destination.parent).free < size + 32 * 1024**2:
                raise OSError(f"Insufficient disk space for {destination.name} ({size} bytes)")
            with partial.open("wb") as f:
                while block := response.read(1024 * 1024):
                    f.write(block)
                    sha.update(block)
            if size and partial.stat().st_size != size:
                raise ValueError(f"Incomplete download: {destination.name}")
            etag = response.headers.get("ETag")
        partial.replace(destination)
        meta = {"url": url, "sha256": sha.hexdigest(), "bytes": destination.stat().st_size,
                "etag": etag, "downloaded_at": datetime.now(timezone.utc).isoformat()}
        receipt.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("datasets", nargs="*", choices=[*SOURCES, "all"], default=["all"])
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "data/raw"))
    p.add_argument("--persona-size", choices=["32k", "128k"], default="32k")
    p.add_argument("--persona-convs", type=int, default=0, help="Limit history files; 0 downloads all")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.persona_convs < 0 or not 1 <= args.workers <= 16:
        p.error("Invalid history limit or workers (1-16)")
    root = Path(args.output_dir)
    requested = list(SOURCES) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    for name in requested:
        source = SOURCES[name]
        if name == "scriptmem":
            for stem in ["angry", "enemy", "friends", "man_earth"]:
                fetch(f"https://raw.githubusercontent.com/memorax-ai/ScriptMem/main/data/raw/{stem}.json",
                      root / "scriptmem" / f"{stem}.json")
            print("scriptmem: questions downloaded; source histories are unavailable upstream", flush=True)
        else:
            result = fetch(source["url"], root / source["file"])
            print(f"{name}: {result['bytes']} bytes verified", flush=True)
        if name == "locomo-refined":
            for relative in ["LICENSE.txt", "NOTICE", "data/public/questions.jsonl"]:
                fetch("https://raw.githubusercontent.com/mem-eval-suite/LoCoMo_refined/main/" + relative,
                      root / "locomo-refined" / Path(relative).name)
        if name == "personamem-v2":
            links = list(dict.fromkeys(r[f"chat_history_{args.persona_size}_link"]
                                       for r in read_records(root / source["file"])))
            links = links[:args.persona_convs or None]
            def history(rel):
                expected = f"data/chat_history_{args.persona_size}/"
                if not rel.startswith(expected) or not rel.endswith(".json"):
                    raise ValueError("Unexpected PersonaMem history path")
                destination = safe_history_path(root / "personamem-v2", rel)
                return fetch(HF + "bowen-upenn/PersonaMem-v2/resolve/main/" + rel, destination)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                for i, _ in enumerate(pool.map(history, links), 1):
                    if i % 20 == 0 or i == len(links):
                        print(f"personamem-v2: histories {i}/{len(links)}", flush=True)
        (root / f"{name}.source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print("Public upstream data only; not a verified copy of the hosted AML evaluation bundle.")


if __name__ == "__main__":
    main()
