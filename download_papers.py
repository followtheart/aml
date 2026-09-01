"""Batch-download agent-memory papers from arXiv into ./papers and extract text."""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent
PDF_DIR = BASE / "papers"
TXT_DIR = PDF_DIR / "text"
PDF_DIR.mkdir(exist_ok=True)
TXT_DIR.mkdir(exist_ok=True)

# slug -> (arxiv_id or None, title query fallback)
PAPERS = {
    "a-mem": ("2502.12110", "A-MEM Agentic Memory for LLM Agents"),
    "hipporag": ("2405.14831", "HippoRAG Neurobiologically Inspired Long-Term Memory"),
    "hipporag2": (None, "From RAG to Memory Non-Parametric Continual Learning"),
    "memgpt": ("2310.08560", "MemGPT Towards LLMs as Operating Systems"),
    "memorybank": ("2305.10250", "MemoryBank Enhancing Large Language Models with Long-Term Memory"),
    "mem0": ("2504.19413", "Mem0 Building Production-Ready AI Agents Scalable Long-Term Memory"),
    "memoryos": ("2506.06326", "Memory OS of AI Agent"),
    "mirix": (None, "MIRIX Multi-Agent Memory System for LLM-Based Agents"),
    "zep": ("2501.13956", "Zep Temporal Knowledge Graph Architecture for Agent Memory"),
    "g-memory": (None, "G-Memory Tracing Hierarchical Memory for Multi-Agent Systems"),
    "nemori": (None, "Nemori Self-Organizing Agent Memory Inspired by Cognitive Science"),
    "memos": (None, "MemOS Operating System for Memory-Augmented Generation"),
    "evermemos": ("2601.02163", "EverMemOS"),
    "expel": ("2308.10144", "ExpeL LLM Agents Are Experiential Learners"),
    "agent-workflow-memory": ("2409.07429", "Agent Workflow Memory"),
    "reasoningbank": (None, "ReasoningBank Scaling Agent Self-Evolving with Reasoning Memory"),
    "ace": (None, "Agentic Context Engineering Evolving Contexts for Self-Improving Language Models"),
    "voyager": ("2305.16291", "Voyager Open-Ended Embodied Agent Large Language Models"),
    "memory-r1": (None, "Memory-R1 Enhancing Large Language Model Agents to Manage and Utilize Memories"),
    "dynamic-cheatsheet": (None, "Dynamic Cheatsheet Test-Time Learning with Adaptive Memory"),
    "larimar": ("2403.11901", "Larimar Large Language Models with Episodic Memory Control"),
    "memoryllm": ("2402.04624", "MEMORYLLM Towards Self-Updatable Large Language Models"),
    "m-plus": (None, "M+ Extending MemoryLLM with Scalable Long-Term Memory"),
    "mem1": (None, "MEM1 Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents"),
    "memagent": (None, "MemAgent Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent"),
    "titans": ("2501.00663", "Titans Learning to Memorize at Test Time"),
    "locomo": ("2402.17753", "Evaluating Very Long-Term Conversational Memory of LLM Agents"),
    "longmemeval": ("2410.10813", "LongMemEval Benchmarking Chat Assistants on Long-Term Interactive Memory"),
    "secom": (None, "SeCom Memory Construction and Retrieval for Long-Term Personalized Conversational Agents"),
    "evo-memory": (None, "Evo-Memory Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory"),
    "survey-memory-mechanism": ("2404.13501", "Survey on the Memory Mechanism of Large Language Model based Agents"),
    "survey-human-to-ai": (None, "From Human Memory to AI Memory Survey on Memory Mechanisms in the Era of LLMs"),
    "survey-security": ("2604.16548", "Survey on the Security of Long-Term Memory in LLM Agents"),
    "structural-memory": ("2412.15266", "On the Structural Memory of LLM Agents"),
}

UA = {"User-Agent": "aml-paper-collector/1.0 (academic use)"}


def fetch(url, binary=False, retries=2):
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read() if binary else r.read().decode("utf-8", "replace")
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(3 * (attempt + 1))


def search_arxiv_id(query):
    q = urllib.parse.quote(f'all:"{query}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=3"
    xml = fetch(url)
    ids = re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", xml)
    titles = re.findall(r"<title>([^<]+)</title>", xml)
    return (ids[0] if ids else None, titles[1] if len(titles) > 1 else "")


def main():
    report = {}
    for slug, (aid, query) in PAPERS.items():
        try:
            if aid is None:
                aid, found_title = search_arxiv_id(query)
                time.sleep(3)  # arXiv API rate limit
                if aid is None:
                    report[slug] = {"status": "not_found", "query": query}
                    continue
            aid_clean = aid.split("v")[0]
            pdf_path = PDF_DIR / f"{slug}.pdf"
            if not pdf_path.exists() or pdf_path.stat().st_size < 10000:
                data = fetch(f"https://arxiv.org/pdf/{aid_clean}", binary=True)
                pdf_path.write_bytes(data)
                time.sleep(1)
            report[slug] = {"status": "ok", "arxiv": aid_clean, "size": pdf_path.stat().st_size}
            print(f"[ok] {slug} <- {aid_clean} ({pdf_path.stat().st_size//1024} KB)")
        except Exception as e:
            report[slug] = {"status": "error", "error": str(e)[:200]}
            print(f"[fail] {slug}: {e}")
    (BASE / "download_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    ok = sum(1 for v in report.values() if v["status"] == "ok")
    print(f"done: {ok}/{len(PAPERS)} downloaded")


if __name__ == "__main__":
    main()
