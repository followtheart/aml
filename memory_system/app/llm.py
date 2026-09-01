"""LLM abstraction over LiteLLM.

All LLM calls in the system go through `complete()`. The model is selected
by AML_LLM_MODEL (default gpt-4o-mini per AML full-gate rules). When
AML_FAKE=1 or the provider credentials are missing, a deterministic offline
FakeLLM is used so plumbing/contract tests never need network access.
"""
import json
import re
from typing import Optional

from . import config


class LLMError(RuntimeError):
    pass


def _fake_available() -> bool:
    return config.FAKE


async def complete(prompt: str, system: Optional[str] = None) -> str:
    """Single text completion, temperature 0. Returns raw text."""
    if _fake_available():
        return FakeLLM().complete(prompt)
    try:
        import litellm
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await litellm.acompletion(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:  # pragma: no cover - depends on provider
        raise LLMError(f"litellm completion failed for model "
                       f"{config.LLM_MODEL}: {e}") from e


def extract_json(text: str):
    """Tolerant JSON extraction from LLM output."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0],
                default=-1)
    if start < 0:
        raise ValueError(f"no JSON in LLM output: {text[:200]}")
    for end in range(len(text), start, -1):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"unparseable JSON in LLM output: {text[:200]}")


class FakeLLM:
    """Deterministic offline LLM for plumbing tests.

    Behaviour is keyed by which prompt template markers appear in the prompt.
    """

    def complete(self, prompt: str) -> str:
        if "memory extraction module" in prompt:
            return json.dumps(self._fake_extract(prompt))
        if "memory governance agent" in prompt:
            return json.dumps({"operation": "ADD", "target_id": None,
                               "merged_content": None, "reason": "fake"})
        if "running summary" in prompt:
            return "fake summary"
        if "query understanding module" in prompt:
            return json.dumps(self._fake_query(prompt))
        if "relevance scoring module" in prompt:
            return json.dumps(self._fake_rerank(prompt))
        if "intelligent memory assistant" in prompt:
            return "fake answer"
        if 'label an answer to a question' in prompt:
            return json.dumps({"label": "CORRECT"})
        return "{}"

    # -- fake behaviours -------------------------------------------------
    def _fake_extract(self, prompt: str):
        # take lines after "New messages to extract from:"
        try:
            section = prompt.split("New messages to extract from:", 1)[1]
            section = section.split("Extract atomic memory units", 1)[0]
        except IndexError:
            return {"facts": [], "triples": []}
        facts, triples = [], []
        for line in section.strip().splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            speaker, content = line.split(":", 1)
            content = content.strip()
            if not content:
                continue
            ents = [w.strip(".,!?\"'") for w in content.split()
                    if w[:1].isupper() and len(w) > 2][:4]
            facts.append({
                "content": f"{speaker.strip()}: {content}",
                "retrieval_key": content[:80],
                "type": "fact",
                "entities": ents,
                "keywords": [w for w in content.split()[:4]],
                "event_time": None,
                "sensitivity": "normal",
            })
            for e in ents:
                triples.append({"subject": speaker.strip(),
                                "relation": "mentioned", "object": e})
        return {"facts": facts, "triples": triples}

    def _fake_query(self, prompt: str):
        q = ""
        try:
            q = prompt.split("Question:", 1)[1].split("Answer options", 1)[0].strip()
        except IndexError:
            pass
        ents = [w.strip(".,!?\"'") for w in q.split()
                if w[:1].isupper() and len(w) > 2][:4]
        return {"intent": "fact",
                "time_scope": {"from": None, "to": None, "note": ""},
                "entities": ents,
                "sub_queries": [q] if q else [],
                "expanded_queries": [q] if q else []}

    def _fake_rerank(self, prompt: str):
        try:
            q = prompt.split("Question:", 1)[1].split("Relevant time scope", 1)[0]
            qwords = {"".join(ch for ch in w if ch.isalnum())
                      for w in q.lower().split()}
            qwords.discard("")
            cand = prompt.split("Candidate memories (id: text):", 1)[1]
            cand = cand.split("For each candidate", 1)[0]
        except IndexError:
            return []
        out = []
        for line in cand.strip().splitlines():
            if ":" not in line:
                continue
            cid, text = line.split(":", 1)
            twords = {"".join(ch for ch in w if ch.isalnum())
                      for w in text.lower().split()}
            overlap = len(qwords & twords)
            rel = min(1.0, overlap / 5.0)
            out.append({"id": cid.strip(), "relevance": rel,
                        "keep": rel >= 0.3})
        return out
