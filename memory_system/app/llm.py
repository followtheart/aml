"""LLM abstraction over LiteLLM.

All LLM calls in the system go through `complete()`. The model is selected
by AML_LLM_MODEL (default gpt-4o-mini per AML full-gate rules). When
AML_FAKE=1 or the provider credentials are missing, a deterministic offline
FakeLLM is used so plumbing/contract tests never need network access.
"""
import json
import re
from typing import Dict, Optional

from . import config


class LLMError(RuntimeError):
    pass


STRUCTURED_SCHEMAS: Dict[str, dict] = {
    "extraction": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": (
                                "A grounded, self-contained fact extracted "
                                "from the new messages; never placeholder text."),
                        },
                        "retrieval_key": {
                            "type": "string",
                            "description": "A question that this fact answers.",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["fact", "preference", "rule",
                                     "workflow", "event", "profile"],
                        },
                        "entities": {"type": "array",
                                     "items": {"type": "string"}},
                        "keywords": {"type": "array",
                                     "items": {"type": "string"}},
                        "event_time": {"type": ["string", "null"]},
                        "sensitivity": {"type": "string",
                                        "enum": ["normal", "sensitive"]},
                    },
                    "required": ["content", "retrieval_key", "type",
                                 "entities", "keywords", "event_time",
                                 "sensitivity"],
                },
            },
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "relation": {"type": "string"},
                        "object": {"type": "string"},
                    },
                    "required": ["subject", "relation", "object"],
                },
            },
        },
        "required": ["facts", "triples"],
    },
    "governance": {
        "type": "object",
        "properties": {
            "operation": {"type": "string",
                          "enum": ["ADD", "UPDATE", "SUPERSEDE", "NOOP"]},
            "target_id": {"type": ["string", "null"]},
            "merged_content": {"type": ["string", "null"]},
            "reason": {"type": "string"},
        },
        "required": ["operation", "target_id", "merged_content", "reason"],
    },
    "query": {
        "type": "object",
        "properties": {
            "intent": {"type": "string"},
            "time_scope": {
                "type": ["object", "null"],
                "properties": {
                    "from": {"type": ["string", "null"]},
                    "to": {"type": ["string", "null"]},
                    "note": {"type": "string"},
                },
            },
            "entities": {"type": "array", "items": {"type": "string"}},
            "sub_queries": {"type": "array",
                            "items": {"type": "string"}},
            "expanded_queries": {"type": "array",
                                 "items": {"type": "string"}},
        },
        "required": ["intent", "time_scope", "entities", "sub_queries",
                     "expanded_queries"],
    },
    "rerank": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "relevance": {"type": "number"},
                        "keep": {"type": "boolean"},
                    },
                    "required": ["id", "relevance", "keep"],
                },
            },
        },
        "required": ["scores"],
    },
    "judge": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["CORRECT", "WRONG"]},
        },
        "required": ["label"],
    },
}


def _fake_available() -> bool:
    return config.FAKE


async def complete(prompt: str, system: Optional[str] = None,
                   response_format: Optional[dict] = None,
                   max_tokens: Optional[int] = None) -> str:
    """Single text completion, temperature 0. Returns raw text."""
    if _fake_available():
        return FakeLLM().complete(prompt)
    try:
        import litellm
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {}
        if config.LLM_API_BASE:
            kwargs["api_base"] = config.LLM_API_BASE
        if config.LLM_API_KEY:
            kwargs["api_key"] = config.LLM_API_KEY
        if response_format:
            kwargs["response_format"] = response_format
        resp = await litellm.acompletion(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            timeout=180,
            num_retries=1,
            **kwargs,
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


async def complete_json(prompt: str, schema_hint: str = "",
                        system: Optional[str] = None,
                        schema: Optional[dict] = None):
    """Structured completion plus parsing, with one clean retry.

    SiliconFlow and OpenAI-compatible providers support JSON mode through
    ``response_format``. If parsing still fails, rerun the complete original
    task with a stronger JSON reminder. Do not feed malformed output back to
    a small model: repeated garbage can reinforce token degeneration.
    """
    if schema and not _fake_available():
        return await _complete_tool_json(prompt, schema, system=system)

    json_system = system or (
        "You are a structured-data generator. Return exactly one valid JSON "
        "object and no markdown, commentary, or HTML entities.")
    json_format = {"type": "json_object"}
    out = await complete(
        prompt,
        system=json_system,
        response_format=json_format,
        max_tokens=config.LLM_JSON_MAX_TOKENS,
    )
    try:
        return extract_json(out)
    except ValueError:
        pass
    retry_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: Generate the result again from the original input. "
        "Return one complete valid JSON object only. Do not copy or repair "
        "the previous response. The required shape is:\n"
        f"{schema_hint or 'the JSON object requested above'}")
    out2 = await complete(
        retry_prompt,
        system=json_system,
        response_format=json_format,
        max_tokens=config.LLM_JSON_MAX_TOKENS,
    )
    return extract_json(out2)


async def _complete_tool_json(prompt: str, schema: dict,
                              system: Optional[str] = None):
    """Force a top-level object through OpenAI-compatible function calling."""
    import litellm

    tool_name = "emit_json_result"
    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Return the requested structured result.",
            "parameters": schema,
        },
    }
    base_messages = []
    if system:
        base_messages.append({"role": "system", "content": system})
    base_messages.append({"role": "user", "content": prompt})

    last_error = None
    for attempt in range(2):
        messages = list(base_messages)
        if attempt:
            messages.append({
                "role": "user",
                "content": (
                    "Call emit_json_result now. Fill every required field "
                    "from the original input; do not return prose."),
            })
        kwargs = {}
        if config.LLM_API_BASE:
            kwargs["api_base"] = config.LLM_API_BASE
        if config.LLM_API_KEY:
            kwargs["api_key"] = config.LLM_API_KEY
        try:
            resp = await litellm.acompletion(
                model=config.LLM_MODEL,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "function",
                             "function": {"name": tool_name}},
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_JSON_MAX_TOKENS,
                timeout=180,
                num_retries=1,
                **kwargs,
            )
            message = resp["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                raise ValueError("model returned no structured tool call")
            arguments = calls[0]["function"]["arguments"]
            result = arguments if isinstance(arguments, dict) \
                else extract_json(arguments)
            if not isinstance(result, dict):
                raise ValueError("tool arguments are not a JSON object")
            return result
        except Exception as e:  # provider/model dependent
            last_error = e
    raise LLMError(
        f"structured completion failed for model {config.LLM_MODEL}: "
        f"{last_error}") from last_error


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
            return json.dumps({"scores": self._fake_rerank(prompt)})
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
