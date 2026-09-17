"""LLM abstraction over LiteLLM.

All LLM calls in the system go through `complete()`. The model is selected
by AML_LLM_MODEL (default gpt-4o-mini per AML full-gate rules). When
AML_FAKE=1 or the provider credentials are missing, a deterministic offline
FakeLLM is used so plumbing/contract tests never need network access.
"""
import json
import re
from typing import Dict, Optional

from . import config, metrics


class LLMError(RuntimeError):
    pass


STRUCTURED_SCHEMAS: Dict[str, dict] = {
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
            "intent": {"type": "string", "enum": ["fact", "multi_hop", "temporal",
                "preference", "rule", "profile", "narrative", "document",
                "procedural", "abstention_check"]},
            "include_history": {"type": "boolean"},
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
        "required": ["intent", "include_history", "time_scope", "entities", "sub_queries",
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
    # ULM §5.5: is the evidence necessary and sufficient? what is missing?
    "verify": {
        "type": "object",
        "properties": {
            "sufficient": {"type": "boolean"},
            "confidence": {"type": "number"},
            "missing": {"type": "string"},
            "follow_up_queries": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["sufficient", "confidence", "missing", "follow_up_queries"],
    },
}


# Evidence and triples belong to their fact; no model-generated cross-array IDs.
def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties)}


_TEXT = {"type": "string"}
_TRIPLE = _object({"subject": _TEXT, "relation": _TEXT, "object": _TEXT})
_EVIDENCE = _object({"message_index": {"type": "integer"}, "quote": _TEXT})
_STATE = _object({"subject": _TEXT, "attribute": _TEXT, "value": _TEXT})
_STATE["type"] = ["object", "null"]
STRUCTURED_SCHEMAS["extraction"] = _object({
    "episode": _object({"narrative": _TEXT, "compressed_chunk": _TEXT}),
    "facts": {"type": "array", "items": _object({
        "content": _TEXT, "retrieval_key": _TEXT,
        "type": {"type": "string", "enum": ["fact", "preference", "rule", "workflow", "event", "profile", "plan"]},
        "entities": {"type": "array", "items": _TEXT},
        "keywords": {"type": "array", "items": _TEXT},
        "time_expression": {"type": ["string", "null"]},
        "state": _STATE,
        "evidence": {"type": "array", "items": _EVIDENCE, "minItems": 1},
        "triples": {"type": "array", "items": _TRIPLE},
        "sensitivity": {"type": "string", "enum": ["normal", "sensitive"]},
    })}
})
STRUCTURED_SCHEMAS["evidence_check"] = _object({
    "valid": {"type": "boolean"}, "reason": _TEXT
})
STRUCTURED_SCHEMAS["experience"] = _object({
    "items": {"type": "array", "maxItems": 3, "items": _object({
        "type": {"type": "string", "enum": ["strategy", "workflow", "skill", "playbook"]},
        "title": _TEXT,
        "description": _TEXT,
        "content": _TEXT,
        "keywords": {"type": "array", "items": _TEXT},
    })}
})
STRUCTURED_SCHEMAS["profile_consolidation"] = _object({
    "items": {"type": "array", "maxItems": 3, "items": _object({
        "content": _TEXT,
        "kind": {"type": "string", "enum": ["interest", "habit", "health", "diet",
                                            "trait", "possession", "relationship", "other"]},
        "basis": {"type": "string", "enum": ["stated", "inferred"]},
        "support_ids": {"type": "array", "items": _TEXT},
    })}
})
STRUCTURED_SCHEMAS["choice_align"] = _object({
    "options": {"type": "array", "items": _object({
        "letter": _TEXT,
        "kind": {"type": "string", "enum": ["persona", "generic"]},
        "match": {"type": "string", "enum": ["strong", "weak", "none"]},
        "option_claim": _TEXT,
        "evidence_id": _TEXT,
        "evidence": _TEXT,
        "unsupported_claims": {"type": "array", "items": _TEXT},
        "forbidden": {"type": "boolean"},
        "constraint_id": _TEXT,
        "constraint_span": _TEXT,
    })}
})
STRUCTURED_SCHEMAS["invalidate"] = _object({
    "invalidate_ids": {"type": "array", "items": _TEXT},
})


def _fake_available() -> bool:
    return config.FAKE


def _provider_kwargs() -> dict:
    """Endpoint credentials plus model-specific request-body switches."""
    kwargs = {}
    if config.LLM_API_BASE:
        kwargs["api_base"] = config.LLM_API_BASE
    if config.LLM_API_KEY:
        kwargs["api_key"] = config.LLM_API_KEY
    if config.LLM_DISABLE_THINKING and "qwen3" in config.LLM_MODEL.lower():
        kwargs["extra_body"] = {"enable_thinking": False}
    return kwargs


async def complete(prompt: str, system: Optional[str] = None,
                   response_format: Optional[dict] = None,
                   max_tokens: Optional[int] = None,
                   stage: str = "llm.complete") -> str:
    """Single text completion, temperature 0. Returns raw text."""
    if _fake_available():
        metrics.log_fake(kind="llm", stage=stage, model="fake")
        return FakeLLM().complete(prompt)
    try:
        import litellm
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = _provider_kwargs()
        if response_format:
            kwargs["response_format"] = response_format
        async def _call(_attempt):
            return await litellm.acompletion(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                timeout=180,
                num_retries=0,
                **kwargs,
            )

        resp = await metrics.measured_call(
            kind="llm", stage=stage, model=config.LLM_MODEL,
            call=_call, attempts=2,
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
                        schema: Optional[dict] = None,
                        stage: str = "llm.structured"):
    """Structured completion plus parsing, with one clean retry.

    SiliconFlow and OpenAI-compatible providers support JSON mode through
    ``response_format``. If parsing still fails, rerun the complete original
    task with a stronger JSON reminder. Do not feed malformed output back to
    a small model: repeated garbage can reinforce token degeneration.
    """
    if schema and not _fake_available():
        return await _complete_tool_json(
            prompt, schema, system=system, stage=stage)

    json_system = system or (
        "You are a structured-data generator. Return exactly one valid JSON "
        "object and no markdown, commentary, or HTML entities.")
    json_format = {"type": "json_object"}
    out = await complete(
        prompt,
        system=json_system,
        response_format=json_format,
        max_tokens=config.LLM_JSON_MAX_TOKENS,
        stage=stage,
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
        stage=f"{stage}.json_repair",
    )
    return extract_json(out2)


async def _complete_tool_json(prompt: str, schema: dict,
                              system: Optional[str] = None,
                              stage: str = "llm.tool"):
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

    async def _call(attempt):
        messages = list(base_messages)
        if attempt > 1:
            messages.append({
                "role": "user",
                "content": (
                    "Call emit_json_result now. Fill every required field "
                    "from the original input; do not return prose."),
            })
        kwargs = _provider_kwargs()
        resp = await litellm.acompletion(
            model=config.LLM_MODEL,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function",
                         "function": {"name": tool_name}},
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_JSON_MAX_TOKENS,
            timeout=180,
            num_retries=0,
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
        return result, resp

    try:
        result, _resp = await metrics.measured_call(
            kind="llm", stage=stage, model=config.LLM_MODEL,
            call=_call, attempts=2, response_getter=lambda item: item[1])
        return result
    except Exception as e:  # provider/model dependent
        raise LLMError(
            f"structured completion failed for model {config.LLM_MODEL}: "
            f"{e}") from e


class FakeLLM:
    """Deterministic offline LLM for plumbing tests.

    Behaviour is keyed by which prompt template markers appear in the prompt.
    """

    def complete(self, prompt: str) -> str:
        if prompt.startswith("Judge whether the proposed answer correctly conveys"):
            return json.dumps({"label": "WRONG"})
        if prompt.startswith("Evaluate the response against each rubric independently."):
            # Plumbing-only rubric response; fake scores have no quality meaning.
            payload = json.loads(prompt[prompt.index("{"):])
            return json.dumps({"scores": [{"index": r["index"], "score": 0.0}
                                           for r in payload["rubrics"]]})
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
        if "sufficiency verifier" in prompt:
            return json.dumps({"sufficient": True, "confidence": 0.9,
                               "missing": "", "follow_up_queries": []})
        if "experience distillation module" in prompt:
            return json.dumps({"items": [{"type": "strategy", "title": "Task lesson",
                "description": "Reusable lesson from explicit task feedback",
                "content": "Review the observed outcome before repeating this task.",
                "keywords": ["task", "feedback"]}]})
        if "profile consolidation module" in prompt:
            return json.dumps({"items": []})
        if "choice alignment module" in prompt:
            letters = re.findall(r"^\s*([A-Z])\.\s", prompt, re.M)
            return json.dumps({"options": [
                {"letter": l, "kind": "generic", "match": "none", "option_claim": "",
                 "evidence_id": "", "evidence": "", "unsupported_claims": [],
                 "forbidden": False, "constraint_id": "", "constraint_span": ""}
                for l in letters]})
        if "memory invalidation module" in prompt:
            ids = re.findall(r"^(amu_\w+):", prompt, re.M)
            return json.dumps({"invalidate_ids": ids[:1]})
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
            return {"episode": {"narrative": "", "compressed_chunk": ""}, "facts": []}
        facts, triples = [], []
        for line in section.strip().splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            marker = re.match(r"\[(\d+)\] (.*)", line)
            source_index = int(marker[1]) if marker else len(facts)
            if marker:
                line = marker[2]
            speaker, content = line.split(":", 1)
            content = content.strip()
            if not content:
                continue
            ents = [w.strip(".,!?\"'") for w in content.split()
                    if w[:1].isupper() and len(w) > 2][:4]
            facts.append({
                "evidence": [{"message_index": source_index, "quote": content}],
                "state": None, "time_expression": None, "triples": [],
                "content": f"{speaker.strip()}: {content}",
                "retrieval_key": content[:80],
                "type": "fact",
                "entities": ents,
                "keywords": [w for w in content.split()[:4]],
                "event_time": None,
                "sensitivity": "normal",
            })
            for e in ents:
                facts[-1]["triples"].append({"subject": speaker.strip(),
                                "relation": "mentioned", "object": e})
        episode_text = " ".join(fact["content"] for fact in facts)
        return {"episode": {"narrative": episode_text,
                    "compressed_chunk": episode_text}, "facts": facts}

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
