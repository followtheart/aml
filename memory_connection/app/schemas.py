from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc(value: str) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamps require a timezone")
    return dt.astimezone(timezone.utc).isoformat()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(Strict):
    role: Literal["user", "assistant", "tool", "document"]
    content: str = Field(min_length=1, max_length=100000)
    message_id: str | None = Field(default=None, min_length=1)
    timestamp: int | None = None
    speaker_id: str | None = None
    source_kind: str = "dialog"
    sensitivity: Literal["normal", "sensitive", "suppressed"] = "normal"
    valid_from: str | None = None

    @field_validator("content")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("blank content")
        return value

    @field_validator("valid_from")
    @classmethod
    def time(cls, value):
        return utc(value) if value else None

    @field_validator("timestamp")
    @classmethod
    def timestamp_range(cls, value):
        if value is not None:
            try:
                datetime.fromtimestamp(value / 1000, timezone.utc)
            except (ValueError, OverflowError, OSError) as exc:
                raise ValueError("timestamp is outside supported range") from exc
        return value


class AddRequest(Strict):
    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1, max_length=100)


class SearchRequest(Strict):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=10, ge=1, le=100)
    options: list[str] = Field(default_factory=list, max_length=20)
    reference_time: str | None = None
    as_of: str | None = None
    include_history: bool = False
    include_sensitive: bool = True
    min_revision: int = Field(default=0, ge=0)
    debug: bool = False

    @field_validator("reference_time", "as_of")
    @classmethod
    def time(cls, value):
        return utc(value) if value else None

    @field_validator("query")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("blank query")
        return value

    @field_validator("options")
    @classmethod
    def option_size(cls, value):
        if sum(len(v.encode("utf-8")) for v in value) > 12000:
            raise ValueError("options exceed byte budget")
        return value


class Span(Strict):
    source_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class View(Strict):
    kind: Literal["original", "paraphrase", "relation", "abstraction", "applicability"]
    text: str = Field(min_length=1, max_length=100000)
    derivation: Literal["faithful", "associative"] = "faithful"
    anchor_spans: list[Span] = Field(default_factory=list, max_length=8)
    facet: str = Field(default="main", min_length=1, max_length=100)


class Expansion(Strict):
    views: list[View] = Field(max_length=16)


class Decision(Strict):
    event_id: str
    verdict: Literal["relevant", "partial", "contradicting", "irrelevant", "uncertain"]
    relation_type: Literal["direct_fact", "constraint", "preference", "temporal", "causal_link", "analogy", "other"]
    subject_match: Literal["yes", "no", "unknown"]
    time_match: Literal["yes", "no", "unknown"]
    polarity: Literal["consistent", "contradicting", "unknown"]
    evidence_spans: list[Span] = Field(default_factory=list, max_length=8)
    reason: str = Field(max_length=2000)


class Verdicts(Strict):
    decisions: list[Decision] = Field(max_length=8)
