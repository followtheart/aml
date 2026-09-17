"""Pydantic schemas for the AML Add/Search synchronous contract."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1)
    timestamp: Optional[int] = None  # unix ms
    message_id: Optional[str] = None
    source_event_id: Optional[str] = None
    speaker_id: Optional[str] = None
    source_kind: str = Field(default='dialog', pattern='^(dialog|tool|document|import|unknown)$')
    trust_scope: List[str] = Field(default_factory=lambda: ['personal'])
    sensitivity: str = Field(default='normal', pattern='^(normal|sensitive|suppressed)$')


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: List[Message]
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str
    write_revision: int = 0
    scope_epoch: int = 0


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    options: Optional[List[str]] = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)
    include_history: Optional[bool] = None
    include_sensitive: bool = False
    # ISO-8601 question/evaluation time used to resolve relative expressions.
    reference_time: Optional[str] = None
    as_of: Optional[str] = None
    min_revision: Optional[int] = Field(default=None, ge=0)
    evidence_token_budget: int = Field(default=32000, ge=256, le=32000)


class SearchItem(BaseModel):
    id: str
    content: str
    score: Optional[float] = None
    created_at: Optional[str] = None
    memory_type: str = "fact"
    personal_evidence: Optional[bool] = None
    sources: List[dict] = Field(default_factory=list)
    source_count: int = 0
    temporal: Optional[dict] = None
    packet_hash: Optional[str] = None
    packet_hash_version: Optional[int] = None


class SearchResponse(BaseModel):
    search_id: Optional[str] = None
    data: List[SearchItem]
    # retrieved means candidates were packed, not that an LLM proved sufficiency.
    evidence_status: str = "not_found"  # retrieved | conflicting | not_found
    verification_status: str = "not_run"
    packet_hash: Optional[str] = None
    read_revision: Optional[int] = None
    read_epoch: Optional[int] = None
    missing_evidence: List[str] = Field(default_factory=list)
    coverage_manifest: dict = Field(default_factory=dict)


class PurgeResponse(BaseModel):
    success: bool = True
    receipt_id: str
    user_id: str
    deleted_at: str
    row_counts: dict


class FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    outcome: str = Field(pattern="^(success|failure)$")
    trace: str = Field(min_length=1)
    task_signature: Optional[str] = None
    environment_verified: bool = False
    feedback_event_id: Optional[str] = None
    search_id: Optional[str] = None
    used_memory_ids: List[str] = Field(default_factory=list)
    attribution_reason: Optional[str] = None
    environment_signature: Optional[str] = None
    code_version: Optional[str] = None
    dependency_versions: dict = Field(default_factory=dict)
    verification_result: Optional[str] = Field(default=None, pattern='^(passed|failed)$')


class FeedbackResponse(BaseModel):
    success: bool = True
    memory_ids: List[str] = Field(default_factory=list)
