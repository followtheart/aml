"""Pydantic schemas for the AML Add/Search synchronous contract."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_serializer, model_validator


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
    top_k: int = Field(default=100, ge=1, le=100, description='Total returned item limit, including constraints')
    include_history: Optional[bool] = None
    include_sensitive: bool = True
    # ISO-8601 question/evaluation time used to resolve relative expressions.
    reference_time: Optional[str] = None
    as_of: Optional[str] = None
    min_revision: Optional[int] = Field(default=None, ge=0)
    # UTF-8 byte bound for model-facing evidence. None selects the intent tier
    # from config (the leaderboard contract never sends this field).
    evidence_token_budget: Optional[int] = Field(default=None, ge=256, le=110000)


class HTTPMessage(Message):
    """External AML contract; internal ingestion may carry other source roles."""
    role: Literal['user', 'assistant']


class HTTPAddRequest(AddRequest):
    messages: List[HTTPMessage]


class HTTPSearchRequest(SearchRequest):
    """Require the external limit while retaining defaults for internal callers."""
    top_k: int = Field(ge=1, le=100, description='Total returned item limit, including constraints')


class SearchItem(BaseModel):
    id: str
    content: str
    score: Optional[float] = None
    score_kind: str = 'unknown'
    coverage_ids: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    memory_type: str = "fact"
    is_constraint: bool = False
    equivalent_ids: List[str] = Field(default_factory=list)
    personal_evidence: Optional[bool] = None
    sources: List[dict] = Field(default_factory=list)
    source_count: int = 0
    temporal: Optional[dict] = None
    packet_hash: Optional[str] = None
    packet_hash_version: Optional[int] = None

    @model_serializer(mode='wrap')
    def serialize_optional_score(self, handler):
        # AML accepts an absent score or a number, not an explicit null.
        # Preserve nulls in provenance: those fields participate in the digest.
        data = handler(self)
        if self.score is None:
            data.pop('score', None)
        return data


class SearchResponse(BaseModel):
    search_id: Optional[str] = None
    data: List[SearchItem]
    # retrieved means candidates were packed, not that an LLM proved sufficiency.
    evidence_status: str = "not_found"  # retrieved | conflicting | not_found | incomplete
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


class TaskMetric(BaseModel):
    name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    baseline: Optional[float] = Field(default=None, allow_inf_nan=False)
    unit: str = ''
    direction: Literal['higher', 'lower', 'neutral'] = 'neutral'


class TaskStep(BaseModel):
    step_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    tool: Optional[str] = None
    input: str = ''
    output: str = ''
    status: Literal['success', 'failure', 'partial', 'unknown'] = 'unknown'
    used_memory_ids: List[str] = Field(default_factory=list)


class TaskTrajectory(BaseModel):
    schema_version: Literal[1] = 1
    task_type: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    constraints: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    dataset_version: Optional[str] = None
    settings: dict = Field(default_factory=dict)
    steps: List[TaskStep] = Field(default_factory=list)
    metrics: List[TaskMetric] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def unique_ids(self):
        for values in ([s.step_id for s in self.steps], [m.name for m in self.metrics]):
            if len(values) != len(set(values)):
                raise ValueError('Step IDs and metric names must be unique')
        return self


class FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    outcome: Literal['success', 'failure', 'partial', 'unknown']
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
    execution_kind: Literal['observed', 'replay', 'simulation'] = 'observed'
    trajectory: Optional[TaskTrajectory] = None
    distill: bool = True


class ExperienceEvidenceRef(BaseModel):
    event_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    relation: Literal['support', 'counterexample', 'context'] = 'context'


class ExperienceReviewRequest(BaseModel):
    user_id: str = Field(min_length=1)
    memory_id: str = Field(min_length=1)
    expected_version: int = Field(ge=1)
    status: Literal['validated', 'adopted', 'rejected', 'stale', 'hypothesis']
    reviewer: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence: List[ExperienceEvidenceRef] = Field(min_length=1)


class FeedbackResponse(BaseModel):
    success: bool = True
    memory_ids: List[str] = Field(default_factory=list)
    feedback_event_id: Optional[str] = None
