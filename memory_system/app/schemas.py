"""Pydantic schemas for the AML Add/Search synchronous contract."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str = Field(min_length=1)
    timestamp: Optional[int] = None  # unix ms


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


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    options: Optional[List[str]] = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=100)
    include_history: Optional[bool] = None
    # ISO-8601 question/evaluation time used to resolve relative expressions.
    reference_time: Optional[str] = None


class SearchItem(BaseModel):
    id: str
    content: str
    score: Optional[float] = None
    created_at: Optional[str] = None
    memory_type: str = "fact"
    sources: List[dict] = Field(default_factory=list)
    source_count: int = 0


class SearchResponse(BaseModel):
    data: List[SearchItem]
