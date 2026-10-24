from pathlib import Path
from pydantic import BaseModel, Field, model_validator
import os


class Settings(BaseModel):
    db_path: str = str(Path(__file__).resolve().parents[1] / "connection.db")
    api_key: str = ""
    provider_api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    model: str = ""
    embed_model: str = ""
    embed_api_base: str = ""
    embed_api_key: str = ""
    embed_batch_size: int = Field(default=10, ge=1, le=2048)
    embed_send_dimensions: bool = False
    embed_dimension: int = Field(default=1536, ge=1)
    rerank_model: str = ""
    rerank_api_url: str = ""
    rerank_api_key: str = ""
    rerank_api_format: str = "dashscope"
    rerank_seconds: float = Field(default=2, gt=0)
    rerank_batch_size: int = Field(default=24, ge=1, le=64)
    rerank_document_bytes: int = Field(default=6000, ge=256)
    rerank_request_bytes: int = Field(default=90000, ge=256)
    fake: bool = False
    memory_expansion: bool = True
    query_expansion: bool = True
    memory_views: int = Field(default=4, ge=0, le=4)
    query_views: int = Field(default=3, ge=0, le=3)
    route_limit: int = Field(default=24, ge=1, le=100)
    candidate_limit: int = Field(default=64, ge=1, le=256)
    verify_limit: int = Field(default=16, ge=1, le=16)
    exclusive_bridge_slots: int = Field(default=0, ge=0, le=1)
    complementary_slots: int = Field(default=2, ge=0, le=2)
    batch_size: int = Field(default=8, ge=1, le=8)
    # Byte-bounded batches may contain fewer than batch_size events. Allow all
    # 16 verification candidates to be considered without enlarging a prompt.
    max_batches: int = Field(default=8, ge=1, le=16)
    batch_bytes: int = Field(default=24000, ge=256)
    evidence_bytes: int = Field(default=32000, ge=256)
    search_seconds: float = Field(default=12, gt=0)
    expansion_seconds: float = Field(default=2, gt=0)
    recall_seconds: float = Field(default=2, gt=0)
    verify_seconds: float = Field(default=7, gt=0)
    add_seconds: float = Field(default=60, gt=0)
    answer_seconds: float = Field(default=30, gt=0)

    @model_validator(mode="after")
    def limits(self):
        if self.verify_limit > self.candidate_limit:
            raise ValueError("verify_limit must not exceed candidate_limit")
        return self

    @classmethod
    def from_env(cls):
        return cls(**{name: os.environ[f"MC_{name.upper()}"]
                      for name in cls.model_fields if f"MC_{name.upper()}" in os.environ})
