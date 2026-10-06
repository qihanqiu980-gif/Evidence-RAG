from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class SystemStatusValue(StrEnum):
    READY = "ready"
    CONFIGURATION_MISSING = "configuration_missing"
    STORAGE_INCONSISTENT = "storage_inconsistent"


class UploadLimits(BaseModel):
    max_files_per_upload: int
    max_file_bytes: int
    accepted_extensions: list[str]


class SystemStatus(BaseModel):
    status: SystemStatusValue
    knowledge_base_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    limits: UploadLimits


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("Knowledge base name cannot be empty")
        return name


class KnowledgeBaseSummary(BaseModel):
    id: str
    name: str
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    created_at: datetime


class DocumentSummary(BaseModel):
    id: str
    kb_id: str
    filename: str
    status: Literal["ready"]
    chunk_count: int = Field(ge=1)
    created_at: datetime


class UploadedDocument(DocumentSummary):
    progress: list[str]


class DocumentUploadError(BaseModel):
    filename: str
    code: str
    message: str


class DocumentBatchResult(BaseModel):
    documents: list[UploadedDocument]
    errors: list[DocumentUploadError]


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        content = value.strip()
        if not content:
            raise ValueError("History message cannot be empty")
        return content


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=100)
    kb_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        return question

    @field_validator("kb_ids")
    @classmethod
    def normalize_kb_ids(cls, value: list[str]) -> list[str]:
        normalized = list(
            dict.fromkeys(kb_id.strip() for kb_id in value if kb_id.strip())
        )
        if not normalized:
            raise ValueError("At least one knowledge base is required")
        return normalized


class SseEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "stage_started",
        "stage_completed",
        "evidence",
        "decision",
        "answer_final",
        "completed",
        "error",
    ]
    stage: (
        Literal[
            "decompose",
            "retrieve",
            "rerank",
            "judge",
            "generate",
            "validate",
            "regenerate",
            "compose",
        ]
        | None
    ) = None
    message: str
    sequence: int = Field(ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stage(self) -> SseEvent:
        if self.type in {"stage_started", "stage_completed"} and self.stage is None:
            raise ValueError("Stage events require a stage")
        return self


class UploadJobCreated(BaseModel):
    job_id: str


class DiscoveryQuestionSummary(BaseModel):
    id: str
    question: str = Field(min_length=1, max_length=4000)
    source_chunk_ids: list[str] = Field(min_length=1)


class DiscoveryTopicSummary(BaseModel):
    id: str
    title: str
    type: str
    summary: str
    document_ids: list[str]
    chunk_count: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    updated_at: datetime
    questions: list[DiscoveryQuestionSummary]


class DiscoverySummary(BaseModel):
    kb_id: str
    status: Literal["not_analyzed", "pending", "processing", "completed", "failed", "cancelled"]
    job_id: str | None = None
    code: str | None = None
    message: str | None = None
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    topic_count: int = Field(ge=0)
    question_count: int = Field(ge=0)
    analyzed_at: datetime | None = None
    topics: list[DiscoveryTopicSummary] = Field(default_factory=list)


class UploadJobItemSummary(BaseModel):
    filename: str
    status: Literal["pending", "processing", "completed", "failed", "cancelled"]
    code: str | None = None
    message: str | None = None
    progress: list[str] = Field(default_factory=list)
    document: DocumentSummary | None = None


class UploadJobSummary(BaseModel):
    job_id: str
    kb_id: str
    status: Literal["pending", "processing", "completed", "cancelled"]
    created_at: datetime
    updated_at: datetime
    items: list[UploadJobItemSummary]
