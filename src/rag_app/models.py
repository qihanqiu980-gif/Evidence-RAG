from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class DocumentStatus(StrEnum):
    READY = "ready"


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    id: str
    name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    kb_id: str
    filename: str
    content_hash: str
    storage_path: str
    chunk_count: int
    status: DocumentStatus
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    kb_id: str
    chunk_index: int
    heading_path: tuple[str, ...]
    content: str
    char_start: int
    char_end: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class VectorCollection:
    id: str
    collection_name: str
    status: str
    embedding_model: str
    embedding_dimension: int
    chunk_count: int
    created_at: datetime
    activated_at: datetime | None
    retired_at: datetime | None
