from datetime import UTC, datetime
from uuid import uuid4

import pytest

from rag_app.errors import (
    DuplicateDocumentError,
    DuplicateKnowledgeBaseError,
    StorageInconsistentError,
)
from rag_app.models import Chunk, Document, DocumentStatus
from rag_app.storage.sqlite import SQLiteStore
from rag_app.storage.uploads import UploadStore


def test_sqlite_migration_is_idempotent(tmp_path):
    database_path = tmp_path / "app.db"
    SQLiteStore(database_path)
    SQLiteStore(database_path)


def test_knowledge_base_and_document_constraints(tmp_path):
    store = SQLiteStore(tmp_path / "app.db")
    kb = store.create_knowledge_base(str(uuid4()), "Product Docs")
    with pytest.raises(DuplicateKnowledgeBaseError):
        store.create_knowledge_base(str(uuid4()), "product docs")

    document_id = str(uuid4())
    chunks = [make_chunk(document_id, kb.id)]
    document = make_document(document_id, kb.id, chunks)
    store.insert_document(document, chunks)

    duplicate = make_document(str(uuid4()), kb.id, chunks)
    with pytest.raises(DuplicateDocumentError):
        store.insert_document(duplicate, [make_chunk(duplicate.id, kb.id)])

    assert store.document_hash_exists(kb.id, document.content_hash)
    assert store.list_documents(kb.id) == [document]
    assert store.list_chunks(kb.id) == chunks
    assert store.counts() == (1, 1, 1)


def test_active_vector_collection_is_unique(tmp_path):
    store = SQLiteStore(tmp_path / "app.db")
    store.create_active_vector_collection(str(uuid4()), "collection_a", "embedding", 8)
    with pytest.raises(StorageInconsistentError):
        store.create_active_vector_collection(
            str(uuid4()), "collection_b", "embedding", 8
        )


def test_upload_store_rejects_escape_and_deletes_kb_directory(tmp_path):
    uploads = UploadStore(tmp_path / "uploads")
    kb_id = str(uuid4())
    document_id = str(uuid4())
    relative = uploads.save(kb_id, document_id, "# Title\n", "manual.md")
    assert relative == f"{kb_id}/{document_id}.md"
    assert uploads.exists(relative)

    with pytest.raises(StorageInconsistentError):
        uploads.read("../../etc/passwd")

    uploads.delete_knowledge_base(kb_id)
    assert not uploads.exists(relative)


def make_document(document_id: str, kb_id: str, chunks: list[Chunk]) -> Document:
    return Document(
        id=document_id,
        kb_id=kb_id,
        filename="manual.md",
        content_hash="a" * 64,
        storage_path=f"{kb_id}/{document_id}.md",
        chunk_count=len(chunks),
        status=DocumentStatus.READY,
        created_at=datetime.now(UTC),
    )


def make_chunk(document_id: str, kb_id: str) -> Chunk:
    return Chunk(
        id=f"{document_id}:0001",
        document_id=document_id,
        kb_id=kb_id,
        chunk_index=1,
        heading_path=("Overview",),
        content="# Overview\nWiFi 6 support",
        char_start=0,
        char_end=24,
        content_hash="b" * 64,
    )
