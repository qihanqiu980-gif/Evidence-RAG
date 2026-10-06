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


def test_keyword_index_syncs_searches_and_deletes(tmp_path):
    store = SQLiteStore(tmp_path / "app.db")
    kb_a = store.create_knowledge_base(str(uuid4()), "Manual")
    kb_b = store.create_knowledge_base(str(uuid4()), "Other")
    document_a = str(uuid4())
    document_b = str(uuid4())
    chunks_a = [
        make_chunk(
            document_a,
            kb_a.id,
            chunk_index=1,
            heading_path=("WiFi",),
            content="AX6000 支持 WiFi 6 标准。",
        ),
        make_chunk(
            document_a,
            kb_a.id,
            chunk_index=2,
            heading_path=("Lights",),
            content="WiFi 指示灯不亮或变红表示故障。",
        ),
    ]
    chunks_b = [
        make_chunk(
            document_b,
            kb_b.id,
            chunk_index=1,
            heading_path=("Other",),
            content="AX6000 是另一台设备的编号。",
        ),
    ]
    store.insert_document(make_document(document_a, kb_a.id, chunks_a), chunks_a)
    store.insert_document(make_document(document_b, kb_b.id, chunks_b), chunks_b)

    assert store.keyword_index_count() == 3
    scoped = store.search_keyword("AX6000 WiFi 标准", [kb_a.id], 10)
    assert scoped
    assert all(match.kb_id == kb_a.id for match in scoped)
    assert scoped[0].chunk_index == 1

    chinese = store.search_keyword("指示灯", [kb_a.id], 10)
    assert [match.chunk_index for match in chinese] == [2]

    short = store.search_keyword("灯", [kb_a.id], 10)
    assert [match.chunk_index for match in short] == [2]

    store.delete_document(kb_a.id, document_a)
    assert store.keyword_index_count() == 1
    assert store.search_keyword("WiFi", [kb_a.id], 10) == []
    assert store.search_keyword("AX6000", [kb_b.id], 10)

    store.delete_knowledge_base(kb_b.id)
    assert store.keyword_index_count() == 0


def test_keyword_index_auto_heals_and_rebuilds(tmp_path):
    database = tmp_path / "app.db"
    store = SQLiteStore(database)
    kb = store.create_knowledge_base(str(uuid4()), "Manual")
    document_id = str(uuid4())
    chunks = [
        make_chunk(document_id, kb.id, content="恢复出厂设置需要长按 Reset。"),
    ]
    store.insert_document(make_document(document_id, kb.id, chunks), chunks)

    with store.connect() as connection:
        connection.execute("DELETE FROM chunks_fts")
    assert store.keyword_index_count() == 0

    store.rebuild_keyword_index()
    assert store.keyword_index_count() == 1
    assert store.search_keyword("出厂设置", [kb.id], 10)

    with store.connect() as connection:
        connection.execute("DELETE FROM chunks_fts")
    reopened = SQLiteStore(database)
    assert reopened.keyword_index_count() == 1
    assert reopened.search_keyword("Reset", [kb.id], 10)


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


def make_chunk(
    document_id: str,
    kb_id: str,
    chunk_index: int = 1,
    heading_path: tuple[str, ...] = ("Overview",),
    content: str = "# Overview\nWiFi 6 support",
) -> Chunk:
    return Chunk(
        id=f"{document_id}:{chunk_index:04d}",
        document_id=document_id,
        kb_id=kb_id,
        chunk_index=chunk_index,
        heading_path=heading_path,
        content=content,
        char_start=0,
        char_end=len(content),
        content_hash="b" * 64,
    )
