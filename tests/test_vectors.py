from uuid import uuid4

import pytest

from rag_app.errors import VectorConfigurationMismatch
from rag_app.models import Chunk
from rag_app.storage.chroma import ChromaVectorStore
from rag_app.storage.sqlite import SQLiteStore


def test_chroma_registry_add_search_and_delete(tmp_path):
    sqlite = SQLiteStore(tmp_path / "app.db")
    vectors = ChromaVectorStore(
        tmp_path / "chroma",
        sqlite,
        embedding_model="test-embedding",
        embedding_dimension=2,
    )
    kb_id = str(uuid4())
    document_id = str(uuid4())
    chunk = Chunk(
        id=f"{document_id}:0001",
        document_id=document_id,
        kb_id=kb_id,
        chunk_index=1,
        heading_path=("Spec",),
        content="The router supports WiFi 6.",
        char_start=0,
        char_end=28,
        content_hash="a" * 64,
    )
    vectors.add_chunks([chunk], [[1.0, 0.0]], "router.md")
    assert vectors.vector_count == 1
    assert vectors.active_collection.chunk_count == 1

    matches = vectors.search([1.0, 0.0], [kb_id], 7)
    assert len(matches) == 1
    assert matches[0].document_name == "router.md"
    assert matches[0].kb_id == kb_id

    vectors.delete_document(document_id)
    assert vectors.vector_count == 0
    assert vectors.search([1.0, 0.0], [kb_id], 7) == []


def test_embedding_configuration_change_blocks_reopen(tmp_path):
    sqlite = SQLiteStore(tmp_path / "app.db")
    ChromaVectorStore(
        tmp_path / "chroma", sqlite, embedding_model="old-model", embedding_dimension=2
    )
    with pytest.raises(VectorConfigurationMismatch):
        ChromaVectorStore(
            tmp_path / "chroma",
            sqlite,
            embedding_model="new-model",
            embedding_dimension=2,
        )
