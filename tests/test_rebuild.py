from rag_app.core.knowledge_base import KnowledgeBaseManager
from rag_app.server import build_container
from rag_app.storage.chroma import ChromaVectorStore
from rag_app.storage.sqlite import SQLiteStore
from rag_app.storage.uploads import UploadStore
from tests.conftest import make_settings
from tests.fakes.provider import FakeProvider


def test_rebuild_switches_embedding_space_and_cleans_old_collection(tmp_path):
    old_settings = make_settings(tmp_path, RAG_APP_EMBEDDING_MODEL="old-model")
    old_container = build_container(
        old_settings,
        FakeProvider(old_settings.embedding_dimension),
    )
    kb = old_container.manager.create("Docs")
    document = old_container.manager.ingest(
        kb.id,
        "router.md",
        b"# Router\nThe router supports WiFi 6.\n",
    )
    assert document.chunk_count == 1

    new_settings = make_settings(
        tmp_path,
        RAG_APP_EMBEDDING_MODEL="new-model",
    )
    sqlite = SQLiteStore(new_settings.sqlite_path)
    uploads = UploadStore(new_settings.uploads_path)
    vectors = ChromaVectorStore(
        path=new_settings.chroma_path,
        sqlite=sqlite,
        embedding_model=new_settings.embedding_model,
        embedding_dimension=new_settings.embedding_dimension,
        allow_configuration_mismatch=True,
    )
    manager = KnowledgeBaseManager(
        settings=new_settings,
        sqlite=sqlite,
        uploads=uploads,
        vectors=vectors,
        provider=FakeProvider(new_settings.embedding_dimension),
    )
    assert manager.rebuild() == []

    active = sqlite.get_active_vector_collection()
    assert active is not None
    assert active.embedding_model == "new-model"
    assert active.chunk_count == 1
    assert sqlite.retired_vector_collections() == []

    rebuilt_container = build_container(
        new_settings,
        FakeProvider(new_settings.embedding_dimension),
    )
    query = rebuilt_container.provider.embed_query("WiFi 6")
    matches = rebuilt_container.vectors.search(query, [kb.id], 7)
    assert len(matches) == 1
