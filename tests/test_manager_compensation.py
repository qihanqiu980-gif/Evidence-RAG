import pytest

from rag_app.errors import StorageInconsistentError
from rag_app.server import build_container
from tests.conftest import make_settings
from tests.fakes.provider import FakeProvider


def test_metadata_failure_compensates_upload_and_vectors(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    fake = FakeProvider(settings.embedding_dimension)
    container = build_container(settings, fake)
    kb = container.manager.create("Docs")

    def fail_insert(document, chunks):
        raise StorageInconsistentError("injected metadata failure")

    monkeypatch.setattr(container.sqlite, "insert_document", fail_insert)
    with pytest.raises(StorageInconsistentError):
        container.manager.ingest(kb.id, "router.md", b"# Router\nWiFi 6\n")

    assert container.sqlite.list_documents(kb.id) == []
    assert container.vectors.vector_count == 0
    assert container.uploads.document_paths(kb.id) == []
