from __future__ import annotations

from rag_app.core.retrieval import EvidenceRetriever
from rag_app.server import build_container
from tests.conftest import make_settings
from tests.fakes.provider import FakeProvider


def test_retrieval_respects_selected_knowledge_bases_and_top_k(tmp_path):
    settings = make_settings(
        tmp_path,
        RAG_APP_RETRIEVAL_TOP_K="1",
        RAG_APP_CANDIDATE_MULTIPLIER="20",
    )
    fake = FakeProvider(settings.embedding_dimension)
    container = build_container(settings, fake)
    manager = container.manager
    kb_a = manager.create("Docs A").id
    kb_b = manager.create("Docs B").id
    manager.ingest(
        kb_a,
        "router.md",
        b"# Router\nThe router supports WiFi 6.\n",
    )
    manager.ingest(
        kb_b,
        "printer.md",
        b"# Printer\nThis printer supports USB printing.\n",
    )
    assert container.workflow is not None

    retriever = EvidenceRetriever(settings, container.vectors, fake)
    result = retriever.retrieve("Does the router support WiFi 6?", [kb_a])

    assert len(result.evidence) == 1
    assert result.evidence[0].kb_id == kb_a
    assert "WiFi 6" in result.evidence[0].content
    assert result.evidence[0].rerank_score > 0
    assert fake.embed_query_calls == 1
    assert fake.rerank_calls == 1
