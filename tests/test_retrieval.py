from __future__ import annotations

from dataclasses import replace

from rag_app.core.retrieval import EvidenceRetriever, _rrf_fuse
from rag_app.server import build_container
from rag_app.storage.chroma import VectorMatch
from rag_app.storage.sqlite import KeywordMatch
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

    retriever = EvidenceRetriever(settings, container.vectors, fake, container.sqlite)
    result = retriever.retrieve("Does the router support WiFi 6?", [kb_a])

    assert len(result.evidence) == 1
    assert result.evidence[0].kb_id == kb_a
    assert "WiFi 6" in result.evidence[0].content
    assert result.evidence[0].rerank_score > 0
    assert fake.embed_query_calls == 1
    assert fake.rerank_calls == 1


def test_rrf_fusion_prioritizes_cross_source_agreement():
    vector = [make_vector_match("a"), make_vector_match("b")]
    keyword = [make_keyword_match("b"), make_keyword_match("c")]

    fused = _rrf_fuse(
        vector,
        keyword,
        vector_weight=1.0,
        keyword_weight=1.0,
        rrf_k=60,
    )
    assert [match.chunk_id for match in fused] == ["b", "a", "c"]

    keyword_only = _rrf_fuse(
        vector,
        keyword,
        vector_weight=0.0,
        keyword_weight=1.0,
        rrf_k=60,
    )
    assert [match.chunk_id for match in keyword_only] == ["b", "c", "a"]


def test_hybrid_retrieval_uses_keyword_candidates_and_can_be_disabled(
    tmp_path,
    monkeypatch,
):
    settings = make_settings(
        tmp_path,
        RAG_APP_RETRIEVAL_TOP_K="1",
        RAG_APP_RETRIEVAL_VECTOR_WEIGHT="0",
        RAG_APP_RETRIEVAL_KEYWORD_WEIGHT="1",
    )
    fake = FakeProvider(settings.embedding_dimension)
    container = build_container(settings, fake)
    manager = container.manager
    kb = manager.create("Docs").id
    manager.ingest(
        kb,
        "spec.md",
        b"# Spec\nAX6000 supports WiFi 6.\n",
    )
    manager.ingest(
        kb,
        "story.md",
        b"# Story\nThe router performs well in daily use.\n",
    )

    retriever = EvidenceRetriever(
        settings,
        container.vectors,
        fake,
        container.sqlite,
    )
    result = retriever.retrieve("AX6000 WiFi 6", [kb])
    assert result.evidence
    assert "AX6000" in result.evidence[0].content

    disabled_settings = replace(settings, hybrid_retrieval=False)

    def fail_keyword_search(*args, **kwargs):
        raise AssertionError("keyword search must not run when hybrid is disabled")

    monkeypatch.setattr(container.sqlite, "search_keyword", fail_keyword_search)
    disabled_retriever = EvidenceRetriever(
        disabled_settings,
        container.vectors,
        fake,
        container.sqlite,
    )
    disabled_result = disabled_retriever.retrieve("AX6000 WiFi 6", [kb])
    assert disabled_result.evidence


def make_vector_match(chunk_id: str) -> VectorMatch:
    return VectorMatch(
        chunk_id=chunk_id,
        kb_id="kb",
        document_id="doc",
        document_name="doc.md",
        heading_path=("Heading",),
        chunk_index=1,
        content=f"content {chunk_id}",
        content_hash=chunk_id,
        similarity=0.9,
    )


def make_keyword_match(chunk_id: str) -> KeywordMatch:
    return KeywordMatch(
        chunk_id=chunk_id,
        kb_id="kb",
        document_id="doc",
        document_name="doc.md",
        heading_path=("Heading",),
        chunk_index=1,
        content=f"content {chunk_id}",
        content_hash=chunk_id,
        keyword_score=-1.0,
    )
