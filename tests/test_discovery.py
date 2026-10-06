from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from rag_app.api.app import build_app
from rag_app.core.discovery import KnowledgeDiscoveryService
from rag_app.models import Chunk, Document, DocumentStatus
from rag_app.storage.sqlite import SQLiteStore
from tests.conftest import make_settings
from tests.fakes.provider import FakeProvider


def test_discovery_persists_source_bound_questions_and_invalidates_on_delete(tmp_path):
    store = SQLiteStore(tmp_path / "app.db")
    kb = store.create_knowledge_base(str(uuid4()), "Manual")
    document_id = str(uuid4())
    chunks = [
        make_chunk(
            document_id,
            kb.id,
            chunk_index=1,
            heading_path=("WiFi",),
            content="AX6000 支持 WiFi 6 标准。",
        ),
        make_chunk(
            document_id,
            kb.id,
            chunk_index=2,
            heading_path=("Troubleshooting",),
            content="WiFi 指示灯红色常亮表示系统故障。",
        ),
    ]
    store.insert_document(make_document(document_id, kb.id, chunks), chunks)
    provider = FakeProvider(8)
    provider.chat_responses = {
        "discovery_topics": {
            "topics": [
                {
                    "title": "无线能力",
                    "type": "概念",
                    "summary": "无线标准与指示灯状态",
                    "confidence": 0.9,
                    "source_chunk_ids": [chunks[0].id, chunks[1].id, "invented:id"],
                }
            ]
        },
        "discovery_questions": {
            "questions": [
                {
                    "topic_id": "t1",
                    "question": "AX6000 支持哪个 WiFi 标准？",
                    "source_chunk_ids": [chunks[0].id],
                },
                {
                    "topic_id": "t1",
                    "question": "这个编造的问题支持吗？",
                    "source_chunk_ids": ["invented:id"],
                },
            ]
        },
    }

    result = KnowledgeDiscoveryService(store, provider, make_settings(tmp_path)).analyze(kb.id)

    assert result.topic_count == 1
    assert result.question_count == 1
    assert result.topics[0].chunk_count == 2
    assert result.topics[0].questions[0].source_chunk_ids == (chunks[0].id,)
    assert provider.chat_calls == 2
    persisted = store.list_discovery(kb.id)
    assert persisted is not None
    assert persisted.topics[0].questions[0].question == "AX6000 支持哪个 WiFi 标准？"

    store.delete_document(kb.id, document_id)
    assert store.list_discovery(kb.id) is None


def test_discovery_api_generates_and_exposes_question_map(tmp_path):
    settings = make_settings(tmp_path)
    fake = FakeProvider(settings.embedding_dimension)
    app = build_app(settings, fake)

    with TestClient(app) as client:
        created = client.post("/api/knowledge-bases", json={"name": "Product"})
        kb_id = created.json()["id"]
        uploaded = client.post(
            f"/api/knowledge-bases/{kb_id}/documents",
            files=[
                (
                    "files",
                    ("router.md", b"# Router\nThe router supports WiFi 6.\n", "text/markdown"),
                )
            ],
        )
        assert uploaded.status_code == 201
        document_id = uploaded.json()["documents"][0]["id"]
        chunk_id = f"{document_id}:0001"
        fake.chat_responses = {
            "discovery_topics": {
                "topics": [
                    {
                        "title": "产品规格",
                        "type": "概念",
                        "summary": "无线标准",
                        "confidence": 0.9,
                        "source_chunk_ids": [chunk_id],
                    }
                ]
            },
            "discovery_questions": {
                "questions": [
                    {
                        "topic_id": "t1",
                        "question": "这台路由器支持哪个 WiFi 标准？",
                        "source_chunk_ids": [chunk_id],
                    }
                ]
            },
        }

        accepted = client.post(f"/api/knowledge-bases/{kb_id}/discovery/analyze")
        assert accepted.status_code == 202

        for _ in range(50):
            payload = client.get(f"/api/knowledge-bases/{kb_id}/discovery").json()
            if payload["status"] == "completed":
                break

        assert payload["status"] == "completed"
        assert payload["document_count"] == 1
        assert payload["chunk_count"] == 1
        assert payload["topic_count"] == 1
        assert payload["topics"][0]["questions"][0]["source_chunk_ids"] == [chunk_id]


def make_document(document_id: str, kb_id: str, chunks: list[Chunk]) -> Document:
    return Document(
        id=document_id,
        kb_id=kb_id,
        filename="manual.md",
        content_hash=uuid4().hex,
        storage_path=f"{kb_id}/{document_id}.md",
        chunk_count=len(chunks),
        status=DocumentStatus.READY,
        created_at=datetime.now(UTC),
    )


def make_chunk(
    document_id: str,
    kb_id: str,
    *,
    chunk_index: int,
    heading_path: tuple[str, ...],
    content: str,
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
        content_hash=uuid4().hex,
    )
