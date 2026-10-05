
import json
import logging
import threading
import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from rag_app.api.app import build_app
from rag_app.models import Document, DocumentStatus
from tests.conftest import make_settings
from tests.fakes.provider import FakeProvider


def test_status_reports_configuration_missing_without_model_or_key(tmp_path):
    settings = make_settings(tmp_path, RAG_APP_API_KEY="")
    app = build_app(settings)
    with TestClient(app) as client:
        response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "configuration_missing"
    assert payload["limits"] == {
        "max_files_per_upload": 20,
        "max_file_bytes": 5242880,
        "accepted_extensions": [".md", ".txt", ".pdf", ".docx", ".html", ".htm"],
    }
    assert "api_key" not in response.text
    assert "embedding_model" not in response.text
    assert "qwen" not in response.text.lower()


def test_app_lifecycle_logs_safe_startup_state_and_stops(tmp_path, caplog):
    settings = make_settings(tmp_path, RAG_APP_API_KEY="")
    app = build_app(settings)

    with caplog.at_level(logging.INFO, logger="rag_app.api.app"), TestClient(app) as client:
        response = client.get("/api/status")
        assert response.status_code == 200

    messages = [record.getMessage() for record in caplog.records]
    assert "service started" in messages
    assert "service stopped" in messages
    startup = next(record for record in caplog.records if record.getMessage() == "service started")
    assert startup.configured is False
    assert startup.knowledge_base_count == 0
    assert "test-api-key" not in caplog.text


def test_status_is_ready_when_provider_configuration_exists(tmp_path):
    settings = make_settings(tmp_path)
    app = build_app(settings)
    with TestClient(app) as client:
        response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_knowledge_base_document_upload_and_delete_flow(tmp_path):
    settings = make_settings(tmp_path)
    app = build_app(settings, FakeProvider(settings.embedding_dimension))
    content = "# Router\n## WiFi\nThe router supports WiFi 6.\n"

    with TestClient(app) as client:
        created = client.post("/api/knowledge-bases", json={"name": "Product Docs"})
        assert created.status_code == 201
        kb_id = created.json()["id"]

        response = client.post(
            f"/api/knowledge-bases/{kb_id}/documents",
            files=[
                ("files", ("router.md", content.encode("utf-8"), "text/markdown")),
                ("files", ("invalid.csv", b"a,b\nc,d", "text/csv")),
            ],
        )
        assert response.status_code == 207
        payload = response.json()
        assert len(payload["documents"]) == 1
        assert payload["documents"][0]["progress"] == ["校验", "切块", "向量化", "保存", "完成"]
        assert payload["errors"] == [
            {
                "filename": "invalid.csv",
                "code": "invalid_extension",
                "message": "Unsupported file type; supported: .md, .txt, .pdf, .docx, .html",
            }
        ]
        document_id = payload["documents"][0]["id"]

        duplicate = client.post(
            f"/api/knowledge-bases/{kb_id}/documents",
            files=[
                (
                    "files",
                    ("renamed.md", content.replace("\n", "\r\n").encode("utf-8"), "text/markdown"),
                )
            ],
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["errors"][0]["code"] == "duplicate_document"

        documents = client.get(f"/api/knowledge-bases/{kb_id}/documents")
        assert len(documents.json()) == 1

        deleted = client.delete(
            f"/api/knowledge-bases/{kb_id}/documents/{document_id}"
        )
        assert deleted.status_code == 204
        assert client.get(f"/api/knowledge-bases/{kb_id}/documents").json() == []


def test_upload_is_disabled_when_configuration_is_missing(tmp_path):
    settings = make_settings(tmp_path, RAG_APP_API_KEY="")
    app = build_app(settings)
    with TestClient(app) as client:
        created = client.post("/api/knowledge-bases", json={"name": "Docs"})
        kb_id = created.json()["id"]
        response = client.post(
            f"/api/knowledge-bases/{kb_id}/documents",
            files=[("files", ("router.md", b"# Router\n", "text/markdown"))],
        )
    assert response.status_code == 503
    assert response.json()["errors"][0]["code"] == "configuration_error"


def test_cross_knowledge_base_duplicate_content_is_allowed(tmp_path):
    settings = make_settings(tmp_path)
    fake = FakeProvider(settings.embedding_dimension)
    app = build_app(settings, fake)
    with TestClient(app) as client:
        ids = []
        for name in ("Docs A", "Docs B"):
            created = client.post("/api/knowledge-bases", json={"name": name})
            ids.append(created.json()["id"])
        for kb_id in ids:
            response = client.post(
                f"/api/knowledge-bases/{kb_id}/documents",
                files=[
                    (
                        "files",
                        ("router.md", b"# Router\nWiFi 6 support\n", "text/markdown"),
                    )
                ],
            )
            assert response.status_code == 201
        assert fake.embed_documents_calls == 2


def test_delete_knowledge_base_removes_only_its_data_plane(tmp_path):
    settings = make_settings(tmp_path)
    app = build_app(settings, FakeProvider(settings.embedding_dimension))
    with TestClient(app) as client:
        ids = []
        for name in ("Docs A", "Docs B"):
            created = client.post("/api/knowledge-bases", json={"name": name})
            ids.append(created.json()["id"])
        for kb_id in ids:
            response = client.post(
                f"/api/knowledge-bases/{kb_id}/documents",
                files=[
                    (
                        "files",
                        (f"{kb_id}.md", f"# {kb_id}\nUnique content\n".encode(), "text/markdown"),
                    )
                ],
            )
            assert response.status_code == 201

        assert client.delete(f"/api/knowledge-bases/{ids[0]}").status_code == 204
        assert client.get(f"/api/knowledge-bases/{ids[0]}/documents").status_code == 404
        assert len(client.get(f"/api/knowledge-bases/{ids[1]}/documents").json()) == 1
        assert not settings.uploads_path.joinpath(ids[0]).exists()
        assert settings.uploads_path.joinpath(ids[1]).is_dir()


def test_import_demo_ingests_six_documents_and_rejects_duplicates(tmp_path):
    settings = make_settings(tmp_path)
    fake = FakeProvider(settings.embedding_dimension)
    app = build_app(settings, fake)
    with TestClient(app) as client:
        kb = client.post("/api/knowledge-bases", json={"name": "Demo"})
        kb_id = kb.json()["id"]
        imported = client.post(f"/api/knowledge-bases/{kb_id}/import-demo")
        assert imported.status_code == 201
        assert len(imported.json()["documents"]) == 6

        duplicate = client.post(f"/api/knowledge-bases/{kb_id}/import-demo")
        assert duplicate.status_code == 409
        assert {item["code"] for item in duplicate.json()["errors"]} == {
            "duplicate_document"
        }


def test_chat_stream_returns_contract_events_and_hides_unverified_answer(tmp_path):
    settings = make_settings(tmp_path)
    fake = FakeProvider(settings.embedding_dimension)
    app = build_app(settings, fake)
    content = "# Router\nThe router supports WiFi 6.\n"
    fake.chat_response_queues = {
        "decompose": [
            {
                "standalone_question": "Does the router support WiFi 6?",
                "subquestions": [{"text": "Does the router support WiFi 6?"}],
            }
        ],
        "judge": [
            {
                "decisions": [
                    {
                        "subquestion_id": "q1",
                        "answerable": True,
                        "evidence_ids": [1],
                        "missing_information": "",
                        "reason": "证据直接回答子问题",
                    }
                ]
            }
        ],
        "generate": [
            {
                "parts": [
                    {
                        "subquestion_id": "q1",
                        "answer": "The router supports WiFi 6. [1]",
                        "citations": [1],
                    }
                ]
            }
        ],
        "validate": [
            {
                "validations": [
                    {
                        "subquestion_id": "q1",
                        "supported": True,
                        "complete": True,
                        "reason": "",
                    }
                ]
            }
        ],
    }

    with TestClient(app) as client:
        kb_id = client.post(
            "/api/knowledge-bases",
            json={"name": "Product Docs"},
        ).json()["id"]
        uploaded = client.post(
            f"/api/knowledge-bases/{kb_id}/documents",
            files=[("files", ("router.md", content.encode(), "text/markdown"))],
        )
        assert uploaded.status_code == 201

        response = client.post(
            "/api/chat/stream",
            json={
                "question": "Does the router support WiFi 6?",
                "history": [],
                "kb_ids": [kb_id],
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1)
    )
    assert events[-2]["type"] == "answer_final"
    assert events[-1]["type"] == "completed"
    answer_index = next(
        index for index, item in enumerate(events) if item["type"] == "answer_final"
    )
    before_answer = json.dumps(events[:answer_index], ensure_ascii=False)
    assert "The router supports WiFi 6. [1]" not in before_answer

    evidence = next(item for item in events if item["type"] == "evidence")
    evidence_item = evidence["payload"]["evidence"][0]
    assert evidence_item["support_status"] == "supporting"
    assert "content_hash" not in evidence_item
    assert "kb_id" not in evidence_item
    assert evidence_item["reference_id"] == 1

    final = events[answer_index]["payload"]
    assert final["refused"] is False
    assert final["parts"][0]["status"] == "verified"
    assert final["parts"][0]["citations"] == [1]

    assert "test-api-key" not in response.text
    assert str(tmp_path) not in response.text
    assert "content_hash" not in response.text
    assert "prompt" not in response.text


def test_chat_stream_rejects_missing_configuration_and_unknown_scope(tmp_path):
    missing_settings = make_settings(tmp_path, RAG_APP_API_KEY="")
    missing_app = build_app(missing_settings)
    with TestClient(missing_app) as client:
        kb_id = client.post("/api/knowledge-bases", json={"name": "Docs"}).json()["id"]
        missing = client.post(
            "/api/chat/stream",
            json={"question": "Does it work?", "history": [], "kb_ids": [kb_id]},
        )
    assert missing.status_code == 503
    assert missing.json()["detail"]["code"] == "configuration_error"

    settings = make_settings(tmp_path)
    app = build_app(settings, FakeProvider(settings.embedding_dimension))
    with TestClient(app) as client:
        unknown = client.post(
            "/api/chat/stream",
            json={"question": "Does it work?", "history": [], "kb_ids": ["missing"]},
        )
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["code"] == "not_found"


def test_chat_stream_midflight_failure_returns_safe_error_event(tmp_path):
    settings = make_settings(tmp_path)
    fake = FakeProvider(settings.embedding_dimension)
    fake.chat_response_queues = {
        "decompose": [
            {
                "standalone_question": "Does the router support WiFi 6?",
                "subquestions": [{"text": "Does the router support WiFi 6?"}],
            }
        ]
    }
    app = build_app(settings, fake)
    with TestClient(app) as client:
        kb_id = client.post(
            "/api/knowledge-bases",
            json={"name": "Product Docs"},
        ).json()["id"]
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
        fake.fail_rerank = True
        response = client.post(
            "/api/chat/stream",
            json={
                "question": "Does the router support WiFi 6?",
                "history": [],
                "kb_ids": [kb_id],
            },
        )
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert events[-1]["payload"]["code"] == "internal_error"
    assert "injected rerank failure" not in response.text


def test_upload_accepts_docx_and_html_and_preserves_extensions(tmp_path):
    import io
    import zipfile
    from xml.etree import ElementTree

    settings = make_settings(tmp_path)
    fake = FakeProvider(settings.embedding_dimension)
    app = build_app(settings, fake)

    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def tag(name: str) -> str:
        return f"{{{namespace}}}{name}"

    root = ElementTree.Element(tag("document"))
    body = ElementTree.SubElement(root, tag("body"))
    paragraph = ElementTree.SubElement(body, tag("p"))
    run = ElementTree.SubElement(paragraph, tag("r"))
    node = ElementTree.SubElement(run, tag("t"))
    node.text = "Router supports WiFi 6."
    xml = ElementTree.tostring(root, encoding="utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)

    with TestClient(app) as client:
        created = client.post("/api/knowledge-bases", json={"name": "Docs"})
        kb_id = created.json()["id"]
        response = client.post(
            f"/api/knowledge-bases/{kb_id}/documents",
            files=[
                (
                    "files",
                    ("manual.docx", buffer.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                ),
                (
                    "files",
                    ("page.html", b"<html><body><p>WiFi guide</p></body></html>", "text/html"),
                ),
            ],
        )
        assert response.status_code == 201
        payload = response.json()
        assert len(payload["documents"]) == 2
        assert [item["filename"] for item in payload["documents"]] == ["manual.docx", "page.html"]

        docs = client.get(f"/api/knowledge-bases/{kb_id}/documents").json()
        assert len(docs) == 2
        assert settings.uploads_path.joinpath(f"{kb_id}").is_dir()

        # Deleting both documents removes the upload files with preserved extensions.
        for doc in docs:
            assert client.delete(f"/api/knowledge-bases/{kb_id}/documents/{doc['id']}").status_code == 204
        assert client.get(f"/api/knowledge-bases/{kb_id}/documents").json() == []


def test_async_upload_returns_job_and_completes(tmp_path):
    import time

    settings = make_settings(tmp_path)
    fake = FakeProvider(settings.embedding_dimension)
    app = build_app(settings, fake)
    content = "# Router\nWiFi 6 support\n"

    with TestClient(app) as client:
        created = client.post("/api/knowledge-bases", json={"name": "Docs"})
        kb_id = created.json()["id"]

        response = client.post(
            f"/api/knowledge-bases/{kb_id}/documents/async",
            files=[("files", ("router.md", content.encode("utf-8"), "text/markdown"))],
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        # Poll until the job completes (bounded).
        status_seen = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            status_seen = job["status"]
            if status_seen == "completed":
                break
            time.sleep(0.05)

        assert status_seen == "completed"
        assert len(job["items"]) == 1
        assert job["items"][0]["status"] == "completed"
        assert job["items"][0]["document"]["filename"] == "router.md"

        docs = client.get(f"/api/knowledge-bases/{kb_id}/documents").json()
        assert len(docs) == 1


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [
        ("invalid.csv", b"a,b\n", "invalid_extension"),
        ("big.md", b"x" * 17, "file_too_large"),
        ("", b"# Router\n", "validation_error"),
    ],
)
def test_async_upload_rejects_invalid_submission_before_job_creation(
    tmp_path,
    filename,
    content,
    code,
):
    settings = make_settings(
        tmp_path,
        RAG_APP_UPLOAD_MAX_BYTES="16",
    )
    app = build_app(settings, FakeProvider(settings.embedding_dimension))

    with TestClient(app) as client:
        kb_id = client.post("/api/knowledge-bases", json={"name": "Docs"}).json()["id"]
        response = client.post(
            f"/api/knowledge-bases/{kb_id}/documents/async",
            files=[("files", (filename, content, "application/octet-stream"))],
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == code
        assert client.get(f"/api/knowledge-bases/{kb_id}/documents").json() == []
        assert list((settings.data_dir / "spool").glob("*")) == []


def test_async_upload_rejects_too_many_files_before_job_creation(tmp_path):
    settings = make_settings(tmp_path, RAG_APP_UPLOAD_MAX_FILES="1")
    app = build_app(settings, FakeProvider(settings.embedding_dimension))

    with TestClient(app) as client:
        kb_id = client.post("/api/knowledge-bases", json={"name": "Docs"}).json()["id"]
        response = client.post(
            f"/api/knowledge-bases/{kb_id}/documents/async",
            files=[
                ("files", ("one.md", b"# One\n", "text/markdown")),
                ("files", ("two.md", b"# Two\n", "text/markdown")),
            ],
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "validation_error"
        assert client.get(f"/api/knowledge-bases/{kb_id}/documents").json() == []
        assert list((settings.data_dir / "spool").glob("*")) == []


def test_async_upload_normalizes_filename_at_submission(tmp_path):
    settings = make_settings(tmp_path)
    app = build_app(settings, FakeProvider(settings.embedding_dimension))

    with TestClient(app) as client:
        kb_id = client.post("/api/knowledge-bases", json={"name": "Docs"}).json()["id"]
        response = client.post(
            f"/api/knowledge-bases/{kb_id}/documents/async",
            files=[("files", ("../router.md", b"# Router\n", "text/markdown"))],
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        deadline = time.monotonic() + 5
        job = None
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "completed":
                break
            time.sleep(0.05)

        assert job is not None
        assert job["items"][0]["filename"] == "router.md"


def test_cancel_upload_job_contract(tmp_path):
    import time

    settings = make_settings(tmp_path)
    app = build_app(settings, FakeProvider(settings.embedding_dimension))

    with TestClient(app) as client:
        unknown = client.post("/api/jobs/missing/cancel")
        assert unknown.status_code == 404
        assert unknown.json()["detail"]["code"] == "not_found"

        kb_id = client.post("/api/knowledge-bases", json={"name": "Docs"}).json()["id"]
        created = client.post(
            f"/api/knowledge-bases/{kb_id}/documents/async",
            files=[("files", ("router.md", b"# Router\n", "text/markdown"))],
        )
        job_id = created.json()["job_id"]

        deadline = time.monotonic() + 5
        job = None
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "completed":
                break
            time.sleep(0.05)

        assert job is not None
        assert job["status"] == "completed"
        finished = client.post(f"/api/jobs/{job_id}/cancel")
        assert finished.status_code == 409
        assert finished.json()["detail"]["code"] == "upload_not_cancelable"


def test_cancel_api_cancels_queued_job_and_cleans_spool(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, RAG_APP_UPLOAD_MAX_WORKERS="1")
    app = build_app(settings, FakeProvider(settings.embedding_dimension))
    started = threading.Event()
    release = threading.Event()
    ingested: list[str] = []

    def blocker(kb_id, filename, content, report):
        ingested.append(filename)
        report("校验")
        started.set()
        assert release.wait(5)
        report("完成")
        return Document(
            id=f"doc-{filename}",
            kb_id=kb_id,
            filename=filename,
            content_hash=f"hash-{filename}",
            storage_path=f"uploads/{filename}",
            chunk_count=1,
            status=DocumentStatus.READY,
            created_at=datetime.now(UTC),
        )

    monkeypatch.setattr(app.state.container.manager, "ingest", blocker)
    with TestClient(app) as client:
        kb_id = client.post("/api/knowledge-bases", json={"name": "Docs"}).json()["id"]
        first_created = client.post(
            f"/api/knowledge-bases/{kb_id}/documents/async",
            files=[("files", ("blocker.md", b"first", "text/markdown"))],
        )
        assert first_created.status_code == 202
        assert started.wait(5)

        second_created = client.post(
            f"/api/knowledge-bases/{kb_id}/documents/async",
            files=[("files", ("queued.md", b"second", "text/markdown"))],
        )
        assert second_created.status_code == 202
        queued_job_id = second_created.json()["job_id"]

        cancelled = client.post(f"/api/jobs/{queued_job_id}/cancel")
        assert cancelled.status_code == 202
        assert cancelled.json()["job_id"] == queued_job_id
        release.set()

        deadline = time.monotonic() + 5
        queued_job = None
        while time.monotonic() < deadline:
            queued_job = client.get(f"/api/jobs/{queued_job_id}").json()
            if queued_job["status"] == "cancelled":
                break
            time.sleep(0.05)

        assert queued_job is not None
        assert queued_job["status"] == "cancelled"
        assert queued_job["items"][0]["code"] == "upload_cancelled"
        assert ingested == ["blocker.md"]
        assert list((settings.data_dir / "spool").glob("*")) == []
