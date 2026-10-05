from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

from rag_app.core.jobs import UploadJobManager
from rag_app.models import Document, DocumentStatus


def _document(filename: str) -> Document:
    return Document(
        id=f"doc-{filename}",
        kb_id="kb",
        filename=filename,
        content_hash=f"hash-{filename}",
        storage_path=f"uploads/{filename}",
        chunk_count=1,
        status=DocumentStatus.READY,
        created_at=datetime.now(UTC),
    )


def _wait_terminal(manager: UploadJobManager, job_id: str) -> str | None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(job_id)
        if snapshot is not None and snapshot.status in {"completed", "cancelled"}:
            return snapshot.status
        time.sleep(0.01)
    final = manager.snapshot(job_id)
    return final.status if final is not None else None


def test_startup_cleanup_removes_stale_spool_files_but_preserves_directories(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    stale_file = spool_dir / "old-job.md"
    stale_file.write_bytes(b"stale")
    stale_target = tmp_path / "target.md"
    stale_target.write_bytes(b"target")
    stale_link = spool_dir / "old-link.md"
    stale_link.symlink_to(stale_target)
    preserved_dir = spool_dir / "nested"
    preserved_dir.mkdir()

    manager = UploadJobManager(spool_dir, max_workers=1)
    try:
        assert not stale_file.exists()
        assert not stale_link.exists()
        assert preserved_dir.is_dir()
    finally:
        manager.close()


def test_queued_job_can_be_cancelled_before_ingestion_starts(tmp_path):
    started = threading.Event()
    release = threading.Event()
    ingested: list[str] = []
    manager = UploadJobManager(tmp_path / "spool", max_workers=1)

    def blocker(kb_id: str, filename: str, raw: bytes, report) -> Document:
        ingested.append(filename)
        report("校验")
        started.set()
        assert release.wait(5)
        report("完成")
        return _document(filename)

    try:
        first_id = manager.submit("kb", [("blocker.md", b"first")], blocker)
        assert started.wait(5)
        second_id = manager.submit(
            "kb",
            [("queued-one.md", b"one"), ("queued-two.md", b"two")],
            blocker,
        )

        assert manager.cancel(second_id) is True
        release.set()

        assert _wait_terminal(manager, first_id) == "completed"
        assert _wait_terminal(manager, second_id) == "cancelled"
        second = manager.snapshot(second_id)
        assert second is not None
        assert [item.status for item in second.items] == ["cancelled", "cancelled"]
        assert [item.code for item in second.items] == ["upload_cancelled"] * 2
        assert ingested == ["blocker.md"]
        assert list((tmp_path / "spool").glob("*")) == []
    finally:
        release.set()
        manager.close()


def test_running_job_cancels_at_progress_stage_boundary(tmp_path):
    started = threading.Event()
    release = threading.Event()
    ingested: list[str] = []
    manager = UploadJobManager(tmp_path / "spool", max_workers=1)

    def first_ingest(kb_id: str, filename: str, raw: bytes, report) -> Document:
        ingested.append(filename)
        report("校验")
        started.set()
        assert release.wait(5)
        report("切块")
        return _document(filename)

    try:
        job_id = manager.submit(
            "kb",
            [("first.md", b"first"), ("second.md", b"second")],
            lambda kb_id, filename, raw, report: (
                first_ingest(kb_id, filename, raw, report)
                if filename == "first.md"
                else _document(filename)
            ),
        )
        assert started.wait(5)
        assert manager.cancel(job_id) is True
        release.set()

        assert _wait_terminal(manager, job_id) == "cancelled"
        snapshot = manager.snapshot(job_id)
        assert snapshot is not None
        assert [item.status for item in snapshot.items] == ["cancelled", "cancelled"]
        assert snapshot.items[0].progress == ("校验",)
        assert snapshot.items[1].progress == ()
        assert ingested == ["first.md"]
        assert list((tmp_path / "spool").glob("*")) == []
    finally:
        release.set()
        manager.close()


def test_cancellation_retains_files_already_completed_before_boundary(tmp_path):
    started = threading.Event()
    release = threading.Event()
    ingested: list[str] = []
    manager = UploadJobManager(tmp_path / "spool", max_workers=1)

    def first_ingest(kb_id: str, filename: str, raw: bytes, report) -> Document:
        ingested.append(filename)
        report("校验")
        report("完成")
        started.set()
        assert release.wait(5)
        return _document(filename)

    try:
        job_id = manager.submit(
            "kb",
            [("first.md", b"first"), ("second.md", b"second")],
            lambda kb_id, filename, raw, report: (
                first_ingest(kb_id, filename, raw, report)
                if filename == "first.md"
                else _document(filename)
            ),
        )
        assert started.wait(5)
        assert manager.cancel(job_id) is True
        release.set()

        assert _wait_terminal(manager, job_id) == "cancelled"
        snapshot = manager.snapshot(job_id)
        assert snapshot is not None
        assert [item.status for item in snapshot.items] == ["completed", "cancelled"]
        assert snapshot.items[0].document is not None
        assert snapshot.items[1].document is None
        assert ingested == ["first.md"]
        assert list((tmp_path / "spool").glob("*")) == []
    finally:
        release.set()
        manager.close()
