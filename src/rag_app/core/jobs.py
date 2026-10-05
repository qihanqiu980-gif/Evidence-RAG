from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..errors import RagAppError
from ..models import Document

logger = logging.getLogger(__name__)
UploadTask = Callable[[str, str, bytes, Callable[[str], None]], Document]


@dataclass(frozen=True, slots=True)
class UploadJobItem:
    filename: str
    status: str
    code: str | None
    message: str | None
    progress: tuple[str, ...]
    document: Document | None


@dataclass(frozen=True, slots=True)
class UploadJobSnapshot:
    job_id: str
    kb_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    items: tuple[UploadJobItem, ...]


class UploadJobManager:
    """In-process, thread-pool upload jobs with a spool for raw bytes.

    Keeps the product's "no Celery/Redis" boundary: jobs are local, bounded,
    and safe to lose across restarts (spooled raw files are removed on close).
    """

    def __init__(self, spool_dir: Path, max_workers: int) -> None:
        self.spool_dir = Path(spool_dir).resolve()
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="upload")
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        kb_id: str,
        files: Sequence[tuple[str, bytes]],
        ingest: UploadTask,
    ) -> str:
        """Queue a batch of raw files for ingestion and return a job id."""
        job_id = str(uuid4())
        now = datetime.now(UTC)
        job = _Job(
            job_id=job_id,
            kb_id=kb_id,
            status="pending",
            created_at=now,
            updated_at=now,
            items=tuple(
                UploadJobItem(
                    filename=filename,
                    status="pending",
                    code=None,
                    message=None,
                    progress=(),
                    document=None,
                )
                for filename, _ in files
            ),
        )
        with self._lock:
            self._jobs[job_id] = job
        for item, (_, raw) in zip(job.items, files, strict=True):
            self._write_spool(job_id, item.filename, raw)
        self._executor.submit(self._run, job_id, files, ingest)
        return job_id

    def snapshot(self, job_id: str) -> UploadJobSnapshot | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.snapshot()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(
        self,
        job_id: str,
        files: Sequence[tuple[str, bytes]],
        ingest: UploadTask,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
        self._mark(job_id, "processing")
        items: list[UploadJobItem] = []
        for index, (filename, _) in enumerate(files):
            raw = self._read_spool(job_id, filename)
            progress: list[str] = []
            try:
                document = ingest(job.kb_id, filename, raw, progress.append)
                items.append(
                    UploadJobItem(
                        filename=filename,
                        status="completed",
                        code=None,
                        message=None,
                        progress=tuple(progress),
                        document=document,
                    )
                )
            except RagAppError as error:
                items.append(
                    UploadJobItem(
                        filename=filename,
                        status="failed",
                        code=error.code,
                        message=str(error),
                        progress=tuple(progress),
                        document=None,
                    )
                )
            except OSError:
                items.append(
                    UploadJobItem(
                        filename=filename,
                        status="failed",
                        code="storage_write_failed",
                        message="Unable to read the uploaded file",
                        progress=tuple(progress),
                        document=None,
                    )
                )
            except Exception as error:  # noqa: BLE001
                logger.warning("upload job item failed", extra={"job_id": job_id, "error": type(error).__name__})
                items.append(
                    UploadJobItem(
                        filename=filename,
                        status="failed",
                        code="internal_error",
                        message="Document processing failed",
                        progress=tuple(progress),
                        document=None,
                    )
                )
            finally:
                self._remove_spool(job_id, filename)

        with self._lock:
            job.items = tuple(items)
            job.status = "completed"
            job.updated_at = datetime.now(UTC)

    def _mark(self, job_id: str, status: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.updated_at = datetime.now(UTC)

    def _spool_path(self, job_id: str, filename: str) -> Path:
        safe = Path(filename).name
        return self.spool_dir / f"{job_id}-{safe}"

    def _write_spool(self, job_id: str, filename: str, raw: bytes) -> None:
        self._spool_path(job_id, filename).write_bytes(raw)

    def _read_spool(self, job_id: str, filename: str) -> bytes:
        return self._spool_path(job_id, filename).read_bytes()

    def _remove_spool(self, job_id: str, filename: str) -> None:
        path = self._spool_path(job_id, filename)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@dataclass
class _Job:
    job_id: str
    kb_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    items: tuple[UploadJobItem, ...]

    def snapshot(self) -> UploadJobSnapshot:
        return UploadJobSnapshot(
            job_id=self.job_id,
            kb_id=self.kb_id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            items=self.items,
        )
