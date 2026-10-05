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
TERMINAL_JOB_STATUSES = frozenset({"completed", "cancelled"})


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


class UploadCancelledError(Exception):
    """Raised internally when an upload crosses a cancellation boundary."""


class UploadJobManager:
    """In-process, thread-pool upload jobs with a spool for raw bytes.

    Jobs intentionally remain in memory. A restart loses their API status, and
    startup removes any raw spool files left behind by the previous process.
    """

    def __init__(self, spool_dir: Path, max_workers: int) -> None:
        self.spool_dir = Path(spool_dir).resolve()
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._clean_stale_spool()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="upload",
        )
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        kb_id: str,
        files: Sequence[tuple[str, bytes]],
        ingest: UploadTask,
    ) -> str:
        """Queue a batch of raw files for ingestion and return a job id."""
        if not files:
            raise ValueError("An upload job requires at least one file")

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

        try:
            for item, (_, raw) in zip(job.items, files, strict=True):
                self._write_spool(job_id, item.filename, raw)
            with self._lock:
                self._jobs[job_id] = job
            self._executor.submit(self._run, job_id, files, ingest)
        except Exception:
            for item, _ in zip(job.items, files, strict=True):
                self._remove_spool(job_id, item.filename)
            with self._lock:
                self._jobs.pop(job_id, None)
            raise
        return job_id

    def snapshot(self, job_id: str) -> UploadJobSnapshot | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.snapshot()

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a queued or running job.

        Cancellation is cooperative: already-persisted documents are retained,
        while work stops at the next file or progress-stage boundary.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_JOB_STATUSES:
                return False
            job.cancel_requested = True
            job.updated_at = datetime.now(UTC)
            return True

    def close(self) -> None:
        with self._lock:
            active_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status not in TERMINAL_JOB_STATUSES
            ]
            now = datetime.now(UTC)
            for job_id in active_ids:
                job = self._jobs[job_id]
                job.cancel_requested = True
                job.updated_at = now
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(
        self,
        job_id: str,
        files: Sequence[tuple[str, bytes]],
        ingest: UploadTask,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]

        if self._cancel_requested(job_id):
            self._cancel_items(job_id, 0)
            self._cleanup_spool(job_id, files)
            self._finalize(job_id, "cancelled")
            return

        self._mark(job_id, "processing")
        for index, (filename, _) in enumerate(files):
            if self._cancel_requested(job_id):
                self._cancel_items(job_id, index)
                self._cleanup_spool(job_id, files)
                self._finalize(job_id, "cancelled")
                return

            self._set_item(
                job_id,
                index,
                UploadJobItem(
                    filename=filename,
                    status="processing",
                    code=None,
                    message=None,
                    progress=(),
                    document=None,
                ),
            )
            progress: list[str] = []

            def report(
                stage: str,
                *,
                index: int = index,
                filename: str = filename,
                item_progress: list[str] = progress,
            ) -> None:
                if stage != "完成" and self._cancel_requested(job_id):
                    raise UploadCancelledError
                item_progress.append(stage)
                self._set_item(
                    job_id,
                    index,
                    UploadJobItem(
                        filename=filename,
                        status="processing",
                        code=None,
                        message=None,
                        progress=tuple(item_progress),
                        document=None,
                    ),
                )
                if stage != "完成" and self._cancel_requested(job_id):
                    raise UploadCancelledError

            try:
                raw = self._read_spool(job_id, filename)
                document = ingest(job.kb_id, filename, raw, report)
                self._set_item(
                    job_id,
                    index,
                    UploadJobItem(
                        filename=filename,
                        status="completed",
                        code=None,
                        message=None,
                        progress=tuple(progress),
                        document=document,
                    ),
                )
            except UploadCancelledError:
                self._set_item(
                    job_id,
                    index,
                    UploadJobItem(
                        filename=filename,
                        status="cancelled",
                        code="upload_cancelled",
                        message="Upload cancelled",
                        progress=tuple(progress),
                        document=None,
                    ),
                )
            except RagAppError as error:
                self._set_item(
                    job_id,
                    index,
                    UploadJobItem(
                        filename=filename,
                        status="failed",
                        code=error.code,
                        message=str(error),
                        progress=tuple(progress),
                        document=None,
                    ),
                )
            except OSError:
                self._set_item(
                    job_id,
                    index,
                    UploadJobItem(
                        filename=filename,
                        status="failed",
                        code="storage_write_failed",
                        message="Unable to read the uploaded file",
                        progress=tuple(progress),
                        document=None,
                    ),
                )
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "upload job item failed",
                    extra={"job_id": job_id, "error": type(error).__name__},
                )
                self._set_item(
                    job_id,
                    index,
                    UploadJobItem(
                        filename=filename,
                        status="failed",
                        code="internal_error",
                        message="Document processing failed",
                        progress=tuple(progress),
                        document=None,
                    ),
                )
            finally:
                self._remove_spool(job_id, filename)

            if self._cancel_requested(job_id):
                self._cancel_items(job_id, index + 1)
                self._cleanup_spool(job_id, files)
                self._finalize(job_id, "cancelled")
                return

        self._finalize(job_id, "completed")

    def _cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return job is not None and job.cancel_requested

    def _cancel_items(self, job_id: str, start_index: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            items = list(job.items)
            for index in range(start_index, len(items)):
                if items[index].status in {"pending", "processing"}:
                    items[index] = UploadJobItem(
                        filename=items[index].filename,
                        status="cancelled",
                        code="upload_cancelled",
                        message="Upload cancelled",
                        progress=items[index].progress,
                        document=None,
                    )
            job.items = tuple(items)
            job.updated_at = datetime.now(UTC)

    def _set_item(self, job_id: str, index: int, item: UploadJobItem) -> None:
        with self._lock:
            job = self._jobs[job_id]
            items = list(job.items)
            items[index] = item
            job.items = tuple(items)
            job.updated_at = datetime.now(UTC)

    def _finalize(self, job_id: str, job_status: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = job_status
            job.updated_at = datetime.now(UTC)

    def _mark(self, job_id: str, job_status: str) -> None:
        self._finalize(job_id, job_status)

    def _clean_stale_spool(self) -> None:
        removed = 0
        failed = 0
        for path in self.spool_dir.iterdir():
            if path.is_dir() and not path.is_symlink():
                continue
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                failed += 1
        if removed or failed:
            logger.warning(
                "upload spool cleanup removed stale files",
                extra={"removed": removed, "failed": failed},
            )

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

    def _cleanup_spool(self, job_id: str, files: Sequence[tuple[str, bytes]]) -> None:
        for filename, _ in files:
            self._remove_spool(job_id, filename)


@dataclass
class _Job:
    job_id: str
    kb_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    items: tuple[UploadJobItem, ...]
    cancel_requested: bool = False

    def snapshot(self) -> UploadJobSnapshot:
        return UploadJobSnapshot(
            job_id=self.job_id,
            kb_id=self.kb_id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            items=self.items,
        )
