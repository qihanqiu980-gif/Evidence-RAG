"""Minimal, portable backup/restore for the local data plane.

A backup is a single ``.tar.gz`` archive containing a consistent snapshot of:

- ``app.db``        (SQLite business facts, copied with the SQLite backup API)
- ``uploads/``      (original uploaded files)
- ``chroma/``       (the vector index)
- ``manifest.json`` (metadata used for restore-time validation)

This module intentionally does not touch ``.env``, prompts, or any model
secrets. It is a manual safety net for migrations such as the document-adapter
work, not a continuous-replication service.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..config import Settings
from ..errors import StorageInconsistentError
from .sqlite import SCHEMA_VERSION, SQLiteStore

APP_VERSION = "0.1.0"
MANIFEST_NAME = "manifest.json"
ARCHIVE_EXTENSION = ".tar.gz"
REQUIRED_PAYLOADS = ("app.db", "uploads", "chroma")


@dataclass(frozen=True, slots=True)
class BackupManifest:
    app_version: str
    created_at: str
    schema_version: int
    counts: dict[str, int]
    embedding_model: str | None
    embedding_dimension: int | None
    collection_name: str | None


@dataclass(frozen=True, slots=True)
class RestoreResult:
    manifest: BackupManifest
    data_dir: Path


def create_backup(settings: Settings, output_path: Path) -> Path:
    """Write a data-plane snapshot to ``output_path`` and return it."""
    if not settings.sqlite_path.is_file():
        raise StorageInconsistentError("Nothing to back up: the SQLite database is missing")

    output_path = Path(output_path).resolve()
    if output_path.is_dir():
        raise StorageInconsistentError("Backup output path must be a file")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sqlite = SQLiteStore(settings.sqlite_path)
    kb_count, document_count, chunk_count = sqlite.counts()
    active = sqlite.get_active_vector_collection()
    manifest = BackupManifest(
        app_version=APP_VERSION,
        created_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        schema_version=SCHEMA_VERSION,
        counts={
            "knowledge_bases": kb_count,
            "documents": document_count,
            "chunks": chunk_count,
            "vectors": active.chunk_count if active is not None else 0,
        },
        embedding_model=active.embedding_model if active is not None else None,
        embedding_dimension=active.embedding_dimension if active is not None else None,
        collection_name=active.collection_name if active is not None else None,
    )

    with tempfile.TemporaryDirectory(prefix="rag-backup-") as staging_name:
        staging = Path(staging_name)
        _snapshot_sqlite(settings.sqlite_path, staging / "app.db")
        _copy_tree_if_present(settings.uploads_path, staging / "uploads")
        _copy_tree_if_present(settings.chroma_path, staging / "chroma")
        (staging / MANIFEST_NAME).write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        _tar_directory(staging, output_path)

    return output_path


def restore_backup(
    settings: Settings,
    archive_path: Path,
    *,
    force: bool = False,
) -> RestoreResult:
    """Restore ``archive_path`` into the configured data directory.

    Refuses to overwrite a non-empty data directory unless ``force`` is set.
    """
    archive_path = Path(archive_path).resolve()
    if not archive_path.is_file():
        raise StorageInconsistentError("Backup archive not found")

    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    if _directory_has_content(data_dir) and not force:
        raise StorageInconsistentError(
            "The data directory is not empty; rerun with --yes to overwrite"
        )

    with tempfile.TemporaryDirectory(prefix="rag-restore-") as staging_name:
        staging = Path(staging_name)
        _extract_archive(archive_path, staging)
        manifest = _load_manifest(staging / MANIFEST_NAME)
        _validate_manifest(manifest)
        for required in REQUIRED_PAYLOADS:
            if not (staging / required).exists():
                raise StorageInconsistentError(
                    f"Backup archive is missing required payload: {required}"
                )

        _replace(staging / "app.db", data_dir / "app.db")
        _replace(staging / "uploads", data_dir / "uploads")
        _replace(staging / "chroma", data_dir / "chroma")
        _remove_stale_sqlite_journals(data_dir)
        if manifest.schema_version < SCHEMA_VERSION:
            SQLiteStore(data_dir / "app.db")

    return RestoreResult(manifest=manifest, data_dir=data_dir)


def _snapshot_sqlite(source: Path, destination: Path) -> None:
    """Copy ``source`` using the SQLite online-backup API for a consistent file."""
    source_conn = sqlite3.connect(source)
    try:
        destination_conn = sqlite3.connect(destination)
        try:
            source_conn.backup(destination_conn)
        finally:
            destination_conn.close()
    finally:
        source_conn.close()


def _copy_tree_if_present(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)


def _tar_directory(source_dir: Path, output_path: Path) -> None:
    with tarfile.open(output_path, "w:gz") as archive:
        for path in sorted(source_dir.iterdir()):
            archive.add(path, arcname=path.name)


def _extract_archive(archive_path: Path, destination: Path) -> None:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination, filter="data")
    except tarfile.TarError as exc:
        raise StorageInconsistentError("Unable to read the backup archive") from exc


def _load_manifest(path: Path) -> BackupManifest:
    if not path.is_file():
        raise StorageInconsistentError("Backup archive is missing its manifest")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageInconsistentError("Backup manifest is unreadable") from exc
    if not isinstance(raw, dict):
        raise StorageInconsistentError("Backup manifest is invalid")
    return BackupManifest(
        app_version=str(raw.get("app_version", "")),
        created_at=str(raw.get("created_at", "")),
        schema_version=int(raw.get("schema_version", 0)),
        counts={
            str(key): int(value)
            for key, value in (raw.get("counts") or {}).items()
        },
        embedding_model=(
            str(raw["embedding_model"]) if raw.get("embedding_model") is not None else None
        ),
        embedding_dimension=(
            int(raw["embedding_dimension"])
            if raw.get("embedding_dimension") is not None
            else None
        ),
        collection_name=(
            str(raw["collection_name"]) if raw.get("collection_name") is not None else None
        ),
    )


def _validate_manifest(manifest: BackupManifest) -> None:
    if manifest.schema_version > SCHEMA_VERSION:
        raise StorageInconsistentError(
            f"Backup schema version {manifest.schema_version} is newer than "
            f"the current version {SCHEMA_VERSION}"
        )


def _replace(source: Path, destination: Path) -> None:
    if destination.is_symlink() or destination.exists():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.move(source, destination)


def _remove_stale_sqlite_journals(data_dir: Path) -> None:
    for suffix in ("-wal", "-shm"):
        path = data_dir / f"app.db{suffix}"
        if path.exists() or path.is_symlink():
            path.unlink()


def _directory_has_content(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(path.iterdir())
    except OSError:
        return True
