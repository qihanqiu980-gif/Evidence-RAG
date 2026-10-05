from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from uuid import UUID

from ..errors import StorageInconsistentError


class UploadStore:
    """Path-constrained storage for original uploaded business documents."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, kb_id: str, document_id: str, content: str, filename: str) -> str:
        relative_path = self.relative_path(kb_id, document_id, filename)
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f".{document_id}-",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, path)
        except OSError as exc:
            raise StorageInconsistentError("Unable to save uploaded document") from exc
        return Path(relative_path).as_posix()

    def read(self, storage_path: str) -> str:
        path = self._resolve(storage_path)
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise StorageInconsistentError("Unable to read uploaded document") from exc

    def delete_document(self, storage_path: str) -> None:
        path = self._resolve(storage_path)
        try:
            path.unlink(missing_ok=True)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise StorageInconsistentError("Unable to delete uploaded document") from exc
        try:
            path.parent.rmdir()
        except OSError:
            # Ignore a non-empty directory; siblings may still exist.
            pass

    def delete_knowledge_base(self, kb_id: str) -> None:
        self._validate_uuid(kb_id, "knowledge base id")
        path = (self.root / kb_id).resolve()
        self._ensure_inside_root(path)
        try:
            if path.exists():
                shutil.rmtree(path)
        except OSError as exc:
            raise StorageInconsistentError("Unable to delete knowledge base uploads") from exc

    def exists(self, storage_path: str) -> bool:
        return self._resolve(storage_path).is_file()

    def document_paths(self, kb_id: str | None = None) -> list[str]:
        base = self.root / kb_id if kb_id is not None else self.root
        self._ensure_inside_root(base.resolve())
        if not base.exists():
            return []
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )

    def relative_path(self, kb_id: str, document_id: str, filename: str) -> str:
        self._validate_uuid(kb_id, "knowledge base id")
        self._validate_uuid(document_id, "document id")
        extension = Path(filename).suffix.lower() or ".md"
        return f"{kb_id}/{document_id}{extension}"

    def _resolve(self, storage_path: str) -> Path:
        if storage_path.startswith(("/", "\\")) or ".." in Path(storage_path).parts:
            raise StorageInconsistentError("Invalid upload storage path")
        path = (self.root / storage_path).resolve()
        self._ensure_inside_root(path)
        return path

    def _ensure_inside_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise StorageInconsistentError("Upload path escapes the storage root") from exc

    @staticmethod
    def _validate_uuid(value: str, label: str) -> None:
        try:
            UUID(value)
        except ValueError as exc:
            raise StorageInconsistentError(f"Invalid {label}") from exc
