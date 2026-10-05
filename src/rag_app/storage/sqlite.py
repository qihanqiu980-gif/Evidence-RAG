from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from ..errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    DuplicateKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    StorageInconsistentError,
)
from ..models import Chunk, Document, DocumentStatus, KnowledgeBase, VectorCollection

SCHEMA_VERSION = 1


class SQLiteStore:
    """SQLite business-fact storage with explicit, versioned migrations."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as connection:
            yield connection

    def _migrate(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            versions = {
                row[0]
                for row in connection.execute("PRAGMA user_version").fetchall()
            }
            current_version = next(iter(versions), 0)
            if current_version > SCHEMA_VERSION:
                raise StorageInconsistentError("SQLite schema is newer than this app")
            if current_version == SCHEMA_VERSION:
                self._validate_schema(connection)
                return
            if current_version != 0:
                raise StorageInconsistentError("Unsupported SQLite schema version")

            connection.executescript(
                """
                CREATE TABLE schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    kb_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL CHECK (chunk_count > 0),
                    status TEXT NOT NULL DEFAULT 'ready' CHECK (status IN ('ready')),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                    UNIQUE (kb_id, content_hash)
                );

                CREATE INDEX idx_documents_kb_id ON documents(kb_id);
                CREATE INDEX idx_documents_content_hash ON documents(kb_id, content_hash);

                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    kb_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL CHECK (chunk_index > 0),
                    heading_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                    UNIQUE (document_id, chunk_index)
                );

                CREATE INDEX idx_chunks_document_id ON chunks(document_id);
                CREATE INDEX idx_chunks_kb_id ON chunks(kb_id);

                CREATE TABLE vector_collections (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN ('building', 'active', 'retired')),
                    embedding_model TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
                    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    retired_at TEXT
                );

                CREATE UNIQUE INDEX one_active_vector_collection
                ON vector_collections(status)
                WHERE status = 'active';

                PRAGMA user_version = 1;
                """,
            )
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (1, ?)",
                (_timestamp(),),
            )
            self._validate_schema(connection)

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        expected = {
            "schema_migrations",
            "knowledge_bases",
            "documents",
            "chunks",
            "vector_collections",
        }
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        actual = {row["name"] for row in rows}
        if not expected.issubset(actual):
            raise StorageInconsistentError("SQLite schema is incomplete")

    def create_knowledge_base(self, kb_id: str, name: str) -> KnowledgeBase:
        self._validate_uuid(kb_id, "knowledge base id")
        knowledge_base = KnowledgeBase(
            id=kb_id,
            name=name.strip(),
            created_at=_now(),
        )
        try:
            with self.transaction() as connection:
                connection.execute(
                    "INSERT INTO knowledge_bases(id, name, created_at) VALUES (?, ?, ?)",
                    (knowledge_base.id, knowledge_base.name, _iso(knowledge_base.created_at)),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateKnowledgeBaseError("Knowledge base name already exists") from exc
        return knowledge_base

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, name, created_at FROM knowledge_bases ORDER BY created_at, id"
            ).fetchall()
        return [_knowledge_base_from_row(row) for row in rows]

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, name, created_at FROM knowledge_bases WHERE id = ?",
                (kb_id,),
            ).fetchone()
        if row is None:
            raise KnowledgeBaseNotFoundError("Knowledge base not found")
        return _knowledge_base_from_row(row)

    def knowledge_base_exists(self, kb_id: str) -> bool:
        try:
            self.get_knowledge_base(kb_id)
        except KnowledgeBaseNotFoundError:
            return False
        return True

    def document_hash_exists(self, kb_id: str, content_hash: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM documents WHERE kb_id = ? AND content_hash = ?",
                (kb_id, content_hash),
            ).fetchone()
        return row is not None

    def insert_document(self, document: Document, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise StorageInconsistentError("A ready document must contain chunks")
        if document.chunk_count != len(chunks):
            raise StorageInconsistentError("Document chunk count does not match chunks")
        if any(
            chunk.document_id != document.id or chunk.kb_id != document.kb_id
            for chunk in chunks
        ):
            raise StorageInconsistentError("Chunk ownership does not match document")

        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO documents(
                        id, kb_id, filename, content_hash, storage_path,
                        chunk_count, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.kb_id,
                        document.filename,
                        document.content_hash,
                        document.storage_path,
                        document.chunk_count,
                        document.status.value,
                        _iso(document.created_at),
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO chunks(
                        id, document_id, kb_id, chunk_index, heading_path,
                        content, char_start, char_end, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.id,
                            chunk.document_id,
                            chunk.kb_id,
                            chunk.chunk_index,
                            json.dumps(chunk.heading_path, ensure_ascii=False),
                            chunk.content,
                            chunk.char_start,
                            chunk.char_end,
                            chunk.content_hash,
                        )
                        for chunk in chunks
                    ],
                )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc):
                raise DuplicateDocumentError(
                    "Document content already exists in this knowledge base"
                ) from exc
            raise StorageInconsistentError("Document metadata is inconsistent") from exc

    def list_documents(self, kb_id: str) -> list[Document]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, kb_id, filename, content_hash, storage_path,
                       chunk_count, status, created_at
                FROM documents
                WHERE kb_id = ?
                ORDER BY created_at, id
                """,
                (kb_id,),
            ).fetchall()
        return [_document_from_row(row) for row in rows]

    def get_document(self, kb_id: str, document_id: str) -> Document:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, kb_id, filename, content_hash, storage_path,
                       chunk_count, status, created_at
                FROM documents
                WHERE kb_id = ? AND id = ?
                """,
                (kb_id, document_id),
            ).fetchone()
        if row is None:
            raise DocumentNotFoundError("Document not found")
        return _document_from_row(row)

    def list_document_ids(self, kb_id: str) -> list[str]:
        with self.connect() as connection:
            return [
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM documents WHERE kb_id = ? ORDER BY created_at, id",
                    (kb_id,),
                ).fetchall()
            ]

    def delete_document(self, kb_id: str, document_id: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE kb_id = ? AND id = ?",
                (kb_id, document_id),
            )
            if cursor.rowcount != 1:
                raise DocumentNotFoundError("Document not found")

    def delete_knowledge_base(self, kb_id: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM knowledge_bases WHERE id = ?",
                (kb_id,),
            )
            if cursor.rowcount != 1:
                raise KnowledgeBaseNotFoundError("Knowledge base not found")

    def list_chunks(self, kb_id: str | None = None) -> list[Chunk]:
        query = """
            SELECT id, document_id, kb_id, chunk_index, heading_path,
                   content, char_start, char_end, content_hash
            FROM chunks
        """
        parameters: tuple[str, ...] = ()
        if kb_id is not None:
            query += " WHERE kb_id = ? ORDER BY kb_id, document_id, chunk_index"
            parameters = (kb_id,)
        else:
            query += " ORDER BY kb_id, document_id, chunk_index"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def counts(self) -> tuple[int, int, int]:
        with self.connect() as connection:
            kb_count = int(
                connection.execute("SELECT COUNT(*) FROM knowledge_bases").fetchone()[0]
            )
            document_count = int(
                connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            )
            chunk_count = int(
                connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            )
        return kb_count, document_count, chunk_count

    def get_active_vector_collection(self) -> VectorCollection | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, collection_name, status, embedding_model,
                       embedding_dimension, chunk_count, created_at,
                       activated_at, retired_at
                FROM vector_collections
                WHERE status = 'active'
                """
            ).fetchone()
        return None if row is None else _vector_collection_from_row(row)

    def create_active_vector_collection(
        self,
        collection_id: str,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> VectorCollection:
        now = _now()
        collection = VectorCollection(
            id=collection_id,
            collection_name=collection_name,
            status="active",
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            chunk_count=0,
            created_at=now,
            activated_at=now,
            retired_at=None,
        )
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO vector_collections(
                        id, collection_name, status, embedding_model,
                        embedding_dimension, chunk_count, created_at,
                        activated_at, retired_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _vector_collection_values(collection),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageInconsistentError("Active vector collection is not unique") from exc
        return collection

    def update_vector_chunk_count(self, collection_id: str, chunk_count: int) -> None:
        if chunk_count < 0:
            raise StorageInconsistentError("Vector chunk count cannot be negative")
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE vector_collections
                SET chunk_count = ?
                WHERE id = ? AND status = 'active'
                """,
                (chunk_count, collection_id),
            )
            if cursor.rowcount != 1:
                raise StorageInconsistentError("Active vector collection not found")

    def create_building_vector_collection(
        self,
        collection_id: str,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> VectorCollection:
        collection = VectorCollection(
            id=collection_id,
            collection_name=collection_name,
            status="building",
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            chunk_count=0,
            created_at=_now(),
            activated_at=None,
            retired_at=None,
        )
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO vector_collections(
                        id, collection_name, status, embedding_model,
                        embedding_dimension, chunk_count, created_at,
                        activated_at, retired_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    _vector_collection_values(collection),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageInconsistentError(
                "Unable to create a building vector collection"
            ) from exc
        return collection

    def activate_vector_collection(self, collection_id: str) -> None:
        now = _iso(_now())
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE vector_collections
                SET status = 'retired', retired_at = ?
                WHERE status = 'active'
                """,
                (now,),
            )
            cursor = connection.execute(
                """
                UPDATE vector_collections
                SET status = 'active', activated_at = ?, retired_at = NULL,
                    chunk_count = (
                        SELECT COUNT(*) FROM chunks
                    )
                WHERE id = ? AND status = 'building'
                """,
                (now, collection_id),
            )
            if cursor.rowcount != 1:
                raise StorageInconsistentError(
                    "Building vector collection not found"
                )

    def delete_vector_collection_record(self, collection_id: str) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                DELETE FROM vector_collections
                WHERE id = ? AND status != 'active'
                """,
                (collection_id,),
            )
            if cursor.rowcount != 1:
                raise StorageInconsistentError("Non-active vector collection not found")

    def retired_vector_collections(self) -> list[VectorCollection]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, collection_name, status, embedding_model,
                       embedding_dimension, chunk_count, created_at,
                       activated_at, retired_at
                FROM vector_collections
                WHERE status = 'retired'
                ORDER BY retired_at, id
                """
            ).fetchall()
        return [_vector_collection_from_row(row) for row in rows]

    @staticmethod
    def _validate_uuid(value: str, label: str) -> None:
        try:
            UUID(value)
        except ValueError as exc:
            raise StorageInconsistentError(f"Invalid {label}") from exc


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp() -> str:
    return _iso(_now())


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _knowledge_base_from_row(row: sqlite3.Row) -> KnowledgeBase:
    return KnowledgeBase(
        id=row["id"],
        name=row["name"],
        created_at=_parse_datetime(row["created_at"]),
    )


def _document_from_row(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        kb_id=row["kb_id"],
        filename=row["filename"],
        content_hash=row["content_hash"],
        storage_path=row["storage_path"],
        chunk_count=int(row["chunk_count"]),
        status=DocumentStatus(row["status"]),
        created_at=_parse_datetime(row["created_at"]),
    )


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        kb_id=row["kb_id"],
        chunk_index=int(row["chunk_index"]),
        heading_path=tuple(json.loads(row["heading_path"])),
        content=row["content"],
        char_start=int(row["char_start"]),
        char_end=int(row["char_end"]),
        content_hash=row["content_hash"],
    )


def _vector_collection_from_row(row: sqlite3.Row) -> VectorCollection:
    return VectorCollection(
        id=row["id"],
        collection_name=row["collection_name"],
        status=row["status"],
        embedding_model=row["embedding_model"],
        embedding_dimension=int(row["embedding_dimension"]),
        chunk_count=int(row["chunk_count"]),
        created_at=_parse_datetime(row["created_at"]),
        activated_at=(
            _parse_datetime(row["activated_at"])
            if row["activated_at"] is not None
            else None
        ),
        retired_at=(
            _parse_datetime(row["retired_at"])
            if row["retired_at"] is not None
            else None
        ),
    )


def _vector_collection_values(collection: VectorCollection) -> tuple[object, ...]:
    return (
        collection.id,
        collection.collection_name,
        collection.status,
        collection.embedding_model,
        collection.embedding_dimension,
        collection.chunk_count,
        _iso(collection.created_at),
        _iso(collection.activated_at) if collection.activated_at else None,
        _iso(collection.retired_at) if collection.retired_at else None,
    )
