from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from ..errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    DuplicateKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    StorageInconsistentError,
)
from ..models import (
    Chunk,
    DiscoveryQuestionRecord,
    DiscoverySnapshot,
    DiscoveryTopicDraft,
    DiscoveryTopicRecord,
    Document,
    DocumentStatus,
    KnowledgeBase,
    VectorCollection,
)

SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class KeywordMatch:
    chunk_id: str
    kb_id: str
    document_id: str
    document_name: str
    heading_path: tuple[str, ...]
    chunk_index: int
    content: str
    content_hash: str
    keyword_score: float


class SQLiteStore:
    """SQLite business-fact storage with explicit, versioned migrations."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()
        self._ensure_keyword_index()

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
            if current_version not in {0, 1}:
                raise StorageInconsistentError("Unsupported SQLite schema version")

            if current_version == 0:
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
            if current_version < SCHEMA_VERSION:
                self._migrate_discovery_schema(connection)
            self._validate_schema(connection)

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        expected = {
            "schema_migrations",
            "knowledge_bases",
            "documents",
            "chunks",
            "vector_collections",
            "discovery_runs",
            "discovery_topics",
            "discovery_questions",
            "discovery_topic_coverage",
        }
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        actual = {row["name"] for row in rows}
        if not expected.issubset(actual):
            raise StorageInconsistentError("SQLite schema is incomplete")

    @staticmethod
    def _migrate_discovery_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE discovery_runs (
                id TEXT PRIMARY KEY,
                kb_id TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                document_count INTEGER NOT NULL CHECK (document_count >= 0),
                chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
                topic_count INTEGER NOT NULL CHECK (topic_count >= 0),
                question_count INTEGER NOT NULL CHECK (question_count >= 0),
                created_at TEXT NOT NULL,
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
            );

            CREATE INDEX idx_discovery_runs_kb_id ON discovery_runs(kb_id, created_at);

            CREATE TABLE discovery_topics (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                kb_id TEXT NOT NULL,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                summary TEXT NOT NULL,
                document_ids TEXT NOT NULL,
                sort_order INTEGER NOT NULL CHECK (sort_order >= 0),
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES discovery_runs(id) ON DELETE CASCADE,
                FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
            );

            CREATE INDEX idx_discovery_topics_run_id ON discovery_topics(run_id, sort_order);
            CREATE INDEX idx_discovery_topics_kb_id ON discovery_topics(kb_id);

            CREATE TABLE discovery_questions (
                id TEXT PRIMARY KEY,
                topic_id TEXT NOT NULL,
                question TEXT NOT NULL,
                source_chunk_ids TEXT NOT NULL,
                sort_order INTEGER NOT NULL CHECK (sort_order >= 0),
                FOREIGN KEY (topic_id) REFERENCES discovery_topics(id) ON DELETE CASCADE
            );

            CREATE INDEX idx_discovery_questions_topic_id
            ON discovery_questions(topic_id, sort_order);

            CREATE TABLE discovery_topic_coverage (
                topic_id TEXT PRIMARY KEY,
                chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
                confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                updated_at TEXT NOT NULL,
                FOREIGN KEY (topic_id) REFERENCES discovery_topics(id) ON DELETE CASCADE
            );

            PRAGMA user_version = 2;
            """,
        )
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (2, ?)",
            (_timestamp(),),
        )

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
                connection.executemany(
                    "INSERT INTO chunks_fts(chunk_id, content) VALUES (?, ?)",
                    [(chunk.id, chunk.content) for chunk in chunks],
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
            connection.execute(
                "DELETE FROM discovery_runs WHERE kb_id = ?",
                (kb_id,),
            )
            connection.execute(
                """
                DELETE FROM chunks_fts
                WHERE chunk_id IN (
                    SELECT id FROM chunks WHERE kb_id = ? AND document_id = ?
                )
                """,
                (kb_id, document_id),
            )
            cursor = connection.execute(
                "DELETE FROM documents WHERE kb_id = ? AND id = ?",
                (kb_id, document_id),
            )
            if cursor.rowcount != 1:
                raise DocumentNotFoundError("Document not found")

    def delete_knowledge_base(self, kb_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                DELETE FROM chunks_fts
                WHERE chunk_id IN (SELECT id FROM chunks WHERE kb_id = ?)
                """,
                (kb_id,),
            )
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

    def discovery_fingerprint(self, kb_id: str) -> str:
        self.get_knowledge_base(kb_id)
        with self.connect() as connection:
            return _discovery_fingerprint(connection, kb_id)

    def save_discovery(
        self,
        kb_id: str,
        source_fingerprint: str,
        topics: Sequence[DiscoveryTopicDraft],
    ) -> DiscoverySnapshot:
        """Atomically replace cached discovery results after verifying the source."""
        if not topics:
            raise StorageInconsistentError("Discovery results require at least one topic")
        run_id = str(uuid4())
        now = _now()
        document_count = len(
            {
                document_id
                for topic in topics
                for document_id in topic.document_ids
            }
        )
        chunk_count = len(
            {
                chunk_id
                for topic in topics
                for chunk_id in topic.source_chunk_ids
            }
        )
        question_count = sum(len(topic.questions) for topic in topics)

        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT 1 FROM knowledge_bases WHERE id = ?",
                    (kb_id,),
                ).fetchone()
                if row is None:
                    raise KnowledgeBaseNotFoundError("Knowledge base not found")
                if _discovery_fingerprint(connection, kb_id) != source_fingerprint:
                    raise StorageInconsistentError(
                        "Knowledge base changed while discovery was running"
                    )

                connection.execute(
                    "DELETE FROM discovery_runs WHERE kb_id = ?",
                    (kb_id,),
                )
                connection.execute(
                    """
                    INSERT INTO discovery_runs(
                        id, kb_id, source_fingerprint, document_count, chunk_count,
                        topic_count, question_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        kb_id,
                        source_fingerprint,
                        document_count,
                        chunk_count,
                        len(topics),
                        question_count,
                        _iso(now),
                    ),
                )

                records: list[DiscoveryTopicRecord] = []
                for topic_order, topic in enumerate(topics):
                    if not topic.questions or not topic.source_chunk_ids:
                        raise StorageInconsistentError(
                            "Discovery topics require at least one question"
                        )
                    topic_id = str(uuid4())
                    connection.execute(
                        """
                        INSERT INTO discovery_topics(
                            id, run_id, kb_id, title, type, summary, document_ids,
                            sort_order, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            topic_id,
                            run_id,
                            kb_id,
                            topic.title,
                            topic.type,
                            topic.summary,
                            json.dumps(topic.document_ids, ensure_ascii=False),
                            topic_order,
                            _iso(now),
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO discovery_topic_coverage(
                            topic_id, chunk_count, confidence, updated_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            topic_id,
                            len(set(topic.source_chunk_ids)),
                            topic.confidence,
                            _iso(now),
                        ),
                    )
                    question_records = [
                        DiscoveryQuestionRecord(
                            id=str(uuid4()),
                            topic_id=topic_id,
                            question=question.question,
                            source_chunk_ids=question.source_chunk_ids,
                            sort_order=index,
                        )
                        for index, question in enumerate(topic.questions)
                    ]
                    connection.executemany(
                        """
                        INSERT INTO discovery_questions(
                            id, topic_id, question, source_chunk_ids, sort_order
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                item.id,
                                item.topic_id,
                                item.question,
                                json.dumps(item.source_chunk_ids, ensure_ascii=False),
                                item.sort_order,
                            )
                            for item in question_records
                        ],
                    )
                    records.append(
                        DiscoveryTopicRecord(
                            id=topic_id,
                            kb_id=kb_id,
                            title=topic.title,
                            type=topic.type,
                            summary=topic.summary,
                            document_ids=topic.document_ids,
                            chunk_count=len(set(topic.source_chunk_ids)),
                            confidence=topic.confidence,
                            updated_at=now,
                            questions=tuple(question_records),
                        )
                    )
        except sqlite3.IntegrityError as exc:
            raise StorageInconsistentError("Discovery results are inconsistent") from exc

        return DiscoverySnapshot(
            kb_id=kb_id,
            source_fingerprint=source_fingerprint,
            document_count=document_count,
            chunk_count=chunk_count,
            topic_count=len(records),
            question_count=question_count,
            created_at=now,
            topics=tuple(records),
        )

    def list_discovery(self, kb_id: str) -> DiscoverySnapshot | None:
        self.get_knowledge_base(kb_id)
        with self.connect() as connection:
            run = connection.execute(
                """
                SELECT id, kb_id, source_fingerprint, document_count, chunk_count,
                       topic_count, question_count, created_at
                FROM discovery_runs
                WHERE kb_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (kb_id,),
            ).fetchone()
            if run is None:
                return None

            topic_rows = connection.execute(
                """
                SELECT t.id, t.kb_id, t.title, t.type, t.summary, t.document_ids,
                       t.created_at, c.chunk_count, c.confidence, c.updated_at
                FROM discovery_topics t
                JOIN discovery_topic_coverage c ON c.topic_id = t.id
                WHERE t.run_id = ?
                ORDER BY t.sort_order, t.id
                """,
                (run["id"],),
            ).fetchall()
            question_rows = connection.execute(
                """
                SELECT q.id, q.topic_id, q.question, q.source_chunk_ids, q.sort_order
                FROM discovery_questions q
                JOIN discovery_topics t ON t.id = q.topic_id
                WHERE t.run_id = ?
                ORDER BY q.sort_order, q.id
                """,
                (run["id"],),
            ).fetchall()

        questions_by_topic: dict[str, list[DiscoveryQuestionRecord]] = {
            row["id"]: [] for row in topic_rows
        }
        for row in question_rows:
            questions_by_topic.setdefault(row["topic_id"], []).append(
                DiscoveryQuestionRecord(
                    id=row["id"],
                    topic_id=row["topic_id"],
                    question=row["question"],
                    source_chunk_ids=tuple(json.loads(row["source_chunk_ids"])),
                    sort_order=int(row["sort_order"]),
                )
            )
        topics = tuple(
            DiscoveryTopicRecord(
                id=row["id"],
                kb_id=row["kb_id"],
                title=row["title"],
                type=row["type"],
                summary=row["summary"],
                document_ids=tuple(json.loads(row["document_ids"])),
                chunk_count=int(row["chunk_count"]),
                confidence=float(row["confidence"]),
                updated_at=_parse_datetime(row["updated_at"]),
                questions=tuple(questions_by_topic.get(row["id"], [])),
            )
            for row in topic_rows
        )
        return DiscoverySnapshot(
            kb_id=kb_id,
            source_fingerprint=run["source_fingerprint"],
            document_count=int(run["document_count"]),
            chunk_count=int(run["chunk_count"]),
            topic_count=int(run["topic_count"]),
            question_count=int(run["question_count"]),
            created_at=_parse_datetime(run["created_at"]),
            topics=topics,
        )

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

    def keyword_index_count(self) -> int:
        with self.connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            )

    def rebuild_keyword_index(self) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM chunks_fts")
            connection.execute(
                "INSERT INTO chunks_fts(chunk_id, content) "
                "SELECT id, content FROM chunks"
            )

    def search_keyword(
        self,
        query: str,
        kb_ids: Sequence[str],
        limit: int,
    ) -> list[KeywordMatch]:
        """Rank chunks by trigram BM25 plus substring fallback for short terms."""
        if not kb_ids or limit <= 0:
            return []
        fts_terms, substring_terms = _keyword_terms(query)
        if not fts_terms and not substring_terms:
            return []

        selected = list(dict.fromkeys(kb_ids))
        placeholders = ", ".join("?" for _ in selected)
        scores: dict[str, float] = {}
        rows: dict[str, sqlite3.Row] = {}

        if fts_terms:
            match = " OR ".join(f'"{term}"' for term in fts_terms)
            sql = f"""
                SELECT c.id, c.document_id, c.kb_id, d.filename,
                       c.heading_path, c.chunk_index, c.content, c.content_hash,
                       bm25(chunks_fts) AS keyword_score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE c.kb_id IN ({placeholders}) AND chunks_fts MATCH ?
                ORDER BY keyword_score
                LIMIT ?
            """
            parameters = [*selected, match, limit]
            with self.connect() as connection:
                for row in connection.execute(sql, parameters).fetchall():
                    scores[row["id"]] = scores.get(row["id"], 0.0) + float(
                        row["keyword_score"]
                    )
                    rows[row["id"]] = row

        for term in substring_terms:
            sql = f"""
                SELECT c.id, c.document_id, c.kb_id, d.filename,
                       c.heading_path, c.chunk_index, c.content, c.content_hash,
                       -(length(c.content) - length(replace(c.content, ?, '')))
                          / max(length(?), 1) AS keyword_score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.kb_id IN ({placeholders}) AND instr(c.content, ?) > 0
            """
            parameters = [term, term, *selected, term]
            with self.connect() as connection:
                for row in connection.execute(sql, parameters).fetchall():
                    scores[row["id"]] = scores.get(row["id"], 0.0) + float(
                        row["keyword_score"]
                    )
                    rows[row["id"]] = row

        ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]))[:limit]
        return [
            KeywordMatch(
                chunk_id=chunk_id,
                kb_id=rows[chunk_id]["kb_id"],
                document_id=rows[chunk_id]["document_id"],
                document_name=rows[chunk_id]["filename"],
                heading_path=tuple(json.loads(rows[chunk_id]["heading_path"])),
                chunk_index=int(rows[chunk_id]["chunk_index"]),
                content=rows[chunk_id]["content"],
                content_hash=rows[chunk_id]["content_hash"],
                keyword_score=score,
            )
            for chunk_id, score in ranked
        ]

    def _ensure_keyword_index(self) -> None:
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        content,
                        tokenize='trigram'
                    )
                    """
                )
            if self.keyword_index_count() != self.counts()[2]:
                self.rebuild_keyword_index()
        except sqlite3.OperationalError as exc:
            raise StorageInconsistentError(
                "SQLite FTS5 is unavailable for the keyword index"
            ) from exc

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


def _discovery_fingerprint(connection: sqlite3.Connection, kb_id: str) -> str:
    rows = connection.execute(
        """
        SELECT id, content_hash
        FROM documents
        WHERE kb_id = ?
        ORDER BY created_at, id
        """,
        (kb_id,),
    ).fetchall()
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["id"].encode("utf-8"))
        digest.update(row["content_hash"].encode("utf-8"))
    return digest.hexdigest()


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


def _keyword_terms(query: str) -> tuple[list[str], list[str]]:
    """Split a query into trigram MATCH terms and short substring terms."""
    normalized = query.casefold()
    fts_terms: list[str] = []
    substring_terms: list[str] = []

    for token in re.findall(r"[a-z0-9]+", normalized):
        if len(token) >= 3:
            fts_terms.append(token)
        elif token:
            substring_terms.append(token)

    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) < 3:
            substring_terms.append(sequence)
            continue
        fts_terms.extend(
            sequence[start : start + 3] for start in range(len(sequence) - 2)
        )

    return (
        list(dict.fromkeys(fts_terms)),
        list(dict.fromkeys(substring_terms)),
    )
