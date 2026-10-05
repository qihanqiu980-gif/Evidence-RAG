from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from uuid import uuid4

from ..config import Settings
from ..errors import (
    ConfigurationError,
    DuplicateDocumentError,
    IngestionError,
    StorageInconsistentError,
)
from ..models import Chunk, Document, DocumentStatus
from ..models import KnowledgeBase as KnowledgeBaseEntity
from ..providers.base import ModelProvider
from ..storage.chroma import ChromaVectorStore
from ..storage.sqlite import SQLiteStore
from ..storage.uploads import UploadStore
from .adapters import DocumentAdapter

logger = logging.getLogger(__name__)
ProgressReporter = Callable[[str], None]


class KnowledgeBaseManager:
    def __init__(
        self,
        settings: Settings,
        sqlite: SQLiteStore,
        uploads: UploadStore,
        vectors: ChromaVectorStore,
        provider: ModelProvider | None,
    ) -> None:
        self.settings = settings
        self.sqlite = sqlite
        self.uploads = uploads
        self.vectors = vectors
        self.provider = provider
        self.ingester = DocumentAdapter(
            max_bytes=settings.upload_max_bytes,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def create(self, name: str) -> KnowledgeBaseEntity:
        return self.sqlite.create_knowledge_base(str(uuid4()), name)

    def list_knowledge_bases(self) -> list[KnowledgeBaseEntity]:
        return self.sqlite.list_knowledge_bases()

    def list_documents(self, kb_id: str) -> list[Document]:
        self.sqlite.get_knowledge_base(kb_id)
        return self.sqlite.list_documents(kb_id)

    def validate_question_scope(self, kb_ids: Sequence[str]) -> None:
        if not self.settings.configured or self.provider is None:
            raise ConfigurationError("Model provider configuration is missing")
        for kb_id in kb_ids:
            self.sqlite.get_knowledge_base(kb_id)

    def ingest(
        self,
        kb_id: str,
        filename: str,
        content: bytes,
        report: ProgressReporter | None = None,
    ) -> Document:
        notify = report or (lambda _: None)
        self.sqlite.get_knowledge_base(kb_id)
        if not self.settings.configured or self.provider is None:
            raise ConfigurationError("Model provider configuration is missing")

        document_id = str(uuid4())
        upload_saved = False
        vectors_attempted = False
        metadata_saved = False

        try:
            notify("校验")
            prepared = self.ingester.prepare(filename, content)
            if self.sqlite.document_hash_exists(kb_id, prepared.content_hash):
                raise DuplicateDocumentError("Document content already exists")

            notify("切块")
            chunks = [
                Chunk(
                    id=f"{document_id}:{item.chunk_index:04d}",
                    document_id=document_id,
                    kb_id=kb_id,
                    chunk_index=item.chunk_index,
                    heading_path=item.heading_path,
                    content=item.content,
                    char_start=item.char_start,
                    char_end=item.char_end,
                    content_hash=item.content_hash,
                )
                for item in prepared.chunks
            ]

            notify("向量化")
            try:
                embeddings = self.provider.embed_documents(
                    [chunk.content for chunk in chunks]
                )
            except Exception as exc:
                raise IngestionError(
                    "embedding_failed", "Embedding generation failed"
                ) from exc
            if len(embeddings) != len(chunks):
                raise IngestionError(
                    "embedding_failed", "Embedding count does not match chunks"
                )

            notify("保存")
            storage_path = self.uploads.save(
                kb_id,
                document_id,
                prepared.normalized_content,
                prepared.filename,
            )
            upload_saved = True
            vectors_attempted = True
            self.vectors.add_chunks(chunks, embeddings, prepared.filename)

            document = Document(
                id=document_id,
                kb_id=kb_id,
                filename=prepared.filename,
                content_hash=prepared.content_hash,
                storage_path=storage_path,
                chunk_count=len(chunks),
                status=DocumentStatus.READY,
                created_at=datetime.now(UTC),
            )
            self.sqlite.insert_document(document, chunks)
            metadata_saved = True
            notify("完成")
            return document
        except Exception as error:
            cleanup_errors = self._compensate(
                kb_id=kb_id,
                document_id=document_id,
                upload_saved=upload_saved,
                storage_path=storage_path if upload_saved else None,
                vectors_attempted=vectors_attempted,
                metadata_saved=metadata_saved,
            )
            if cleanup_errors:
                logger.warning(
                    "ingestion cleanup failed",
                    extra={"document_id": document_id, "cleanup_targets": len(cleanup_errors)},
                )
                raise StorageInconsistentError(
                    "Document processing failed and cleanup was incomplete"
                ) from error
            raise

    def delete_document(self, kb_id: str, document_id: str) -> None:
        document = self.sqlite.get_document(kb_id, document_id)
        self.vectors.delete_document(document_id)
        self.uploads.delete_document(document.storage_path)
        self.sqlite.delete_document(kb_id, document_id)

    def delete(self, kb_id: str) -> None:
        self.sqlite.get_knowledge_base(kb_id)
        self.vectors.delete_knowledge_base(kb_id)
        self.uploads.delete_knowledge_base(kb_id)
        self.sqlite.delete_knowledge_base(kb_id)

    def rebuild(self) -> list[str]:
        if not self.settings.configured or self.provider is None:
            raise ConfigurationError("Model provider configuration is missing")
        chunks = self.sqlite.list_chunks()
        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), 10):
            embeddings.extend(
                self.provider.embed_documents(
                    [chunk.content for chunk in chunks[start : start + 10]]
                )
            )
        document_names = {
            document.id: document.filename
            for knowledge_base in self.list_knowledge_bases()
            for document in self.list_documents(knowledge_base.id)
        }
        self.vectors.rebuild(chunks, embeddings, document_names)
        return self.vectors.cleanup_retired_collections()

    def _compensate(
        self,
        *,
        kb_id: str,
        document_id: str,
        upload_saved: bool,
        storage_path: str | None,
        vectors_attempted: bool,
        metadata_saved: bool,
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        actions: list[tuple[str, Callable[[], None]]] = []
        if metadata_saved:
            actions.append(
                ("sqlite", lambda: self.sqlite.delete_document(kb_id, document_id))
            )
        if vectors_attempted:
            actions.append(("vectors", lambda: self.vectors.delete_document(document_id)))
        if upload_saved and storage_path is not None:
            actions.append(
                ("uploads", lambda: self.uploads.delete_document(storage_path))
            )
        for target, action in actions:
            try:
                action()
            except Exception as error:  # noqa: BLE001
                errors.append(error)
                logger.warning("ingestion cleanup failed for %s", target)
        return errors
