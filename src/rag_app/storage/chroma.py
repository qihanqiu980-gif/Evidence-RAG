from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from ..errors import (
    ProviderResponseError,
    StorageInconsistentError,
    VectorConfigurationMismatch,
)
from ..models import Chunk, VectorCollection
from .sqlite import SQLiteStore

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class VectorMatch:
    chunk_id: str
    kb_id: str
    document_id: str
    document_name: str
    heading_path: tuple[str, ...]
    chunk_index: int
    content: str
    content_hash: str
    similarity: float


class ChromaVectorStore:
    """Active-collection registry and a thin Chroma data boundary."""

    def __init__(
        self,
        path: Path,
        sqlite: SQLiteStore,
        embedding_model: str,
        embedding_dimension: int,
        allow_configuration_mismatch: bool = False,
    ) -> None:
        self.path = Path(path)
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.allow_configuration_mismatch = allow_configuration_mismatch
        self.sqlite = sqlite
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection: Collection = self._initialize_collection()

    @property
    def active_collection(self) -> VectorCollection:
        collection = self.sqlite.get_active_vector_collection()
        if collection is None:
            raise StorageInconsistentError("No active vector collection")
        return collection

    @property
    def vector_count(self) -> int:
        return int(self._collection.count())

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        document_name: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ProviderResponseError("Embedding count does not match chunk count")
        if not chunks:
            return
        self._validate_embeddings(embeddings)
        try:
            self._collection.add(
                ids=[chunk.id for chunk in chunks],
                embeddings=cast(
                    list[Sequence[float]],
                    [list(embedding) for embedding in embeddings],
                ),
                documents=[chunk.content for chunk in chunks],
                metadatas=[self._metadata(chunk, document_name) for chunk in chunks],
            )
            self._sync_registry_count()
        except Exception as exc:
            if isinstance(exc, ProviderResponseError):
                raise
            raise StorageInconsistentError("Unable to write vector index") from exc

    def search(
        self,
        query_embedding: Sequence[float],
        kb_ids: Sequence[str],
        top_k: int,
    ) -> list[VectorMatch]:
        if not kb_ids or top_k <= 0:
            return []
        self._validate_embeddings([query_embedding])
        where: Any = (
            {"kb_id": kb_ids[0]}
            if len(kb_ids) == 1
            else {"kb_id": {"$in": list(dict.fromkeys(kb_ids))}}
        )
        try:
            result = self._collection.query(
                query_embeddings=cast(list[Sequence[float]], [list(query_embedding)]),
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise StorageInconsistentError("Unable to query vector index") from exc

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        matches: list[VectorMatch] = []
        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            data = metadata or {}
            matches.append(
                VectorMatch(
                    chunk_id=chunk_id,
                    kb_id=str(data["kb_id"]),
                    document_id=str(data["document_id"]),
                    document_name=str(data["document_name"]),
                    heading_path=tuple(json.loads(str(data["heading_path"]))),
                    chunk_index=int(cast(str, data["chunk_index"])),
                    content=document,
                    content_hash=str(data["content_hash"]),
                    similarity=1.0 - float(distance),
                )
            )
        return matches

    def delete_document(self, document_id: str) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
            self._sync_registry_count()
        except Exception as exc:
            raise StorageInconsistentError("Unable to delete document vectors") from exc

    def delete_knowledge_base(self, kb_id: str) -> None:
        try:
            self._collection.delete(where={"kb_id": kb_id})
            self._sync_registry_count()
        except Exception as exc:
            raise StorageInconsistentError("Unable to delete knowledge base vectors") from exc

    def doctor_metadata(self) -> dict[str, int | str]:
        return dict(self._collection.metadata or {})

    def _initialize_collection(self) -> Collection:
        active = self.sqlite.get_active_vector_collection()
        if active is None:
            collection_id = str(uuid4())
            collection_name = f"rag_chunks_{collection_id.replace('-', '')}"
            metadata = self._collection_metadata()
            try:
                collection = self._client.create_collection(
                    name=collection_name,
                    metadata=metadata,
                    embedding_function=None,
                )
            except Exception as exc:
                raise StorageInconsistentError("Unable to create vector collection") from exc
            try:
                active = self.sqlite.create_active_vector_collection(
                    collection_id,
                    collection_name,
                    self.embedding_model,
                    self.embedding_dimension,
                )
            except Exception as exc:
                try:
                    self._client.delete_collection(collection_name)
                except Exception:  # noqa: BLE001, S110
                    pass
                raise StorageInconsistentError(
                    "Unable to register the vector collection"
                ) from exc
            return collection

        if not self.allow_configuration_mismatch:
            self._validate_registry(active)
        try:
            collection = self._client.get_collection(
                name=active.collection_name,
                embedding_function=None,
            )
        except NotFoundError as exc:
            raise StorageInconsistentError(
                "Active vector collection is missing; run rag-app doctor"
            ) from exc
        except Exception as exc:
            raise StorageInconsistentError("Unable to open vector collection") from exc

        metadata = dict(collection.metadata or {})
        expected = self._collection_metadata()
        if not self.allow_configuration_mismatch and any(
            metadata.get(key) != value for key, value in expected.items()
        ):
            raise VectorConfigurationMismatch(
                "Embedding model or dimension has changed; run doctor and rebuild"
            )
        if collection.count() != active.chunk_count:
            raise StorageInconsistentError(
                "Vector registry count does not match the physical index; run doctor"
            )
        return collection

    def rebuild(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        document_names: Mapping[str, str],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ProviderResponseError("Embedding count does not match chunk count")
        self._validate_embeddings(embeddings)

        old = self.active_collection
        collection_id = str(uuid4())
        collection_name = f"rag_chunks_{collection_id.replace('-', '')}"
        building = self.sqlite.create_building_vector_collection(
            collection_id,
            collection_name,
            self.embedding_model,
            self.embedding_dimension,
        )
        try:
            collection = self._client.create_collection(
                name=collection_name,
                metadata=self._collection_metadata(),
                embedding_function=None,
            )
            if chunks:
                collection.add(
                    ids=[chunk.id for chunk in chunks],
                    embeddings=cast(
                        list[Sequence[float]],
                        [list(embedding) for embedding in embeddings],
                    ),
                    documents=[chunk.content for chunk in chunks],
                    metadatas=[
                        self._metadata(
                            chunk,
                            document_names.get(chunk.document_id, chunk.document_id),
                        )
                        for chunk in chunks
                    ],
                )
            if collection.count() != len(chunks):
                raise StorageInconsistentError("Rebuilt vector count is invalid")
            self.sqlite.activate_vector_collection(collection_id)
        except Exception:
            try:
                self._client.delete_collection(collection_name)
            except Exception:  # noqa: BLE001, S110
                pass
            try:
                self.sqlite.delete_vector_collection_record(building.id)
            except Exception:  # noqa: BLE001, S110
                pass
            raise

        self._collection = collection
        self._cleanup_retired_collection(old)

    def cleanup_retired_collections(self) -> list[str]:
        residues: list[str] = []
        for retired in self.sqlite.retired_vector_collections():
            if self._cleanup_retired_collection(retired):
                residues.append(retired.id)
        return residues

    def _cleanup_retired_collection(self, collection: VectorCollection) -> bool:
        """Return True when a retired registry record remains."""
        try:
            self._client.delete_collection(collection.collection_name)
        except NotFoundError:
            pass
        except Exception:  # noqa: BLE001
            return True
        try:
            self.sqlite.delete_vector_collection_record(collection.id)
        except Exception:  # noqa: BLE001
            return True
        return False

    def _validate_registry(self, collection: VectorCollection) -> None:
        if (
            collection.embedding_model != self.embedding_model
            or collection.embedding_dimension != self.embedding_dimension
        ):
            raise VectorConfigurationMismatch(
                "Embedding model or dimension has changed; rebuild is required"
            )

    def _collection_metadata(self) -> dict[str, int | str]:
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "hnsw:space": "cosine",
            "schema_version": SCHEMA_VERSION,
        }

    def _sync_registry_count(self) -> None:
        self.sqlite.update_vector_chunk_count(
            self.active_collection.id,
            self.vector_count,
        )

    def _validate_embeddings(self, embeddings: Sequence[Sequence[float]]) -> None:
        if any(len(embedding) != self.embedding_dimension for embedding in embeddings):
            raise ProviderResponseError("Embedding dimension does not match configuration")

    @staticmethod
    def _metadata(chunk: Chunk, document_name: str) -> dict[str, int | str]:
        return {
            "kb_id": chunk.kb_id,
            "document_id": chunk.document_id,
            "document_name": document_name,
            "heading_path": json.dumps(chunk.heading_path, ensure_ascii=False),
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
            "schema_version": SCHEMA_VERSION,
        }
