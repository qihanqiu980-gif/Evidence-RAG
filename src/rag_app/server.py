from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .core.jobs import UploadJobManager
from .core.knowledge_base import KnowledgeBaseManager
from .core.registry import WorkflowRegistry
from .core.workflow import EvidenceQAWorkflow
from .providers.base import ModelProvider
from .providers.online import OnlineModelProvider
from .storage.chroma import ChromaVectorStore
from .storage.sqlite import SQLiteStore
from .storage.uploads import UploadStore


@dataclass
class AppContainer:
    settings: Settings
    sqlite: SQLiteStore
    uploads: UploadStore
    vectors: ChromaVectorStore
    provider: ModelProvider | None
    manager: KnowledgeBaseManager
    workflow: EvidenceQAWorkflow | None
    workflow_registry: WorkflowRegistry
    jobs: UploadJobManager
    owns_provider: bool = False
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.jobs.close()
        if self.owns_provider and self.provider is not None:
            close = getattr(self.provider, "close", None)
            if callable(close):
                close()


def build_container(
    settings: Settings,
    provider: ModelProvider | None = None,
) -> AppContainer:
    sqlite = SQLiteStore(settings.sqlite_path)
    uploads = UploadStore(settings.uploads_path)
    vectors = ChromaVectorStore(
        path=settings.chroma_path,
        sqlite=sqlite,
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
    )
    owns_provider = False
    selected_provider = provider
    if settings.configured and selected_provider is None:
        selected_provider = OnlineModelProvider(settings)
        owns_provider = True
    workflow = (
        EvidenceQAWorkflow(settings, vectors, selected_provider, sqlite)
        if selected_provider is not None
        else None
    )
    jobs = UploadJobManager(
        settings.data_dir / "spool",
        settings.upload_max_workers,
    )
    registry = WorkflowRegistry()
    if workflow is not None:
        registry.register("evidence_qa", workflow)

    return AppContainer(
        settings=settings,
        sqlite=sqlite,
        uploads=uploads,
        vectors=vectors,
        provider=selected_provider,
        manager=KnowledgeBaseManager(
            settings=settings,
            sqlite=sqlite,
            uploads=uploads,
            vectors=vectors,
            provider=selected_provider,
        ),
        workflow=workflow,
        workflow_registry=registry,
        jobs=jobs,
        owns_provider=owns_provider,
    )
