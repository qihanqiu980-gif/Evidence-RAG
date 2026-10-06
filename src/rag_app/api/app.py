from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from ..core.adapters import validate_upload_submission
from ..core.jobs import UploadJobSnapshot
from ..core.workflow import WorkflowEvent
from ..errors import RagAppError
from ..logging import safe_event_fields
from ..models import DiscoveryTopicRecord, Document, KnowledgeBase
from ..providers.base import ModelProvider
from ..server import AppContainer, build_container
from .contracts import (
    ChatRequest,
    DiscoveryQuestionSummary,
    DiscoverySummary,
    DiscoveryTopicSummary,
    DocumentBatchResult,
    DocumentSummary,
    DocumentUploadError,
    KnowledgeBaseCreate,
    KnowledgeBaseSummary,
    SseEvent,
    SystemStatus,
    SystemStatusValue,
    UploadedDocument,
    UploadJobCreated,
    UploadJobItemSummary,
    UploadJobSummary,
    UploadLimits,
)
from .sse import encode_sse_event

DEMO_DOCUMENTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "demo-docs"
logger = logging.getLogger(__name__)


def build_app(
    settings: Settings,
    provider: ModelProvider | None = None,
    frontend_dir: Path | None = None,
) -> FastAPI:
    container = build_container(settings, provider)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        kb_count, document_count, chunk_count = container.sqlite.counts()
        logger.info(
            "service started",
            extra=safe_event_fields(
                configured=container.settings.configured,
                knowledge_base_count=kb_count,
                document_count=document_count,
                chunk_count=chunk_count,
            ),
        )
        try:
            yield
        finally:
            logger.info("service stopped")
            container.close()

    app = FastAPI(
        title="Local RAG Knowledge Workbench",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.container = container

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                }
            },
        )

    @app.exception_handler(RagAppError)
    async def domain_error(request: Request, error: RagAppError) -> JSONResponse:
        return JSONResponse(
            status_code=_error_status(error.code),
            content={"detail": {"code": error.code, "message": str(error)}},
        )

    @app.get("/api/status", response_model=SystemStatus)
    def get_status() -> SystemStatus:
        kb_count, document_count, chunk_count = container.sqlite.counts()
        system_status = (
            SystemStatusValue.READY
            if container.settings.configured
            else SystemStatusValue.CONFIGURATION_MISSING
        )
        return SystemStatus(
            status=system_status,
            knowledge_base_count=kb_count,
            document_count=document_count,
            chunk_count=chunk_count,
            limits=UploadLimits(
                max_files_per_upload=settings.upload_max_files,
                max_file_bytes=settings.upload_max_bytes,
                accepted_extensions=[".md", ".txt", ".pdf", ".docx", ".html", ".htm"],
            ),
        )

    @app.get("/api/knowledge-bases", response_model=list[KnowledgeBaseSummary])
    def list_knowledge_bases() -> list[KnowledgeBaseSummary]:
        return [
            _knowledge_base_summary(item, container.manager.list_documents(item.id))
            for item in container.manager.list_knowledge_bases()
        ]

    @app.post(
        "/api/knowledge-bases",
        status_code=status.HTTP_201_CREATED,
        response_model=KnowledgeBaseSummary,
    )
    def create_knowledge_base(request: KnowledgeBaseCreate) -> KnowledgeBaseSummary:
        item = container.manager.create(request.name)
        return _knowledge_base_summary(item, [])

    @app.delete(
        "/api/knowledge-bases/{kb_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_knowledge_base(kb_id: str) -> Response:
        container.discovery.cancel_for_kb(kb_id)
        container.manager.delete(kb_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get(
        "/api/knowledge-bases/{kb_id}/documents",
        response_model=list[DocumentSummary],
    )
    def list_documents(kb_id: str) -> list[DocumentSummary]:
        return [
            _document_summary(item) for item in container.manager.list_documents(kb_id)
        ]

    @app.delete(
        "/api/knowledge-bases/{kb_id}/documents/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_document(kb_id: str, document_id: str) -> Response:
        container.manager.delete_document(kb_id, document_id)
        _schedule_discovery(container, kb_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post(
        "/api/knowledge-bases/{kb_id}/documents",
        response_model=DocumentBatchResult,
    )
    async def upload_documents(
        kb_id: str,
        files: list[UploadFile] = File(...),  # noqa: B008
    ) -> JSONResponse:
        if not 1 <= len(files) <= settings.upload_max_files:
            raise _DomainResponse("validation_error", "Invalid upload file count")
        result = await _process_files(container, kb_id, files)
        if result.documents:
            _schedule_discovery(container, kb_id)
        return JSONResponse(
            status_code=_batch_status(result),
            content=result.model_dump(mode="json"),
        )

    @app.post(
        "/api/knowledge-bases/{kb_id}/documents/async",
        response_model=UploadJobCreated,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def upload_documents_async(
        kb_id: str,
        files: list[UploadFile] = File(...),  # noqa: B008
    ) -> UploadJobCreated:
        container.manager.sqlite.get_knowledge_base(kb_id)
        if not settings.configured or container.manager.provider is None:
            raise _DomainResponse("configuration_missing", "Model provider configuration is missing")
        if not 1 <= len(files) <= settings.upload_max_files:
            raise _DomainResponse("validation_error", "Invalid upload file count")

        pending: list[tuple[str, bytes]] = []
        for upload in files:
            content = await upload.read()
            filename = validate_upload_submission(
                upload.filename or "",
                content,
                settings.upload_max_bytes,
            )
            pending.append((filename, content))
        job_id = container.jobs.submit(
            kb_id,
            pending,
            container.manager.ingest,
            lambda snapshot: _schedule_discovery_after_upload(container, snapshot),
        )
        return UploadJobCreated(job_id=job_id)

    @app.get(
        "/api/knowledge-bases/{kb_id}/discovery",
        response_model=DiscoverySummary,
    )
    def get_discovery(kb_id: str) -> DiscoverySummary:
        return _discovery_summary(container, kb_id)

    @app.post(
        "/api/knowledge-bases/{kb_id}/discovery/analyze",
        response_model=DiscoverySummary,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def analyze_discovery(kb_id: str) -> DiscoverySummary:
        container.sqlite.get_knowledge_base(kb_id)
        if not container.settings.configured or container.provider is None:
            raise _DomainResponse(
                "configuration_missing",
                "Model provider configuration is missing",
            )
        if not container.sqlite.list_documents(kb_id):
            raise _DomainResponse("validation_error", "Knowledge base has no documents")
        container.discovery.submit(kb_id, force=True)
        return _discovery_summary(container, kb_id)

    @app.get("/api/jobs/{job_id}", response_model=UploadJobSummary)
    def get_upload_job(job_id: str) -> UploadJobSummary:
        snapshot = container.jobs.snapshot(job_id)
        if snapshot is None:
            raise _DomainResponse("not_found", "Upload job not found")
        return _job_summary(snapshot)

    @app.post(
        "/api/jobs/{job_id}/cancel",
        response_model=UploadJobSummary,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def cancel_upload_job(job_id: str) -> UploadJobSummary:
        if container.jobs.snapshot(job_id) is None:
            raise _DomainResponse("not_found", "Upload job not found")
        if not container.jobs.cancel(job_id):
            raise _DomainResponse("upload_not_cancelable", "Upload job is already finished")
        snapshot = container.jobs.snapshot(job_id)
        if snapshot is None:
            raise _DomainResponse("not_found", "Upload job not found")
        return _job_summary(snapshot)

    @app.post(
        "/api/knowledge-bases/{kb_id}/import-demo",
        response_model=DocumentBatchResult,
    )
    async def import_demo(kb_id: str) -> JSONResponse:
        if not DEMO_DOCUMENTS_DIR.is_dir():
            raise _DomainResponse("storage_inconsistent", "Demo documents are unavailable")
        paths = sorted(DEMO_DOCUMENTS_DIR.glob("*.md"))
        if not paths:
            raise _DomainResponse("storage_inconsistent", "Demo documents are unavailable")
        documents: list[UploadedDocument] = []
        errors: list[DocumentUploadError] = []
        progress: list[str] = []
        for path in paths:
            try:
                content = path.read_bytes()
                document = container.manager.ingest(
                    kb_id,
                    path.name,
                    content,
                    progress.append,
                )
                documents.append(_uploaded_document(document, progress.copy()))
            except RagAppError as error:
                errors.append(
                    DocumentUploadError(
                        filename=path.name,
                        code=error.code,
                        message=str(error),
                    )
                )
            except OSError:
                errors.append(
                    DocumentUploadError(
                        filename=path.name,
                        code="storage_write_failed",
                        message="Unable to read demo document",
                    )
                )
            finally:
                progress.clear()
        result = DocumentBatchResult(documents=documents, errors=errors)
        if documents:
            _schedule_discovery(container, kb_id)
        return JSONResponse(
            status_code=_batch_status(result),
            content=result.model_dump(mode="json"),
        )

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        container.manager.validate_question_scope(request.kb_ids)
        workflow = container.workflow_registry.get("evidence_qa")
        history = [item.model_dump() for item in request.history]

        def event_stream() -> Iterator[str]:
            sequence = 0
            try:
                events = workflow.stream(
                    request.question,
                    history,
                    request.kb_ids,
                )
                for item in events:
                    sequence += 1
                    yield _sse_event(item, sequence)
            except RagAppError as error:
                sequence += 1
                logger.warning(
                    "chat stream failed",
                    extra={"code": error.code},
                )
                yield _sse_error_event(error.code, sequence)
            except Exception as error:  # noqa: BLE001
                sequence += 1
                logger.warning(
                    "chat stream failed unexpectedly",
                    extra={"error_type": type(error).__name__},
                )
                yield _sse_error_event("internal_error", sequence)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    if frontend_dir is not None:
        path = Path(frontend_dir)
        if (path / "index.html").is_file():
            app.mount("/", StaticFiles(directory=path, html=True), name="frontend")

    return app


class _DomainResponse(RagAppError):
    """Trigger FastAPI's shared domain-error response."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _job_summary(snapshot: UploadJobSnapshot) -> UploadJobSummary:
    items = [
        UploadJobItemSummary(
            filename=item.filename,
            status=item.status,  # type: ignore[arg-type]
            code=item.code,
            message=item.message,
            progress=list(item.progress),
            document=_document_summary(item.document) if item.document is not None else None,
        )
        for item in snapshot.items
    ]
    return UploadJobSummary(
        job_id=snapshot.job_id,
        kb_id=snapshot.kb_id,
        status=snapshot.status,  # type: ignore[arg-type]
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
        items=items,
    )


def _schedule_discovery_after_upload(
    container: AppContainer,
    snapshot: UploadJobSnapshot,
) -> None:
    if snapshot.status == "completed" and any(
        item.document is not None for item in snapshot.items
    ):
        _schedule_discovery(container, snapshot.kb_id, force=True)


def _schedule_discovery(
    container: AppContainer,
    kb_id: str,
    *,
    force: bool = False,
) -> None:
    if not container.settings.configured or container.provider is None:
        return
    if not container.sqlite.list_documents(kb_id):
        return
    container.discovery.submit(kb_id, force=force)


def _discovery_summary(container: AppContainer, kb_id: str) -> DiscoverySummary:
    container.sqlite.get_knowledge_base(kb_id)
    documents = container.sqlite.list_documents(kb_id)
    chunks = container.sqlite.list_chunks(kb_id)
    snapshot = container.sqlite.list_discovery(kb_id)
    job = container.discovery.latest_for_kb(kb_id)
    current_fingerprint = container.sqlite.discovery_fingerprint(kb_id)
    fingerprint_matches = (
        snapshot is not None and snapshot.source_fingerprint == current_fingerprint
    )
    in_progress = job is not None and job.status in {"pending", "processing"}
    topics = snapshot.topics if snapshot is not None and (fingerprint_matches or in_progress) else ()

    status = "not_analyzed"
    if job is not None and job.status in {
        "pending",
        "processing",
        "failed",
        "cancelled",
    }:
        status = job.status
    elif snapshot is not None and fingerprint_matches:
        status = "completed"

    return DiscoverySummary(
        kb_id=kb_id,
        status=status,  # type: ignore[arg-type]
        job_id=job.job_id if job is not None else None,
        code=job.code if job is not None else None,
        message=job.message if job is not None else None,
        document_count=len(documents),
        chunk_count=len(chunks),
        topic_count=len(topics),
        question_count=sum(len(topic.questions) for topic in topics),
        analyzed_at=snapshot.created_at if snapshot is not None else None,
        topics=[_discovery_topic_summary(topic) for topic in topics],
    )


def _discovery_topic_summary(
    topic: DiscoveryTopicRecord,
) -> DiscoveryTopicSummary:
    return DiscoveryTopicSummary(
        id=topic.id,
        title=topic.title,
        type=topic.type,
        summary=topic.summary,
        document_ids=list(topic.document_ids),
        chunk_count=topic.chunk_count,
        confidence=topic.confidence,
        updated_at=topic.updated_at,
        questions=[
            DiscoveryQuestionSummary(
                id=item.id,
                question=item.question,
                source_chunk_ids=list(item.source_chunk_ids),
            )
            for item in topic.questions
        ],
    )


async def _process_files(
    container: AppContainer,
    kb_id: str,
    files: Sequence[UploadFile],
) -> DocumentBatchResult:
    documents: list[UploadedDocument] = []
    errors: list[DocumentUploadError] = []
    for upload in files:
        filename = upload.filename or ""
        progress: list[str] = []
        try:
            content = await upload.read()
            document = container.manager.ingest(
                kb_id,
                filename,
                content,
                progress.append,
            )
            documents.append(_uploaded_document(document, progress.copy()))
        except RagAppError as error:
            errors.append(
                DocumentUploadError(
                    filename=filename or "unnamed.md",
                    code=error.code,
                    message=str(error),
                )
            )
        except OSError:
            errors.append(
                DocumentUploadError(
                    filename=filename or "unnamed.md",
                    code="storage_write_failed",
                    message="Unable to read the uploaded file",
                )
            )
    return DocumentBatchResult(documents=documents, errors=errors)


def _knowledge_base_summary(
    item: KnowledgeBase,
    documents: Sequence[Document],
) -> KnowledgeBaseSummary:
    return KnowledgeBaseSummary(
        id=item.id,
        name=item.name,
        document_count=len(documents),
        chunk_count=sum(document.chunk_count for document in documents),
        created_at=item.created_at,
    )


def _document_summary(item: Document) -> DocumentSummary:
    return DocumentSummary(
        id=item.id,
        kb_id=item.kb_id,
        filename=item.filename,
        status=item.status.value,
        chunk_count=item.chunk_count,
        created_at=item.created_at,
    )


def _uploaded_document(item: Document, progress: list[str]) -> UploadedDocument:
    return UploadedDocument(
        id=item.id,
        kb_id=item.kb_id,
        filename=item.filename,
        status=item.status.value,
        chunk_count=item.chunk_count,
        created_at=item.created_at,
        progress=progress,
    )


def _batch_status(result: DocumentBatchResult) -> int:
    if result.errors and result.documents:
        return status.HTTP_207_MULTI_STATUS
    if not result.errors:
        return status.HTTP_201_CREATED
    return max(_error_status(error.code) for error in result.errors)


def _error_status(code: str) -> int:
    if code in {
        "validation_error",
        "invalid_extension",
        "file_too_large",
        "invalid_utf8",
        "empty_document",
        "chunking_failed",
        "pdf_parse_failed",
        "docx_parse_failed",
        "html_parse_failed",
        "unsupported_parser",
    }:
        return status.HTTP_400_BAD_REQUEST
    if code in {"duplicate_knowledge_base", "duplicate_document"}:
        return status.HTTP_409_CONFLICT
    if code == "upload_not_cancelable":
        return status.HTTP_409_CONFLICT
    if code == "not_found":
        return status.HTTP_404_NOT_FOUND
    if code in {
        "configuration_error",
        "configuration_missing",
        "provider_unavailable",
        "embedding_failed",
        "provider_response_error",
    }:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _sse_event(item: WorkflowEvent, sequence: int) -> str:
    event = SseEvent(
        type=item.type,  # type: ignore[arg-type]
        stage=item.stage,  # type: ignore[arg-type]
        message=item.message,
        sequence=sequence,
        payload=item.payload,
    )
    return encode_sse_event(event)


def _sse_error_event(code: str, sequence: int) -> str:
    event = SseEvent(
        type="error",
        stage=None,
        message=_safe_error_message(code),
        sequence=sequence,
        payload={"code": code},
    )
    return encode_sse_event(event)


def _safe_error_message(code: str) -> str:
    if code in {"configuration_error", "configuration_missing"}:
        return "模型配置缺失，暂时无法问答"
    if code in {"provider_unavailable", "provider_response_error"}:
        return "模型服务暂不可用，请稍后重试"
    if code == "storage_inconsistent":
        return "本地数据索引不一致，请先运行一致性检查"
    if code == "not_found":
        return "所选知识库不存在"
    return "问答处理失败，请稍后重试"
