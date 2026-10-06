from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from ..config import Settings
from ..errors import ProviderResponseError, RagAppError
from ..models import Chunk, DiscoveryQuestion, DiscoverySnapshot, DiscoveryTopicDraft
from ..providers.base import ModelProvider
from ..storage.sqlite import SQLiteStore

logger = logging.getLogger(__name__)
MAX_TOPICS = 8
MAX_PLANNER_CHUNKS = 240
MAX_QUESTION_CHUNKS_PER_TOPIC = 8
MAX_QUESTIONS_PER_TOPIC = 4
TERMINAL_DISCOVERY_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class _TopicPlan:
    temporary_id: str
    title: str
    type: str
    summary: str
    source_chunk_ids: frozenset[str]
    document_ids: frozenset[str]
    confidence: float


class KnowledgeDiscoveryService:
    """Generate evidence-bound topic maps from persisted chunks."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        provider: ModelProvider,
        settings: Settings,
    ) -> None:
        self.sqlite = sqlite
        self.provider = provider
        self.discovery_model = settings.discovery_model

    def analyze(self, kb_id: str) -> DiscoverySnapshot:
        self.sqlite.get_knowledge_base(kb_id)
        fingerprint = self.sqlite.discovery_fingerprint(kb_id)
        documents = self.sqlite.list_documents(kb_id)
        chunks = self.sqlite.list_chunks(kb_id)
        if not documents or not chunks:
            raise ProviderResponseError("Knowledge base has no analyzed documents")

        document_names = {document.id: document.filename for document in documents}
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        topic_plans = self._plan_topics(kb_id, document_names, chunks)
        questions = self._generate_questions(topic_plans, chunks_by_id, document_names)
        drafts: list[DiscoveryTopicDraft] = []
        for plan in topic_plans:
            topic_questions = [
                DiscoveryQuestion(
                    question=item.question,
                    source_chunk_ids=tuple(item.source_chunk_ids),
                )
                for item in questions.get(plan.temporary_id, [])
            ]
            if not topic_questions:
                continue
            drafts.append(
                DiscoveryTopicDraft(
                    title=plan.title,
                    type=plan.type,
                    summary=plan.summary,
                    document_ids=tuple(sorted(plan.document_ids)),
                    source_chunk_ids=tuple(sorted(plan.source_chunk_ids)),
                    confidence=plan.confidence,
                    questions=tuple(topic_questions),
                )
            )

        if not drafts:
            raise ProviderResponseError("Discovery output contains no valid questions")
        return self.sqlite.save_discovery(kb_id, fingerprint, drafts)

    def _plan_topics(
        self,
        kb_id: str,
        document_names: Mapping[str, str],
        chunks: Sequence[Chunk],
    ) -> list[_TopicPlan]:
        selected = _select_planner_chunks(chunks)
        catalog = [
            {
                "chunk_id": chunk.id,
                "document_name": document_names.get(chunk.document_id, ""),
                "heading_path": list(chunk.heading_path),
                "excerpt": _snippet(chunk.content, 220),
            }
            for chunk in selected
        ]
        system = (
            "你是知识库信息架构师。根据真实 chunk 目录归纳主题，不得发明资料中不存在的内容。"
            '只输出 JSON：{"topics":[{"title":"...","type":"...",'
            '"summary":"...","confidence":0.0,'
            '"source_chunk_ids":["..."]}]}。'
        )
        user = json.dumps(
            {
                "knowledge_base_id": kb_id,
                "topic_count_range": {"min": 4, "max": MAX_TOPICS},
                "requirements": [
                    "主题必须来自 chunk 的文档名、标题路径和摘录",
                    "source_chunk_ids 只能使用输入 chunk_id",
                    "合并重复或相邻的小主题，输出面向使用者的问题域",
                    "type 使用简短中文标签，例如概念、任务、故障排查或政策",
                    "confidence 是 0 到 1 的覆盖信心，不是相关度分数",
                ],
                "chunks": catalog,
            },
            ensure_ascii=False,
        )
        payload = self.provider.chat_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            task="discovery_topics",
            model=self.discovery_model,
            response_format={"type": "json_object"},
        )
        return self._parse_topic_plan(payload, selected, chunks)

    def _parse_topic_plan(
        self,
        payload: Mapping[str, object],
        selected_chunks: Sequence[Chunk],
        all_chunks: Sequence[Chunk],
    ) -> list[_TopicPlan]:
        raw_topics = payload.get("topics")
        if not isinstance(raw_topics, list):
            raise ProviderResponseError("Discovery topic response is invalid")

        selected_ids = {chunk.id for chunk in selected_chunks}
        all_chunks_by_id = {chunk.id: chunk for chunk in all_chunks}
        plans: dict[str, _TopicPlan] = {}
        for index, raw in enumerate(raw_topics, start=1):
            if not isinstance(raw, dict):
                continue
            title = _text(raw.get("title"), max_length=80)
            type_name = _text(raw.get("type"), max_length=30) or "主题"
            summary = _text(raw.get("summary"), max_length=240)
            confidence = _confidence(raw.get("confidence"))
            source_ids = {
                item
                for item in _string_list(raw.get("source_chunk_ids"))
                if item in selected_ids
            }
            if not title or not source_ids:
                continue

            normalized_title = _normalize_title(title)
            existing = next(
                (
                    plan
                    for plan in plans.values()
                    if _normalize_title(plan.title) == normalized_title
                ),
                None,
            )
            if existing is not None:
                source_ids.update(existing.source_chunk_ids)
                plans[existing.temporary_id] = _TopicPlan(
                    temporary_id=existing.temporary_id,
                    title=existing.title,
                    type=existing.type,
                    summary=existing.summary or summary,
                    source_chunk_ids=frozenset(source_ids),
                    document_ids=frozenset(
                        all_chunks_by_id[chunk_id].document_id
                        for chunk_id in source_ids
                    ),
                    confidence=max(existing.confidence, confidence),
                )
                continue

            temporary_id = f"t{index}"
            plans[temporary_id] = _TopicPlan(
                temporary_id=temporary_id,
                title=title,
                type=type_name,
                summary=summary,
                source_chunk_ids=frozenset(source_ids),
                document_ids=frozenset(
                    all_chunks_by_id[chunk_id].document_id for chunk_id in source_ids
                ),
                confidence=confidence,
            )

        ordered = list(plans.values())[:MAX_TOPICS]
        if not ordered:
            raise ProviderResponseError("Discovery topic output contains no valid topics")
        return ordered

    def _generate_questions(
        self,
        topics: Sequence[_TopicPlan],
        chunks_by_id: Mapping[str, Chunk],
        document_names: Mapping[str, str],
    ) -> dict[str, list[DiscoveryQuestion]]:
        topic_inputs = []
        for topic in topics:
            source_chunks = [
                chunks_by_id[chunk_id]
                for chunk_id in sorted(topic.source_chunk_ids)
                if chunk_id in chunks_by_id
            ][:MAX_QUESTION_CHUNKS_PER_TOPIC]
            topic_inputs.append(
                {
                    "topic_id": topic.temporary_id,
                    "title": topic.title,
                    "chunks": [
                        {
                            "chunk_id": chunk.id,
                            "document_name": document_names.get(chunk.document_id, ""),
                            "heading_path": list(chunk.heading_path),
                            "content": _snippet(chunk.content, 700),
                        }
                        for chunk in source_chunks
                    ],
                }
            )

        system = (
            "你是知识库问题策划师。只基于给定 chunk 内容生成用户可提出的问题。"
            '只输出 JSON：{"questions":[{"topic_id":"...","question":"...",'
            '"source_chunk_ids":["..."]}]}。'
        )
        user = json.dumps(
            {
                "max_questions_per_topic": MAX_QUESTIONS_PER_TOPIC,
                "requirements": [
                    "问题必须能由对应 source_chunk_ids 的内容回答",
                    "source_chunk_ids 只能使用输入 chunk_id，且必须属于同一 topic_id",
                    "问题要具体，不使用“资料中是否提到”这类元问题",
                    "不要生成价格、日期、政策等 chunk 未提供的问题",
                    "每个 topic_id 输出 2 到 4 个问题",
                ],
                "topics": topic_inputs,
            },
            ensure_ascii=False,
        )
        payload = self.provider.chat_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            task="discovery_questions",
            model=self.discovery_model,
            response_format={"type": "json_object"},
        )
        return self._parse_questions(payload, topics)

    def _parse_questions(
        self,
        payload: Mapping[str, object],
        topics: Sequence[_TopicPlan],
    ) -> dict[str, list[DiscoveryQuestion]]:
        raw_questions = payload.get("questions")
        if not isinstance(raw_questions, list):
            raise ProviderResponseError("Discovery question response is invalid")
        plans_by_id = {topic.temporary_id: topic for topic in topics}
        questions: dict[str, list[DiscoveryQuestion]] = {
            topic.temporary_id: [] for topic in topics
        }
        seen: set[str] = set()
        for raw in raw_questions:
            if not isinstance(raw, dict):
                continue
            topic_id = raw.get("topic_id")
            plan = plans_by_id.get(topic_id) if isinstance(topic_id, str) else None
            question = _text(raw.get("question"), max_length=240)
            source_ids = tuple(
                dict.fromkeys(
                    item
                    for item in _string_list(raw.get("source_chunk_ids"))
                    if plan is not None and item in plan.source_chunk_ids
                )
            )
            normalized = _normalize_title(question)
            if plan is None or not question or not source_ids or normalized in seen:
                continue
            seen.add(normalized)
            questions[plan.temporary_id].append(
                DiscoveryQuestion(
                    question=question,
                    source_chunk_ids=source_ids,
                )
            )

        return {
            topic_id: items[:MAX_QUESTIONS_PER_TOPIC]
            for topic_id, items in questions.items()
        }


@dataclass(frozen=True, slots=True)
class DiscoveryJobSnapshot:
    job_id: str
    kb_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    topic_count: int | None
    question_count: int | None
    code: str | None
    message: str | None


@dataclass
class _DiscoveryJob:
    job_id: str
    kb_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    topic_count: int | None = None
    question_count: int | None = None
    code: str | None = None
    message: str | None = None

    def snapshot(self) -> DiscoveryJobSnapshot:
        return DiscoveryJobSnapshot(
            job_id=self.job_id,
            kb_id=self.kb_id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            topic_count=self.topic_count,
            question_count=self.question_count,
            code=self.code,
            message=self.message,
        )


class DiscoveryJobManager:
    """In-process serialized discovery jobs; completed results persist in SQLite."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        provider: ModelProvider | None,
        settings: Settings,
    ) -> None:
        self.sqlite = sqlite
        self.provider = provider
        self.discovery_settings = settings
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="discovery",
        )
        self._jobs: dict[str, _DiscoveryJob] = {}
        self._lock = threading.Lock()

    def submit(self, kb_id: str, *, force: bool = False) -> str:
        self.sqlite.get_knowledge_base(kb_id)
        if not self.sqlite.list_documents(kb_id):
            raise ProviderResponseError("Knowledge base has no documents")
        fingerprint = self.sqlite.discovery_fingerprint(kb_id)
        now = datetime.now(UTC)
        with self._lock:
            active = next(
                (
                    job
                    for job in reversed(self._jobs.values())
                    if job.kb_id == kb_id and job.status not in TERMINAL_DISCOVERY_STATUSES
                ),
                None,
            )
            if active is not None and not force:
                return active.job_id

            job_id = str(uuid4())
            self._jobs[job_id] = _DiscoveryJob(
                job_id=job_id,
                kb_id=kb_id,
                status="pending",
                created_at=now,
                updated_at=now,
            )
        self._executor.submit(self._run, job_id, fingerprint)
        return job_id

    def snapshot(self, job_id: str) -> DiscoveryJobSnapshot | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.snapshot() if job is not None else None

    def latest_for_kb(self, kb_id: str) -> DiscoveryJobSnapshot | None:
        with self._lock:
            job = next(
                (job for job in reversed(self._jobs.values()) if job.kb_id == kb_id),
                None,
            )
            return job.snapshot() if job is not None else None

    def cancel_for_kb(self, kb_id: str) -> None:
        now = datetime.now(UTC)
        with self._lock:
            for job in self._jobs.values():
                if job.kb_id == kb_id and job.status not in TERMINAL_DISCOVERY_STATUSES:
                    job.status = "cancelled"
                    job.updated_at = now

    def close(self) -> None:
        with self._lock:
            now = datetime.now(UTC)
            for job in self._jobs.values():
                if job.status not in TERMINAL_DISCOVERY_STATUSES:
                    job.status = "cancelled"
                    job.updated_at = now
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, expected_fingerprint: str) -> None:
        if not self._mark(job_id, "processing"):
            return
        try:
            if self.provider is None:
                raise ProviderResponseError("Model provider configuration is missing")
            kb_id = self._kb_id(job_id)
            fingerprint = self.sqlite.discovery_fingerprint(kb_id)
            if fingerprint != expected_fingerprint:
                raise ProviderResponseError("Knowledge base changed during discovery")
            service = KnowledgeDiscoveryService(
                self.sqlite, self.provider, self.discovery_settings
            )
            result = service.analyze(kb_id)
            with self._lock:
                job = self._jobs[job_id]
                if job.status == "cancelled":
                    return
                job.status = "completed"
                job.topic_count = result.topic_count
                job.question_count = result.question_count
                job.updated_at = datetime.now(UTC)
        except RagAppError as error:
            self._fail(job_id, error.code, str(error))
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "discovery job failed",
                extra={"job_id": job_id, "error_type": type(error).__name__},
            )
            self._fail(job_id, "internal_error", "Knowledge discovery failed")

    def _kb_id(self, job_id: str) -> str:
        with self._lock:
            return self._jobs[job_id].kb_id

    def _mark(self, job_id: str, status: str) -> bool:
        with self._lock:
            job = self._jobs[job_id]
            if job.status in TERMINAL_DISCOVERY_STATUSES:
                return False
            job.status = status
            job.updated_at = datetime.now(UTC)
            return True

    def _fail(self, job_id: str, code: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status == "cancelled":
                return
            job.status = "failed"
            job.code = code
            job.message = message
            job.updated_at = datetime.now(UTC)


def _select_planner_chunks(chunks: Sequence[Chunk]) -> Sequence[Chunk]:
    if len(chunks) <= MAX_PLANNER_CHUNKS:
        return chunks
    step = len(chunks) / MAX_PLANNER_CHUNKS
    indexes = {round(index * step) for index in range(MAX_PLANNER_CHUNKS)}
    indexes.update({0, len(chunks) - 1})
    return [chunks[index] for index in sorted(indexes)]


def _snippet(content: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", content).strip()
    return normalized[:limit]


def _text(value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:max_length]


def _confidence(value: object) -> float:
    try:
        confidence = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    if confidence <= 0:
        return 0.0
    return min(confidence, 1.0)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _normalize_title(value: str) -> str:
    return re.sub(r"[\s：:，,。.？?]+", "", value.casefold())
