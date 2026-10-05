from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from ..config import Settings
from ..errors import ProviderResponseError
from ..providers.base import ModelProvider
from ..storage.chroma import ChromaVectorStore


class EvidenceSupportStatus(StrEnum):
    SUPPORTING = "supporting"
    RELATED = "related"


@dataclass(frozen=True, slots=True)
class Evidence:
    reference_id: int
    kb_id: str
    document_id: str
    document_name: str
    heading_path: tuple[str, ...]
    chunk_index: int
    content: str
    similarity: float
    rerank_score: float
    subquestion_id: str
    subquestion: str
    support_status: EvidenceSupportStatus = EvidenceSupportStatus.RELATED


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    evidence: tuple[Evidence, ...]


class EvidenceRetriever:
    """Retrieve a bounded candidate pool, then rerank it for one subquestion."""

    def __init__(
        self,
        settings: Settings,
        vectors: ChromaVectorStore,
        provider: ModelProvider,
    ) -> None:
        self.settings = settings
        self.vectors = vectors
        self.provider = provider

    def retrieve(self, query: str, kb_ids: Sequence[str]) -> RetrievalResult:
        candidate_limit = (
            self.settings.retrieval_top_k * self.settings.candidate_multiplier
        )
        query_embedding = self.provider.embed_query(query)
        matches = self.vectors.search(query_embedding, kb_ids, candidate_limit)
        if not matches:
            return RetrievalResult(query=query, evidence=())

        ranked = self.provider.rerank(
            query,
            [match.content for match in matches],
            self.settings.retrieval_top_k,
        )
        evidence: list[Evidence] = []
        seen_indexes: set[int] = set()
        for item in ranked:
            if item.index in seen_indexes or not 0 <= item.index < len(matches):
                raise ProviderResponseError("Rerank returned an invalid candidate index")
            seen_indexes.add(item.index)
            match = matches[item.index]
            evidence.append(
                Evidence(
                    reference_id=0,
                    kb_id=match.kb_id,
                    document_id=match.document_id,
                    document_name=match.document_name,
                    heading_path=match.heading_path,
                    chunk_index=match.chunk_index,
                    content=match.content,
                    similarity=match.similarity,
                    rerank_score=item.score,
                    subquestion_id="",
                    subquestion=query,
                )
            )
        return RetrievalResult(
            query=query,
            evidence=tuple(evidence[: self.settings.retrieval_top_k]),
        )


def assign_reference_ids(
    results: dict[str, RetrievalResult],
    question_by_id: dict[str, str],
) -> dict[str, RetrievalResult]:
    """Assign stable frontend reference ids across all subquestions."""
    assigned: dict[str, RetrievalResult] = {}
    next_id = 1
    for subquestion_id, result in results.items():
        evidence = []
        for item in result.evidence:
            evidence.append(
                replace(
                    item,
                    reference_id=next_id,
                    subquestion_id=subquestion_id,
                    subquestion=question_by_id[subquestion_id],
                )
            )
            next_id += 1
        assigned[subquestion_id] = replace(result, evidence=tuple(evidence))
    return assigned


def mark_supporting_evidence(
    results: dict[str, RetrievalResult],
    supporting_ids: set[int],
) -> dict[str, RetrievalResult]:
    marked: dict[str, RetrievalResult] = {}
    for subquestion_id, result in results.items():
        evidence = tuple(
            replace(
                item,
                support_status=(
                    EvidenceSupportStatus.SUPPORTING
                    if item.reference_id in supporting_ids
                    else EvidenceSupportStatus.RELATED
                ),
            )
            for item in result.evidence
        )
        marked[subquestion_id] = replace(result, evidence=evidence)
    return marked
