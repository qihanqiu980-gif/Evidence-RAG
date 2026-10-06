from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from ..config import Settings
from ..errors import ProviderResponseError
from ..providers.base import ModelProvider
from ..storage.chroma import ChromaVectorStore, VectorMatch
from ..storage.sqlite import KeywordMatch, SQLiteStore


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
        sqlite: SQLiteStore,
    ) -> None:
        self.settings = settings
        self.vectors = vectors
        self.provider = provider
        self.sqlite = sqlite

    def retrieve(self, query: str, kb_ids: Sequence[str]) -> RetrievalResult:
        candidate_limit = (
            self.settings.retrieval_top_k * self.settings.candidate_multiplier
        )
        query_embedding = self.provider.embed_query(query)
        vector_matches = self.vectors.search(query_embedding, kb_ids, candidate_limit)
        if self.settings.hybrid_retrieval:
            keyword_matches = self.sqlite.search_keyword(
                query,
                kb_ids,
                candidate_limit,
            )
            matches = _rrf_fuse(
                vector_matches,
                keyword_matches,
                vector_weight=self.settings.retrieval_vector_weight,
                keyword_weight=self.settings.retrieval_keyword_weight,
                rrf_k=self.settings.retrieval_rrf_k,
            )
        else:
            matches = vector_matches
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


def _rrf_fuse(
    vector_matches: Sequence[VectorMatch],
    keyword_matches: Sequence[KeywordMatch],
    *,
    vector_weight: float,
    keyword_weight: float,
    rrf_k: int,
) -> list[VectorMatch]:
    """Fuse two ranked lists with weighted Reciprocal Rank Fusion."""
    candidates: dict[str, VectorMatch] = {}
    scores: dict[str, float] = {}

    for rank, match in enumerate(vector_matches, start=1):
        candidates.setdefault(match.chunk_id, match)
        scores[match.chunk_id] = (
            scores.get(match.chunk_id, 0.0) + vector_weight / (rrf_k + rank)
        )

    for rank, keyword_match in enumerate(keyword_matches, start=1):
        candidates.setdefault(
            keyword_match.chunk_id,
            VectorMatch(
                chunk_id=keyword_match.chunk_id,
                kb_id=keyword_match.kb_id,
                document_id=keyword_match.document_id,
                document_name=keyword_match.document_name,
                heading_path=keyword_match.heading_path,
                chunk_index=keyword_match.chunk_index,
                content=keyword_match.content,
                content_hash=keyword_match.content_hash,
                similarity=0.0,
            ),
        )
        scores[keyword_match.chunk_id] = (
            scores.get(keyword_match.chunk_id, 0.0) + keyword_weight / (rrf_k + rank)
        )

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [candidates[chunk_id] for chunk_id in ordered]


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
