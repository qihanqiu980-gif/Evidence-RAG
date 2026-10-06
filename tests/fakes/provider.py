from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from rag_app.providers.base import ProviderMetricsTracker, RerankItem


class FakeProvider:
    """Deterministic provider for tests only; never expose it as product mode."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self._metrics_tracker = ProviderMetricsTracker()
        self.embed_documents_calls = 0
        self.embed_query_calls = 0
        self.rerank_calls = 0
        self.chat_calls = 0
        self.fail_embeddings = False
        self.fail_rerank = False
        self.chat_responses: dict[str, dict[str, Any]] = {}
        self.chat_response_queues: dict[str, list[dict[str, Any]]] = {}

    @property
    def metrics_tracker(self) -> ProviderMetricsTracker:
        return self._metrics_tracker

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._metrics_tracker.record_request("embedding")
        self._metrics_tracker.record_attempt("embedding")
        self.embed_documents_calls += 1
        try:
            if self.fail_embeddings:
                raise RuntimeError("injected embedding failure")
            return [self._embedding(text) for text in texts]
        except Exception:
            self._metrics_tracker.record_failure("embedding", unavailable=False)
            raise

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls += 1
        if self.fail_embeddings:
            raise RuntimeError("injected embedding failure")
        return self._embedding(text)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int,
    ) -> list[RerankItem]:
        self._metrics_tracker.record_request("rerank")
        self._metrics_tracker.record_attempt("rerank")
        self.rerank_calls += 1
        try:
            if self.fail_rerank:
                raise RuntimeError("injected rerank failure")
            query_terms = self._terms(query)
            scored = [
                (
                    index,
                    len(query_terms & self._terms(document)) / max(len(query_terms), 1),
                )
                for index, document in enumerate(documents)
            ]
            scored.sort(key=lambda item: (-item[1], item[0]))
            return [
                RerankItem(index=index, score=score)
                for index, score in scored[:top_k]
            ]
        except Exception:
            self._metrics_tracker.record_failure("rerank", unavailable=False)
            raise

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        task: str,
        model: str | None = None,
        response_format: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        self._metrics_tracker.record_request("chat")
        self._metrics_tracker.record_attempt("chat")
        self.chat_calls += 1
        try:
            queue = self.chat_response_queues.get(task)
            response = queue.pop(0) if queue else self.chat_responses.get(task)
            if response is None:
                raise AssertionError(f"No fake chat response configured for task {task}")
            return json.loads(json.dumps(response))
        except Exception:
            self._metrics_tracker.record_failure("chat", unavailable=False)
            raise

    def _embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for term in self._terms(text):
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        else:
            vector[0] = 1.0
        return vector

    @staticmethod
    def _terms(text: str) -> set[str]:
        normalized = text.casefold()
        words = set(re.findall(r"[a-z0-9]+", normalized))
        cjk = re.findall(r"[\u4e00-\u9fff]", normalized)
        bigrams = {
            normalized[start : start + 2]
            for start in range(len(normalized) - 1)
            if any("\u4e00" <= char <= "\u9fff" for char in normalized[start : start + 2])
        }
        return words | set(cjk) | bigrams
