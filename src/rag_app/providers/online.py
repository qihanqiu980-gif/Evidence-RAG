from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from ..config import Settings
from ..errors import ProviderResponseError, ProviderUnavailableError, RagAppError
from .base import ProviderMetricsTracker, ProviderTokenUsage, RerankItem


class OnlineModelProvider:
    """Lightweight online adapter; workflow code never sees HTTP details."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        sleep_function: Callable[[float], None] | None = None,
    ) -> None:
        self._settings = settings
        self._metrics_tracker = ProviderMetricsTracker()
        self._sleep = sleep_function or time.sleep
        self._client = client or httpx.Client(
            base_url=settings.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            timeout=settings.request_timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    @property
    def metrics_tracker(self) -> ProviderMetricsTracker:
        return self._metrics_tracker

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._metrics_tracker.record_request("embedding")
        embeddings: list[list[float]] = []
        try:
            for start in range(0, len(texts), 10):
                batch = list(texts[start : start + 10])
                payload = {
                    "model": self._settings.embedding_model,
                    "input": batch,
                }
                data = self._request("/embeddings", payload, operation="embedding")
                self._record_usage("embedding", data)
                rows = data.get("data")
                if not isinstance(rows, list) or len(rows) != len(batch):
                    raise ProviderResponseError(
                        "Embedding response has an invalid item count"
                    )
                try:
                    for row in rows:
                        embedding = row["embedding"]
                        if not isinstance(embedding, list):
                            raise TypeError
                        embeddings.append([float(value) for value in embedding])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ProviderResponseError(
                        "Embedding response format is invalid"
                    ) from exc
            self._validate_dimensions(embeddings)
        except RagAppError as error:
            self._record_failure("embedding", unavailable=self._is_unavailable(error))
            raise
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0]

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int,
    ) -> list[RerankItem]:
        if not documents or top_k <= 0:
            return []
        self._metrics_tracker.record_request("rerank")
        payload = {
            "model": self._settings.rerank_model,
            "query": query,
            "documents": list(documents),
            "top_n": min(top_k, len(documents)),
        }
        try:
            data = self._request(
                self._settings.rerank_url,
                payload,
                absolute_url=True,
                operation="rerank",
            )
            self._record_usage("rerank", data)
            rows = data.get("results")
            if not isinstance(rows, list):
                raise ProviderResponseError("Rerank response format is invalid")
            items: list[RerankItem] = []
            for row in rows:
                index = int(row["index"])
                score = float(row["relevance_score"])
                if not 0 <= index < len(documents):
                    raise ValueError
                items.append(RerankItem(index=index, score=score))
        except (RagAppError, KeyError, TypeError, ValueError) as error:
            self._record_failure("rerank", unavailable=self._is_unavailable(error))
            if isinstance(error, RagAppError):
                raise
            raise ProviderResponseError("Rerank response contains invalid indexes") from error
        return items

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        task: str,
    ) -> dict[str, Any]:
        self._metrics_tracker.record_request("chat")
        payload = {
            "model": self._settings.chat_model,
            "temperature": 0,
            "messages": [dict(message) for message in messages],
        }
        try:
            data = self._request("/chat/completions", payload, operation="chat")
            self._record_usage("chat", data)
            content = data["choices"][0]["message"]["content"]
            parsed = self._parse_json_content(content)
        except RagAppError as error:
            self._record_failure("chat", unavailable=self._is_unavailable(error))
            raise
        except (KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
            response_error = ProviderResponseError(
                f"Model JSON response is invalid for {task}"
            )
            self._record_failure("chat", unavailable=False)
            raise response_error from exc
        if not isinstance(parsed, dict):
            response_error = ProviderResponseError(
                f"Model JSON response is invalid for {task}"
            )
            self._record_failure("chat", unavailable=False)
            raise response_error
        return parsed

    def _request(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        absolute_url: bool = False,
        operation: str,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries + 1):
            self._metrics_tracker.record_attempt(operation)
            try:
                response = self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ProviderResponseError("Provider response is not a JSON object")
                return data
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if 500 <= status_code < 600 and attempt < self._settings.max_retries:
                    self._metrics_tracker.record_retry(operation, "http_5xx")
                    self._wait_before_retry(attempt)
                    last_error = exc
                    continue
                raise ProviderResponseError("Provider rejected the request") from exc
            except httpx.TransportError as exc:
                last_error = exc
                if attempt < self._settings.max_retries:
                    self._metrics_tracker.record_retry(operation, "transport")
                    self._wait_before_retry(attempt)
                    continue
                raise ProviderUnavailableError("Provider is unreachable") from exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise ProviderResponseError("Provider returned invalid JSON") from exc
        raise ProviderUnavailableError("Provider is unreachable") from last_error

    def _record_failure(self, operation: str, *, unavailable: bool) -> None:
        self._metrics_tracker.record_failure(operation, unavailable=unavailable)

    def _wait_before_retry(self, attempt: int) -> None:
        delay = self._settings.retry_backoff_seconds * (2**attempt)
        self._sleep(min(delay, 30.0))

    def _record_usage(self, operation: str, data: dict[str, Any]) -> None:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return

        input_tokens = self._usage_integer(usage, ("prompt_tokens", "input_tokens"))
        output_tokens = self._usage_integer(
            usage, ("completion_tokens", "output_tokens")
        )
        total_tokens = self._usage_integer(usage, ("total_tokens",))
        if (
            operation == "embedding"
            and input_tokens is None
            and total_tokens is not None
        ):
            input_tokens = total_tokens
        if total_tokens is None:
            input_tokens = input_tokens or 0
            output_tokens = output_tokens or 0
            total_tokens = input_tokens + output_tokens

        self._metrics_tracker.record_usage(
            operation,
            ProviderTokenUsage(
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
                total_tokens=total_tokens or 0,
            ),
        )

    @staticmethod
    def _is_unavailable(error: BaseException) -> bool:
        return isinstance(error, ProviderUnavailableError)

    @staticmethod
    def _usage_integer(
        usage: dict[str, Any],
        names: tuple[str, ...],
    ) -> int | None:
        for name in names:
            value = usage.get(name)
            if type(value) is int and value >= 0:
                return value
        return None

    def _validate_dimensions(self, embeddings: Sequence[Sequence[float]]) -> None:
        if any(
            len(embedding) != self._settings.embedding_dimension
            for embedding in embeddings
        ):
            raise ProviderResponseError("Embedding dimension does not match configuration")

    @staticmethod
    def _parse_json_content(content: Any) -> Any:
        candidate = content.strip() if isinstance(content, str) else content
        if isinstance(candidate, str):
            if candidate.startswith("```"):
                lines = candidate.splitlines()
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    candidate = "\n".join(lines[1:-1]).strip()
            # Some compatible-mode models occasionally return a JSON-encoded string.
            if candidate.startswith('"') and candidate.endswith('"'):
                try:
                    decoded = json.loads(candidate)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, str):
                    candidate = decoded.strip()
            start = candidate.find("{")
            end = candidate.rfind("}")
            if (start > 0 or end < len(candidate) - 1) and 0 <= start < end:
                candidate = candidate[start : end + 1]
        return json.loads(candidate)
