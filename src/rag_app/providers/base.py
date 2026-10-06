from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

PROVIDER_OPERATIONS = ("embedding", "chat", "rerank")


@dataclass(frozen=True, slots=True)
class RerankItem:
    index: int
    score: float


@dataclass(frozen=True, slots=True)
class ProviderOperationMetrics:
    operation: str
    request_count: int
    attempt_count: int
    retry_count: int
    transport_retry_count: int
    http_5xx_retry_count: int
    failed_request_count: int
    provider_unavailable_failure_count: int
    provider_response_failure_count: int
    usage_report_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ProviderMetricsSnapshot:
    operations: tuple[ProviderOperationMetrics, ...]

    def operation(self, name: str) -> ProviderOperationMetrics:
        for item in self.operations:
            if item.operation == name:
                return item
        return _empty_operation(name)


@dataclass(frozen=True, slots=True)
class ProviderTokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class _OperationCounters:
    request_count: int = 0
    attempt_count: int = 0
    retry_count: int = 0
    transport_retry_count: int = 0
    http_5xx_retry_count: int = 0
    failed_request_count: int = 0
    provider_unavailable_failure_count: int = 0
    provider_response_failure_count: int = 0
    usage_report_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ProviderMetricsTracker:
    """Aggregate provider observability without retaining request payloads."""

    def __init__(self) -> None:
        self._operations = {
            operation: _OperationCounters()
            for operation in PROVIDER_OPERATIONS
        }

    def record_request(self, operation: str) -> None:
        self._operation(operation).request_count += 1

    def record_attempt(self, operation: str) -> None:
        self._operation(operation).attempt_count += 1

    def record_retry(self, operation: str, reason: str) -> None:
        values = self._operation(operation)
        values.retry_count += 1
        if reason == "transport":
            values.transport_retry_count += 1
        elif reason == "http_5xx":
            values.http_5xx_retry_count += 1

    def record_failure(
        self,
        operation: str,
        *,
        unavailable: bool,
    ) -> None:
        values = self._operation(operation)
        values.failed_request_count += 1
        if unavailable:
            values.provider_unavailable_failure_count += 1
        else:
            values.provider_response_failure_count += 1

    def record_usage(
        self,
        operation: str,
        usage: ProviderTokenUsage,
    ) -> None:
        if usage.total_tokens <= 0 and usage.input_tokens <= 0 and usage.output_tokens <= 0:
            return
        values = self._operation(operation)
        values.usage_report_count += 1
        values.input_tokens += max(usage.input_tokens, 0)
        values.output_tokens += max(usage.output_tokens, 0)
        values.total_tokens += max(usage.total_tokens, 0)

    def snapshot(self) -> ProviderMetricsSnapshot:
        return ProviderMetricsSnapshot(
            operations=tuple(
                ProviderOperationMetrics(
                    operation=operation,
                    request_count=self._operations[operation].request_count,
                    attempt_count=self._operations[operation].attempt_count,
                    retry_count=self._operations[operation].retry_count,
                    transport_retry_count=self._operations[operation].transport_retry_count,
                    http_5xx_retry_count=self._operations[operation].http_5xx_retry_count,
                    failed_request_count=self._operations[operation].failed_request_count,
                    provider_unavailable_failure_count=(
                        self._operations[operation].provider_unavailable_failure_count
                    ),
                    provider_response_failure_count=(
                        self._operations[operation].provider_response_failure_count
                    ),
                    usage_report_count=self._operations[operation].usage_report_count,
                    input_tokens=self._operations[operation].input_tokens,
                    output_tokens=self._operations[operation].output_tokens,
                    total_tokens=self._operations[operation].total_tokens,
                )
                for operation in PROVIDER_OPERATIONS
            )
        )

    def _operation(self, operation: str) -> _OperationCounters:
        values = self._operations.get(operation)
        if values is None:
            raise ValueError(f"unknown provider operation: {operation}")
        return values


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def metrics_tracker(self) -> ProviderMetricsTracker:
        """Expose aggregate call metrics for evaluation and diagnostics."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks in caller-provided order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query using the document embedding model."""

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int,
    ) -> list[RerankItem]:
        """Return original candidate indexes and provider scores."""

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        task: str,
        model: str | None = None,
        response_format: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Return a parsed JSON object for a workflow task."""


def subtract_provider_metrics(
    after: ProviderMetricsSnapshot,
    before: ProviderMetricsSnapshot,
) -> ProviderMetricsSnapshot:
    before_by_name = {item.operation: item for item in before.operations}
    differences: list[ProviderOperationMetrics] = []
    for after_item in after.operations:
        before_item = before_by_name.get(after_item.operation, _empty_operation(after_item.operation))
        differences.append(
            ProviderOperationMetrics(
                operation=after_item.operation,
                request_count=_nonnegative_delta(
                    after_item.request_count, before_item.request_count
                ),
                attempt_count=_nonnegative_delta(
                    after_item.attempt_count, before_item.attempt_count
                ),
                retry_count=_nonnegative_delta(
                    after_item.retry_count, before_item.retry_count
                ),
                transport_retry_count=_nonnegative_delta(
                    after_item.transport_retry_count,
                    before_item.transport_retry_count,
                ),
                http_5xx_retry_count=_nonnegative_delta(
                    after_item.http_5xx_retry_count,
                    before_item.http_5xx_retry_count,
                ),
                failed_request_count=_nonnegative_delta(
                    after_item.failed_request_count,
                    before_item.failed_request_count,
                ),
                provider_unavailable_failure_count=_nonnegative_delta(
                    after_item.provider_unavailable_failure_count,
                    before_item.provider_unavailable_failure_count,
                ),
                provider_response_failure_count=_nonnegative_delta(
                    after_item.provider_response_failure_count,
                    before_item.provider_response_failure_count,
                ),
                usage_report_count=_nonnegative_delta(
                    after_item.usage_report_count,
                    before_item.usage_report_count,
                ),
                input_tokens=_nonnegative_delta(
                    after_item.input_tokens, before_item.input_tokens
                ),
                output_tokens=_nonnegative_delta(
                    after_item.output_tokens, before_item.output_tokens
                ),
                total_tokens=_nonnegative_delta(
                    after_item.total_tokens, before_item.total_tokens
                ),
            )
        )
    return ProviderMetricsSnapshot(operations=tuple(differences))


def add_provider_metrics(
    left: ProviderMetricsSnapshot,
    right: ProviderMetricsSnapshot,
) -> ProviderMetricsSnapshot:
    right_by_name = {item.operation: item for item in right.operations}
    combined = [
        _add_operations(item, right_by_name.get(item.operation, _empty_operation(item.operation)))
        for item in left.operations
    ]
    for item in right.operations:
        if all(item.operation != existing.operation for existing in combined):
            combined.append(item)
    return ProviderMetricsSnapshot(operations=tuple(sorted(combined, key=lambda item: item.operation)))


def _empty_operation(operation: str) -> ProviderOperationMetrics:
    return ProviderOperationMetrics(
        operation=operation,
        request_count=0,
        attempt_count=0,
        retry_count=0,
        transport_retry_count=0,
        http_5xx_retry_count=0,
        failed_request_count=0,
        provider_unavailable_failure_count=0,
        provider_response_failure_count=0,
        usage_report_count=0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
    )


def _nonnegative_delta(after: int, before: int) -> int:
    return max(0, after - before)


def _add_operations(
    left: ProviderOperationMetrics,
    right: ProviderOperationMetrics,
) -> ProviderOperationMetrics:
    return ProviderOperationMetrics(
        operation=left.operation,
        request_count=left.request_count + right.request_count,
        attempt_count=left.attempt_count + right.attempt_count,
        retry_count=left.retry_count + right.retry_count,
        transport_retry_count=left.transport_retry_count + right.transport_retry_count,
        http_5xx_retry_count=left.http_5xx_retry_count + right.http_5xx_retry_count,
        failed_request_count=left.failed_request_count + right.failed_request_count,
        provider_unavailable_failure_count=(
            left.provider_unavailable_failure_count
            + right.provider_unavailable_failure_count
        ),
        provider_response_failure_count=(
            left.provider_response_failure_count
            + right.provider_response_failure_count
        ),
        usage_report_count=left.usage_report_count + right.usage_report_count,
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )
