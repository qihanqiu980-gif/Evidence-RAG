from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config import Settings
from .core.workflow import EvidenceQAWorkflow
from .errors import RagAppError
from .providers.base import (
    ProviderMetricsSnapshot,
    add_provider_metrics,
    subtract_provider_metrics,
)


class RefusalExpectation(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    question: str
    history: tuple[dict[str, str], ...]
    expected_document_prefixes: tuple[str, ...]
    expected_refusal: RefusalExpectation


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    case_id: str
    passed: bool
    retrieval_hit: bool | None
    refusal_pass: bool
    scope_leak_count: int
    evidence_count: int
    supporting_evidence_count: int
    verified_part_count: int
    refused_part_count: int
    error_code: str | None
    duration_ms: float
    provider_metrics: ProviderMetricsSnapshot
    estimated_cost: float | None
    cost_currency: str


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    case_count: int
    retrieval_applicable_count: int
    retrieval_hit_count: int
    refusal_pass_count: int
    overall_pass_count: int
    retrieval_hit_rate: float
    refusal_pass_rate: float
    overall_pass_rate: float
    scope_leak_count: int
    total_duration_ms: float
    average_case_duration_ms: float
    p50_case_duration_ms: float
    p95_case_duration_ms: float
    provider_metrics: ProviderMetricsSnapshot
    estimated_cost: float | None
    cost_currency: str
    retry_count: int
    failed_request_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    failed_case_count: int


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            case = _parse_case(raw, path, line_number)
            if case.id in seen_ids:
                raise ValueError(f"duplicate evaluation id: {case.id}")
            seen_ids.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"evaluation dataset is empty: {path}")
    return tuple(cases)


def evaluate_case(
    workflow: EvidenceQAWorkflow,
    case: EvaluationCase,
    kb_ids: Sequence[str],
) -> EvaluationCaseResult:
    evidence: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    error_code: str | None = None
    started = time.perf_counter()
    metrics_before = workflow.provider.metrics_tracker.snapshot()

    try:
        for event in workflow.stream(case.question, case.history, kb_ids):
            if event.type == "evidence":
                raw_evidence = event.payload.get("evidence")
                if isinstance(raw_evidence, list):
                    evidence = [item for item in raw_evidence if isinstance(item, dict)]
            elif event.type == "answer_final":
                final = event.payload
            elif event.type == "error":
                raw_code = event.payload.get("code")
                error_code = raw_code if isinstance(raw_code, str) else "internal_error"
    except RagAppError as error:
        error_code = error.code
    duration_ms = (time.perf_counter() - started) * 1000
    provider_metrics = subtract_provider_metrics(
        workflow.provider.metrics_tracker.snapshot(),
        metrics_before,
    )

    scope = set(kb_ids)
    scope_leak_count = sum(
        1
        for item in evidence
        if isinstance(item.get("kb_id"), str) and item["kb_id"] not in scope
    )
    retrieval_hit: bool | None = None
    if case.expected_document_prefixes:
        retrieval_hit = any(
            _matches_prefix(item.get("document_name"), case.expected_document_prefixes)
            for item in evidence
        )

    parts = final.get("parts") if final is not None else None
    part_statuses = (
        [item.get("status") for item in parts if isinstance(item, dict)]
        if isinstance(parts, list)
        else []
    )
    verified_count = sum(status == "verified" for status in part_statuses)
    refused_count = sum(status == "refused" for status in part_statuses)
    refusal_pass = _refusal_matches(
        case.expected_refusal,
        refused_all=bool(final.get("refused")) if final is not None else False,
        verified_count=verified_count,
        refused_count=refused_count,
    )
    supporting_count = sum(
        item.get("support_status") == "supporting" for item in evidence
    )
    passed = (
        error_code is None
        and scope_leak_count == 0
        and retrieval_hit is not False
        and refusal_pass
    )

    return EvaluationCaseResult(
        case_id=case.id,
        passed=passed,
        retrieval_hit=retrieval_hit,
        refusal_pass=refusal_pass,
        scope_leak_count=scope_leak_count,
        evidence_count=len(evidence),
        supporting_evidence_count=supporting_count,
        verified_part_count=verified_count,
        refused_part_count=refused_count,
        error_code=error_code,
        duration_ms=round(duration_ms, 3),
        provider_metrics=provider_metrics,
        estimated_cost=estimate_provider_cost(provider_metrics, workflow.settings),
        cost_currency=workflow.settings.cost_currency,
    )


def summarize_evaluation(
    results: Sequence[EvaluationCaseResult],
) -> EvaluationSummary:
    retrieval_applicable = [item for item in results if item.retrieval_hit is not None]
    retrieval_hits = sum(item.retrieval_hit is True for item in retrieval_applicable)
    refusal_passes = sum(item.refusal_pass for item in results)
    overall_passes = sum(item.passed for item in results)
    case_count = len(results)
    retrieval_count = len(retrieval_applicable)
    durations = sorted(item.duration_ms for item in results)
    provider_metrics = _combined_metrics(results)
    estimated_cost = (
        sum(
            item.estimated_cost
            for item in results
            if item.estimated_cost is not None
        )
        if all(item.estimated_cost is not None for item in results)
        else None
    )
    return EvaluationSummary(
        case_count=case_count,
        retrieval_applicable_count=retrieval_count,
        retrieval_hit_count=retrieval_hits,
        refusal_pass_count=refusal_passes,
        overall_pass_count=overall_passes,
        retrieval_hit_rate=retrieval_hits / retrieval_count if retrieval_count else 1.0,
        refusal_pass_rate=refusal_passes / case_count if case_count else 0.0,
        overall_pass_rate=overall_passes / case_count if case_count else 0.0,
        scope_leak_count=sum(item.scope_leak_count for item in results),
        total_duration_ms=round(sum(durations), 3),
        average_case_duration_ms=round(sum(durations) / case_count, 3)
        if case_count
        else 0.0,
        p50_case_duration_ms=round(_percentile(durations, 0.50), 3),
        p95_case_duration_ms=round(_percentile(durations, 0.95), 3),
        provider_metrics=provider_metrics,
        estimated_cost=round(estimated_cost, 8)
        if estimated_cost is not None
        else None,
        cost_currency=results[0].cost_currency if results else "",
        retry_count=sum(item.retry_count for item in provider_metrics.operations),
        failed_request_count=sum(
            item.failed_request_count for item in provider_metrics.operations
        ),
        input_tokens=sum(item.input_tokens for item in provider_metrics.operations),
        output_tokens=sum(item.output_tokens for item in provider_metrics.operations),
        total_tokens=sum(item.total_tokens for item in provider_metrics.operations),
        failed_case_count=case_count - overall_passes,
    )


def evaluation_payload(
    summary: EvaluationSummary,
    results: Sequence[EvaluationCaseResult],
) -> dict[str, Any]:
    return {
        "summary": {
            "case_count": summary.case_count,
            "retrieval_applicable_count": summary.retrieval_applicable_count,
            "retrieval_hit_count": summary.retrieval_hit_count,
            "refusal_pass_count": summary.refusal_pass_count,
            "overall_pass_count": summary.overall_pass_count,
            "retrieval_hit_rate": round(summary.retrieval_hit_rate, 4),
            "refusal_pass_rate": round(summary.refusal_pass_rate, 4),
            "overall_pass_rate": round(summary.overall_pass_rate, 4),
            "scope_leak_count": summary.scope_leak_count,
            "total_duration_ms": summary.total_duration_ms,
            "average_case_duration_ms": summary.average_case_duration_ms,
            "p50_case_duration_ms": summary.p50_case_duration_ms,
            "p95_case_duration_ms": summary.p95_case_duration_ms,
            "estimated_cost": summary.estimated_cost,
            "cost_currency": summary.cost_currency,
            "retry_count": summary.retry_count,
            "failed_request_count": summary.failed_request_count,
            "input_tokens": summary.input_tokens,
            "output_tokens": summary.output_tokens,
            "total_tokens": summary.total_tokens,
            "failed_case_count": summary.failed_case_count,
            "provider_metrics": dataclass_asdict(summary.provider_metrics),
        },
        "cases": [dataclass_asdict(item) for item in results],
    }


def estimate_provider_cost(
    metrics: ProviderMetricsSnapshot,
    settings: Settings,
) -> float | None:
    cost = 0.0
    for item in metrics.operations:
        successful_requests = item.request_count - item.failed_request_count
        if successful_requests <= 0:
            continue
        if item.usage_report_count < successful_requests:
            return None
        if item.total_tokens > item.input_tokens + item.output_tokens:
            return None

        if item.operation == "embedding":
            if item.input_tokens > 0:
                price = settings.embedding_input_price_per_1k
                if price is None:
                    return None
                cost += item.input_tokens / 1000 * price
            continue

        if item.operation == "chat":
            input_price = settings.chat_input_price_per_1k
            output_price = settings.chat_output_price_per_1k
        elif item.operation == "rerank":
            input_price = settings.rerank_input_price_per_1k
            output_price = settings.rerank_output_price_per_1k
        else:
            return None

        if item.input_tokens > 0 and input_price is None:
            return None
        if item.output_tokens > 0 and output_price is None:
            return None
        cost += item.input_tokens / 1000 * (input_price or 0.0)
        cost += item.output_tokens / 1000 * (output_price or 0.0)

    return cost


def _combined_metrics(
    results: Sequence[EvaluationCaseResult],
) -> ProviderMetricsSnapshot:
    combined = ProviderMetricsSnapshot(operations=())
    for result in results:
        combined = add_provider_metrics(combined, result.provider_metrics)
    return combined


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    rank = math.ceil(quantile * len(values))
    return values[max(0, min(rank, len(values)) - 1)]


def _parse_case(
    raw: Any,
    path: Path,
    line_number: int,
) -> EvaluationCase:
    if not isinstance(raw, dict):
        raise TypeError(f"evaluation case must be an object at {path}:{line_number}")
    case_id = raw.get("id")
    question = raw.get("question")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"evaluation id is required at {path}:{line_number}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"evaluation question is required at {path}:{line_number}")

    prefixes_raw = raw.get("expected_document_prefixes", [])
    if not isinstance(prefixes_raw, list) or any(
        not isinstance(item, str) or not item for item in prefixes_raw
    ):
        raise ValueError(f"invalid expected_document_prefixes at {path}:{line_number}")
    refusal_raw = raw.get("expected_refusal", "none")
    try:
        refusal = RefusalExpectation(refusal_raw)
    except ValueError as exc:
        raise ValueError(f"invalid expected_refusal at {path}:{line_number}") from exc

    history_raw = raw.get("history", [])
    if not isinstance(history_raw, list):
        raise TypeError(f"invalid history at {path}:{line_number}")
    history: list[dict[str, str]] = []
    for item in history_raw:
        if (
            not isinstance(item, dict)
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
        ):
            raise ValueError(f"invalid history item at {path}:{line_number}")
        history.append({"role": item["role"], "content": item["content"]})

    return EvaluationCase(
        id=case_id.strip(),
        question=question.strip(),
        history=tuple(history),
        expected_document_prefixes=tuple(prefixes_raw),
        expected_refusal=refusal,
    )


def _matches_prefix(value: Any, prefixes: Sequence[str]) -> bool:
    return isinstance(value, str) and any(value.startswith(prefix) for prefix in prefixes)


def _refusal_matches(
    expectation: RefusalExpectation,
    *,
    refused_all: bool,
    verified_count: int,
    refused_count: int,
) -> bool:
    if expectation is RefusalExpectation.ALL:
        return refused_all and verified_count == 0 and refused_count > 0
    if expectation is RefusalExpectation.PARTIAL:
        return not refused_all and verified_count > 0 and refused_count > 0
    return not refused_all and verified_count > 0 and refused_count == 0
