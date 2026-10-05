import json
from pathlib import Path

from rag_app.evaluation import (
    evaluate_case,
    evaluation_payload,
    load_evaluation_cases,
    summarize_evaluation,
)
from rag_app.server import build_container
from tests.conftest import make_settings
from tests.fakes.provider import FakeProvider


def test_fixed_dataset_passes_fake_provider_thresholds(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = make_settings(tmp_path, RAG_APP_EMBEDDING_DIMENSION="8")
    provider = FakeProvider(settings.embedding_dimension)
    container = build_container(settings, provider)
    knowledge_base = container.manager.create("Evaluation")
    for path in sorted((root / "assets" / "demo-docs").glob("*.md")):
        container.manager.ingest(knowledge_base.id, path.name, path.read_bytes())

    cases = load_evaluation_cases(root / "eval" / "cases.jsonl")
    results = []
    workflow = container.workflow
    assert workflow is not None
    try:
        for case in cases:
            _configure_provider(provider, case)
            results.append(evaluate_case(workflow, case, [knowledge_base.id]))
    finally:
        container.close()

    summary = summarize_evaluation(results)
    assert len(cases) >= 12
    assert summary.retrieval_hit_rate >= 0.85
    assert summary.refusal_pass_rate == 1.0
    assert summary.overall_pass_rate == 1.0
    assert summary.scope_leak_count == 0
    assert all(result.duration_ms >= 0 for result in results)
    assert summary.total_duration_ms >= 0
    assert summary.estimated_cost is None
    assert summary.cost_currency == "CNY"
    assert summary.retry_count == 0
    assert summary.failed_request_count == 0
    payload = evaluation_payload(summary, results)
    json.dumps(payload, ensure_ascii=False)
    assert payload["summary"]["p95_case_duration_ms"] >= 0
    assert payload["summary"]["provider_metrics"]["operations"]


def _configure_provider(provider: FakeProvider, case) -> None:
    subquestions = (
        ["AX6000 Mesh 最多几台？", "官方零售价是多少？"]
        if case.expected_refusal == "partial"
        else [case.question]
    )
    answerable = case.expected_refusal != "all"
    provider.chat_response_queues = {
        "decompose": [
            {
                "standalone_question": case.question,
                "subquestions": [{"text": text} for text in subquestions],
            }
        ],
        "judge": [
            {
                "decisions": [
                    {
                        "subquestion_id": f"q{index}",
                        "answerable": answerable if index == 1 else False,
                        "evidence_ids": [1] if index == 1 and answerable else [],
                        "missing_information": "" if index == 1 else "价格",
                        "reason": "证据直接回答子问题" if index == 1 else "资料未提供价格",
                    }
                ]
            }
            for index in range(1, len(subquestions) + 1)
        ],
        "generate": [
            {
                "parts": [
                    {
                        "subquestion_id": "q1",
                        "answer": "资料已给出对应结论。 [1]",
                        "citations": [1],
                    }
                ]
            }
        ]
        if answerable
        else [],
        "validate": [
            {
                "validations": [
                    {
                        "subquestion_id": "q1",
                        "supported": True,
                        "complete": True,
                        "reason": "",
                    }
                ]
            }
        ]
        if answerable
        else [],
    }
