from __future__ import annotations

import json

from rag_app.core.retrieval import Evidence
from rag_app.core.workflow import EvidenceQAWorkflow
from rag_app.server import AppContainer, build_container
from tests.conftest import make_settings
from tests.fakes.provider import FakeProvider


def _workflow(
    tmp_path,
    *,
    fake: FakeProvider | None = None,
) -> tuple[EvidenceQAWorkflow, FakeProvider, AppContainer]:
    settings = make_settings(tmp_path)
    provider = fake or FakeProvider(settings.embedding_dimension)
    container = build_container(settings, provider)
    assert container.workflow is not None
    assert container.workflow_registry.names() == ("evidence_qa",)
    return container.workflow, provider, container


def _ingest_router(container: AppContainer) -> str:
    manager = container.manager
    kb = manager.create("Product Docs")
    manager.ingest(
        kb.id,
        "router.md",
        b"# Router\nThe router supports WiFi 6.\n",
    )
    return kb.id


def test_workflow_emits_safe_stages_and_verified_final_answer(tmp_path):
    workflow, fake, container = _workflow(tmp_path)
    kb_id = _ingest_router(container)
    fake.chat_response_queues = {
        "decompose": [
            {
                "standalone_question": "Does the router support WiFi 6?",
                "subquestions": [{"text": "Does the router support WiFi 6?"}],
            }
        ],
        "judge": [
            {
                "decisions": [
                    {
                        "subquestion_id": "q1",
                        "answerable": True,
                        "evidence_ids": [1],
                        "missing_information": "",
                        "reason": "证据直接回答子问题",
                    }
                ]
            }
        ],
        "generate": [
            {
                "parts": [
                    {
                        "subquestion_id": "q1",
                        "answer": "The router supports WiFi 6. [1]",
                        "citations": [1],
                    }
                ]
            }
        ],
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
        ],
    }

    events = list(
        workflow.stream(
            "Does the router support WiFi 6?",
            [],
            [kb_id],
        )
    )
    types = [item.type for item in events]

    assert types == [
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "stage_started",
        "evidence",
        "decision",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "answer_final",
        "completed",
    ]
    answer_index = types.index("answer_final")
    before_answer = json.dumps(
        [item.payload for item in events[:answer_index]],
        ensure_ascii=False,
    )
    assert "The router supports WiFi 6. [1]" not in before_answer

    evidence_event = next(item for item in events if item.type == "evidence")
    assert evidence_event.payload["evidence"][0]["support_status"] == "supporting"
    decision_event = next(item for item in events if item.type == "decision")
    assert decision_event.payload["decisions"][0]["evidence_ids"] == [1]

    final = events[answer_index].payload
    assert final["refused"] is False
    assert final["truncated"] is False
    assert final["parts"][0]["status"] == "verified"
    assert final["parts"][0]["citations"] == [1]
    assert final["answer"] == "**Does the router support WiFi 6?**\nThe router supports WiFi 6. [1]"


def test_invalid_decomposition_falls_back_and_six_questions_are_truncated(tmp_path):
    workflow, fake, _ = _workflow(tmp_path)
    fake.chat_responses = {
        "decompose": {
            "standalone_question": "",
            "subquestions": [],
        }
    }
    fallback = workflow.decompose("它支持 WiFi 6 吗？", [])
    assert fallback.standalone_question == "它支持 WiFi 6 吗？"
    assert [item.id for item in fallback.subquestions] == ["q1"]
    assert fallback.truncated is False

    fake.chat_responses = {
        "decompose": {
            "standalone_question": "路由器问题",
            "subquestions": [{"text": f"问题 {index}"} for index in range(1, 7)],
        }
    }
    truncated = workflow.decompose("路由器问题", [])
    assert len(truncated.subquestions) == 5
    assert [item.id for item in truncated.subquestions] == ["q1", "q2", "q3", "q4", "q5"]
    assert truncated.truncated is True


def test_unauthorized_citation_is_regenerated_once_and_can_recover(tmp_path):
    workflow, fake, container = _workflow(tmp_path)
    kb_id = _ingest_router(container)
    fake.chat_response_queues = {
        "decompose": [
            {
                "standalone_question": "Does it support WiFi 6?",
                "subquestions": [{"text": "Does it support WiFi 6?"}],
            }
        ],
        "judge": [
            {
                "decisions": [
                    {
                        "subquestion_id": "q1",
                        "answerable": True,
                        "evidence_ids": [1],
                        "missing_information": "",
                        "reason": "direct",
                    }
                ]
            }
        ],
        "generate": [
            {
                "parts": [
                    {
                        "subquestion_id": "q1",
                        "answer": "The router supports WiFi 6. [2]",
                        "citations": [2],
                    }
                ]
            },
            {
                "parts": [
                    {
                        "subquestion_id": "q1",
                        "answer": "The router supports WiFi 6. [1]",
                        "citations": [1],
                    }
                ]
            },
        ],
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
        ],
    }

    events = list(workflow.stream("Does it support WiFi 6?", [], [kb_id]))
    regenerate = next(item for item in events if item.stage == "regenerate")
    assert regenerate.type == "stage_started"
    final = next(item for item in events if item.type == "answer_final").payload
    assert final["parts"][0]["status"] == "verified"
    assert final["parts"][0]["citations"] == [1]


def test_second_number_validation_failure_becomes_local_refusal(tmp_path):
    workflow, fake, container = _workflow(tmp_path)
    kb_id = _ingest_router(container)
    invalid_part = {
        "parts": [
            {
                "subquestion_id": "q1",
                "answer": "The speed is 100 Mbps. [1]",
                "citations": [1],
            }
        ]
    }
    fake.chat_response_queues = {
        "decompose": [
            {
                "standalone_question": "What is the speed?",
                "subquestions": [{"text": "What is the speed?"}],
            }
        ],
        "judge": [
            {
                "decisions": [
                    {
                        "subquestion_id": "q1",
                        "answerable": True,
                        "evidence_ids": [1],
                        "missing_information": "",
                        "reason": "direct",
                    }
                ]
            }
        ],
        "generate": [invalid_part, invalid_part],
    }

    events = list(workflow.stream("What is the speed?", [], [kb_id]))
    final = next(item for item in events if item.type == "answer_final").payload

    assert final["refused"] is True
    assert final["parts"][0]["status"] == "refused"
    assert "100 Mbps" not in final["answer"]
    assert fake.chat_response_queues["generate"] == []


def test_judge_with_out_of_scope_evidence_id_refuses_conservatively(tmp_path):
    workflow, fake, container = _workflow(tmp_path)
    kb_id = _ingest_router(container)
    fake.chat_response_queues = {
        "decompose": [
            {
                "standalone_question": "Does it support WiFi 6?",
                "subquestions": [{"text": "Does it support WiFi 6?"}],
            }
        ],
        "judge": [
            {
                "decisions": [
                    {
                        "subquestion_id": "q1",
                        "answerable": True,
                        "evidence_ids": [1, 999],
                        "missing_information": "",
                        "reason": "invalid references",
                    }
                ]
            }
        ],
    }

    events = list(workflow.stream("Does it support WiFi 6?", [], [kb_id]))
    final = next(item for item in events if item.type == "answer_final").payload

    assert final["refused"] is True
    assert final["parts"][0]["refusal_reason"] == "找到相关资料，但缺少足以回答的直接证据"


def test_judge_normalises_compatible_qwen_shapes(tmp_path):
    workflow, _, _ = _workflow(tmp_path)
    evidence = Evidence(
        reference_id=1,
        kb_id="kb",
        document_id="doc",
        document_name="router.md",
        heading_path=("Spec",),
        chunk_index=0,
        content="The router supports WiFi 6.",
        similarity=0.9,
        rerank_score=0.9,
        subquestion_id="q1",
        subquestion="Does it support WiFi 6?",
    )
    raw_shapes = [
        {"decisions": {"subquestion_id": "q1", "answerable": True, "evidence_ids": [1]}},
        {
            "decisions": (
                '{"subquestion_id":"q1","answerable":"true",'
                '"evidence_ids":["1"]}'
            )
        },
        {"subquestion_id": "q1", "answerable": True, "evidence_ids": [1]},
    ]

    decisions = [
        workflow._parse_judge_decision(payload, "q1", (evidence,))[0]
        for payload in raw_shapes
    ]

    assert all(item.answerable for item in decisions)
    assert all(item.evidence_ids == (1,) for item in decisions)


def test_partial_evidence_refusal_keeps_answerable_section(tmp_path):
    workflow, fake, container = _workflow(tmp_path)
    kb_id = _ingest_router(container)
    fake.chat_response_queues = {
        "decompose": [
            {
                "standalone_question": "WiFi and price",
                "subquestions": [
                    {"text": "Does the router support WiFi 6?"},
                    {"text": "What is the price?"},
                ],
            }
        ],
        "judge": [
            {
                "decisions": [
                    {
                        "subquestion_id": "q1",
                        "answerable": True,
                        "evidence_ids": [1],
                        "missing_information": "",
                        "reason": "direct",
                    }
                ]
            },
            {
                "decisions": [
                    {
                        "subquestion_id": "q2",
                        "answerable": False,
                        "evidence_ids": [],
                        "missing_information": "价格",
                        "reason": "相关资料未包含价格",
                    }
                ]
            },
        ],
        "generate": [
            {
                "parts": [
                    {
                        "subquestion_id": "q1",
                        "answer": "The router supports WiFi 6. [1]",
                        "citations": [1],
                    }
                ]
            }
        ],
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
        ],
    }

    events = list(workflow.stream("WiFi 6 and price?", [], [kb_id]))
    evidence = next(item for item in events if item.type == "evidence").payload["evidence"]
    statuses = {item["subquestion_id"]: item["support_status"] for item in evidence}
    final = next(item for item in events if item.type == "answer_final").payload

    assert statuses["q1"] == "supporting"
    assert statuses["q2"] == "related"
    assert final["refused"] is False
    assert [item["status"] for item in final["parts"]] == ["verified", "refused"]
