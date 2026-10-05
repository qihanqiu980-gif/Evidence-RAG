import pytest
from pydantic import ValidationError

from rag_app.api.contracts import SseEvent


def test_stage_events_require_a_stage():
    event = SseEvent(
        type="stage_started",
        stage="retrieve",
        message="开始检索子问题",
        sequence=1,
    )
    assert event.stage == "retrieve"


def test_stage_event_without_stage_is_rejected():
    with pytest.raises(ValidationError):
        SseEvent(
            type="stage_started",
            message="开始检索子问题",
            sequence=1,
        )
