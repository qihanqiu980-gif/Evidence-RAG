from __future__ import annotations

from collections.abc import Iterator

from .contracts import SseEvent


def encode_sse_event(event: SseEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def encode_sse_events(events: Iterator[SseEvent]) -> Iterator[str]:
    for event in events:
        yield encode_sse_event(event)


__all__ = ["encode_sse_event", "encode_sse_events"]
