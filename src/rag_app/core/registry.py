from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol

from ..errors import RagAppError
from .workflow import WorkflowEvent


class StreamingWorkflow(Protocol):
    def stream(
        self,
        question: str,
        history: Sequence[Mapping[str, str]],
        kb_ids: Sequence[str],
    ) -> Iterator[WorkflowEvent]:
        """Execute a named workflow and return ordered, safe events."""


class WorkflowRegistry:
    """Small replacement point for future non-evidence workflows."""

    def __init__(self) -> None:
        self._workflows: dict[str, StreamingWorkflow] = {}

    def register(self, name: str, workflow: StreamingWorkflow) -> None:
        self._workflows[name] = workflow

    def get(self, name: str) -> StreamingWorkflow:
        workflow = self._workflows.get(name)
        if workflow is None:
            raise WorkflowNotFoundError(f"Workflow {name} is not registered")
        return workflow

    def names(self) -> tuple[str, ...]:
        return tuple(self._workflows)


class WorkflowNotFoundError(RagAppError):
    code = "not_found"
