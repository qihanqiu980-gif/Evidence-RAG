from __future__ import annotations

import logging

SENSITIVE_FIELDS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "password",
        "token",
        "RAG_APP_API_KEY",
    }
)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def safe_event_fields(**fields: object) -> dict[str, object]:
    """Return log fields with credentials and raw request bodies removed."""
    return {
        key: value
        for key, value in fields.items()
        if key.lower() not in SENSITIVE_FIELDS
        and key not in SENSITIVE_FIELDS
        and key not in {"prompt", "response", "question", "content"}
    }
