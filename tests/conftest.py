from __future__ import annotations

from pathlib import Path

import pytest

from rag_app.config import Settings


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def settings(project_root: Path) -> Settings:
    return make_settings(project_root)


def make_settings(project_root: Path, **overrides: str) -> Settings:
    values = {
        "RAG_APP_API_KEY": "test-api-key",
        "RAG_APP_DATA_DIR": str(project_root / "data"),
    }
    values.update(overrides)
    return Settings.from_env(project_root, values)
