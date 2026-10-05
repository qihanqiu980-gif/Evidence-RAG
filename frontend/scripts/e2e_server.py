from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

from rag_app.api.app import build_app
from rag_app.config import Settings
from tests.fakes.provider import FakeProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--frontend-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = args.data_dir or Path(tempfile.mkdtemp(prefix="rag-app-e2e-"))
    frontend_dir = args.frontend_dir or project_root / "frontend" / "dist"
    settings = Settings.from_env(
        project_root,
        env={
            "RAG_APP_API_KEY": "e2e-test-key",
            "RAG_APP_DATA_DIR": str(data_dir),
            "RAG_APP_EMBEDDING_DIMENSION": "8",
        },
    )
    provider = FakeProvider(settings.embedding_dimension)
    provider.chat_responses = {
        "decompose": {
            "standalone_question": "星云智联 AX6000 是否支持 WiFi 6？",
            "subquestions": [{"text": "星云智联 AX6000 是否支持 WiFi 6？"}],
        },
        "judge": {
            "decisions": [
                {
                    "subquestion_id": "q1",
                    "answerable": True,
                    "evidence_ids": [1],
                    "missing_information": "",
                    "reason": "证据直接说明 WiFi 6 支持",
                }
            ]
        },
        "generate": {
            "parts": [
                {
                    "subquestion_id": "q1",
                    "answer": "星云智联 AX6000 支持 WiFi 6 双频。 [1]",
                    "citations": [1],
                }
            ]
        },
        "validate": {
            "validations": [
                {
                    "subquestion_id": "q1",
                    "supported": True,
                    "complete": True,
                    "reason": "",
                }
            ]
        },
    }
    app = build_app(settings, provider, frontend_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
