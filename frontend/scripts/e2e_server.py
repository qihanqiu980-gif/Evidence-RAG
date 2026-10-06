from __future__ import annotations

import argparse
import json
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


class E2EProvider(FakeProvider):
    def chat_json(self, messages, *, task, model=None, response_format=None):
        if task == "discovery_topics":
            payload = json.loads(messages[1]["content"])
            chunk_ids = [item["chunk_id"] for item in payload["chunks"][:2]]
            return {
                "topics": [
                    {
                        "title": "产品规格",
                        "type": "概念",
                        "summary": "无线能力",
                        "confidence": 0.9,
                        "source_chunk_ids": chunk_ids,
                    }
                ]
            }
        if task == "discovery_questions":
            payload = json.loads(messages[1]["content"])
            source_chunks = payload["topics"][0]["chunks"]
            return {
                "questions": [
                    {
                        "topic_id": "t1",
                        "question": "星云智联 AX6000 支持 WiFi 6 吗？",
                        "source_chunk_ids": [source_chunks[0]["chunk_id"]],
                    }
                ]
            }
        return super().chat_json(messages, task=task)


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
    provider = E2EProvider(settings.embedding_dimension)
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
