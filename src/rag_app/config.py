from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str
    embedding_dimension: int
    rerank_model: str
    rerank_url: str
    data_dir: Path
    upload_max_bytes: int
    upload_max_files: int
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    candidate_multiplier: int
    max_subquestions: int
    upload_max_workers: int
    request_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    log_level: str
    embedding_input_price_per_1k: float | None
    chat_input_price_per_1k: float | None
    chat_output_price_per_1k: float | None
    rerank_input_price_per_1k: float | None
    rerank_output_price_per_1k: float | None
    cost_currency: str

    @property
    def configured(self) -> bool:
        return bool(
            self.api_key.strip()
            and self.base_url.strip()
            and self.chat_model.strip()
            and self.embedding_model.strip()
            and self.rerank_model.strip()
            and self.rerank_url.strip()
            and self.embedding_dimension > 0
        )

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_path(self) -> Path:
        return self.data_dir / "uploads"

    @classmethod
    def from_env(
        cls,
        base_dir: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> Settings:
        root = Path(base_dir or Path.cwd()).resolve()
        source = env if env is not None else os.environ
        if env is None:
            load_dotenv(root / ".env", override=False)

        def text(name: str, default: str = "") -> str:
            return source.get(name, default).strip()

        def integer(name: str, default: int) -> int:
            try:
                return int(text(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be an integer") from exc

        def number(name: str, default: float) -> float:
            try:
                return float(text(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a number") from exc

        def optional_price(name: str) -> float | None:
            value = text(name)
            if not value:
                return None
            try:
                price = float(value)
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a number") from exc
            if not math.isfinite(price) or price < 0:
                raise ConfigurationError(f"{name} must be a non-negative finite number")
            return price

        data_dir_value = Path(text("RAG_APP_DATA_DIR", "data"))
        data_dir = data_dir_value if data_dir_value.is_absolute() else root / data_dir_value
        embedding_dimension = integer("RAG_APP_EMBEDDING_DIMENSION", 1024)
        upload_max_bytes = integer("RAG_APP_UPLOAD_MAX_BYTES", 5_242_880)
        upload_max_files = integer("RAG_APP_UPLOAD_MAX_FILES", 20)
        chunk_size = integer("RAG_APP_CHUNK_SIZE", 800)
        chunk_overlap = integer("RAG_APP_CHUNK_OVERLAP", 100)
        retrieval_top_k = integer("RAG_APP_RETRIEVAL_TOP_K", 7)
        candidate_multiplier = integer("RAG_APP_CANDIDATE_MULTIPLIER", 20)
        max_subquestions = integer("RAG_APP_MAX_SUBQUESTIONS", 5)
        upload_max_workers = integer("RAG_APP_UPLOAD_MAX_WORKERS", 2)
        request_timeout = number("RAG_APP_REQUEST_TIMEOUT_SECONDS", 60.0)
        max_retries = integer("RAG_APP_MAX_RETRIES", 1)
        retry_backoff = number("RAG_APP_RETRY_BACKOFF_SECONDS", 0.2)
        cost_currency = text("RAG_APP_COST_CURRENCY", "CNY").upper()
        if not re.fullmatch(r"[A-Z]{3}", cost_currency):
            raise ConfigurationError("RAG_APP_COST_CURRENCY must be a 3-letter currency code")

        if embedding_dimension <= 0:
            raise ConfigurationError("RAG_APP_EMBEDDING_DIMENSION must be positive")
        if upload_max_bytes <= 0:
            raise ConfigurationError("RAG_APP_UPLOAD_MAX_BYTES must be positive")
        if upload_max_files <= 0:
            raise ConfigurationError("RAG_APP_UPLOAD_MAX_FILES must be positive")
        if chunk_size <= 0:
            raise ConfigurationError("RAG_APP_CHUNK_SIZE must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ConfigurationError("RAG_APP_CHUNK_OVERLAP must be between 0 and chunk size")
        if retrieval_top_k <= 0:
            raise ConfigurationError("RAG_APP_RETRIEVAL_TOP_K must be positive")
        if candidate_multiplier <= 0:
            raise ConfigurationError("RAG_APP_CANDIDATE_MULTIPLIER must be positive")
        if not 1 <= max_subquestions <= 5:
            raise ConfigurationError("RAG_APP_MAX_SUBQUESTIONS must be between 1 and 5")
        if not 1 <= upload_max_workers <= 8:
            raise ConfigurationError("RAG_APP_UPLOAD_MAX_WORKERS must be between 1 and 8")
        if request_timeout <= 0:
            raise ConfigurationError("RAG_APP_REQUEST_TIMEOUT_SECONDS must be positive")
        if not 0 <= max_retries <= 5:
            raise ConfigurationError("RAG_APP_MAX_RETRIES must be between 0 and 5")
        if not math.isfinite(retry_backoff) or not 0 <= retry_backoff <= 10:
            raise ConfigurationError(
                "RAG_APP_RETRY_BACKOFF_SECONDS must be between 0 and 10 seconds"
            )

        return cls(
            api_key=text("RAG_APP_API_KEY"),
            base_url=text(
                "RAG_APP_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            chat_model=text("RAG_APP_CHAT_MODEL", "qwen3.7-plus"),
            embedding_model=text("RAG_APP_EMBEDDING_MODEL", "text-embedding-v4"),
            embedding_dimension=embedding_dimension,
            rerank_model=text("RAG_APP_RERANK_MODEL", "qwen3-rerank"),
            rerank_url=text(
                "RAG_APP_RERANK_URL",
                "https://maas.qianwenaiapi.com/compatible-api/v1/reranks",
            ),
            data_dir=data_dir.resolve(),
            upload_max_bytes=upload_max_bytes,
            upload_max_files=upload_max_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_top_k=retrieval_top_k,
            candidate_multiplier=candidate_multiplier,
            max_subquestions=max_subquestions,
            upload_max_workers=upload_max_workers,
            request_timeout_seconds=request_timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff,
            log_level=text("RAG_APP_LOG_LEVEL", "INFO").upper(),
            embedding_input_price_per_1k=optional_price(
                "RAG_APP_EMBEDDING_INPUT_PRICE_PER_1K"
            ),
            chat_input_price_per_1k=optional_price("RAG_APP_CHAT_INPUT_PRICE_PER_1K"),
            chat_output_price_per_1k=optional_price(
                "RAG_APP_CHAT_OUTPUT_PRICE_PER_1K"
            ),
            rerank_input_price_per_1k=optional_price(
                "RAG_APP_RERANK_INPUT_PRICE_PER_1K"
            ),
            rerank_output_price_per_1k=optional_price(
                "RAG_APP_RERANK_OUTPUT_PRICE_PER_1K"
            ),
            cost_currency=cost_currency,
        )
