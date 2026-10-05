from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import IngestionError

HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
NATURAL_BOUNDARIES = (
    "\n\n",
    "\n",
    "。",
    "；",
    "！",
    "？",
    ".",
    ";",
    "!",
    "?",
    "，",
    ",",
)
MARKDOWN_EXTENSION = ".md"


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    content: str
    heading_path: tuple[str, ...]
    chunk_index: int
    char_start: int
    char_end: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    filename: str
    normalized_content: str
    content_hash: str
    chunks: tuple[PreparedChunk, ...]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Split ``text`` into ``(start, end)`` ranges bounded by chunk_size/overlap.

    Prefers natural sentence/paragraph boundaries inside the window and falls
    back to a hard cut when a boundary cannot be found. Shared by every
    document adapter so Markdown and plain-text sources chunk consistently.
    """
    if len(text) <= chunk_size:
        return [(0, len(text))]

    ranges: list[tuple[int, int]] = []
    start = 0
    minimum_progress = max(1, min(chunk_size // 4, 32))
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        natural_end = _last_natural_end(text, start, hard_end)
        end = natural_end if natural_end is not None else hard_end
        if end - start < minimum_progress:
            end = hard_end
        ranges.append((start, end))
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return ranges


def _last_natural_end(text: str, start: int, end: int) -> int | None:
    window = text[start:end]
    candidates: list[int] = []
    for marker in NATURAL_BOUNDARIES:
        index = window.rfind(marker)
        if index >= 0:
            candidates.append(start + index + len(marker))
    if not candidates:
        return None
    return max(candidates)


class MarkdownIngester:
    """Validate, normalize, fingerprint, and heading-aware chunk Markdown."""

    def __init__(self, max_bytes: int, chunk_size: int, chunk_overlap: int) -> None:
        self.max_bytes = max_bytes
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def prepare(self, filename: str, content: bytes) -> PreparedDocument:
        display_name = _safe_filename(filename)
        if Path(display_name).suffix.lower() != MARKDOWN_EXTENSION:
            raise IngestionError("invalid_extension", "Only Markdown .md files are accepted")
        normalized = _decode_text(content, self.max_bytes)
        chunks: list[PreparedChunk] = []
        try:
            for section in self._sections(normalized):
                body = section.content
                stripped = body.strip()
                if not stripped:
                    continue
                body_start = section.char_start + (len(body) - len(body.lstrip()))
                for local_start, local_end in split_text(
                    stripped, self.chunk_size, self.chunk_overlap
                ):
                    chunk_content = stripped[local_start:local_end]
                    if not chunk_content.strip():
                        continue
                    chunks.append(
                        PreparedChunk(
                            content=chunk_content,
                            heading_path=section.heading_path,
                            chunk_index=len(chunks) + 1,
                            char_start=body_start + local_start,
                            char_end=body_start + local_end,
                            content_hash=_hash(chunk_content),
                        )
                    )
        except Exception as exc:
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError("chunking_failed", "Unable to split the document") from exc

        if not chunks:
            raise IngestionError("chunking_failed", "The document produced no chunks")
        return PreparedDocument(
            filename=display_name,
            normalized_content=normalized,
            content_hash=_hash(normalized),
            chunks=tuple(chunks),
        )

    def _sections(self, content: str) -> list[_Section]:
        sections: list[_Section] = []
        heading_path: list[str] = []
        section_start = 0
        section_end = 0
        cursor = 0

        def append_section(end: int) -> None:
            nonlocal section_end
            body = content[section_start:end]
            if body.strip():
                sections.append(_Section(tuple(heading_path), body, section_start, end))

        lines = content.splitlines(keepends=True)
        for line in lines:
            match = HEADING_PATTERN.match(line.rstrip("\n"))
            if match is None:
                if not sections and not heading_path and section_start == cursor:
                    section_start = cursor
                cursor += len(line)
                section_end = cursor
                continue

            if heading_path or section_end:
                append_section(cursor)
            level = len(match.group(1))
            heading_path[level - 1 :] = [match.group(2).strip()]
            cursor += len(line)
            section_start = cursor

        append_section(len(content))
        return sections


class PlainTextIngester:
    """Chunk already-normalized text without markdown heading semantics.

    Used by the PDF / DOCX / HTML adapters, which produce a single flat text
    stream. Chunk offsets are relative to the normalized text, mirroring the
    Markdown ingester's contract.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def prepare(self, normalized_content: str, filename: str) -> PreparedDocument:
        normalized = normalized_content.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
        if not normalized.strip():
            raise IngestionError("empty_document", "The document is empty")

        chunks: list[PreparedChunk] = []
        try:
            for local_start, local_end in split_text(
                normalized.strip(), self.chunk_size, self.chunk_overlap
            ):
                chunk_content = normalized[local_start:local_end]
                if not chunk_content.strip():
                    continue
                chunks.append(
                    PreparedChunk(
                        content=chunk_content,
                        heading_path=(),
                        chunk_index=len(chunks) + 1,
                        char_start=local_start,
                        char_end=local_end,
                        content_hash=_hash(chunk_content),
                    )
                )
        except Exception as exc:
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError("chunking_failed", "Unable to split the document") from exc

        if not chunks:
            raise IngestionError("chunking_failed", "The document produced no chunks")
        return PreparedDocument(
            filename=_safe_filename(filename),
            normalized_content=normalized,
            content_hash=_hash(normalized),
            chunks=tuple(chunks),
        )


@dataclass(frozen=True, slots=True)
class _Section:
    heading_path: tuple[str, ...]
    content: str
    char_start: int
    char_end: int


def _decode_text(content: bytes, max_bytes: int) -> str:
    if len(content) > max_bytes:
        raise IngestionError("file_too_large", "The file exceeds the upload size limit")
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError("invalid_utf8", "The file must be UTF-8 encoded") from exc
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    if not normalized.strip():
        raise IngestionError("empty_document", "The document is empty")
    return normalized


def _safe_filename(filename: str) -> str:
    name = filename.strip()
    if not name:
        raise IngestionError("validation_error", "Filename cannot be empty")
    return Path(name).name


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
