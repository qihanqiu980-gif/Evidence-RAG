from __future__ import annotations

import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from ..errors import IngestionError
from .ingestion import MarkdownIngester, PlainTextIngester, PreparedDocument

SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf", ".docx", ".html", ".htm")


@dataclass(frozen=True, slots=True)
class AdapterResult:
    prepared: PreparedDocument


class DocumentAdapter:
    """Dispatch raw upload bytes to a format-specific text extractor.

    Every extractor returns ``(display_name, normalized_text)``. The shared
    ``PlainTextIngester`` then fingerprints and chunks the result, so all
    formats follow the same chunk/overlap/hash semantics as Markdown.
    """

    def __init__(self, max_bytes: int, chunk_size: int, chunk_overlap: int) -> None:
        self.max_bytes = max_bytes
        self.markdown = MarkdownIngester(max_bytes, chunk_size, chunk_overlap)
        self.plain = PlainTextIngester(chunk_size, chunk_overlap)
        self._extractors: dict[str, Callable[[str, bytes], tuple[str, str]]] = {
            ".md": self._extract_markdown,
            ".txt": self._extract_text,
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".html": self._extract_html,
            ".htm": self._extract_html,
        }

    def prepare(self, filename: str, content: bytes) -> PreparedDocument:
        display_name = Path(filename.strip()).name
        if not display_name:
            raise IngestionError("validation_error", "Filename cannot be empty")
        extension = Path(display_name).suffix.lower()
        if extension not in self._extractors:
            raise IngestionError(
                "invalid_extension",
                "Unsupported file type; supported: .md, .txt, .pdf, .docx, .html",
            )
        if len(content) > self.max_bytes:
            raise IngestionError("file_too_large", "The file exceeds the upload size limit")

        if extension == ".md":
            # Markdown keeps its heading-aware chunking and content hash.
            return self.markdown.prepare(display_name, content)

        extractor = self._extractors[extension]
        resolved_name, text = extractor(display_name, content)
        return self.plain.prepare(text, resolved_name)

    def _extract_markdown(self, filename: str, content: bytes) -> tuple[str, str]:
        prepared = self.markdown.prepare(filename, content)
        return prepared.filename, prepared.normalized_content

    def _extract_text(self, filename: str, content: bytes) -> tuple[str, str]:
        try:
            return filename, content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IngestionError("invalid_utf8", "The file must be UTF-8 encoded") from exc

    def _extract_pdf(self, filename: str, content: bytes) -> tuple[str, str]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - guarded for portability
            raise IngestionError(
                "unsupported_parser", "PDF parsing is unavailable"
            ) from exc

        try:
            from io import BytesIO

            reader = PdfReader(BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError("pdf_parse_failed", "Unable to parse the PDF document") from exc

        text = "\n\n".join(page.strip() for page in pages if page.strip())
        if not text:
            raise IngestionError("empty_document", "The PDF produced no extractable text")
        return filename, text

    def _extract_docx(self, filename: str, content: bytes) -> tuple[str, str]:
        try:
            with zipfile.ZipFile(_as_bytes_io(content)) as archive:
                xml = archive.read("word/document.xml")
        except (zipfile.BadZipFile, KeyError) as exc:
            raise IngestionError("docx_parse_failed", "Unable to parse the Word document") from exc

        try:
            text = _docx_to_text(xml)
        except Exception as exc:
            raise IngestionError("docx_parse_failed", "Unable to parse the Word document") from exc
        if not text.strip():
            raise IngestionError("empty_document", "The Word document produced no extractable text")
        return filename, text

    def _extract_html(self, filename: str, content: bytes) -> tuple[str, str]:
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IngestionError("invalid_utf8", "The file must be UTF-8 encoded") from exc
        parser = _HtmlTextParser()
        try:
            parser.feed(decoded)
        except Exception as exc:  # pragma: no cover - parser is lenient
            raise IngestionError("html_parse_failed", "Unable to parse the HTML document") from exc
        text = parser.text()
        if not text.strip():
            raise IngestionError("empty_document", "The HTML document produced no extractable text")
        return filename, text


def _as_bytes_io(content: bytes):
    from io import BytesIO

    return BytesIO(content)


_TAG_RE = re.compile(r"<[^>]+>")


class _HtmlTextParser(HTMLParser):
    """Collect visible text from HTML, honouring block-level line breaks."""

    _block_tags: frozenset[str] = frozenset({
        "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
        "tr", "section", "article", "blockquote", "table", "pre",
    })
    _skip_tags: frozenset[str] = frozenset(
        {"script", "style", "noscript", "head", "title", "meta", "link"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        elif tag in self._block_tags and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._block_tags and self._skip_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = _TAG_RE.sub("", raw)
        collapsed = re.sub(r"[ \t]+", " ", raw)
        lines = [line.strip() for line in collapsed.splitlines()]
        return "\n".join(line for line in lines if line).strip()


def _docx_to_text(document_xml: bytes) -> str:
    from xml.etree import ElementTree

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    root = ElementTree.fromstring(document_xml)
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        runs = [
            (node.text or "")
            for node in paragraph.iter(f"{namespace}t")
            if node.text
        ]
        text = "".join(runs)
        paragraphs.append(text)
    return "\n".join(paragraphs).strip()
