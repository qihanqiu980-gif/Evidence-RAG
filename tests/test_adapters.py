"""文档 Adapter（PDF / Word / HTML / 纯文本）的行为测试。"""

from __future__ import annotations

import io
import zipfile
from xml.etree import ElementTree

import pytest

from rag_app.core.adapters import SUPPORTED_EXTENSIONS, DocumentAdapter
from rag_app.errors import IngestionError


def make_adapter() -> DocumentAdapter:
    return DocumentAdapter(max_bytes=10_000, chunk_size=800, chunk_overlap=100)


def make_docx(text: str) -> bytes:
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def tag(name: str) -> str:
        return f"{{{namespace}}}{name}"

    root = ElementTree.Element(tag("document"))
    body = ElementTree.SubElement(root, tag("body"))
    paragraph = ElementTree.SubElement(body, tag("p"))
    run = ElementTree.SubElement(paragraph, tag("r"))
    node = ElementTree.SubElement(run, tag("t"))
    node.text = text
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", ElementTree.tostring(root, encoding="utf-8"))
    return buffer.getvalue()


def make_pdf(text: str) -> bytes:
    # 一个极小的合法 PDF，包含可提取的文本流。
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(content)).encode() + b">>stream\n" + content + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    body = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{index} 0 obj".encode() + obj + b"endobj\n"
    xref = len(body)
    body += b"xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets:
        body += f"{offset:010d} 00000 n \n".encode()
    body += b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n" + str(xref).encode() + b"\n%%EOF\n"
    return bytes(body)


def test_supported_extensions_cover_target_formats():
    assert set(SUPPORTED_EXTENSIONS) == {".md", ".txt", ".pdf", ".docx", ".html", ".htm"}


def test_markdown_still_produces_heading_paths():
    prepared = make_adapter().prepare("manual.md", b"# Router\nWiFi 6\n")
    assert prepared.filename == "manual.md"
    assert [chunk.heading_path for chunk in prepared.chunks] == [("Router",)]


def test_txt_is_extracted_as_plain_text():
    prepared = make_adapter().prepare("notes.txt", "第一行\n第二行\n".encode())
    assert prepared.chunks
    assert prepared.chunks[0].heading_path == ()


def test_html_strips_tags_and_keeps_text():
    prepared = make_adapter().prepare(
        "page.html",
        "<html><head><title>忽略</title><style>.x{}</style></head>"
        "<body><h1>标题</h1><p>正文内容</p></body></html>".encode(),
    )
    text = prepared.normalized_content
    assert "正文内容" in text
    assert "标题" in text
    assert "忽略" not in text
    assert ".x{}" not in text


def test_docx_extracts_paragraph_text():
    prepared = make_adapter().prepare("manual.docx", make_docx("路由器支持 WiFi 6。"))
    assert "路由器支持 WiFi 6" in prepared.normalized_content


def test_pdf_extracts_text_stream():
    prepared = make_adapter().prepare("guide.pdf", make_pdf("Factory reset"))
    assert "Factory reset" in prepared.normalized_content


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [
        ("manual.xyz", b"data", "invalid_extension"),
        ("manual.pdf", b"not a pdf", "pdf_parse_failed"),
        ("manual.docx", b"not a zip", "docx_parse_failed"),
        ("empty.html", b"<html><body></body></html>", "empty_document"),
        ("big.pdf", b"x" * 10_001, "file_too_large"),
    ],
)
def test_adapter_validation_failures_are_classified(
    filename: str,
    content: bytes,
    code: str,
):
    adapter = make_adapter()
    with pytest.raises(IngestionError) as error:
        adapter.prepare(filename, content)
    assert error.value.code == code
