from __future__ import annotations

import pytest

from rag_app.core.adapters import DocumentAdapter
from rag_app.errors import IngestionError


def _make_text_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF carrying extractable text."""
    text_escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    content = f"BT /F1 12 Tf 72 720 Td ({text_escaped}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("latin-1")
        out += obj.encode("latin-1")
        out += b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


def test_pdf_extracts_text_and_chunks():
    adapter = DocumentAdapter(max_bytes=100_000, chunk_size=200, chunk_overlap=20)
    prepared = adapter.prepare("router.pdf", _make_text_pdf("Router supports WiFi 6."))
    assert prepared.filename == "router.pdf"
    assert "WiFi 6" in prepared.normalized_content
    assert prepared.chunks


def test_corrupt_pdf_is_rejected():
    adapter = DocumentAdapter(max_bytes=100_000, chunk_size=200, chunk_overlap=20)
    with pytest.raises(IngestionError) as error:
        adapter.prepare("broken.pdf", b"%PDF-1.4\nnot really a pdf")
    assert error.value.code in {"pdf_parse_failed", "empty_document"}
