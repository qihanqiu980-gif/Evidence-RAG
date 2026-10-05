import pytest

from rag_app.core.ingestion import MarkdownIngester
from rag_app.errors import IngestionError


def test_normalization_and_content_hash_ignore_line_endings():
    ingester = MarkdownIngester(max_bytes=100, chunk_size=80, chunk_overlap=10)
    unix = ingester.prepare("manual.md", b"# Title\nWiFi 6\n")
    windows = ingester.prepare("copy.md", b"# Title\r\nWiFi 6\r\n")
    assert unix.content_hash == windows.content_hash
    assert unix.normalized_content == "# Title\nWiFi 6\n"


def test_heading_paths_and_original_offsets_are_preserved():
    content = "Preamble\n# Router\n## Specs\nThe router supports WiFi 6.\n"
    ingester = MarkdownIngester(max_bytes=200, chunk_size=80, chunk_overlap=10)
    prepared = ingester.prepare("manual.md", content.encode("utf-8"))

    assert [item.heading_path for item in prepared.chunks] == [
        (),
        ("Router", "Specs"),
    ]
    for chunk in prepared.chunks:
        assert prepared.normalized_content[chunk.char_start : chunk.char_end] == chunk.content


def test_overflow_splitting_keeps_size_and_overlap():
    text = "".join(f"Sentence {index:02d} about WiFi specifications. " for index in range(20))
    ingester = MarkdownIngester(max_bytes=10_000, chunk_size=100, chunk_overlap=15)
    prepared = ingester.prepare("long.md", text.encode("utf-8"))

    assert len(prepared.chunks) > 1
    for chunk in prepared.chunks:
        assert len(chunk.content) <= 100
    for left, right in zip(prepared.chunks, prepared.chunks[1:], strict=False):
        assert right.char_start <= left.char_end - 10


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [
        ("manual.txt", b"text", "invalid_extension"),
        ("manual.md", b"x" * 11, "file_too_large"),
        ("manual.md", b"\xff", "invalid_utf8"),
        ("manual.md", b" \n", "empty_document"),
    ],
)
def test_validation_failures_are_classified(
    filename: str,
    content: bytes,
    code: str,
):
    ingester = MarkdownIngester(max_bytes=10, chunk_size=80, chunk_overlap=10)
    with pytest.raises(IngestionError) as error:
        ingester.prepare(filename, content)
    assert error.value.code == code
