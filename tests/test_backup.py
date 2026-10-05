from __future__ import annotations

import pytest

from rag_app.errors import StorageInconsistentError
from rag_app.server import build_container
from rag_app.storage.backup import create_backup, restore_backup
from tests.conftest import make_settings
from tests.fakes.provider import FakeProvider


def _ingest(settings):
    fake = FakeProvider(settings.embedding_dimension)
    container = build_container(settings, fake)
    kb = container.manager.create("Product Docs")
    document = container.manager.ingest(
        kb.id,
        "router.md",
        b"# Router\nThe router supports WiFi 6.\n",
    )
    container.close()
    return kb, document


def test_backup_and_restore_round_trip_preserves_data_plane(tmp_path):
    settings = make_settings(tmp_path)
    kb, _document = _ingest(settings)

    archive = create_backup(settings, tmp_path / "snap.tar.gz")
    assert archive.is_file()

    # Corrupt the live data plane, then restore it.
    for path in (settings.sqlite_path, settings.uploads_path, settings.chroma_path):
        if path.is_dir():
            for child in path.iterdir():
                if child.is_file():
                    child.unlink()
        else:
            path.unlink(missing_ok=True)

    result = restore_backup(settings, archive, force=True)
    assert result.manifest.counts["documents"] == 1

    container = build_container(settings, FakeProvider(settings.embedding_dimension))
    try:
        docs = container.manager.list_documents(kb.id)
        assert [item.filename for item in docs] == ["router.md"]
        matches = container.vectors.search(
            FakeProvider(settings.embedding_dimension).embed_query("WiFi 6"),
            [kb.id],
            7,
        )
        assert len(matches) == 1
    finally:
        container.close()


def test_backup_refuses_missing_database(tmp_path):
    settings = make_settings(tmp_path)
    with pytest.raises(StorageInconsistentError):
        create_backup(settings, tmp_path / "snap.tar.gz")


def test_restore_refuses_to_overwrite_nonempty_data_dir(tmp_path):
    settings = make_settings(tmp_path)
    _ingest(settings)
    archive = create_backup(settings, tmp_path / "snap.tar.gz")

    with pytest.raises(StorageInconsistentError):
        restore_backup(settings, archive, force=False)


def test_restore_rejects_schema_mismatch(tmp_path):
    settings = make_settings(tmp_path)
    _ingest(settings)
    archive = create_backup(settings, tmp_path / "snap.tar.gz")

    import tarfile
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as staging:
        with tarfile.open(archive, "r:gz") as src:
            src.extractall(staging, filter="data")
        manifest_path = Path(staging) / "manifest.json"
        import json

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["schema_version"] = 999
        manifest_path.write_text(json.dumps(raw), encoding="utf-8")
        bad_archive = tmp_path / "bad.tar.gz"
        with tarfile.open(bad_archive, "w:gz") as dst:
            for child in Path(staging).iterdir():
                dst.add(child, arcname=child.name)

    with pytest.raises(StorageInconsistentError):
        restore_backup(settings, bad_archive, force=True)


def test_backup_archive_does_not_contain_env_secrets(tmp_path):
    settings = make_settings(tmp_path)
    _ingest(settings)
    archive = create_backup(settings, tmp_path / "snap.tar.gz")

    import tarfile

    names = set()
    with tarfile.open(archive, "r:gz") as src:
        names = {member.name for member in src.getmembers()}
    assert "manifest.json" in names
    assert "app.db" in names
    assert "uploads" in names
    assert "chroma" in names
    assert ".env" not in names
