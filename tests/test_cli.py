import errno
from types import SimpleNamespace

import pytest

from rag_app.cli import _serve, doctor
from rag_app.server import build_container
from tests.conftest import make_settings


def test_doctor_reports_initialized_data_plane(tmp_path):
    settings = make_settings(tmp_path)
    container = build_container(settings)
    container.close()
    payload = doctor(settings)
    assert payload["status"] == "ok"
    assert payload["counts"] == {
        "knowledge_bases": 0,
        "documents": 0,
        "chunks": 0,
        "vectors": 0,
        "keyword_index": 0,
    }


def test_serve_preflights_existing_data_and_closes_container(
    tmp_path,
    monkeypatch,
    capsys,
):
    settings = make_settings(tmp_path)
    initialized = build_container(settings)
    initialized.close()
    calls = {"configure": 0, "run": 0, "build": 0}
    container = SimpleNamespace(closed=0)

    def close():
        container.closed += 1

    container.close = close
    app = SimpleNamespace(state=SimpleNamespace(container=container))

    monkeypatch.setattr(
        "rag_app.cli.configure_logging",
        lambda level: calls.__setitem__("configure", calls["configure"] + 1),
    )
    monkeypatch.setattr(
        "rag_app.cli.doctor",
        lambda value: {
            "errors": [],
            "warnings": ["retired_vector_collection:test"],
        },
    )
    monkeypatch.setattr(
        "rag_app.cli.build_app",
        lambda *args, **kwargs: calls.__setitem__("build", calls["build"] + 1) or app,
    )

    def run(*args, **kwargs):
        calls["run"] += 1
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8010
        assert kwargs["log_config"] is None

    monkeypatch.setattr("rag_app.cli.run", run)

    result = _serve(
        settings,
        host="127.0.0.1",
        port=8010,
        frontend_dir=tmp_path / "missing-frontend",
    )

    assert result == 0
    assert calls == {"configure": 1, "run": 1, "build": 1}
    assert container.closed == 1
    output = capsys.readouterr().out
    assert "startup warning: retired_vector_collection:test" in output
    assert "frontend build is missing" in output


def test_serve_blocks_startup_diagnostics_errors(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    initialized = build_container(settings)
    initialized.close()
    build_calls = []
    monkeypatch.setattr(
        "rag_app.cli.doctor",
        lambda value: {"errors": ["sqlite_schema_version_mismatch"], "warnings": []},
    )
    monkeypatch.setattr(
        "rag_app.cli.build_app",
        lambda *args, **kwargs: build_calls.append(args),
    )

    result = _serve(
        settings,
        host="127.0.0.1",
        port=8010,
        frontend_dir=tmp_path,
    )

    assert result == 1
    assert build_calls == []


@pytest.mark.parametrize(
    "failure",
    [
        OSError(errno.EADDRINUSE, "address in use"),
        SystemExit(3),
    ],
)
def test_serve_handles_startup_failure_and_closes_container(
    tmp_path,
    monkeypatch,
    failure,
):
    settings = make_settings(tmp_path)
    container = SimpleNamespace(closed=0)

    def close():
        container.closed += 1

    container.close = close
    app = SimpleNamespace(state=SimpleNamespace(container=container))
    monkeypatch.setattr("rag_app.cli.doctor", lambda value: {"errors": [], "warnings": []})
    monkeypatch.setattr("rag_app.cli.build_app", lambda *args, **kwargs: app)

    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr("rag_app.cli.run", run)
    result = _serve(
        settings,
        host="127.0.0.1",
        port=8010,
        frontend_dir=tmp_path,
    )
    assert result == 1
    assert container.closed == 1
