from __future__ import annotations

import argparse
import errno
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
from uvicorn import run

from .api.app import build_app
from .config import Settings
from .core.knowledge_base import KnowledgeBaseManager
from .errors import RagAppError
from .evaluation import (
    evaluate_case,
    evaluation_payload,
    load_evaluation_cases,
    summarize_evaluation,
)
from .logging import configure_logging
from .providers.online import OnlineModelProvider
from .server import build_container
from .storage.backup import create_backup, restore_backup
from .storage.chroma import ChromaVectorStore
from .storage.sqlite import SCHEMA_VERSION, SQLiteStore
from .storage.uploads import UploadStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Start the local workbench")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8010)

    doctor_parser = subparsers.add_parser("doctor", help="Check data-plane consistency")
    doctor_parser.add_argument("--json", action="store_true")

    rebuild_parser = subparsers.add_parser(
        "rebuild",
        help="Rebuild vectors from SQLite chunks (implemented in M2)",
    )
    rebuild_parser.add_argument("--yes", action="store_true")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run the online evaluation set (implemented with QA workflow)",
    )
    eval_parser.add_argument("--dataset", default="eval/cases.jsonl")
    eval_parser.add_argument("--top-k", type=int, default=7)
    eval_parser.add_argument("--kb-id", action="append", dest="kb_ids")
    eval_parser.add_argument("--json", action="store_true")

    backup_parser = subparsers.add_parser("backup", help="Snapshot the local data plane")
    backup_parser.add_argument("output", nargs="?", default="backups")

    restore_parser = subparsers.add_parser("restore", help="Restore a data-plane snapshot")
    restore_parser.add_argument("archive")
    restore_parser.add_argument("--yes", action="store_true")

    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    settings = Settings.from_env(root)

    if args.command == "backup":
        return _backup(settings, Path(args.output))
    if args.command == "restore":
        return _restore(settings, Path(args.archive), force=args.yes)
    if args.command == "serve":
        return _serve(settings, args.host, args.port, root / "frontend" / "dist")
    if args.command == "doctor":
        payload = doctor(settings)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            print(f"status: {payload['status']}")
            for issue in payload["errors"]:
                print(f"error: {issue}")
            for issue in payload["warnings"]:
                print(f"warning: {issue}")
            print(
                "counts: "
                f"knowledge_bases={payload['counts']['knowledge_bases']} "
                f"documents={payload['counts']['documents']} "
                f"chunks={payload['counts']['chunks']} "
                f"vectors={payload['counts']['vectors']}"
            )
        return 0 if payload["status"] == "ok" else 1
    if args.command == "rebuild":
        if not args.yes:
            print("rebuild changes the vector index; rerun with --yes to confirm")
            return 2
        if not settings.configured:
            print("rebuild requires a valid model provider configuration")
            return 2
        sqlite = SQLiteStore(settings.sqlite_path)
        uploads = UploadStore(settings.uploads_path)
        vectors = ChromaVectorStore(
            path=settings.chroma_path,
            sqlite=sqlite,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            allow_configuration_mismatch=True,
        )
        provider = OnlineModelProvider(settings)
        manager = KnowledgeBaseManager(
            settings=settings,
            sqlite=sqlite,
            uploads=uploads,
            vectors=vectors,
            provider=provider,
        )
        try:
            residues = manager.rebuild()
        finally:
            provider.close()
        _, _, chunk_count = sqlite.counts()
        print(f"rebuilt vectors: {chunk_count}")
        if residues:
            print("retired vector cleanup is incomplete; run rag-app doctor")
            return 1
        return 0
    if args.command == "eval":
        if not settings.configured:
            print("eval requires a valid model provider configuration")
            return 2
        if args.top_k <= 0:
            print("top-k must be positive")
            return 2
        dataset = Path(args.dataset)
        if not dataset.is_absolute():
            dataset = root / dataset
        try:
            cases = load_evaluation_cases(dataset)
        except (OSError, TypeError, ValueError) as error:
            print(f"unable to load evaluation dataset: {error}")
            return 2

        evaluation_settings = replace(settings, retrieval_top_k=args.top_k)
        container = build_container(evaluation_settings)
        try:
            available_ids = [item.id for item in container.manager.list_knowledge_bases()]
            selected_ids = args.kb_ids or available_ids
            unknown_ids = [item for item in selected_ids if item not in available_ids]
            if unknown_ids:
                print("evaluation scope contains unknown knowledge bases")
                return 2
            if not selected_ids:
                print("create a knowledge base and import documents before evaluation")
                return 2
            if container.workflow is None:
                print("model provider configuration is missing")
                return 2

            results = [
                evaluate_case(container.workflow, case, selected_ids)
                for case in cases
            ]
        finally:
            container.close()

        summary = summarize_evaluation(results)
        payload = evaluation_payload(summary, results)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            print(
                f"cases: {summary.case_count} "
                f"passed: {summary.overall_pass_count} "
                f"retrieval_hit_rate: {summary.retrieval_hit_rate:.2%} "
                f"refusal_pass_rate: {summary.refusal_pass_rate:.2%}"
            )
            cost = (
                f"{summary.estimated_cost:.8f} {summary.cost_currency}"
                if summary.estimated_cost is not None
                else "unavailable"
            )
            print(
                f"duration_ms: total={summary.total_duration_ms:.0f} "
                f"avg={summary.average_case_duration_ms:.0f} "
                f"p50={summary.p50_case_duration_ms:.0f} "
                f"p95={summary.p95_case_duration_ms:.0f}"
            )
            print(
                f"tokens: input={summary.input_tokens} "
                f"output={summary.output_tokens} total={summary.total_tokens}"
            )
            print(
                f"provider: retries={summary.retry_count} "
                f"failed_requests={summary.failed_request_count} "
                f"estimated_cost={cost}"
            )
            for result in results:
                status = "pass" if result.passed else "fail"
                retry_count = sum(
                    item.retry_count for item in result.provider_metrics.operations
                )
                failed_requests = sum(
                    item.failed_request_count
                    for item in result.provider_metrics.operations
                )
                print(
                    f"{status}: {result.case_id} "
                    f"retrieval={result.retrieval_hit} "
                    f"refusal={result.refusal_pass} "
                    f"evidence={result.evidence_count} "
                    f"duration={result.duration_ms:.0f}ms "
                    f"retries={retry_count} "
                    f"failed_requests={failed_requests}"
                )
        return 0
    raise AssertionError("unreachable command")


def _serve(
    settings: Settings,
    host: str,
    port: int,
    frontend_dir: Path,
) -> int:
    configure_logging(settings.log_level)
    diagnostics = _startup_diagnostics(settings)
    if diagnostics is not None and diagnostics["errors"]:
        print("startup diagnostics failed; run rag-app doctor for details")
        return 1
    if diagnostics is not None:
        for issue in diagnostics["warnings"]:
            print(f"startup warning: {issue}")
    if not (frontend_dir / "index.html").is_file():
        print("startup warning: frontend build is missing; API remains available")

    try:
        app = build_app(settings, frontend_dir=frontend_dir)
    except RagAppError as error:
        print(f"startup failed: {error.code}; run rag-app doctor for details")
        return 1
    try:
        run(app, host=host, port=port, log_config=None)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            print("startup failed: the local address is already in use")
        else:
            print("startup failed: unable to start the local service")
        return 1
    except SystemExit as error:
        if error.code == 0:
            return 0
        print(
            "startup failed: the local service exited before accepting requests; "
            "check whether the port is already in use"
        )
        return 1
    finally:
        container = getattr(app.state, "container", None)
        if container is not None:
            container.close()
    return 0


def _startup_diagnostics(settings: Settings) -> dict[str, Any] | None:
    if not settings.sqlite_path.is_file():
        print("startup: initializing an empty local data directory")
        return None
    try:
        return doctor(settings)
    except RagAppError as error:
        print(f"startup diagnostics failed: {error.code}")
        return {"errors": [error.code], "warnings": []}
    except Exception:  # noqa: BLE001
        print("startup diagnostics failed: storage_inconsistent")
        return {"errors": ["storage_inconsistent"], "warnings": []}


def doctor(settings: Settings) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    sqlite = SQLiteStore(settings.sqlite_path)
    kb_count, document_count, chunk_count = sqlite.counts()
    vector_count = 0
    active = sqlite.get_active_vector_collection()

    with sqlite.connect() as connection:
        schema_version = int(
            connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        )
    if schema_version != SCHEMA_VERSION:
        errors.append("sqlite_schema_version_mismatch")

    if active is None:
        errors.append("active_vector_collection_missing")
    else:
        if active.embedding_model != settings.embedding_model:
            errors.append("embedding_model_mismatch")
        if active.embedding_dimension != settings.embedding_dimension:
            errors.append("embedding_dimension_mismatch")
        try:
            client = chromadb.PersistentClient(path=str(settings.chroma_path))
            collection = client.get_collection(
                name=active.collection_name,
                embedding_function=None,
            )
        except NotFoundError:
            errors.append("chroma_collection_missing")
        except Exception:  # noqa: BLE001
            errors.append("chroma_collection_unreadable")
        else:
            metadata = dict(collection.metadata or {})
            if metadata.get("embedding_model") != settings.embedding_model:
                errors.append("chroma_embedding_model_mismatch")
            if metadata.get("embedding_dimension") != settings.embedding_dimension:
                errors.append("chroma_embedding_dimension_mismatch")
            if metadata.get("schema_version") != 1:
                errors.append("chroma_schema_version_mismatch")
            vector_count = int(collection.count())
            if vector_count != active.chunk_count:
                errors.append("vector_registry_count_mismatch")

    for retired in sqlite.retired_vector_collections():
        warnings.append(f"retired_vector_collection:{retired.id}")

    if chunk_count != vector_count:
        errors.append("sqlite_chunk_vector_count_mismatch")

    documents = [
        document
        for knowledge_base in sqlite.list_knowledge_bases()
        for document in sqlite.list_documents(knowledge_base.id)
    ]
    registered_paths: set[str] = set()
    for document in documents:
        registered_paths.add(document.storage_path)
        if not settings.uploads_path.joinpath(document.storage_path).is_file():
            errors.append(f"missing_upload:{document.id}")

    upload_store = UploadStore(settings.uploads_path)
    unregistered = sorted(set(upload_store.document_paths()) - registered_paths)
    for storage_path in unregistered:
        warnings.append(f"unregistered_upload:{storage_path}")

    return {
        "status": "error" if errors else ("warning" if warnings else "ok"),
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "knowledge_bases": kb_count,
            "documents": document_count,
            "chunks": chunk_count,
            "vectors": vector_count,
        },
    }


def _backup(settings: Settings, output: Path) -> int:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if output.name.endswith(".tar.gz"):
        target = output
    elif output.is_dir():
        target = output / f"rag-backup-{stamp}.tar.gz"
    else:
        target = output.parent / f"{output.name}-{stamp}.tar.gz"
    try:
        written = create_backup(settings, target)
    except RagAppError as error:
        print(f"backup failed: {error.code}")
        return 1
    print(f"backup written: {written}")
    return 0


def _restore(settings: Settings, archive: Path, *, force: bool) -> int:
    if not force:
        print("restore overwrites the current data directory; rerun with --yes to confirm")
        return 2
    try:
        result = restore_backup(settings, archive, force=True)
    except RagAppError as error:
        print(f"restore failed: {error.code}")
        return 1
    manifest = result.manifest
    print(f"restored data directory: {result.data_dir}")
    print(
        "counts: "
        f"knowledge_bases={manifest.counts['knowledge_bases']} "
        f"documents={manifest.counts['documents']} "
        f"chunks={manifest.counts['chunks']} "
        f"vectors={manifest.counts['vectors']}"
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
