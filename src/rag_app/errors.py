"""Domain errors shared by storage, providers, and application layers."""

from __future__ import annotations


class RagAppError(Exception):
    """Base class for safe, domain-level errors."""

    code = "internal_error"


class ConfigurationError(RagAppError):
    code = "configuration_error"


class VectorConfigurationMismatch(RagAppError):
    code = "vector_configuration_mismatch"


class StorageInconsistentError(RagAppError):
    code = "storage_inconsistent"


class DuplicateKnowledgeBaseError(RagAppError):
    code = "duplicate_knowledge_base"


class DuplicateDocumentError(RagAppError):
    code = "duplicate_document"


class IngestionError(RagAppError):
    """A safe ingestion failure with a stable API error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class KnowledgeBaseNotFoundError(RagAppError):
    code = "not_found"


class DocumentNotFoundError(RagAppError):
    code = "not_found"


class ProviderUnavailableError(RagAppError):
    code = "provider_unavailable"


class ProviderResponseError(RagAppError):
    code = "provider_response_error"
