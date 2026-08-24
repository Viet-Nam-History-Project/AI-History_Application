"""Lightweight release marker shared by FastAPI and local process guards."""

from pathlib import Path


_REVISION_FILE = Path(__file__).with_name("rag_revision.txt")


def source_rag_revision() -> str:
    """Read the revision requested by the code currently present on disk."""
    value = _REVISION_FILE.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"RAG revision file is empty: {_REVISION_FILE}")
    return value


RAG_REVISION = source_rag_revision()


def rag_restart_required() -> bool:
    """Detect a process that predates the revision currently on disk."""
    return source_rag_revision() != RAG_REVISION
