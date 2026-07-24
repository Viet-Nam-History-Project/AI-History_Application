"""Durable checkpoints for resumable PDF indexing jobs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .graph_extraction import ChunkGraphExtraction, TokenUsage
from .pdf_ingestion import ParsedPdf, PdfChunk, PdfPage


def _now() -> str:
    return datetime.now(UTC).isoformat()


class IndexJobPaused(Exception):
    """Raised at a safe checkpoint when an administrator stops a job."""


class IndexCheckpointStore:
    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS index_sessions (
                    source_id TEXT PRIMARY KEY,
                    document_hash TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    entity_model TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL DEFAULT 'queued',
                    total_chunks INTEGER NOT NULL DEFAULT 0,
                    completed_chunks INTEGER NOT NULL DEFAULT 0,
                    current_page INTEGER NOT NULL DEFAULT 0,
                    embedding_input_tokens INTEGER NOT NULL DEFAULT 0,
                    entity_input_tokens INTEGER NOT NULL DEFAULT 0,
                    entity_output_tokens INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    result_json TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS index_sessions_job_id
                ON index_sessions(job_id);

                CREATE TABLE IF NOT EXISTS parsed_documents (
                    source_id TEXT PRIMARY KEY,
                    document_hash TEXT NOT NULL,
                    parsed_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS checkpoint_chunks (
                    source_id TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    embedding_json TEXT,
                    graph_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source_id, content_hash)
                );
                """
            )

    def recover_interrupted_jobs(self) -> None:
        """A process restart turns in-flight jobs into resumable paused jobs."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_sessions
                SET status = 'paused',
                    phase = CASE
                        WHEN phase = 'queued' THEN 'paused'
                        ELSE phase
                    END,
                    error = '',
                    updated_at = ?
                WHERE status IN ('queued', 'running', 'stopping')
                """,
                (_now(),),
            )

    def active_job_count(self) -> int:
        """Return jobs that must reach a checkpoint before a service restart."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT count(*) AS total
                FROM index_sessions
                WHERE status IN ('queued', 'running', 'stopping')
                """
            ).fetchone()
        return int(row["total"] if row else 0)

    def begin_job(
        self,
        *,
        source_id: str,
        document_hash: str,
        pipeline_version: str,
        embedding_model: str,
        entity_model: str,
        job_id: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM index_sessions WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            compatible = bool(
                existing
                and existing["document_hash"] == document_hash
                and existing["pipeline_version"] == pipeline_version
                and existing["embedding_model"] == embedding_model
                and existing["entity_model"] == entity_model
            )
            if not compatible:
                connection.execute(
                    "DELETE FROM checkpoint_chunks WHERE source_id = ?",
                    (source_id,),
                )
                connection.execute(
                    "DELETE FROM parsed_documents WHERE source_id = ?",
                    (source_id,),
                )
                connection.execute(
                    "DELETE FROM index_sessions WHERE source_id = ?",
                    (source_id,),
                )
                existing = None

            reset_usage = existing is None or existing["status"] == "ready"
            embedding_tokens = (
                0 if reset_usage else int(existing["embedding_input_tokens"])
            )
            entity_input_tokens = (
                0 if reset_usage else int(existing["entity_input_tokens"])
            )
            entity_output_tokens = (
                0 if reset_usage else int(existing["entity_output_tokens"])
            )
            connection.execute(
                """
                INSERT INTO index_sessions (
                    source_id, document_hash, pipeline_version,
                    embedding_model, entity_model, job_id, status, phase,
                    total_chunks, completed_chunks, current_page,
                    embedding_input_tokens, entity_input_tokens,
                    entity_output_tokens, error, result_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 'queued', 0, 0, 0,
                          ?, ?, ?, '', NULL, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    job_id = excluded.job_id,
                    status = 'queued',
                    phase = 'queued',
                    total_chunks = 0,
                    completed_chunks = 0,
                    current_page = 0,
                    embedding_input_tokens = excluded.embedding_input_tokens,
                    entity_input_tokens = excluded.entity_input_tokens,
                    entity_output_tokens = excluded.entity_output_tokens,
                    error = '',
                    result_json = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    source_id,
                    document_hash,
                    pipeline_version,
                    embedding_model,
                    entity_model,
                    job_id,
                    embedding_tokens,
                    entity_input_tokens,
                    entity_output_tokens,
                    _now(),
                ),
            )
            connection.execute("COMMIT")

    def set_progress(
        self,
        source_id: str,
        *,
        status: str | None = None,
        phase: str | None = None,
        total_chunks: int | None = None,
        completed_chunks: int | None = None,
        current_page: int | None = None,
        error: str | None = None,
    ) -> None:
        updates: list[str] = ["updated_at = ?"]
        values: list[Any] = [_now()]
        for column, value in (
            ("status", status),
            ("phase", phase),
            ("total_chunks", total_chunks),
            ("completed_chunks", completed_chunks),
            ("current_page", current_page),
            ("error", error),
        ):
            if value is not None:
                updates.append(f"{column} = ?")
                values.append(value)
        values.append(source_id)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE index_sessions SET {', '.join(updates)} "
                "WHERE source_id = ?",
                values,
            )

    def save_parsed_pdf(
        self,
        source_id: str,
        document_hash: str,
        parsed_pdf: ParsedPdf,
    ) -> None:
        payload = json.dumps(asdict(parsed_pdf), ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO parsed_documents (
                    source_id, document_hash, parsed_json, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    document_hash = excluded.document_hash,
                    parsed_json = excluded.parsed_json,
                    updated_at = excluded.updated_at
                """,
                (source_id, document_hash, payload, _now()),
            )

    def load_parsed_pdf(
        self,
        source_id: str,
        document_hash: str,
    ) -> ParsedPdf | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT parsed_json FROM parsed_documents
                WHERE source_id = ? AND document_hash = ?
                """,
                (source_id, document_hash),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["parsed_json"])
        return ParsedPdf(
            page_count=int(payload["page_count"]),
            text_page_count=int(payload["text_page_count"]),
            content_start_page=int(payload["content_start_page"]),
            content_end_page=int(payload["content_end_page"]),
            skipped_page_count=int(payload["skipped_page_count"]),
            boundary_detection=str(payload["boundary_detection"]),
            footnote_page_count=int(payload["footnote_page_count"]),
            removed_footnote_chars=int(payload["removed_footnote_chars"]),
            running_header_page_count=int(payload["running_header_page_count"]),
            removed_running_header_chars=int(
                payload["removed_running_header_chars"]
            ),
            chunks=[
                PdfChunk(
                    **{
                        **item,
                        "facets": tuple(item.get("facets", [])),
                    }
                )
                for item in payload["chunks"]
            ],
            pages=[PdfPage(**item) for item in payload["pages"]],
            duplicate_chunk_count=int(payload["duplicate_chunk_count"]),
            ocr_applied=bool(payload["ocr_applied"]),
            ocr_languages=str(payload["ocr_languages"]),
            ocr_engine=str(payload["ocr_engine"]),
            ocr_failed_pages=[
                int(value) for value in payload["ocr_failed_pages"]
            ],
        )

    def load_chunk_results(
        self,
        source_id: str,
        document_hash: str,
    ) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT content_hash, embedding_json, graph_json
                FROM checkpoint_chunks
                WHERE source_id = ? AND document_hash = ?
                """,
                (source_id, document_hash),
            ).fetchall()
        return {
            str(row["content_hash"]): {
                "embedding": (
                    json.loads(row["embedding_json"])
                    if row["embedding_json"] is not None
                    else None
                ),
                "graph": (
                    json.loads(row["graph_json"])
                    if row["graph_json"] is not None
                    else None
                ),
            }
            for row in rows
        }

    def save_embeddings(
        self,
        source_id: str,
        document_hash: str,
        chunks: Iterable[PdfChunk],
        embeddings: Iterable[list[float]],
        usage: TokenUsage,
    ) -> None:
        rows = [
            (
                source_id,
                document_hash,
                chunk.content_hash,
                chunk.page_start,
                json.dumps(embedding, separators=(",", ":")),
                _now(),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                """
                INSERT INTO checkpoint_chunks (
                    source_id, document_hash, content_hash, page_number,
                    embedding_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, content_hash) DO UPDATE SET
                    document_hash = excluded.document_hash,
                    page_number = excluded.page_number,
                    embedding_json = excluded.embedding_json,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
            connection.execute(
                """
                UPDATE index_sessions
                SET embedding_input_tokens = embedding_input_tokens + ?,
                    updated_at = ?
                WHERE source_id = ?
                """,
                (usage.input_tokens, _now(), source_id),
            )
            connection.execute("COMMIT")

    def save_graph(
        self,
        source_id: str,
        document_hash: str,
        chunks: Iterable[PdfChunk],
        extractions: Iterable[ChunkGraphExtraction],
        usage: TokenUsage,
    ) -> None:
        rows = [
            (
                source_id,
                document_hash,
                chunk.content_hash,
                chunk.page_start,
                json.dumps(
                    extraction.as_row(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                _now(),
            )
            for chunk, extraction in zip(chunks, extractions, strict=True)
        ]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                """
                INSERT INTO checkpoint_chunks (
                    source_id, document_hash, content_hash, page_number,
                    graph_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, content_hash) DO UPDATE SET
                    document_hash = excluded.document_hash,
                    page_number = excluded.page_number,
                    graph_json = excluded.graph_json,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
            connection.execute(
                """
                UPDATE index_sessions
                SET entity_input_tokens = entity_input_tokens + ?,
                    entity_output_tokens = entity_output_tokens + ?,
                    updated_at = ?
                WHERE source_id = ?
                """,
                (
                    usage.input_tokens,
                    usage.output_tokens,
                    _now(),
                    source_id,
                ),
            )
            connection.execute("COMMIT")

    def token_usage(self, source_id: str) -> tuple[int, int, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT embedding_input_tokens, entity_input_tokens,
                       entity_output_tokens
                FROM index_sessions WHERE source_id = ?
                """,
                (source_id,),
            ).fetchone()
        if not row:
            return 0, 0, 0
        return (
            int(row["embedding_input_tokens"]),
            int(row["entity_input_tokens"]),
            int(row["entity_output_tokens"]),
        )

    def finish(self, source_id: str, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_sessions
                SET status = 'ready', phase = 'complete',
                    completed_chunks = total_chunks,
                    result_json = ?, error = '', updated_at = ?
                WHERE source_id = ?
                """,
                (
                    json.dumps(result, ensure_ascii=False),
                    _now(),
                    source_id,
                ),
            )

    def persisted_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM index_sessions WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        status = str(row["status"])
        if status in {"queued", "running", "stopping"}:
            status = "paused"
        return {
            "job_id": str(row["job_id"]),
            "source_id": str(row["source_id"]),
            "status": status,
            "result": (
                json.loads(row["result_json"])
                if row["result_json"]
                else None
            ),
            "error": str(row["error"] or ""),
            "phase": str(row["phase"] or ""),
            "total_chunks": int(row["total_chunks"]),
            "completed_chunks": int(row["completed_chunks"]),
            "current_page": int(row["current_page"]),
            "resumable": status in {"paused", "failed"},
        }

    def delete_source(self, source_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM checkpoint_chunks WHERE source_id = ?",
                (source_id,),
            )
            connection.execute(
                "DELETE FROM parsed_documents WHERE source_id = ?",
                (source_id,),
            )
            connection.execute(
                "DELETE FROM index_sessions WHERE source_id = ?",
                (source_id,),
            )
            connection.execute("COMMIT")
