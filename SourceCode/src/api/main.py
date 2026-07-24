import json
import asyncio
import hashlib
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Callable
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .auth import require_admin_key, require_firebase_user
from .config import get_settings
from .graph_extraction import (
    ChunkGraphExtraction,
    ExtractedEntity,
    ExtractedRelationship,
    PdfGraphExtractionService,
    TokenUsage,
)
from .index_checkpoint import (
    IndexCheckpointStore,
    IndexJobPaused,
)
from .models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    KnowledgeIndexJobResponse,
    KnowledgeIndexMetadata,
    KnowledgeIndexResponse,
    KnowledgeMetadataUpdate,
    PromptActivationRequest,
    RevisionHealthResponse,
)
from .neo4j_repository import Neo4jKnowledgeRepository
from .pdf_ingestion import parse_pdf
from .rag_service import (
    ANSWER_PLANNING_INSTRUCTION,
    DEFAULT_SYSTEM_PROMPT,
    OUTPUT_CONTRACT,
    PRESENTATION_INSTRUCTION,
    QUERY_NORMALIZATION_INSTRUCTION,
    RagService,
    effective_answer_planning_instruction,
)
from .runtime_revision import RAG_REVISION


settings = get_settings()
repository = Neo4jKnowledgeRepository()
rag_service = RagService(repository)
graph_extraction_service = PdfGraphExtractionService(
    rag_service.client,
    settings.openai_entity_model,
    settings.ai_entity_batch_size,
)
checkpoint_store = IndexCheckpointStore(settings.ai_index_checkpoint_path)
index_jobs: dict[str, dict] = {}
active_source_jobs: dict[str, str] = {}
index_jobs_lock = threading.Lock()
index_tasks: set[asyncio.Task] = set()
active_chat_requests = 0
active_chat_lock = threading.Lock()


def _graph_from_cache(
    chunk,
    cached: dict,
) -> ChunkGraphExtraction:
    return ChunkGraphExtraction(
        chunk_id=chunk.chunk_id,
        page_number=chunk.page_start,
        entities=[
            ExtractedEntity(
                canonical_id=item["canonicalId"],
                name=item["name"],
                entity_type=item["entityType"],
                aliases=tuple(item.get("aliases", [])),
                search_keys=tuple(item.get("searchKeys", [])),
                surface_forms=tuple(item.get("surfaceForms", [])),
            )
            for item in cached.get("entities", [])
        ],
        relationships=[
            ExtractedRelationship(
                source_canonical_id=item["sourceCanonicalId"],
                target_canonical_id=item["targetCanonicalId"],
                relationship_type=item["type"],
                evidence=item["evidence"],
                raw_predicate=item.get("rawPredicate", ""),
            )
            for item in cached.get("relationships", [])
        ],
    )


ProgressCallback = Callable[[str, int, int, int], None]


def _index_result(
    metadata: KnowledgeIndexMetadata,
    pdf_bytes: bytes,
    *,
    job_id: str = "",
    document_hash: str = "",
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> KnowledgeIndexResponse:
    use_checkpoint = bool(job_id and document_hash)

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise IndexJobPaused("Tác vụ đã được dừng tại checkpoint an toàn.")

    def report(
        phase: str,
        total_chunks: int,
        completed_chunks: int,
        current_page: int,
    ) -> None:
        if progress_callback is not None:
            progress_callback(
                phase,
                total_chunks,
                completed_chunks,
                current_page,
            )

    check_cancelled()
    report("parsing", 0, 0, 0)
    parsed_pdf = (
        checkpoint_store.load_parsed_pdf(
            metadata.source_id,
            document_hash,
        )
        if use_checkpoint
        else None
    )
    if parsed_pdf is None:
        parsed_pdf = parse_pdf(pdf_bytes, metadata.source_id)
        if use_checkpoint:
            checkpoint_store.save_parsed_pdf(
                metadata.source_id,
                document_hash,
                parsed_pdf,
            )

    total_chunks = len(parsed_pdf.chunks)
    check_cancelled()
    cache = repository.source_chunk_cache(metadata.source_id)
    cached_chunks = cache.get("chunks", {})
    checkpoint_chunks = (
        checkpoint_store.load_chunk_results(
            metadata.source_id,
            document_hash,
        )
        if use_checkpoint
        else {}
    )
    can_reuse_embeddings = (
        cache.get("embeddingModel") == settings.openai_embedding_model
    )
    can_reuse_graph = (
        cache.get("entityModel") == settings.openai_entity_model
        and cache.get("pipelineVersion") == settings.ai_pipeline_version
    )

    embeddings_by_hash: dict[str, list[float]] = {}
    for chunk in parsed_pdf.chunks:
        committed = cached_chunks.get(chunk.content_hash, {})
        checkpoint = checkpoint_chunks.get(chunk.content_hash, {})
        if can_reuse_embeddings and committed.get("embedding"):
            embeddings_by_hash[chunk.content_hash] = committed["embedding"]
        elif checkpoint.get("embedding") is not None:
            embeddings_by_hash[chunk.content_hash] = checkpoint["embedding"]

    chunks_needing_embeddings = [
        chunk
        for chunk in parsed_pdf.chunks
        if chunk.content_hash not in embeddings_by_hash
    ]
    embedding_usage = TokenUsage()
    completed_embeddings = total_chunks - len(chunks_needing_embeddings)
    report("embeddings", total_chunks, completed_embeddings, 0)
    embedding_batch_size = 64
    for offset in range(0, len(chunks_needing_embeddings), embedding_batch_size):
        check_cancelled()
        batch = chunks_needing_embeddings[
            offset:offset + embedding_batch_size
        ]
        batch_embeddings, batch_usage = rag_service.embed_chunks_with_usage(
            [chunk.text for chunk in batch],
            batch_size=embedding_batch_size,
        )
        embedding_usage.add(batch_usage)
        if use_checkpoint:
            checkpoint_store.save_embeddings(
                metadata.source_id,
                document_hash,
                batch,
                batch_embeddings,
                batch_usage,
            )
        for chunk, embedding in zip(batch, batch_embeddings, strict=True):
            embeddings_by_hash[chunk.content_hash] = embedding
        completed_embeddings += len(batch)
        report(
            "embeddings",
            total_chunks,
            completed_embeddings,
            batch[-1].page_start,
        )

    embeddings = [
        embeddings_by_hash[chunk.content_hash]
        for chunk in parsed_pdf.chunks
    ]

    graph_extractions: list[ChunkGraphExtraction] = []
    entity_usage = TokenUsage()
    if settings.ai_enable_entity_extraction:
        graph_by_hash: dict[str, ChunkGraphExtraction] = {}
        for chunk in parsed_pdf.chunks:
            committed = cached_chunks.get(chunk.content_hash, {})
            checkpoint = checkpoint_chunks.get(chunk.content_hash, {})
            if can_reuse_graph and committed.get("graphExtracted"):
                graph_by_hash[chunk.content_hash] = _graph_from_cache(
                    chunk,
                    committed,
                )
            elif checkpoint.get("graph") is not None:
                graph_by_hash[chunk.content_hash] = _graph_from_cache(
                    chunk,
                    checkpoint["graph"],
                )

        chunks_needing_graph = [
            chunk
            for chunk in parsed_pdf.chunks
            if chunk.content_hash not in graph_by_hash
        ]
        completed_graph = total_chunks - len(chunks_needing_graph)
        report("entities", total_chunks, completed_graph, 0)
        graph_batch_size = graph_extraction_service.batch_size
        for offset in range(0, len(chunks_needing_graph), graph_batch_size):
            check_cancelled()
            batch = chunks_needing_graph[offset:offset + graph_batch_size]
            batch_extractions, batch_usage = graph_extraction_service.extract(
                batch
            )
            entity_usage.add(batch_usage)
            if use_checkpoint:
                checkpoint_store.save_graph(
                    metadata.source_id,
                    document_hash,
                    batch,
                    batch_extractions,
                    batch_usage,
                )
            for chunk, extraction in zip(
                batch,
                batch_extractions,
                strict=True,
            ):
                graph_by_hash[chunk.content_hash] = extraction
            completed_graph += len(batch)
            report(
                "entities",
                total_chunks,
                completed_graph,
                batch[-1].page_start,
            )
        graph_extractions = [
            graph_by_hash[chunk.content_hash]
            for chunk in parsed_pdf.chunks
        ]
    else:
        chunks_needing_graph = []

    reused_chunk_count = sum(
        1
        for chunk in parsed_pdf.chunks
        if (
            chunk not in chunks_needing_embeddings
            and (
                not settings.ai_enable_entity_extraction
                or chunk not in chunks_needing_graph
            )
        )
    )
    openai_chunk_count = len({
        chunk.chunk_id
        for chunk in chunks_needing_embeddings + chunks_needing_graph
    })
    entity_count = len({
        entity.canonical_id
        for extraction in graph_extractions
        for entity in extraction.entities
    })
    relationship_count = sum(
        len(extraction.relationships)
        for extraction in graph_extractions
    )
    if use_checkpoint:
        (
            embedding_input_tokens,
            entity_input_tokens,
            entity_output_tokens,
        ) = checkpoint_store.token_usage(metadata.source_id)
    else:
        embedding_input_tokens = embedding_usage.input_tokens
        entity_input_tokens = entity_usage.input_tokens
        entity_output_tokens = entity_usage.output_tokens

    check_cancelled()
    report("saving", total_chunks, total_chunks, parsed_pdf.content_end_page)
    repository.replace_source(
        metadata,
        parsed_pdf.pages,
        parsed_pdf.chunks,
        embeddings,
        graph_extractions,
        {
            "embeddingModel": settings.openai_embedding_model,
            "entityModel": (
                settings.openai_entity_model
                if settings.ai_enable_entity_extraction
                else ""
            ),
            "entityExtractionEnabled": settings.ai_enable_entity_extraction,
            "pipelineVersion": settings.ai_pipeline_version,
            "embeddingInputTokens": embedding_input_tokens,
            "entityInputTokens": entity_input_tokens,
            "entityOutputTokens": entity_output_tokens,
            "totalTokens": (
                embedding_input_tokens
                + entity_input_tokens
                + entity_output_tokens
            ),
            "entityCount": entity_count,
            "relationshipCount": relationship_count,
            "reusedChunkCount": reused_chunk_count,
            "openaiChunkCount": openai_chunk_count,
            "pageCount": parsed_pdf.page_count,
            "indexedPageCount": parsed_pdf.text_page_count,
            "contentStartPage": parsed_pdf.content_start_page,
            "contentEndPage": parsed_pdf.content_end_page,
            "skippedPageCount": parsed_pdf.skipped_page_count,
            "boundaryDetection": parsed_pdf.boundary_detection,
            "footnotePageCount": parsed_pdf.footnote_page_count,
            "removedFootnoteChars": parsed_pdf.removed_footnote_chars,
            "runningHeaderPageCount": (
                parsed_pdf.running_header_page_count
            ),
            "removedRunningHeaderChars": (
                parsed_pdf.removed_running_header_chars
            ),
        },
    )
    integrity = repository.ensure_source_structure(
        metadata.source_id,
        expected_page_count=parsed_pdf.page_count,
        expected_chunk_count=len(parsed_pdf.chunks),
    )
    if not integrity or not integrity["valid"]:
        raise RuntimeError(
            "Neo4j chưa lưu đủ cấu trúc nguồn tri thức sau index. "
            f"Kết quả hậu kiểm: {integrity or 'không tìm thấy source'}."
        )
    response = KnowledgeIndexResponse(
        source_id=metadata.source_id,
        page_count=parsed_pdf.page_count,
        text_page_count=parsed_pdf.text_page_count,
        content_start_page=parsed_pdf.content_start_page,
        content_end_page=parsed_pdf.content_end_page,
        skipped_page_count=parsed_pdf.skipped_page_count,
        boundary_detection=parsed_pdf.boundary_detection,
        footnote_page_count=parsed_pdf.footnote_page_count,
        removed_footnote_chars=parsed_pdf.removed_footnote_chars,
        running_header_page_count=parsed_pdf.running_header_page_count,
        removed_running_header_chars=(
            parsed_pdf.removed_running_header_chars
        ),
        chunk_count=len(parsed_pdf.chunks),
        duplicate_chunk_count=parsed_pdf.duplicate_chunk_count,
        ocr_applied=parsed_pdf.ocr_applied,
        ocr_languages=parsed_pdf.ocr_languages,
        ocr_engine=parsed_pdf.ocr_engine,
        ocr_failed_pages=parsed_pdf.ocr_failed_pages,
        entity_count=entity_count,
        relationship_count=relationship_count,
        embedding_input_tokens=embedding_input_tokens,
        entity_input_tokens=entity_input_tokens,
        entity_output_tokens=entity_output_tokens,
        total_tokens=(
            embedding_input_tokens
            + entity_input_tokens
            + entity_output_tokens
        ),
        reused_chunk_count=reused_chunk_count,
        openai_chunk_count=openai_chunk_count,
    )
    if use_checkpoint:
        checkpoint_store.finish(
            metadata.source_id,
            response.model_dump(),
        )
    report("complete", total_chunks, total_chunks, parsed_pdf.content_end_page)
    return response


def _job_response(job: dict) -> KnowledgeIndexJobResponse:
    return KnowledgeIndexJobResponse(
        job_id=job["job_id"],
        source_id=job["source_id"],
        status=job["status"],
        result=job.get("result"),
        error=job.get("error", ""),
        phase=job.get("phase", "queued"),
        total_chunks=int(job.get("total_chunks", 0)),
        completed_chunks=int(job.get("completed_chunks", 0)),
        current_page=int(job.get("current_page", 0)),
        resumable=bool(job.get("resumable", False)),
    )


async def _run_index_job(
    job_id: str,
    metadata: KnowledgeIndexMetadata,
    pdf_bytes: bytes,
) -> None:
    document_hash = hashlib.sha256(pdf_bytes).hexdigest()
    with index_jobs_lock:
        job = index_jobs[job_id]
        cancel_event = job.get("cancel_event")
        if cancel_event is None:
            cancel_event = threading.Event()
            job["cancel_event"] = cancel_event
        job["status"] = "running"
        job["phase"] = "parsing"
        job["updated_at"] = datetime.now(UTC)

    def update_progress(
        phase: str,
        total_chunks: int,
        completed_chunks: int,
        current_page: int,
    ) -> None:
        progress_status = (
            "ready"
            if phase == "complete"
            else "stopping"
            if cancel_event.is_set()
            else "running"
        )
        visible_phase = (
            "complete"
            if phase == "complete"
            else "stopping"
            if cancel_event.is_set()
            else phase
        )
        with index_jobs_lock:
            current_job = index_jobs.get(job_id)
            if current_job:
                current_job.update(
                    status=progress_status,
                    phase=visible_phase,
                    total_chunks=total_chunks,
                    completed_chunks=completed_chunks,
                    current_page=current_page,
                    updated_at=datetime.now(UTC),
                )
        checkpoint_store.set_progress(
            metadata.source_id,
            status=progress_status,
            phase=visible_phase,
            total_chunks=total_chunks,
            completed_chunks=completed_chunks,
            current_page=current_page,
        )

    try:
        checkpoint_store.begin_job(
            source_id=metadata.source_id,
            document_hash=document_hash,
            pipeline_version=settings.ai_pipeline_version,
            embedding_model=settings.openai_embedding_model,
            entity_model=(
                settings.openai_entity_model
                if settings.ai_enable_entity_extraction
                else ""
            ),
            job_id=job_id,
        )
        checkpoint_store.set_progress(
            metadata.source_id,
            status="running",
            phase="parsing",
        )
        result = await asyncio.to_thread(
            _index_result,
            metadata,
            pdf_bytes,
            job_id=job_id,
            document_hash=document_hash,
            cancel_event=cancel_event,
            progress_callback=update_progress,
        )
        with index_jobs_lock:
            index_jobs[job_id].update(
                status="ready",
                phase="complete",
                result=result,
                resumable=False,
                updated_at=datetime.now(UTC),
            )
    except IndexJobPaused:
        checkpoint_store.set_progress(
            metadata.source_id,
            status="paused",
            phase="paused",
            error="",
        )
        with index_jobs_lock:
            index_jobs[job_id].update(
                status="paused",
                phase="paused",
                error="",
                resumable=True,
                updated_at=datetime.now(UTC),
            )
    except Exception as exc:
        checkpoint_store.set_progress(
            metadata.source_id,
            status="failed",
            error=str(exc)[:1000],
        )
        with index_jobs_lock:
            index_jobs[job_id].update(
                status="failed",
                error=str(exc),
                resumable=True,
                updated_at=datetime.now(UTC),
            )
    finally:
        with index_jobs_lock:
            if active_source_jobs.get(metadata.source_id) == job_id:
                active_source_jobs.pop(metadata.source_id, None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    checkpoint_store.recover_interrupted_jobs()
    repository.verify_connectivity()
    repository.ensure_schema()
    yield
    repository.close()


app = FastAPI(
    title="Vietnam History AI API",
    version="1.0.0",
    description="RAG API dùng chung cho app học lịch sử và web-admin.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        repository.verify_connectivity()
        stats = repository.stats()
        with active_chat_lock:
            chat_count = active_chat_requests
        return HealthResponse(
            status="ready",
            neo4j="connected",
            model=settings.openai_chat_model,
            rag_revision=RAG_REVISION,
            active_chat_requests=chat_count,
            active_index_jobs=checkpoint_store.active_job_count(),
            indexed_sources=stats["sources"],
            indexed_chunks=stats["chunks"],
            indexed_pages=stats["pages"],
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Neo4j chưa sẵn sàng: {exc}") from exc


@app.get("/health/revision", response_model=RevisionHealthResponse)
def revision_health() -> RevisionHealthResponse:
    """Lightweight endpoint for the local idle-restart watchdog."""
    with active_chat_lock:
        chat_count = active_chat_requests
    return RevisionHealthResponse(
        rag_revision=RAG_REVISION,
        active_chat_requests=chat_count,
        active_index_jobs=checkpoint_store.active_job_count(),
    )


@app.post("/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: dict = Depends(require_firebase_user),
) -> ChatResponse:
    global active_chat_requests
    with active_chat_lock:
        active_chat_requests += 1
    try:
        return rag_service.answer(request, str(user.get("uid") or user.get("sub") or ""))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể tạo câu trả lời: {exc}") from exc
    finally:
        with active_chat_lock:
            active_chat_requests = max(0, active_chat_requests - 1)


@app.post("/v1/admin/knowledge/index", response_model=KnowledgeIndexResponse)
async def index_knowledge(
    metadata: str = Form(...),
    file: UploadFile = File(...),
    _: None = Depends(require_admin_key),
) -> KnowledgeIndexResponse:
    try:
        parsed_metadata = KnowledgeIndexMetadata.model_validate(json.loads(metadata))
        pdf_bytes = await file.read()
        return await asyncio.to_thread(_index_result, parsed_metadata, pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index PDF thất bại: {exc}") from exc


@app.post("/v1/admin/knowledge/index/jobs", response_model=KnowledgeIndexJobResponse)
async def create_index_job(
    metadata: str = Form(...),
    file: UploadFile = File(...),
    _: None = Depends(require_admin_key),
) -> KnowledgeIndexJobResponse:
    try:
        parsed_metadata = KnowledgeIndexMetadata.model_validate(json.loads(metadata))
        pdf_bytes = await file.read()
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with index_jobs_lock:
        active_job_id = active_source_jobs.get(parsed_metadata.source_id)
        if active_job_id and active_job_id in index_jobs:
            return _job_response(index_jobs[active_job_id])

        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "source_id": parsed_metadata.source_id,
            "status": "queued",
            "phase": "queued",
            "total_chunks": 0,
            "completed_chunks": 0,
            "current_page": 0,
            "resumable": False,
            "cancel_event": threading.Event(),
            "result": None,
            "error": "",
            "updated_at": datetime.now(UTC),
        }
        index_jobs[job_id] = job
        active_source_jobs[parsed_metadata.source_id] = job_id

    task = asyncio.create_task(_run_index_job(job_id, parsed_metadata, pdf_bytes))
    index_tasks.add(task)
    task.add_done_callback(index_tasks.discard)
    return _job_response(job)


@app.get(
    "/v1/admin/knowledge/index/jobs/{job_id}",
    response_model=KnowledgeIndexJobResponse,
)
def get_index_job(
    job_id: str,
    _: None = Depends(require_admin_key),
) -> KnowledgeIndexJobResponse:
    with index_jobs_lock:
        job = index_jobs.get(job_id)
        if not job:
            persisted = checkpoint_store.persisted_job(job_id)
            if persisted:
                return _job_response(persisted)
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy job index hoặc checkpoint tương ứng.",
            )
        return _job_response(job)


@app.post(
    "/v1/admin/knowledge/index/jobs/{job_id}/cancel",
    response_model=KnowledgeIndexJobResponse,
)
def cancel_index_job(
    job_id: str,
    _: None = Depends(require_admin_key),
) -> KnowledgeIndexJobResponse:
    with index_jobs_lock:
        job = index_jobs.get(job_id)
        if not job:
            persisted = checkpoint_store.persisted_job(job_id)
            if persisted:
                checkpoint_store.set_progress(
                    persisted["source_id"],
                    status="paused",
                    phase="paused",
                )
                persisted.update(
                    status="paused",
                    phase="paused",
                    resumable=True,
                )
                return _job_response(persisted)
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy job index để dừng.",
            )
        if job["status"] in {"ready", "failed", "paused"}:
            return _job_response(job)
        cancel_event = job.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        job.update(
            status="stopping",
            phase="stopping",
            resumable=True,
            updated_at=datetime.now(UTC),
        )
        checkpoint_store.set_progress(
            job["source_id"],
            status="stopping",
            phase="stopping",
        )
        return _job_response(job)


@app.delete("/v1/admin/knowledge/{source_id}")
def delete_knowledge(
    source_id: str,
    _: None = Depends(require_admin_key),
) -> dict[str, str]:
    repository.delete_source(source_id)
    checkpoint_store.delete_source(source_id)
    return {"source_id": source_id, "status": "deleted"}


@app.put("/v1/admin/knowledge/{source_id}/metadata")
def update_knowledge_metadata(
    source_id: str,
    metadata: KnowledgeMetadataUpdate,
    _: None = Depends(require_admin_key),
) -> dict[str, str]:
    updated = repository.update_source_metadata(source_id, metadata.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Không tìm thấy nguồn trong Neo4j.")
    return {"source_id": source_id, "status": "updated"}


@app.get("/v1/admin/knowledge/{source_id}")
def get_knowledge_detail(
    source_id: str,
    page_number: int | None = Query(default=None, ge=1),
    query: str = Query(default="", max_length=300),
    limit: int = Query(default=80, ge=1, le=200),
    _: None = Depends(require_admin_key),
) -> dict:
    try:
        detail = repository.source_detail(source_id, page_number, query, limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if detail is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy nguồn trong Neo4j.")
    return detail


@app.get("/v1/admin/graph/summary")
def graph_summary(_: None = Depends(require_admin_key)) -> dict:
    return repository.graph_summary()


@app.get("/v1/admin/graph/legacy/preview")
def graph_legacy_preview(_: None = Depends(require_admin_key)) -> dict:
    return repository.legacy_graph_preview()


@app.post("/v1/admin/graph/legacy/cleanup")
def graph_legacy_cleanup(
    payload: dict = Body(...),
    _: None = Depends(require_admin_key),
) -> dict:
    if payload.get("confirmation") != "DELETE_LEGACY_GRAPH":
        raise HTTPException(
            status_code=400,
            detail="Cần xác nhận DELETE_LEGACY_GRAPH trước khi xóa dữ liệu cũ.",
        )
    return repository.cleanup_legacy_graph()


@app.get("/v1/admin/graph/ontology/preview")
def graph_ontology_preview(_: None = Depends(require_admin_key)) -> dict:
    return repository.relationship_ontology_preview()


@app.post("/v1/admin/graph/ontology/migrate")
def graph_ontology_migrate(
    payload: dict = Body(...),
    _: None = Depends(require_admin_key),
) -> dict:
    if payload.get("confirmation") != "NORMALIZE_RELATIONSHIPS":
        raise HTTPException(
            status_code=400,
            detail="Cần xác nhận NORMALIZE_RELATIONSHIPS trước khi chuẩn hóa graph.",
        )
    return repository.migrate_relationship_ontology()


@app.get("/v1/admin/graph/search")
def graph_search(
    query: str = Query(default="", max_length=300),
    label: str = Query(default="", max_length=80),
    limit: int = Query(default=40, ge=1, le=100),
    _: None = Depends(require_admin_key),
) -> dict:
    return {"items": repository.graph_search(query, label, limit)}


@app.get("/v1/admin/graph/export")
def graph_export(
    component_type: str = Query(pattern="^(node|relationship)$"),
    name: str = Query(min_length=1, max_length=160),
    include_embeddings: bool = Query(default=False),
    _: None = Depends(require_admin_key),
) -> dict:
    try:
        return repository.export_graph_component(
            component_type,
            name,
            include_embeddings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/admin/prompt")
def active_prompt(_: None = Depends(require_admin_key)) -> dict:
    config = repository.get_active_prompt() or {}
    return {
        "versionId": str(config.get("versionId") or "builtin-v1"),
        "name": str(config.get("name") or "Prompt mặc định"),
        "systemPrompt": str(config.get("systemPrompt") or DEFAULT_SYSTEM_PROMPT),
        "queryNormalizationInstruction": str(
            config.get("queryNormalizationInstruction")
            or QUERY_NORMALIZATION_INSTRUCTION
        ),
        "answerPlanningInstruction": str(
            effective_answer_planning_instruction(
                config.get("answerPlanningInstruction")
                or ANSWER_PLANNING_INSTRUCTION
            )
        ),
        "presentationInstruction": str(
            config.get("presentationInstruction") or PRESENTATION_INSTRUCTION
        ),
        "outputContract": str(config.get("outputContract") or OUTPUT_CONTRACT),
    }


@app.put("/v1/admin/prompt")
def activate_prompt(
    request: PromptActivationRequest,
    _: None = Depends(require_admin_key),
) -> dict[str, str]:
    repository.set_active_prompt(
        request.version_id,
        request.name,
        request.system_prompt,
        request.query_normalization_instruction,
        request.answer_planning_instruction,
        request.presentation_instruction,
        request.output_contract,
    )
    return {"status": "active", "version_id": request.version_id}


@app.post("/v1/admin/retrieval/debug", response_model=ChatResponse)
def debug_retrieval(
    request: ChatRequest,
    _: None = Depends(require_admin_key),
) -> ChatResponse:
    return rag_service.answer(request, "admin-evaluation")


@app.get("/v1/admin/query-insights")
def query_insights(
    limit: int = Query(default=80, ge=1, le=200),
    _: None = Depends(require_admin_key),
) -> dict:
    return repository.query_insights(limit)
