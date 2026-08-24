from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=6000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1200)
    messages: list[ConversationMessage] = Field(default_factory=list, max_length=8)
    include_citations: bool = True


class Citation(BaseModel):
    chunk_id: str = ""
    source_id: str
    source_title: str
    page_start: int | None = None
    page_end: int | None = None
    excerpt: str
    score: float
    facets: list[str] = Field(default_factory=list)


class RetrievalDiagnostics(BaseModel):
    strategy: str = "hybrid"
    candidate_count: int = 0
    selected_count: int = 0
    intent: str = ""
    scope: str = ""
    date_range: str = ""
    query_terms: list[str] = Field(default_factory=list)
    primary_entity: str = ""
    exact_match_count: int = 0
    channels: list[str] = Field(default_factory=list)
    confidence_factors: dict[str, float] = Field(default_factory=dict)
    original_question: str = ""
    standalone_question: str = ""
    conversation_resolved: bool = False
    normalized_question: str = ""
    query_corrections: list[dict[str, str]] = Field(default_factory=list)
    rewrite_confidence: float = 1.0
    query_ambiguous: bool = False
    ambiguity_notes: list[str] = Field(default_factory=list)
    required_facets: list[str] = Field(default_factory=list)
    optional_facets: list[str] = Field(default_factory=list)
    covered_facets: list[str] = Field(default_factory=list)
    missing_facets: list[str] = Field(default_factory=list)
    facet_coverage: dict[str, list[str]] = Field(default_factory=dict)
    requirement_statuses: dict[str, str] = Field(default_factory=dict)
    requirement_evidence: dict[str, list[str]] = Field(default_factory=dict)
    requirement_types: dict[str, str] = Field(default_factory=dict)
    semantic_requirements: dict[str, dict[str, object]] = Field(
        default_factory=dict
    )
    coverage_gate_outcome: str = ""
    coverage_gate_limitations: list[str] = Field(default_factory=list)
    claim_verification_status: str = "not_run"
    checked_claim_count: int = 0
    unsupported_high_risk_claims: list[str] = Field(default_factory=list)
    answer_structure: str = "direct"


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    related_entities: list[str] = Field(default_factory=list)
    retrieval: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)


class KnowledgeIndexMetadata(BaseModel):
    source_id: str
    title: str
    author: str = ""
    publisher: str = ""
    publication_year: int | None = None
    source_type: str = "book"
    trust_level: str = "reference"
    storage_path: str = ""
    related_periods: list[str] = Field(default_factory=list)
    related_stages: list[str] = Field(default_factory=list)
    related_events: list[str] = Field(default_factory=list)
    related_persons: list[str] = Field(default_factory=list)


class KnowledgeMetadataUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    author: str = Field(default="", max_length=240)
    publisher: str = Field(default="", max_length=240)
    publication_year: int | None = Field(default=None, ge=1000, le=2100)
    source_type: str = Field(default="book", max_length=80)
    trust_level: str = Field(default="reference", max_length=80)
    related_periods: list[str] = Field(default_factory=list, max_length=50)
    related_stages: list[str] = Field(default_factory=list, max_length=100)
    related_events: list[str] = Field(default_factory=list, max_length=200)
    related_persons: list[str] = Field(default_factory=list, max_length=200)


class KnowledgeIndexResponse(BaseModel):
    source_id: str
    page_count: int
    text_page_count: int
    content_start_page: int = 1
    content_end_page: int = 1
    skipped_page_count: int = 0
    boundary_detection: str = "full_document"
    footnote_page_count: int = 0
    removed_footnote_chars: int = 0
    running_header_page_count: int = 0
    removed_running_header_chars: int = 0
    chunk_count: int
    duplicate_chunk_count: int
    ocr_applied: bool = False
    ocr_languages: str = ""
    ocr_engine: str = ""
    ocr_failed_pages: list[int] = Field(default_factory=list)
    entity_count: int = 0
    relationship_count: int = 0
    embedding_input_tokens: int = 0
    entity_input_tokens: int = 0
    entity_output_tokens: int = 0
    total_tokens: int = 0
    reused_chunk_count: int = 0
    openai_chunk_count: int = 0
    status: str = "ready"


class KnowledgeIndexJobResponse(BaseModel):
    job_id: str
    source_id: str
    status: str
    result: KnowledgeIndexResponse | None = None
    error: str = ""
    phase: str = "queued"
    total_chunks: int = 0
    completed_chunks: int = 0
    current_page: int = 0
    resumable: bool = False


class PromptActivationRequest(BaseModel):
    version_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=2, max_length=120)
    system_prompt: str = Field(min_length=100, max_length=20000)
    query_normalization_instruction: str = Field(min_length=40, max_length=12000)
    answer_planning_instruction: str = Field(min_length=40, max_length=12000)
    presentation_instruction: str = Field(min_length=40, max_length=8000)
    output_contract: str = Field(min_length=40, max_length=8000)


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    model: str
    rag_revision: str
    source_rag_revision: str
    restart_required: bool = False
    active_chat_requests: int = 0
    active_index_jobs: int = 0
    indexed_sources: int
    indexed_chunks: int
    indexed_pages: int


class RevisionHealthResponse(BaseModel):
    status: str = "ready"
    rag_revision: str
    source_rag_revision: str
    restart_required: bool = False
    active_chat_requests: int = 0
    active_index_jobs: int = 0
