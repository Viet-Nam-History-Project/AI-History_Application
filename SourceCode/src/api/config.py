from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


SOURCE_CODE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    openai_api_key: str
    openai_chat_model: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_entity_model: str = "gpt-5.4-nano"
    ai_pipeline_version: str = "pdf-openai-v2"

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"

    firebase_project_id: str = "historyapplication-de20b"
    firebase_service_account_path: str = ""
    ai_admin_api_key: str
    ai_allow_unauthenticated: bool = False
    ai_cors_origins: str = "http://localhost:3000,http://localhost:3001"

    ai_chunk_size: int = 1600
    ai_chunk_overlap: int = 220
    ai_max_pdf_size_mb: int = 200
    ai_enable_pdf_ocr: bool = True
    ai_ocr_languages: str = "vie+eng"
    ai_ocr_jobs: int = 2
    ai_ocr_fallback_dpi: int = 220
    ai_ocr_page_timeout_seconds: int = 180
    ai_ocr_process_timeout_seconds: int = 14400
    ai_retrieval_top_k: int = 8
    ai_retrieval_candidate_k: int = 24
    ai_min_relevance_score: float = 0.35
    ai_enable_rerank: bool = True
    ai_enable_entity_extraction: bool = True
    ai_entity_batch_size: int = 6
    # Published, administrator-reviewed history content is a second grounded
    # corpus beside PDF/Neo4j.  The local path is useful in the monorepo; a
    # deployed backend falls back to the immutable Firebase Hosting manifest.
    ai_published_content_enabled: bool = True
    ai_published_content_manifest_path: str = ""
    ai_published_content_manifest_url: str = ""
    ai_published_content_cache_ttl_seconds: int = 300
    ai_index_checkpoint_path: str = str(
        SOURCE_CODE_DIR / ".data" / "index-checkpoints.sqlite3"
    )

    model_config = SettingsConfigDict(
        env_file=(SOURCE_CODE_DIR / ".env", SOURCE_CODE_DIR / "env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.ai_cors_origins.split(",") if origin.strip()]

    @property
    def published_content_manifest_url(self) -> str:
        configured = self.ai_published_content_manifest_url.strip()
        if configured:
            return configured
        return (
            f"https://{self.firebase_project_id}.web.app/"
            "content/manifest.json"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
