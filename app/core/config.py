from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://localhost:5432/rag"
    storage_dir: Path = Path("data/uploads")

    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout_seconds: float = 60

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "qwen3-embedding:0.6b"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 16
    ocr_enabled: bool = True
    ocr_backend: str = "auto"
    ocr_min_text_chars: int = 30
    ocr_min_confidence: float = 0.75
    vlm_enabled: bool = False
    ollama_vlm_model: str = "moondream:latest"
    vlm_on_images: bool = True
    vlm_on_pdf_pages_with_images: bool = True

    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_batch_size: int = 8

    chunk_size: int = 700
    chunk_overlap: int = 100
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 30
    rerank_top_k: int = 8
    max_context_chunks: int = 6

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 700)
        if value < 0 or value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须大于等于 0 且小于 CHUNK_SIZE")
        return value

    def prepare_directories(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
