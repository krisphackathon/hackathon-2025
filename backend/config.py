"""Configuration management for the application."""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Krisp Meeting Assistant Chat"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API Keys
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "meeting_transcriptions"
    qdrant_api_key: Optional[str] = None
    
    # Redis Cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour
    
    # Performance Settings
    max_context_length: int = 128000  # GPT-4 Turbo context window
    chunk_size: int = 1000  # characters per chunk
    chunk_overlap: int = 200
    batch_size: int = 100
    max_workers: int = 4
    
    # Search Settings
    top_k_retrieval: int = 20  # Initial retrieval count
    top_k_rerank: int = 5  # Final context chunks
    similarity_threshold: float = 0.7
    hybrid_search_weight: float = 0.7  # Weight for vector vs keyword search
    
    # API Settings
    api_rate_limit: int = 100  # requests per minute
    api_timeout: int = 60  # seconds
    cors_origins: list[str] = ["http://localhost:3000"]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Derived settings
EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimension
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
