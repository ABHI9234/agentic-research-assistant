from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    app_name: str = "Agentic Enterprise Research Assistant"
    app_version: str = "1.0.0"
    app_mode: Literal["development", "production"] = Field(default="development", alias="APP_MODE")
    backend_url: str = Field(default="http://localhost:8000", alias="BACKEND_URL")

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.0
    groq_max_tokens: int = 1024

    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection_chunks: str = "document_chunks"
    qdrant_collection_entities: str = "entities"
    embedding_dimension: int = 384

    neo4j_uri: str = Field(default="", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")

    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="agentic-research-assistant", alias="LANGSMITH_PROJECT")
    langchain_tracing_v2: str = Field(default="true", alias="LANGCHAIN_TRACING_V2")

    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    batch_size: int = 16
    max_file_size_mb: int = 50
    upload_dir: str = "uploads"

    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    graph_depth: int = Field(default=2, alias="GRAPH_DEPTH")
    hybrid_vector_weight: float = 0.6
    hybrid_graph_weight: float = 0.4

    log_level: str = "DEBUG"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True

    @property
    def is_development(self):
        return self.app_mode == "development"

    @property
    def is_production(self):
        return self.app_mode == "production"

    def get_langchain_env(self):
        return {
            "LANGCHAIN_TRACING_V2": self.langchain_tracing_v2,
            "LANGCHAIN_API_KEY": self.langsmith_api_key,
            "LANGCHAIN_PROJECT": self.langsmith_project,
        }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
