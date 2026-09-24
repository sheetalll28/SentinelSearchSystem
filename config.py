"""
Central configuration for SentinelSearch.
Loads from .env (or real environment variables) so the same codebase
runs against Claude, OpenAI, or a local Ollama model without code changes.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Search
    search_provider: str = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")

    # RAG
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    documents_path: str = os.getenv("DOCUMENTS_PATH", "./data/documents")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # Safety / ops
    enable_moderation: bool = _bool("ENABLE_MODERATION", True)
    max_agent_iterations: int = int(os.getenv("MAX_AGENT_ITERATIONS", "6"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


CONFIG = Config()
