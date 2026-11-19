"""Configuration module for the Document-Based AI Question Answering (RAG) system.

This module centralizes all project configuration including directory paths,
model identifiers, document chunking parameters, and vector retrieval/reranking
settings. It loads environment variables using python-dotenv with sensible defaults
for non-secret values and separates configuration from application business logic.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Load environment variables from .env located at the project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATA_DIR: Path = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")).resolve()
VECTORSTORE_DIR: Path = Path(
    os.getenv("VECTORSTORE_DIR", PROJECT_ROOT / "vectorstore")
).resolve()

# ---------------------------------------------------------------------------
# Model Identifiers
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)

DEFAULT_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MODEL_NAME: str = os.getenv(
    "CROSS_ENCODER_MODEL_NAME", DEFAULT_CROSS_ENCODER_MODEL
)

# ---------------------------------------------------------------------------
# Chunking & Retrieval Hyperparameters
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RERANKING_TOP_K: int = int(os.getenv("RERANKING_TOP_K", "3"))

# ---------------------------------------------------------------------------
# Groq LLM Configurations
# ---------------------------------------------------------------------------
DEFAULT_GROQ_MODEL: str = "openai/gpt-oss-120b"
GROQ_MODEL: str = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

# GROQ_API_KEY is read without mandatory validation at import time.
# Validation is deferred to LLM service initialization.
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------
def validate_config(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    reranking_top_k: int = RERANKING_TOP_K,
) -> None:
    """Validate core non-secret configuration parameters.

    Ensures that text chunking and retrieval parameters adhere to logical
    constraints (positive sizes, overlap strictly less than chunk size, and
    reranking top-k not exceeding initial retrieval top-k).

    Args:
        chunk_size: Size of each text chunk in characters or tokens.
        chunk_overlap: Overlapping characters or tokens between consecutive chunks.
        retrieval_top_k: Number of candidate documents retrieved from vector store.
        reranking_top_k: Number of documents retained after cross-encoder reranking.

    Raises:
        ValueError: If any configuration value violates domain constraints.
    """
    if chunk_size <= 0:
        raise ValueError(f"CHUNK_SIZE must be greater than 0, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"CHUNK_OVERLAP must be non-negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({chunk_overlap}) must be strictly less than "
            f"CHUNK_SIZE ({chunk_size})."
        )
    if retrieval_top_k <= 0:
        raise ValueError(
            f"RETRIEVAL_TOP_K must be greater than 0, got {retrieval_top_k}."
        )
    if reranking_top_k <= 0:
        raise ValueError(
            f"RERANKING_TOP_K must be greater than 0, got {reranking_top_k}."
        )
    if reranking_top_k > retrieval_top_k:
        raise ValueError(
            f"RERANKING_TOP_K ({reranking_top_k}) cannot exceed "
            f"RETRIEVAL_TOP_K ({retrieval_top_k})."
        )


def validate_groq_api_key(api_key: Optional[str] = None) -> str:
    """Validate and return the Groq API key.

    This function should only be invoked when the LLM service is initialized.
    It ensures that importing config.py does not fail when the key is not yet set.

    Args:
        api_key: Optional explicit API key string. Defaults to the GROQ_API_KEY
            read from the environment.

    Returns:
        str: The validated, non-empty Groq API key.

    Raises:
        ValueError: If no valid Groq API key is found.
    """
    key = api_key if api_key is not None else GROQ_API_KEY
    if not key or not key.strip() or key.strip() == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set or invalid. Please define a valid GROQ_API_KEY "
            "in your .env file or environment variables before initializing the LLM service."
        )
    return key.strip()


# Validate non-secret hyperparameters at module import time
validate_config()


# ---------------------------------------------------------------------------
# AppConfig Container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """Immutable container for application settings."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    vectorstore_dir: Path = VECTORSTORE_DIR
    embedding_model_name: str = EMBEDDING_MODEL_NAME
    cross_encoder_model_name: str = CROSS_ENCODER_MODEL_NAME
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    retrieval_top_k: int = RETRIEVAL_TOP_K
    reranking_top_k: int = RERANKING_TOP_K
    groq_model: str = GROQ_MODEL
    groq_api_key: Optional[str] = GROQ_API_KEY


# Default singleton instance for convenience
config: AppConfig = AppConfig()
