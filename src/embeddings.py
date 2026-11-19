"""Embeddings service module for the Document-Based RAG system.

This module provides text embedding generation using Hugging Face models via
LangChain's HuggingFaceEmbeddings integration. It supports lazy initialization
and dependency injection for reliable testing without downloading models.
"""

from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME


class EmbeddingService(Embeddings):
    """Service responsible for generating dense vector embeddings from text."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        """Initialize the EmbeddingService.

        Args:
            model_name: The Hugging Face model identifier to load. Defaults to
                EMBEDDING_MODEL_NAME from config.py.
            embeddings: Optional pre-configured LangChain Embeddings instance.
                When provided, this instance is used directly (dependency injection),
                preventing automatic model downloading or initialization.
        """
        self.model_name: str = model_name or EMBEDDING_MODEL_NAME
        self._embeddings: Optional[Embeddings] = embeddings

    @property
    def embeddings(self) -> Embeddings:
        """Access the underlying LangChain Embeddings instance.

        Initializes the model lazily on first access if not already injected or loaded.

        Returns:
            Embeddings: The active LangChain Embeddings instance.

        Raises:
            RuntimeError: If model initialization fails, preserving the original cause.
        """
        if self._embeddings is None:
            try:
                self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize embedding model '{self.model_name}': {exc}"
                ) from exc
        return self._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate numerical embedding vectors for a list of document strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: A list of numerical embedding vectors. Returns an
                empty list if input texts is empty.

        Raises:
            TypeError: If texts is not a list or contains non-string items.
            RuntimeError: If embedding generation fails, preserving the original cause.
        """
        if texts is None or not isinstance(texts, list):
            raise TypeError(
                f"Expected texts to be a list of strings, got: {type(texts).__name__}"
            )

        if not texts:
            return []

        for idx, item in enumerate(texts):
            if not isinstance(item, str):
                raise TypeError(
                    f"Item at index {idx} is not a string: got {type(item).__name__}"
                )

        try:
            return self.embeddings.embed_documents(texts)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate embeddings for documents: {exc}"
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        """Generate a numerical embedding vector for a single query string.

        Args:
            text: Query text string to embed.

        Returns:
            list[float]: Numerical embedding vector for the query.

        Raises:
            TypeError: If text is not a string.
            ValueError: If text is empty or whitespace only.
            RuntimeError: If embedding generation fails, preserving the original cause.
        """
        if text is None or not isinstance(text, str):
            raise TypeError(
                f"Expected query text to be a string, got: {type(text).__name__}"
            )

        if not text.strip():
            raise ValueError("Query text cannot be empty or whitespace only.")

        try:
            return self.embeddings.embed_query(text)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate embedding for query: {exc}"
            ) from exc
