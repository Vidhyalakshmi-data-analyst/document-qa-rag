"""Retriever module for the Document-Based RAG system.

This module provides vector similarity search against the persisted FAISS
vector index to retrieve relevant document chunks for user questions.
"""

from typing import Optional

from langchain_core.documents import Document

from config import RETRIEVAL_TOP_K
from src.vector_store import FAISSVectorStore


class DocumentRetriever:
    """Service responsible for retrieving relevant document chunks from the FAISS vector store."""

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Initialize DocumentRetriever with injectable dependencies.

        Args:
            vector_store: Injected FAISSVectorStore instance. Defaults to FAISSVectorStore().
            top_k: Default number of documents to retrieve. Defaults to RETRIEVAL_TOP_K from config.py.

        Raises:
            ValueError: If top_k is not a positive integer.
        """
        self.vector_store: FAISSVectorStore = (
            vector_store if vector_store is not None else FAISSVectorStore()
        )
        self.top_k: int = top_k if top_k is not None else RETRIEVAL_TOP_K

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {self.top_k}")

    def retrieve(self, question: str, k: Optional[int] = None) -> list[Document]:
        """Retrieve the top-k most similar document chunks for a given question.

        Args:
            question: User's natural-language query string.
            k: Optional number of document chunks to retrieve. If None, uses configured top_k.

        Returns:
            list[Document]: Retrieved LangChain Document objects. Returns an empty list
                if no matching documents are found.

        Raises:
            TypeError: If question is not a string.
            ValueError: If question is empty/whitespace or k is non-positive or non-integer.
            RuntimeError: If vector store search fails.
        """
        if question is None or not isinstance(question, str):
            raise TypeError(
                f"Expected question to be a string, got: {type(question).__name__}"
            )

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question cannot be empty or whitespace only.")

        effective_k = k if k is not None else self.top_k
        if not isinstance(effective_k, int) or effective_k <= 0:
            raise ValueError(f"k must be a positive integer, got: {effective_k}")

        # Ensure index is loaded if not already loaded in memory
        if getattr(self.vector_store, "vector_store", None) is None:
            try:
                self.vector_store.load_index()
            except Exception:
                # If loading fails (e.g. mock or not yet persisted), similarity_search will handle/raise
                pass

        try:
            results = self.vector_store.similarity_search(
                query=cleaned_question,
                k=effective_k,
            )
            return results if results is not None else []
        except Exception as exc:
            raise RuntimeError(
                f"Failed to retrieve documents for question '{cleaned_question}': {exc}"
            ) from exc
