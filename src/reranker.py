"""Reranker module for the Document-Based RAG system.

This module provides Cross-Encoder document reranking using sentence-transformers,
scoring (question, document) candidate pairs to reorder retrieved documents by
relevance and retain the most pertinent passages.
"""

from typing import Any, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import CROSS_ENCODER_MODEL_NAME, RERANKING_TOP_K


class DocumentReranker:
    """Service responsible for scoring and reranking retrieved documents using a Cross-Encoder."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Initialize DocumentReranker with model hyperparameters and optional injected model.

        Args:
            model_name: Hugging Face model identifier for the Cross-Encoder.
                Defaults to CROSS_ENCODER_MODEL_NAME from config.py.
            model: Optional pre-initialized CrossEncoder model instance for dependency injection.
            top_k: Default number of top documents to retain after reranking.
                Defaults to RERANKING_TOP_K from config.py.

        Raises:
            ValueError: If top_k is not a positive integer.
        """
        self.model_name: str = model_name or CROSS_ENCODER_MODEL_NAME
        self.top_k: int = top_k if top_k is not None else RERANKING_TOP_K

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {self.top_k}")

        self._model: Optional[CrossEncoder] = model

    @property
    def model(self) -> CrossEncoder:
        """Access the underlying Cross-Encoder model, initializing it lazily if not provided.

        Returns:
            CrossEncoder: The active CrossEncoder instance.

        Raises:
            RuntimeError: If model loading fails, preserving the original cause.
        """
        if self._model is None:
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize Cross-Encoder model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def rerank(
        self,
        question: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """Rerank retrieved document chunks against the user question using the Cross-Encoder.

        Forms (question, document.page_content) pairs, computes raw relevance scores,
        and returns the top-k highest scoring Document objects in descending order.

        Args:
            question: User's natural-language question string.
            documents: Candidate LangChain Document objects to rerank.
            top_k: Optional override for the number of top documents to return.

        Returns:
            list[Document]: Reranked documents ordered by descending relevance score.
                Returns an empty list if input documents is empty.

        Raises:
            TypeError: If question is not a string, documents is not a list, or documents
                contains non-Document items.
            ValueError: If question is empty/whitespace or top_k is not a positive integer.
            RuntimeError: If Cross-Encoder prediction or scoring fails.
        """
        if question is None or not isinstance(question, str):
            raise TypeError(
                f"Expected question to be a string, got: {type(question).__name__}"
            )

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question cannot be empty or whitespace only.")

        if documents is None or not isinstance(documents, list):
            raise TypeError(
                f"Expected documents to be a list of Document objects, got: {type(documents).__name__}"
            )

        effective_k = top_k if top_k is not None else self.top_k
        if not isinstance(effective_k, int) or effective_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {effective_k}")

        if not documents:
            return []

        for idx, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise TypeError(
                    f"Item at index {idx} is not an instance of LangChain Document: "
                    f"got {type(doc).__name__}"
                )

        # 1. Create (question, document_text) pairs
        pairs = [(cleaned_question, doc.page_content) for doc in documents]

        # 2. Score pairs using Cross-Encoder predict()
        # Note: Scores are unbounded relevance logits/scores, not probabilities.
        try:
            scores = self.model.predict(pairs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to rerank documents using Cross-Encoder: {exc}"
            ) from exc

        # 3. Sort documents by descending relevance score
        scored_documents = sorted(
            zip(documents, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        # 4. Extract top-k reranked documents preserving original Document instances
        reranked_docs = [doc for doc, _ in scored_documents[:effective_k]]
        return reranked_docs
