"""Unit tests for src/retriever.py."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from config import RETRIEVAL_TOP_K
from src.retriever import DocumentRetriever
from src.vector_store import FAISSVectorStore


@pytest.fixture
def sample_documents() -> list[Document]:
    """Fixture providing sample retrieved documents with metadata."""
    return [
        Document(
            page_content="LangChain provides modular components for RAG pipelines.",
            metadata={"source": "rag_architecture.pdf", "page": 1},
        ),
        Document(
            page_content="FAISS facilitates high-throughput vector indexing and search.",
            metadata={"source": "rag_architecture.pdf", "page": 2},
        ),
        Document(
            page_content="Embeddings translate semantic meaning into dense vectors.",
            metadata={"source": "embeddings_guide.pdf", "page": 3},
        ),
        Document(
            page_content="Cross-encoder models rerank retrieved passages accurately.",
            metadata={"source": "reranking_notes.pdf", "page": 4},
        ),
        Document(
            page_content="Groq accelerates LLM text generation with custom LPUs.",
            metadata={"source": "hardware_specs.pdf", "page": 5},
        ),
    ]


@pytest.fixture
def mock_vector_store(sample_documents: list[Document]) -> MagicMock:
    """Fixture providing a mock FAISSVectorStore."""
    store = MagicMock(spec=FAISSVectorStore)
    store.similarity_search.return_value = list(sample_documents)
    # Mock vector_store attribute so auto-load logic does not trigger
    store.vector_store = MagicMock()
    return store


@pytest.fixture
def retriever(mock_vector_store: MagicMock) -> DocumentRetriever:
    """Fixture providing DocumentRetriever with an injected mock vector store."""
    return DocumentRetriever(vector_store=mock_vector_store)


def test_successful_retrieval(
    retriever: DocumentRetriever,
    mock_vector_store: MagicMock,
    sample_documents: list[Document],
):
    """Test Case 1: Successfully retrieves relevant documents for a question."""
    results = retriever.retrieve("What is RAG?")

    assert isinstance(results, list)
    assert len(results) == len(sample_documents)
    mock_vector_store.similarity_search.assert_called_once_with(
        query="What is RAG?",
        k=RETRIEVAL_TOP_K,
    )


def test_default_retrieval_count_is_5(
    retriever: DocumentRetriever, mock_vector_store: MagicMock
):
    """Test Case 2: Uses configured default top_k (5) when k is not provided."""
    assert retriever.top_k == 5
    assert retriever.top_k == RETRIEVAL_TOP_K

    retriever.retrieve("Explain vector embeddings")
    mock_vector_store.similarity_search.assert_called_once_with(
        query="Explain vector embeddings",
        k=5,
    )


def test_custom_k_value_works_correctly(
    retriever: DocumentRetriever, mock_vector_store: MagicMock
):
    """Test Case 3: Custom k parameter overrides default retrieval count."""
    retriever.retrieve("How does FAISS work?", k=3)
    mock_vector_store.similarity_search.assert_called_once_with(
        query="How does FAISS work?",
        k=3,
    )

    retriever.retrieve("How does FAISS work?", k=10)
    mock_vector_store.similarity_search.assert_called_with(
        query="How does FAISS work?",
        k=10,
    )


def test_empty_question_is_rejected(retriever: DocumentRetriever):
    """Test Case 4: Empty string, None, or non-string queries are rejected."""
    with pytest.raises(
        ValueError, match="question cannot be empty or whitespace only"
    ):
        retriever.retrieve("")

    with pytest.raises(TypeError, match="Expected question to be a string"):
        retriever.retrieve(None)  # type: ignore

    with pytest.raises(TypeError, match="Expected question to be a string"):
        retriever.retrieve(12345)  # type: ignore


def test_whitespace_only_question_is_rejected(retriever: DocumentRetriever):
    """Test Case 5: Questions with only spaces, tabs, or newlines are rejected."""
    with pytest.raises(
        ValueError, match="question cannot be empty or whitespace only"
    ):
        retriever.retrieve("     \t \n  ")


def test_invalid_k_values_are_rejected(retriever: DocumentRetriever):
    """Test Case 6: Non-positive or non-integer k values are rejected with ValueError."""
    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.retrieve("valid question", k=0)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.retrieve("valid question", k=-5)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.retrieve("valid question", k="three")  # type: ignore

    # Also verify constructor validation
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        DocumentRetriever(top_k=0)


def test_empty_retrieval_results_handled(
    retriever: DocumentRetriever, mock_vector_store: MagicMock
):
    """Test Case 7: Returns an empty list when no documents match."""
    mock_vector_store.similarity_search.return_value = []
    results = retriever.retrieve("non-matching rare terms")

    assert results == []


def test_vector_store_errors_handled_meaningfully(
    retriever: DocumentRetriever, mock_vector_store: MagicMock
):
    """Test Case 8: Wraps vector store search failures in RuntimeError."""
    mock_vector_store.similarity_search.side_effect = RuntimeError(
        "Corrupted FAISS index memory map"
    )

    with pytest.raises(
        RuntimeError, match="Failed to retrieve documents for question"
    ) as exc_info:
        retriever.retrieve("trigger search error")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_retrieved_documents_and_metadata_preserved(
    retriever: DocumentRetriever, sample_documents: list[Document]
):
    """Test Case 9: Verifies retrieved Document objects preserve page content and metadata."""
    results = retriever.retrieve("Check metadata preservation")

    assert len(results) == len(sample_documents)
    for original, retrieved in zip(sample_documents, results):
        assert retrieved.page_content == original.page_content
        assert retrieved.metadata["source"] == original.metadata["source"]
        assert retrieved.metadata["page"] == original.metadata["page"]


def test_dependency_injection_works_with_mocked_vector_store():
    """Test Case 10: Injected vector store is utilized without triggering real FAISS/models."""
    custom_vs = MagicMock(spec=FAISSVectorStore)
    custom_vs.similarity_search.return_value = [
        Document(page_content="Injected mock content", metadata={"source": "mock.pdf"})
    ]
    custom_vs.vector_store = MagicMock()

    custom_retriever = DocumentRetriever(vector_store=custom_vs, top_k=7)
    assert custom_retriever.vector_store is custom_vs
    assert custom_retriever.top_k == 7

    results = custom_retriever.retrieve("dependency injection test")
    assert len(results) == 1
    assert results[0].page_content == "Injected mock content"
    custom_vs.similarity_search.assert_called_once_with(
        query="dependency injection test",
        k=7,
    )
