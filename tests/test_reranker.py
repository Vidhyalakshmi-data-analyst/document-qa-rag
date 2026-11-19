"""Unit tests for src/reranker.py."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from config import CROSS_ENCODER_MODEL_NAME, RERANKING_TOP_K
from src.reranker import DocumentReranker


@pytest.fixture
def sample_documents() -> list[Document]:
    """Fixture providing 5 candidate Document objects with distinct content and metadata."""
    return [
        Document(
            page_content="Document 0: Python standard library documentation.",
            metadata={"source": "python.pdf", "page": 10},
        ),
        Document(
            page_content="Document 1: Detailed explanation of Retrieval-Augmented Generation.",
            metadata={"source": "rag_paper.pdf", "page": 1},
        ),
        Document(
            page_content="Document 2: General machine learning overview and training loops.",
            metadata={"source": "ml_intro.pdf", "page": 5},
        ),
        Document(
            page_content="Document 3: Advanced RAG retrieval and Cross-Encoder reranking.",
            metadata={"source": "rag_advanced.pdf", "page": 2},
        ),
        Document(
            page_content="Document 4: Unrelated cooking recipes and baking instructions.",
            metadata={"source": "cookbook.pdf", "page": 12},
        ),
    ]


@pytest.fixture
def mock_cross_encoder() -> MagicMock:
    """Fixture providing a mock CrossEncoder instance with predefined relevance scores."""
    mock = MagicMock()
    # Scores for documents [0, 1, 2, 3, 4]:
    # Highest to lowest: doc3 (4.8) > doc1 (3.5) > doc2 (0.2) > doc0 (-1.1) > doc4 (-5.0)
    mock.predict.return_value = [-1.1, 3.5, 0.2, 4.8, -5.0]
    return mock


@pytest.fixture
def reranker(mock_cross_encoder: MagicMock) -> DocumentReranker:
    """Fixture providing a DocumentReranker with an injected mock model."""
    return DocumentReranker(model=mock_cross_encoder)


def test_successful_reranking_of_documents(
    reranker: DocumentReranker, sample_documents: list[Document]
):
    """Test Case 1: Successfully reranks documents and returns Document instances."""
    results = reranker.rerank("What is RAG and reranking?", sample_documents)
    assert isinstance(results, list)
    assert len(results) == 3
    for doc in results:
        assert isinstance(doc, Document)


def test_documents_sorted_in_descending_score_order(
    reranker: DocumentReranker, sample_documents: list[Document]
):
    """Test Case 2: Documents are strictly sorted in descending score order."""
    # Expected ranking based on scores [-1.1, 3.5, 0.2, 4.8, -5.0]:
    # 1st: doc3 (score 4.8)
    # 2nd: doc1 (score 3.5)
    # 3rd: doc2 (score 0.2)
    results = reranker.rerank("Query", sample_documents)

    assert results[0] is sample_documents[3]
    assert results[1] is sample_documents[1]
    assert results[2] is sample_documents[2]


def test_default_top_3_selection(
    reranker: DocumentReranker, sample_documents: list[Document]
):
    """Test Case 3: Returns top 3 documents by default, matching RERANKING_TOP_K."""
    assert reranker.top_k == 3
    assert reranker.top_k == RERANKING_TOP_K

    results = reranker.rerank("Query", sample_documents)
    assert len(results) == 3


def test_custom_top_k_selection(
    reranker: DocumentReranker, sample_documents: list[Document]
):
    """Test Case 4: Supports custom top_k override during rerank call."""
    top_1 = reranker.rerank("Query", sample_documents, top_k=1)
    assert len(top_1) == 1
    assert top_1[0] is sample_documents[3]

    top_4 = reranker.rerank("Query", sample_documents, top_k=4)
    assert len(top_4) == 4
    assert top_4 == [
        sample_documents[3],
        sample_documents[1],
        sample_documents[2],
        sample_documents[0],
    ]


def test_empty_document_list(
    reranker: DocumentReranker, mock_cross_encoder: MagicMock
):
    """Test Case 5: Empty document list returns an empty list without calling predict."""
    results = reranker.rerank("Query", [])
    assert results == []
    mock_cross_encoder.predict.assert_not_called()


def test_invalid_or_empty_question(
    reranker: DocumentReranker, sample_documents: list[Document]
):
    """Test Case 6: Empty, whitespace-only, or non-string questions are rejected."""
    with pytest.raises(
        ValueError, match="question cannot be empty or whitespace only"
    ):
        reranker.rerank("", sample_documents)

    with pytest.raises(
        ValueError, match="question cannot be empty or whitespace only"
    ):
        reranker.rerank("   \t \n  ", sample_documents)

    with pytest.raises(TypeError, match="Expected question to be a string"):
        reranker.rerank(None, sample_documents)  # type: ignore

    with pytest.raises(TypeError, match="Expected question to be a string"):
        reranker.rerank(12345, sample_documents)  # type: ignore


def test_invalid_top_k_values(
    reranker: DocumentReranker, sample_documents: list[Document]
):
    """Test Case 7: Non-positive or non-integer top_k values raise ValueError."""
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        reranker.rerank("Query", sample_documents, top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        reranker.rerank("Query", sample_documents, top_k=-2)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        reranker.rerank("Query", sample_documents, top_k="invalid")  # type: ignore

    # Also test constructor validation
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        DocumentReranker(top_k=0)


def test_correct_question_document_pairs_passed_to_predict(
    reranker: DocumentReranker,
    mock_cross_encoder: MagicMock,
    sample_documents: list[Document],
):
    """Test Case 8: Verifies (question, document.page_content) pairs passed to CrossEncoder."""
    question = "How does reranking improve precision?"
    reranker.rerank(question, sample_documents)

    expected_pairs = [(question, doc.page_content) for doc in sample_documents]
    mock_cross_encoder.predict.assert_called_once_with(expected_pairs)


def test_original_document_metadata_preserved(
    reranker: DocumentReranker, sample_documents: list[Document]
):
    """Test Case 9: Verifies document objects and metadata remain intact after reranking."""
    results = reranker.rerank("Query", sample_documents)

    for doc in results:
        assert "source" in doc.metadata
        assert "page" in doc.metadata

    # Specific metadata check for the top-ranked document (doc3)
    top_doc = results[0]
    assert top_doc.metadata["source"] == "rag_advanced.pdf"
    assert top_doc.metadata["page"] == 2


def test_model_initialization_errors():
    """Test Case 10: Model loading failure raises RuntimeError preserving original cause."""
    with patch(
        "src.reranker.CrossEncoder",
        side_effect=OSError("Network failure downloading weights"),
    ):
        service = DocumentReranker()
        with pytest.raises(
            RuntimeError, match="Failed to initialize Cross-Encoder model"
        ) as exc_info:
            _ = service.model

        assert isinstance(exc_info.value.__cause__, OSError)


def test_prediction_errors(
    reranker: DocumentReranker,
    mock_cross_encoder: MagicMock,
    sample_documents: list[Document],
):
    """Test Case 11: Cross-Encoder predict() failure raises RuntimeError with cause."""
    mock_cross_encoder.predict.side_effect = RuntimeError("CUDA runtime crash")

    with pytest.raises(
        RuntimeError, match="Failed to rerank documents using Cross-Encoder"
    ) as exc_info:
        reranker.rerank("Query", sample_documents)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_dependency_injection_works_without_model_downloads(
    mock_cross_encoder: MagicMock, sample_documents: list[Document]
):
    """Test Case 12: Injected model avoids any CrossEncoder instantiation or download."""
    with patch("src.reranker.CrossEncoder") as mock_cls:
        reranker = DocumentReranker(model=mock_cross_encoder)
        assert reranker.model is mock_cross_encoder
        assert reranker.model_name == CROSS_ENCODER_MODEL_NAME

        results = reranker.rerank("Query", sample_documents)
        assert len(results) == 3
        mock_cls.assert_not_called()
