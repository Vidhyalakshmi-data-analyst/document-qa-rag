"""Unit tests for src/embeddings.py."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL_NAME
from src.embeddings import EmbeddingService


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Fixture providing a mock LangChain Embeddings instance."""
    mock = MagicMock(spec=Embeddings)
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def service_with_mock(mock_embeddings: MagicMock) -> EmbeddingService:
    """Fixture providing an EmbeddingService with injected mock embeddings."""
    return EmbeddingService(embeddings=mock_embeddings)


def test_embedding_service_initialization():
    """Test Case 1: EmbeddingService initializes lazily without loading the model."""
    service = EmbeddingService()
    assert service.model_name == EMBEDDING_MODEL_NAME
    assert service._embeddings is None


def test_correct_configured_model_name_used():
    """Test Case 2: Uses the configured model name from config.py or custom override."""
    # Default uses configured model name
    service = EmbeddingService()
    assert service.model_name == EMBEDDING_MODEL_NAME
    assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    # Custom model name override
    custom_service = EmbeddingService(model_name="custom-org/custom-model")
    assert custom_service.model_name == "custom-org/custom-model"

    # Verifies lazy loader passes the configured model_name to HuggingFaceEmbeddings
    with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf_cls:
        mock_instance = MagicMock()
        mock_hf_cls.return_value = mock_instance

        _ = service.embeddings
        mock_hf_cls.assert_called_once_with(model_name=EMBEDDING_MODEL_NAME)


def test_embed_documents_returns_numerical_vectors(
    service_with_mock: EmbeddingService, mock_embeddings: MagicMock
):
    """Test Case 3: embed_documents returns numerical vectors."""
    vectors = service_with_mock.embed_documents(["sample document text"])
    assert isinstance(vectors, list)
    assert len(vectors) > 0
    for vec in vectors:
        assert isinstance(vec, list)
        for val in vec:
            assert isinstance(val, (int, float))


def test_embed_query_returns_numerical_vector(
    service_with_mock: EmbeddingService, mock_embeddings: MagicMock
):
    """Test Case 4: embed_query returns a single numerical vector."""
    vector = service_with_mock.embed_query("sample query text")
    assert isinstance(vector, list)
    assert len(vector) > 0
    for val in vector:
        assert isinstance(val, (int, float))


def test_multiple_texts_handled_correctly(
    service_with_mock: EmbeddingService, mock_embeddings: MagicMock
):
    """Test Case 5: Multiple document texts are passed to the embedder and processed."""
    input_texts = ["Doc 1 text", "Doc 2 text", "Doc 3 text"]
    mock_embeddings.embed_documents.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]

    vectors = service_with_mock.embed_documents(input_texts)
    assert len(vectors) == 3
    mock_embeddings.embed_documents.assert_called_once_with(input_texts)


def test_empty_document_input_handled_correctly(
    service_with_mock: EmbeddingService, mock_embeddings: MagicMock
):
    """Test Case 6: Empty document input returns an empty list without calling embedder."""
    result = service_with_mock.embed_documents([])
    assert result == []
    mock_embeddings.embed_documents.assert_not_called()


def test_invalid_query_and_document_input_rejected(
    service_with_mock: EmbeddingService,
):
    """Test Case 7: Invalid query and document inputs are rejected with clear exceptions."""
    # Query validation
    with pytest.raises(TypeError, match="Expected query text to be a string"):
        service_with_mock.embed_query(None)  # type: ignore

    with pytest.raises(TypeError, match="Expected query text to be a string"):
        service_with_mock.embed_query(12345)  # type: ignore

    with pytest.raises(ValueError, match="Query text cannot be empty or whitespace only"):
        service_with_mock.embed_query("")

    with pytest.raises(ValueError, match="Query text cannot be empty or whitespace only"):
        service_with_mock.embed_query("    \t\n  ")

    # Document list validation
    with pytest.raises(TypeError, match="Expected texts to be a list of strings"):
        service_with_mock.embed_documents(None)  # type: ignore

    with pytest.raises(TypeError, match="Expected texts to be a list of strings"):
        service_with_mock.embed_documents("not a list")  # type: ignore

    with pytest.raises(TypeError, match="is not a string"):
        service_with_mock.embed_documents(["valid string", None, "another string"])  # type: ignore

    with pytest.raises(TypeError, match="is not a string"):
        service_with_mock.embed_documents(["valid", 999])  # type: ignore


def test_model_initialization_errors_handled():
    """Test Case 8: Model initialization errors are wrapped in RuntimeError with original cause."""
    with patch(
        "src.embeddings.HuggingFaceEmbeddings",
        side_effect=OSError("Network connection failure or model weights missing"),
    ):
        service = EmbeddingService()
        with pytest.raises(RuntimeError, match="Failed to initialize embedding model") as exc_info:
            _ = service.embeddings

        assert isinstance(exc_info.value.__cause__, OSError)


def test_embedding_generation_errors_handled(
    service_with_mock: EmbeddingService, mock_embeddings: MagicMock
):
    """Test Case 9: Embedding generation failures raise RuntimeError preserving original cause."""
    # Document embedding failure
    mock_embeddings.embed_documents.side_effect = RuntimeError("GPU out of memory")
    with pytest.raises(RuntimeError, match="Failed to generate embeddings for documents") as doc_exc:
        service_with_mock.embed_documents(["sample text"])
    assert isinstance(doc_exc.value.__cause__, RuntimeError)

    # Query embedding failure
    mock_embeddings.embed_query.side_effect = ValueError("Inference dimension mismatch")
    with pytest.raises(RuntimeError, match="Failed to generate embedding for query") as q_exc:
        service_with_mock.embed_query("sample query")
    assert isinstance(q_exc.value.__cause__, ValueError)


def test_dependency_injection_works_without_downloading_model(
    mock_embeddings: MagicMock,
):
    """Test Case 10: Injected embeddings dependency works with zero model downloads or imports."""
    with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf_cls:
        service = EmbeddingService(embeddings=mock_embeddings)
        assert service.embeddings is mock_embeddings

        doc_res = service.embed_documents(["text 1"])
        query_res = service.embed_query("query 1")

        assert doc_res == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert query_res == [0.1, 0.2, 0.3]

        # HuggingFaceEmbeddings was never instantiated
        mock_hf_cls.assert_not_called()
