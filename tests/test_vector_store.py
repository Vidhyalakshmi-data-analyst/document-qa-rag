"""Unit tests for src/vector_store.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings

from config import VECTORSTORE_DIR
from src.vector_store import FAISSVectorStore


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    """Fixture providing deterministic in-memory fake embeddings."""
    return FakeEmbeddings(size=4)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Fixture providing sample document chunks with metadata."""
    return [
        Document(
            page_content="LangChain is a framework for developing LLM applications.",
            metadata={"source": "doc1.pdf", "page": 1},
        ),
        Document(
            page_content="FAISS provides fast similarity search for high-dimensional vectors.",
            metadata={"source": "doc2.pdf", "page": 2},
        ),
        Document(
            page_content="RAG pipelines augment language models with retrieved knowledge.",
            metadata={"source": "doc3.pdf", "page": 3},
        ),
        Document(
            page_content="Cross-encoder models rerank retrieved candidates for higher precision.",
            metadata={"source": "doc4.pdf", "page": 4},
        ),
        Document(
            page_content="Groq accelerates inference latency for large language models.",
            metadata={"source": "doc5.pdf", "page": 5},
        ),
        Document(
            page_content="Dense vector representations capture semantic similarity.",
            metadata={"source": "doc6.pdf", "page": 6},
        ),
    ]


@pytest.fixture
def vector_store_service(
    fake_embeddings: FakeEmbeddings, tmp_path: Path
) -> FAISSVectorStore:
    """Fixture providing FAISSVectorStore initialized with a temporary directory."""
    return FAISSVectorStore(
        embedding_service=fake_embeddings,
        vectorstore_dir=tmp_path / "vectorstore",
    )


def test_successful_faiss_index_creation(
    vector_store_service: FAISSVectorStore, sample_documents: list[Document]
):
    """Test Case 1: Successfully creates a FAISS index from document chunks."""
    index = vector_store_service.create_index(sample_documents)
    assert isinstance(index, FAISS)
    assert vector_store_service.vector_store is index


def test_empty_and_invalid_document_input(
    vector_store_service: FAISSVectorStore,
):
    """Test Case 2: Empty or invalid document input raises appropriate exceptions."""
    with pytest.raises(
        ValueError, match="Cannot create a vector index from an empty document list"
    ):
        vector_store_service.create_index([])

    with pytest.raises(
        TypeError, match="Expected documents to be a list of Document objects"
    ):
        vector_store_service.create_index(None)  # type: ignore

    with pytest.raises(
        TypeError, match="Expected documents to be a list of Document objects"
    ):
        vector_store_service.create_index("not a list")  # type: ignore

    with pytest.raises(
        TypeError, match="is not an instance of LangChain Document"
    ):
        vector_store_service.create_index([Document(page_content="Valid"), "Invalid"])  # type: ignore


def test_successful_index_persistence(
    vector_store_service: FAISSVectorStore,
    sample_documents: list[Document],
    tmp_path: Path,
):
    """Test Case 3: Successfully persists FAISS index to disk."""
    index = vector_store_service.create_index(sample_documents)
    save_dir = tmp_path / "saved_index"
    vector_store_service.save_index(index, path=save_dir)

    assert (save_dir / "index.faiss").exists()
    assert (save_dir / "index.pkl").exists()

    # Verify default path persistence
    vector_store_service.save_index(index)
    assert (vector_store_service.vectorstore_dir / "index.faiss").exists()
    assert (vector_store_service.vectorstore_dir / "index.pkl").exists()


def test_successful_loading_saved_index(
    vector_store_service: FAISSVectorStore,
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
    tmp_path: Path,
):
    """Test Case 4: Successfully loads a previously saved FAISS index."""
    index = vector_store_service.create_index(sample_documents)
    save_dir = tmp_path / "persist_store"
    vector_store_service.save_index(index, path=save_dir)

    new_service = FAISSVectorStore(
        embedding_service=fake_embeddings,
        vectorstore_dir=save_dir,
    )
    loaded_index = new_service.load_index()

    assert isinstance(loaded_index, FAISS)
    assert new_service.vector_store is loaded_index

    # Verify loaded index can perform similarity search
    results = new_service.similarity_search("LangChain", k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_missing_index_handling(
    vector_store_service: FAISSVectorStore, tmp_path: Path
):
    """Test Case 5: Missing index directory or missing files raise FileNotFoundError."""
    missing_dir = tmp_path / "non_existent_folder"
    with pytest.raises(FileNotFoundError, match="Vector store directory does not exist"):
        vector_store_service.load_index(path=missing_dir)

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="FAISS index file.*not found"):
        vector_store_service.load_index(path=empty_dir)


def test_similarity_search_returns_document_objects(
    vector_store_service: FAISSVectorStore, sample_documents: list[Document]
):
    """Test Case 6: Similarity search returns list of Document objects with metadata."""
    vector_store_service.create_index(sample_documents)
    results = vector_store_service.similarity_search("vector search", k=3)

    assert isinstance(results, list)
    assert len(results) == 3
    for doc in results:
        assert isinstance(doc, Document)
        assert "source" in doc.metadata
        assert "page" in doc.metadata


def test_similarity_search_respects_requested_k(
    vector_store_service: FAISSVectorStore, sample_documents: list[Document]
):
    """Test Case 7: Similarity search strictly respects the requested k value."""
    vector_store_service.create_index(sample_documents)

    res_1 = vector_store_service.similarity_search("LLM query", k=1)
    assert len(res_1) == 1

    res_4 = vector_store_service.similarity_search("LLM query", k=4)
    assert len(res_4) == 4

    res_5 = vector_store_service.similarity_search("LLM query", k=5)
    assert len(res_5) == 5


def test_invalid_query_handling(
    vector_store_service: FAISSVectorStore, sample_documents: list[Document]
):
    """Test Case 8: Invalid queries or uninitialized index raise appropriate exceptions."""
    # Uninitialized index check
    uninitialized_service = FAISSVectorStore(embedding_service=FakeEmbeddings(size=4))
    with pytest.raises(RuntimeError, match="No active FAISS index found"):
        uninitialized_service.similarity_search("sample query")

    # Invalid query types and empty content
    vector_store_service.create_index(sample_documents)

    with pytest.raises(TypeError, match="Expected query to be a string"):
        vector_store_service.similarity_search(None)  # type: ignore

    with pytest.raises(TypeError, match="Expected query to be a string"):
        vector_store_service.similarity_search(12345)  # type: ignore

    with pytest.raises(
        ValueError, match="Query string cannot be empty or whitespace only"
    ):
        vector_store_service.similarity_search("")

    with pytest.raises(
        ValueError, match="Query string cannot be empty or whitespace only"
    ):
        vector_store_service.similarity_search("    \t \n ")


def test_invalid_k_handling(
    vector_store_service: FAISSVectorStore, sample_documents: list[Document]
):
    """Test Case 9: Non-positive or non-integer k values raise ValueError."""
    vector_store_service.create_index(sample_documents)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        vector_store_service.similarity_search("test", k=0)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        vector_store_service.similarity_search("test", k=-3)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        vector_store_service.similarity_search("test", k="five")  # type: ignore


def test_storage_and_loading_error_handling(
    vector_store_service: FAISSVectorStore,
    sample_documents: list[Document],
    tmp_path: Path,
):
    """Test Case 10: Storage and loading errors raise RuntimeError with original cause."""
    with pytest.raises(
        TypeError, match="Expected vector_store to be a FAISS instance"
    ):
        vector_store_service.save_index("not_a_faiss_instance")  # type: ignore

    # Mock storage error
    mock_vs = MagicMock(spec=FAISS)
    mock_vs.save_local.side_effect = PermissionError("Permission denied during write")
    with pytest.raises(RuntimeError, match="Failed to persist FAISS index") as save_exc:
        vector_store_service.save_index(mock_vs, path=tmp_path / "denied")
    assert isinstance(save_exc.value.__cause__, PermissionError)

    # Mock load deserialization error
    valid_dir = tmp_path / "valid_save_dir"
    index = vector_store_service.create_index(sample_documents)
    vector_store_service.save_index(index, path=valid_dir)

    with patch(
        "src.vector_store.FAISS.load_local",
        side_effect=RuntimeError("Corrupt pickle stream"),
    ):
        with pytest.raises(RuntimeError, match="Failed to load FAISS index") as load_exc:
            vector_store_service.load_index(path=valid_dir)
        assert isinstance(load_exc.value.__cause__, RuntimeError)


def test_dependency_injection_works_with_mocked_embeddings():
    """Test Case 11: Dependency injection works seamlessly with mocked embeddings."""
    mock_embeddings = MagicMock(spec=Embeddings)
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_embeddings.embed_query.return_value = [0.1, 0.2]

    service = FAISSVectorStore(embedding_service=mock_embeddings)
    docs = [
        Document(page_content="Mock doc 1", metadata={"source": "m1.pdf"}),
        Document(page_content="Mock doc 2", metadata={"source": "m2.pdf"}),
    ]

    index = service.create_index(docs)
    assert isinstance(index, FAISS)
    mock_embeddings.embed_documents.assert_called_once()

    results = service.similarity_search("query text", k=1)
    assert len(results) == 1
    mock_embeddings.embed_query.assert_called_once_with("query text")
