"""Unit tests for src/ingestion_pipeline.py."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, call

import pytest
from langchain_core.documents import Document

from config import VECTORSTORE_DIR
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingService
from src.ingestion_pipeline import DocumentIngestionPipeline
from src.text_splitter import DocumentTextSplitter
from src.vector_store import FAISSVectorStore


@pytest.fixture
def sample_pages() -> list[Document]:
    """Fixture providing loaded sample document pages."""
    return [
        Document(
            page_content="Page 1: Overview of system architecture.",
            metadata={"source": "doc1.pdf", "page": 0},
        ),
        Document(
            page_content="Page 2: Detailed ingestion pipeline design.",
            metadata={"source": "doc1.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Fixture providing split document chunks."""
    return [
        Document(
            page_content="Chunk 1: Overview.",
            metadata={"source": "doc1.pdf", "page": 0},
        ),
        Document(
            page_content="Chunk 2: Architecture.",
            metadata={"source": "doc1.pdf", "page": 0},
        ),
        Document(
            page_content="Chunk 3: Ingestion.",
            metadata={"source": "doc1.pdf", "page": 1},
        ),
        Document(
            page_content="Chunk 4: Pipeline.",
            metadata={"source": "doc1.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def mock_loader(sample_pages: list[Document]) -> MagicMock:
    """Fixture providing a mock DocumentLoader."""
    loader = MagicMock(spec=DocumentLoader)
    loader.load_pdf.return_value = list(sample_pages)
    return loader


@pytest.fixture
def mock_splitter(sample_chunks: list[Document]) -> MagicMock:
    """Fixture providing a mock DocumentTextSplitter."""
    splitter = MagicMock(spec=DocumentTextSplitter)
    splitter.split_documents.return_value = list(sample_chunks)
    return splitter


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Fixture providing a mock FAISSVectorStore."""
    store = MagicMock(spec=FAISSVectorStore)
    store.create_index.return_value = MagicMock()
    store.vectorstore_dir = Path("/mock/vectorstore/dir")
    return store


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Fixture providing a mock EmbeddingService."""
    return MagicMock(spec=EmbeddingService)


@pytest.fixture
def pipeline(
    mock_loader: MagicMock,
    mock_splitter: MagicMock,
    mock_vector_store: MagicMock,
    mock_embedding_service: MagicMock,
) -> DocumentIngestionPipeline:
    """Fixture providing DocumentIngestionPipeline with injected mock components."""
    return DocumentIngestionPipeline(
        document_loader=mock_loader,
        text_splitter=mock_splitter,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        vectorstore_dir="/mock/vectorstore/dir",
    )


def test_successful_ingestion_single_pdf(
    pipeline: DocumentIngestionPipeline,
    mock_loader: MagicMock,
    mock_splitter: MagicMock,
    mock_vector_store: MagicMock,
):
    """Test Case 1: Successfully ingests a single PDF and returns structured metadata."""
    result = pipeline.ingest_documents(["guide.pdf"])

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["files_processed"] == 1
    assert result["pages_loaded"] == 2
    assert result["chunks_generated"] == 4
    assert result["vector_store_path"] == str(mock_vector_store.vectorstore_dir)
    mock_loader.load_pdf.assert_called_once_with("guide.pdf")


def test_successful_ingestion_multiple_pdfs(
    pipeline: DocumentIngestionPipeline,
    mock_loader: MagicMock,
    mock_splitter: MagicMock,
):
    """Test Case 2: Successfully processes multiple PDFs and aggregates pages."""
    mock_loader.load_pdf.side_effect = [
        [Document(page_content="Doc A page", metadata={"source": "a.pdf"})],
        [Document(page_content="Doc B page", metadata={"source": "b.pdf"})],
    ]
    mock_splitter.split_documents.return_value = [
        Document(page_content="Chunk A", metadata={"source": "a.pdf"}),
        Document(page_content="Chunk B", metadata={"source": "b.pdf"}),
    ]

    result = pipeline.ingest_documents(["a.pdf", "b.pdf"])

    assert result["files_processed"] == 2
    assert result["pages_loaded"] == 2
    assert result["chunks_generated"] == 2
    assert mock_loader.load_pdf.call_count == 2


def test_correct_order_of_component_execution(
    pipeline: DocumentIngestionPipeline,
    mock_loader: MagicMock,
    mock_splitter: MagicMock,
    mock_vector_store: MagicMock,
    sample_pages: list[Document],
    sample_chunks: list[Document],
):
    """Test Case 3: Ensures components are executed in strict sequential order."""
    execution_tracker = Mock()
    execution_tracker.attach_mock(mock_loader.load_pdf, "load_pdf")
    execution_tracker.attach_mock(mock_splitter.split_documents, "split_documents")
    execution_tracker.attach_mock(mock_vector_store.create_index, "create_index")
    execution_tracker.attach_mock(mock_vector_store.save_index, "save_index")

    pipeline.ingest_documents(["doc1.pdf"])

    expected_calls = [
        call.load_pdf("doc1.pdf"),
        call.split_documents(sample_pages),
        call.create_index(sample_chunks),
        call.save_index(mock_vector_store.create_index.return_value),
    ]
    assert execution_tracker.mock_calls == expected_calls


def test_correct_number_of_loaded_documents_and_generated_chunks(
    pipeline: DocumentIngestionPipeline,
    mock_loader: MagicMock,
    mock_splitter: MagicMock,
):
    """Test Case 4: Accurately records loaded document and chunk counts."""
    docs = [Document(page_content=f"Page {i}") for i in range(5)]
    chunks = [Document(page_content=f"Chunk {i}") for i in range(12)]
    mock_loader.load_pdf.return_value = docs
    mock_splitter.split_documents.return_value = chunks

    result = pipeline.ingest_documents(["doc_large.pdf"])

    assert result["pages_loaded"] == 5
    assert result["chunks_generated"] == 12


def test_faiss_index_creation_and_persistence_are_invoked(
    pipeline: DocumentIngestionPipeline,
    mock_splitter: MagicMock,
    mock_vector_store: MagicMock,
    sample_chunks: list[Document],
):
    """Test Case 5: Confirms FAISS index is created from chunks and persisted to disk."""
    pipeline.ingest_documents(["manual.pdf"])

    mock_vector_store.create_index.assert_called_once_with(sample_chunks)
    mock_vector_store.save_index.assert_called_once_with(
        mock_vector_store.create_index.return_value
    )


def test_empty_pdf_path_list_rejected(pipeline: DocumentIngestionPipeline):
    """Test Case 6: Empty, non-list, or invalid path inputs raise appropriate exceptions."""
    with pytest.raises(ValueError, match="pdf_paths list cannot be empty"):
        pipeline.ingest_documents([])

    with pytest.raises(TypeError, match="Expected pdf_paths to be a list"):
        pipeline.ingest_documents(None)  # type: ignore

    with pytest.raises(TypeError, match="Expected pdf_paths to be a list"):
        pipeline.ingest_documents("single_file.pdf")  # type: ignore

    with pytest.raises(TypeError, match="is not a str or Path"):
        pipeline.ingest_documents(["valid.pdf", 42])  # type: ignore


def test_empty_loaded_documents_handled(
    pipeline: DocumentIngestionPipeline, mock_loader: MagicMock
):
    """Test Case 7: Raises ValueError when loaded documents list is empty."""
    mock_loader.load_pdf.return_value = []

    with pytest.raises(
        ValueError, match="No documents or pages could be extracted"
    ):
        pipeline.ingest_documents(["empty_document.pdf"])


def test_empty_chunks_handled(
    pipeline: DocumentIngestionPipeline, mock_splitter: MagicMock
):
    """Test Case 8: Raises ValueError when text splitter returns zero chunks."""
    mock_splitter.split_documents.return_value = []

    with pytest.raises(
        ValueError, match="Text splitting produced zero chunks"
    ):
        pipeline.ingest_documents(["blank_pages.pdf"])


def test_pdf_loading_errors_propagated_meaningfully(
    pipeline: DocumentIngestionPipeline, mock_loader: MagicMock
):
    """Test Case 9: Propagates PDF loader errors directly without suppression."""
    mock_loader.load_pdf.side_effect = FileNotFoundError(
        "PDF file not found at path: /missing.pdf"
    )
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        pipeline.ingest_documents(["/missing.pdf"])

    mock_loader.load_pdf.side_effect = RuntimeError(
        "Failed to load and parse PDF file 'corrupt.pdf'"
    )
    with pytest.raises(RuntimeError, match="Failed to load and parse PDF file"):
        pipeline.ingest_documents(["corrupt.pdf"])


def test_chunking_errors_handled(
    pipeline: DocumentIngestionPipeline, mock_splitter: MagicMock
):
    """Test Case 10: Propagates text splitting exceptions without silent suppression."""
    mock_splitter.split_documents.side_effect = RuntimeError(
        "Text splitter failed unexpectedly"
    )

    with pytest.raises(RuntimeError, match="Text splitter failed unexpectedly"):
        pipeline.ingest_documents(["doc.pdf"])


def test_vector_index_creation_and_persistence_errors_handled(
    pipeline: DocumentIngestionPipeline, mock_vector_store: MagicMock
):
    """Test Case 11: Propagates vector index creation and persistence errors."""
    # Index creation error
    mock_vector_store.create_index.side_effect = RuntimeError(
        "FAISS failed to build index"
    )
    with pytest.raises(RuntimeError, match="FAISS failed to build index"):
        pipeline.ingest_documents(["doc.pdf"])

    # Persistence error
    mock_vector_store.create_index.side_effect = None
    mock_vector_store.save_index.side_effect = RuntimeError(
        "Permission denied writing index.faiss"
    )
    with pytest.raises(RuntimeError, match="Permission denied writing index.faiss"):
        pipeline.ingest_documents(["doc.pdf"])


def test_dependency_injection_works_correctly(
    mock_loader: MagicMock,
    mock_splitter: MagicMock,
    mock_vector_store: MagicMock,
    mock_embedding_service: MagicMock,
    tmp_path: Path,
):
    """Test Case 12: Dependency injection properly assigns and uses components."""
    custom_dir = tmp_path / "custom_vectorstore"
    pipeline = DocumentIngestionPipeline(
        document_loader=mock_loader,
        text_splitter=mock_splitter,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        vectorstore_dir=custom_dir,
    )

    assert pipeline.document_loader is mock_loader
    assert pipeline.text_splitter is mock_splitter
    assert pipeline.embedding_service is mock_embedding_service
    assert pipeline.vector_store is mock_vector_store
    assert pipeline.vectorstore_dir == custom_dir.resolve()

    # Verify default initialization behaves cleanly without external calls
    default_pipeline = DocumentIngestionPipeline()
    assert isinstance(default_pipeline.document_loader, DocumentLoader)
    assert isinstance(default_pipeline.text_splitter, DocumentTextSplitter)
    assert isinstance(default_pipeline.embedding_service, EmbeddingService)
    assert isinstance(default_pipeline.vector_store, FAISSVectorStore)
    assert default_pipeline.vectorstore_dir == VECTORSTORE_DIR
