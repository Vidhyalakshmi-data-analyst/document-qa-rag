"""Unit tests for src/text_splitter.py."""

import pytest
from langchain_core.documents import Document

from config import CHUNK_OVERLAP, CHUNK_SIZE
from src.text_splitter import DocumentTextSplitter


@pytest.fixture
def splitter() -> DocumentTextSplitter:
    """Fixture providing a DocumentTextSplitter with default config parameters."""
    return DocumentTextSplitter()


@pytest.fixture
def long_synthetic_text() -> str:
    """Fixture providing a long synthetic text string for chunking tests."""
    sentences = [
        f"This is synthetic sentence {i} providing structured context to test "
        f"the recursive character text splitter in our modular RAG pipeline."
        for i in range(50)
    ]
    return " ".join(sentences)


@pytest.fixture
def sample_document(long_synthetic_text: str) -> Document:
    """Fixture providing a sample LangChain Document with metadata."""
    return Document(
        page_content=long_synthetic_text,
        metadata={"source": "docs/architecture_guide.pdf", "page": 3},
    )


def test_successfully_splits_long_document(
    splitter: DocumentTextSplitter, sample_document: Document
):
    """Test Case 1: Successfully splits a long document into multiple chunks."""
    chunks = splitter.split_documents([sample_document])
    assert len(chunks) > 1


def test_returns_langchain_document_objects(
    splitter: DocumentTextSplitter, sample_document: Document
):
    """Test Case 2: Returns a list of LangChain Document objects."""
    chunks = splitter.split_documents([sample_document])
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Document)


def test_respects_configured_chunk_size(
    splitter: DocumentTextSplitter, sample_document: Document
):
    """Test Case 3: Respects configured chunk size (default 500 characters)."""
    assert splitter.chunk_size == CHUNK_SIZE
    assert splitter.chunk_size == 500

    chunks = splitter.split_documents([sample_document])
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.page_content) <= splitter.chunk_size


def test_applies_configured_overlap(
    splitter: DocumentTextSplitter, sample_document: Document
):
    """Test Case 4: Applies configured overlap between consecutive chunks."""
    assert splitter.chunk_overlap == CHUNK_OVERLAP
    assert splitter.chunk_overlap == 100

    chunks = splitter.split_documents([sample_document])
    assert len(chunks) >= 2

    # Verify that consecutive chunks share non-empty overlapping text
    overlap_detected = False
    for i in range(len(chunks) - 1):
        tail = chunks[i].page_content[-splitter.chunk_overlap :]
        head = chunks[i + 1].page_content[: splitter.chunk_overlap]
        # Check if words from the tail of chunk i appear in the head of chunk i+1
        tail_words = set(tail.split())
        head_words = set(head.split())
        if tail_words.intersection(head_words):
            overlap_detected = True
            break
    assert overlap_detected, "Overlap was not detected between consecutive chunks."

    # Compare total character length with non-overlapping splitter
    no_overlap_splitter = DocumentTextSplitter(chunk_size=500, chunk_overlap=0)
    no_overlap_chunks = no_overlap_splitter.split_documents([sample_document])
    total_len_with_overlap = sum(len(c.page_content) for c in chunks)
    total_len_no_overlap = sum(len(c.page_content) for c in no_overlap_chunks)
    assert total_len_with_overlap > total_len_no_overlap


def test_preserves_metadata_in_every_chunk(
    splitter: DocumentTextSplitter, sample_document: Document
):
    """Test Case 5: Preserves source and page metadata in every chunk."""
    chunks = splitter.split_documents([sample_document])
    assert len(chunks) > 1

    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
        assert chunk.metadata["source"] == "docs/architecture_guide.pdf"
        assert chunk.metadata["page"] == 3


def test_handles_multiple_input_documents(
    splitter: DocumentTextSplitter, long_synthetic_text: str
):
    """Test Case 6: Handles multiple input documents and preserves distinct metadata."""
    doc1 = Document(
        page_content=long_synthetic_text,
        metadata={"source": "report_a.pdf", "page": 1},
    )
    doc2 = Document(
        page_content=long_synthetic_text,
        metadata={"source": "report_b.pdf", "page": 2},
    )

    chunks = splitter.split_documents([doc1, doc2])
    assert len(chunks) > 2

    sources = {c.metadata.get("source") for c in chunks}
    pages = {c.metadata.get("page") for c in chunks}
    assert sources == {"report_a.pdf", "report_b.pdf"}
    assert pages == {1, 2}


def test_returns_empty_list_for_empty_input(splitter: DocumentTextSplitter):
    """Test Case 7: Returns an empty list when input is empty."""
    chunks = splitter.split_documents([])
    assert chunks == []


def test_handles_invalid_input_appropriately(splitter: DocumentTextSplitter):
    """Test Case 8: Raises appropriate exceptions for invalid input and bad parameters."""
    # Non-list input
    with pytest.raises(TypeError, match="Expected a list of Document objects"):
        splitter.split_documents(None)  # type: ignore

    with pytest.raises(TypeError, match="Expected a list of Document objects"):
        splitter.split_documents("invalid string input")  # type: ignore

    # List containing non-Document items
    with pytest.raises(TypeError, match="is not an instance of LangChain Document"):
        splitter.split_documents([Document(page_content="Valid"), "Invalid item"])  # type: ignore

    # Invalid initialization parameters
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        DocumentTextSplitter(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        DocumentTextSplitter(chunk_size=-100)

    with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
        DocumentTextSplitter(chunk_size=500, chunk_overlap=-10)

    with pytest.raises(ValueError, match="chunk_overlap.*must be strictly less than"):
        DocumentTextSplitter(chunk_size=200, chunk_overlap=200)

    with pytest.raises(ValueError, match="chunk_overlap.*must be strictly less than"):
        DocumentTextSplitter(chunk_size=200, chunk_overlap=250)
