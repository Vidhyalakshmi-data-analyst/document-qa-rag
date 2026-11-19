"""Unit tests for src/document_loader.py."""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from src.document_loader import DocumentLoader


@pytest.fixture
def loader() -> DocumentLoader:
    """Fixture providing an instance of DocumentLoader."""
    return DocumentLoader()


@pytest.fixture
def valid_pdf_path(tmp_path: Path) -> Path:
    """Fixture providing a valid multi-page PDF file."""
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    writer.write(str(pdf_path))
    return pdf_path


@pytest.fixture
def valid_text_pdf_path(tmp_path: Path) -> Path:
    """Fixture providing a valid PDF containing extractable text."""
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT\n/F1 12 Tf\n72 712 Td\n(Hello RAG World) Tj\nET\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000338 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n417\n%%EOF\n"
    )
    pdf_path = tmp_path / "sample_with_text.pdf"
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path


def test_successfully_loads_valid_pdf(loader: DocumentLoader, valid_pdf_path: Path):
    """Test Case 1: Successfully loads a valid PDF file."""
    documents = loader.load_pdf(str(valid_pdf_path))
    assert documents is not None
    assert len(documents) == 2


def test_returns_list_of_langchain_documents(
    loader: DocumentLoader, valid_pdf_path: Path
):
    """Test Case 2: Returns a list of LangChain Document objects."""
    documents = loader.load_pdf(str(valid_pdf_path))
    assert isinstance(documents, list)
    for doc in documents:
        assert isinstance(doc, Document)


def test_preserves_page_content_and_metadata(
    loader: DocumentLoader, valid_text_pdf_path: Path
):
    """Test Case 3: Preserves page content and metadata (source and page number)."""
    documents = loader.load_pdf(str(valid_text_pdf_path))
    assert len(documents) == 1

    doc = documents[0]
    assert "Hello RAG World" in doc.page_content
    assert "source" in doc.metadata
    assert "page" in doc.metadata
    assert doc.metadata["source"] == str(valid_text_pdf_path.resolve())
    assert doc.metadata["page"] == 0


def test_rejects_non_existent_file(loader: DocumentLoader, tmp_path: Path):
    """Test Case 4: Rejects a non-existent file with FileNotFoundError."""
    missing_path = tmp_path / "non_existent_document.pdf"
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        loader.load_pdf(str(missing_path))


def test_rejects_non_pdf_extension(loader: DocumentLoader, tmp_path: Path):
    """Test Case 5: Rejects a file with a non-PDF extension with ValueError."""
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("plain text content", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid file extension"):
        loader.load_pdf(str(txt_file))


def test_rejects_empty_pdf_file(loader: DocumentLoader, tmp_path: Path):
    """Test Case 6: Rejects an empty PDF file (0 bytes) with ValueError."""
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.touch()

    with pytest.raises(ValueError, match="PDF file is empty \\(0 bytes\\)"):
        loader.load_pdf(str(empty_pdf))


def test_handles_corrupted_pdf_with_meaningful_exception(
    loader: DocumentLoader, tmp_path: Path
):
    """Test Case 7: Handles corrupted or invalid PDF with a meaningful RuntimeError."""
    corrupted_pdf = tmp_path / "corrupted.pdf"
    corrupted_pdf.write_bytes(b"This is corrupted, non-PDF arbitrary content.")

    with pytest.raises(RuntimeError, match="Failed to load and parse PDF file"):
        loader.load_pdf(str(corrupted_pdf))
