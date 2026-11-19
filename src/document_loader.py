"""Document loader module for the Document-Based RAG system.

This module is solely responsible for validating and loading PDF documents
into LangChain Document objects using PyPDFLoader while preserving metadata.
"""

from pathlib import Path
from typing import Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class DocumentLoader:
    """Loader service responsible for validating and reading PDF files."""

    def load_pdf(self, file_path: Union[str, Path]) -> list[Document]:
        """Load a PDF file and return its pages as LangChain Document objects.

        Validates that the file exists, has a .pdf extension, and is not empty.
        Extracts each page as an independent Document while preserving metadata
        such as source file path and page number.

        Args:
            file_path: The filesystem path to the target PDF file (str or Path).

        Returns:
            list[Document]: A list of LangChain Document objects representing pages.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file path is not a regular file, does not have
                a .pdf extension, or the file is empty (0 bytes).
            RuntimeError: If PyPDFLoader fails to parse or extract the PDF content.
        """
        path = Path(file_path).resolve()

        # 1. Validate that the file exists
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at path: {path}")

        # 2. Validate that the path points to a regular file
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        # 3. Validate file extension (.pdf)
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Invalid file extension '{path.suffix}'. Only '.pdf' files are supported: {path.name}"
            )

        # 4. Validate that the file is not empty (greater than 0 bytes)
        if path.stat().st_size == 0:
            raise ValueError(f"PDF file is empty (0 bytes): {path.name}")

        # 5. Load PDF using LangChain PyPDFLoader
        try:
            loader = PyPDFLoader(file_path=str(path))
            documents = loader.load()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load and parse PDF file '{path.name}': {exc}"
            ) from exc

        # 6. Ensure metadata integrity (source and page preservation)
        for idx, doc in enumerate(documents):
            if "source" not in doc.metadata:
                doc.metadata["source"] = str(path)
            if "page" not in doc.metadata:
                doc.metadata["page"] = idx

        return documents
