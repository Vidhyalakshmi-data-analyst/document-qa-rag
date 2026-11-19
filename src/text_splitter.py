"""Text splitter module for the Document-Based RAG system.

This module is responsible for breaking down loaded LangChain Document objects
into smaller, overlapping chunks using RecursiveCharacterTextSplitter while
preserving metadata such as source file paths and page numbers.
"""

from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


class DocumentTextSplitter:
    """Service responsible for chunking documents into manageable text pieces."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        """Initialize DocumentTextSplitter with chunking hyperparameters.

        Args:
            chunk_size: Maximum size of each text chunk in characters.
                Defaults to CHUNK_SIZE from config.py.
            chunk_overlap: Character overlap between consecutive chunks.
                Defaults to CHUNK_OVERLAP from config.py.

        Raises:
            ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
        """
        self.chunk_size: int = chunk_size if chunk_size is not None else CHUNK_SIZE
        self.chunk_overlap: int = (
            chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
        )

        self._validate_parameters(self.chunk_size, self.chunk_overlap)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @staticmethod
    def _validate_parameters(chunk_size: int, chunk_overlap: int) -> None:
        """Validate chunk size and overlap parameters.

        Args:
            chunk_size: The target chunk size.
            chunk_overlap: The target chunk overlap.

        Raises:
            ValueError: If parameters violate domain constraints.
        """
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be a positive integer, got: {chunk_size}"
            )
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be a non-negative integer, got: {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})."
            )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split a list of Document objects into smaller chunks with preserved metadata.

        Args:
            documents: List of LangChain Document objects to split.

        Returns:
            list[Document]: List of split Document objects. Returns an empty list
                when the input list is empty.

        Raises:
            TypeError: If input is not a list or elements within the list are not
                instances of LangChain Document.
        """
        if documents is None or not isinstance(documents, list):
            raise TypeError(
                f"Expected a list of Document objects, got: {type(documents).__name__}"
            )

        if not documents:
            return []

        for idx, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise TypeError(
                    f"Item at index {idx} is not an instance of LangChain Document: "
                    f"got {type(doc).__name__}"
                )

        chunks = self._splitter.split_documents(documents)
        return chunks
