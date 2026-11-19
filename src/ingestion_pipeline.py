"""Document ingestion pipeline module for the Document-Based RAG system.

This module coordinates document loading, text splitting, embedding generation,
and FAISS vector store indexing and persistence into a coherent, reusable pipeline.
"""

from pathlib import Path
from typing import Any, Optional, Union

from langchain_core.documents import Document

from config import VECTORSTORE_DIR
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingService
from src.text_splitter import DocumentTextSplitter
from src.vector_store import FAISSVectorStore


class DocumentIngestionPipeline:
    """Orchestrator responsible for processing PDF files and building a searchable FAISS index."""

    def __init__(
        self,
        document_loader: Optional[DocumentLoader] = None,
        text_splitter: Optional[DocumentTextSplitter] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        vectorstore_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize DocumentIngestionPipeline with injectable dependencies.

        Args:
            document_loader: Injected DocumentLoader instance. Defaults to DocumentLoader().
            text_splitter: Injected DocumentTextSplitter instance. Defaults to DocumentTextSplitter().
            embedding_service: Injected EmbeddingService instance. Defaults to EmbeddingService().
            vector_store: Injected FAISSVectorStore instance. Defaults to FAISSVectorStore()
                configured with the provided or default embedding service and vector store directory.
            vectorstore_dir: Storage directory for the FAISS index. Defaults to VECTORSTORE_DIR from config.py.
        """
        self.document_loader: DocumentLoader = (
            document_loader if document_loader is not None else DocumentLoader()
        )
        self.text_splitter: DocumentTextSplitter = (
            text_splitter if text_splitter is not None else DocumentTextSplitter()
        )
        self.embedding_service: EmbeddingService = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )
        self.vectorstore_dir: Path = (
            Path(vectorstore_dir).resolve()
            if vectorstore_dir is not None
            else VECTORSTORE_DIR
        )
        self.vector_store: FAISSVectorStore = (
            vector_store
            if vector_store is not None
            else FAISSVectorStore(
                embedding_service=self.embedding_service,
                vectorstore_dir=self.vectorstore_dir,
            )
        )

    def ingest_documents(self, pdf_paths: list[Union[str, Path]]) -> dict[str, Any]:
        """Ingest a list of PDF documents into a persistent FAISS vector index.

        Coordinates the workflow:
        1. Validates input PDF paths.
        2. Loads each PDF document using DocumentLoader.
        3. Combines loaded pages and splits them using DocumentTextSplitter.
        4. Validates that usable text chunks were generated.
        5. Creates and persists a FAISS vector index via FAISSVectorStore.
        6. Returns structured ingestion metrics.

        Args:
            pdf_paths: Non-empty list of filesystem paths to PDF documents.

        Returns:
            dict[str, Any]: Structured summary containing ingestion statistics:
                - status (str): Ingestion outcome status ('success').
                - files_processed (int): Total number of PDF files parsed.
                - pages_loaded (int): Total number of document pages extracted.
                - chunks_generated (int): Total number of text chunks created and indexed.
                - vector_store_path (str): File path to the persisted FAISS index directory.

        Raises:
            TypeError: If pdf_paths is not a list or contains elements that are not str/Path.
            ValueError: If pdf_paths is empty, no document pages are loaded, or no text chunks
                are generated.
            FileNotFoundError: If any specified PDF file does not exist.
            RuntimeError: If document loading, chunking, or FAISS indexing/saving fails.
        """
        # 1. Validate PDF path list
        if pdf_paths is None or not isinstance(pdf_paths, list):
            raise TypeError(
                f"Expected pdf_paths to be a list, got: {type(pdf_paths).__name__}"
            )

        if len(pdf_paths) == 0:
            raise ValueError("pdf_paths list cannot be empty.")

        for idx, path in enumerate(pdf_paths):
            if not isinstance(path, (str, Path)):
                raise TypeError(
                    f"Item at index {idx} in pdf_paths is not a str or Path: {type(path).__name__}"
                )

        # 2. Load PDF files using DocumentLoader
        all_documents: list[Document] = []
        for path in pdf_paths:
            docs = self.document_loader.load_pdf(path)
            if docs:
                all_documents.extend(docs)

        if len(all_documents) == 0:
            raise ValueError(
                "No documents or pages could be extracted from the provided PDF files."
            )

        # 3. Split documents into chunks using DocumentTextSplitter
        chunks = self.text_splitter.split_documents(all_documents)
        if len(chunks) == 0:
            raise ValueError(
                "Text splitting produced zero chunks. Ingestion cannot proceed without usable chunks."
            )

        # 4. Create and persist FAISS vector index
        faiss_index = self.vector_store.create_index(chunks)
        self.vector_store.save_index(faiss_index)

        # 5. Return structured ingestion summary
        return {
            "status": "success",
            "files_processed": len(pdf_paths),
            "pages_loaded": len(all_documents),
            "chunks_generated": len(chunks),
            "vector_store_path": str(self.vector_store.vectorstore_dir),
        }
