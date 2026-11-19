"""Vector store module for the Document-Based RAG system.

This module provides FAISS vector storage, index persistence, local loading,
and vector similarity search using LangChain's FAISS integration and injected
embedding services.
"""

from pathlib import Path
from typing import Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import VECTORSTORE_DIR
from src.embeddings import EmbeddingService


class FAISSVectorStore:
    """Service responsible for managing FAISS vector index creation, persistence, and search."""

    def __init__(
        self,
        embedding_service: Optional[Embeddings] = None,
        vectorstore_dir: Optional[Union[str, Path]] = None,
        vector_store: Optional[FAISS] = None,
    ) -> None:
        """Initialize the FAISSVectorStore service.

        Args:
            embedding_service: Injected LangChain Embeddings instance. Defaults to
                instantiating EmbeddingService() from src/embeddings.py.
            vectorstore_dir: Directory where FAISS indices are persisted. Defaults
                to VECTORSTORE_DIR configured in config.py.
            vector_store: Optional pre-existing FAISS instance to manage.
        """
        self.embedding_service: Embeddings = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )
        self.vectorstore_dir: Path = (
            Path(vectorstore_dir).resolve()
            if vectorstore_dir is not None
            else VECTORSTORE_DIR
        )
        self.vector_store: Optional[FAISS] = vector_store

    def create_index(self, documents: list[Document]) -> FAISS:
        """Create a new FAISS vector index from document chunks.

        Args:
            documents: Non-empty list of LangChain Document chunks to index.

        Returns:
            FAISS: The constructed LangChain FAISS vector store.

        Raises:
            TypeError: If documents is not a list or contains non-Document elements.
            ValueError: If documents list is empty.
            RuntimeError: If index creation fails.
        """
        if documents is None or not isinstance(documents, list):
            raise TypeError(
                f"Expected documents to be a list of Document objects, got: {type(documents).__name__}"
            )

        if len(documents) == 0:
            raise ValueError("Cannot create a vector index from an empty document list.")

        for idx, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise TypeError(
                    f"Item at index {idx} is not an instance of LangChain Document: "
                    f"got {type(doc).__name__}"
                )

        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_service,
            )
            return self.vector_store
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create FAISS vector index from documents: {exc}"
            ) from exc

    def save_index(
        self,
        vector_store: FAISS,
        path: Union[str, Path, None] = None,
    ) -> None:
        """Persist a FAISS index to the local filesystem.

        Args:
            vector_store: The FAISS vector store instance to save.
            path: Target directory path for persistence. If None, uses self.vectorstore_dir.

        Raises:
            TypeError: If vector_store is not an instance of FAISS.
            RuntimeError: If saving fails due to filesystem or serialization errors.
        """
        if vector_store is None or not isinstance(vector_store, FAISS):
            raise TypeError(
                f"Expected vector_store to be a FAISS instance, got: {type(vector_store).__name__}"
            )

        target_dir = Path(path).resolve() if path is not None else self.vectorstore_dir

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(folder_path=str(target_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to persist FAISS index to '{target_dir}': {exc}"
            ) from exc

    def load_index(self, path: Union[str, Path, None] = None) -> FAISS:
        """Load a previously persisted FAISS index from local disk.

        Args:
            path: Directory path containing the saved index files ('index.faiss'
                and 'index.pkl'). If None, defaults to self.vectorstore_dir.

        Returns:
            FAISS: The loaded FAISS vector store instance.

        Raises:
            FileNotFoundError: If the target path or required index files do not exist.
            RuntimeError: If loading or deserialization fails.
        """
        target_dir = Path(path).resolve() if path is not None else self.vectorstore_dir

        if not target_dir.exists():
            raise FileNotFoundError(
                f"Vector store directory does not exist: {target_dir}"
            )

        index_file = target_dir / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index file ('index.faiss') not found in: {target_dir}"
            )

        # SECURITY CONSIDERATION:
        # LangChain FAISS persistence serializes document store metadata using Python's
        # pickle format into 'index.pkl'. Deserializing untrusted pickle files is insecure
        # and could lead to arbitrary code execution.
        # allow_dangerous_deserialization=True is enabled strictly and solely for indexes
        # created and saved locally by this application within its dedicated storage directory.
        # Never point this loader to unverified or third-party index files.
        try:
            self.vector_store = FAISS.load_local(
                folder_path=str(target_dir),
                embeddings=self.embedding_service,
                allow_dangerous_deserialization=True,
            )
            return self.vector_store
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FAISS index from '{target_dir}': {exc}"
            ) from exc

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Perform vector similarity search against the active FAISS index.

        Args:
            query: User question or search query string.
            k: Number of most similar document chunks to retrieve. Defaults to 5.

        Returns:
            list[Document]: List of top-k retrieved LangChain Document chunks.

        Raises:
            RuntimeError: If no index has been created or loaded, or if search fails.
            TypeError: If query is not a string.
            ValueError: If query is empty/whitespace or k is not a positive integer.
        """
        if self.vector_store is None:
            raise RuntimeError(
                "No active FAISS index found. Please create or load an index "
                "before executing similarity_search."
            )

        if query is None or not isinstance(query, str):
            raise TypeError(
                f"Expected query to be a string, got: {type(query).__name__}"
            )

        if not query.strip():
            raise ValueError("Query string cannot be empty or whitespace only.")

        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got: {k}")

        try:
            return self.vector_store.similarity_search(query=query, k=k)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to execute similarity search for query '{query}': {exc}"
            ) from exc
