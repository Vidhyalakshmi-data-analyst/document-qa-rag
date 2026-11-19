
"""Streamlit interface for the Document-Based RAG QA system."""

import uuid
import hashlib
import re
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.documents import Document

from config import DATA_DIR, VECTORSTORE_DIR
from src.embeddings import EmbeddingService
from src.ingestion_pipeline import DocumentIngestionPipeline
from src.llm_service import GroqLLMService
from src.qa_pipeline import QAResult, QAPipeline, QAPipelineError
from src.reranker import DocumentReranker
from src.retriever import DocumentRetriever
from src.vector_store import FAISSVectorStore

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Document-Based RAG QA",
    page_icon="📚",
    layout="wide",
)

UPLOAD_DIR = DATA_DIR / "uploads"
MIN_PDF_FILES = 2
MAX_PDF_FILES = 3

# ---------------------------------------------------------------------------
# Service initialization
# ---------------------------------------------------------------------------

def initialize_services() -> dict[str, Any]:
    """Create session-specific RAG services and share their dependencies."""

    if "rag_session_id" not in st.session_state:
        st.session_state.rag_session_id = uuid.uuid4().hex

        session_vectorstore_dir = (
            VECTORSTORE_DIR / st.session_state.rag_session_id
        )
    
    if "rag_services" not in st.session_state:
        embedding_service = EmbeddingService()

        vector_store = FAISSVectorStore(
            embedding_service=embedding_service,
            vectorstore_dir=session_vectorstore_dir,
        )

        ingestion_pipeline = DocumentIngestionPipeline(
            embedding_service=embedding_service,
            vector_store=vector_store,
            vectorstore_dir=session_vectorstore_dir,
        )

        retriever = DocumentRetriever(
            vector_store=vector_store,
        )

        reranker = DocumentReranker()

        llm_service = GroqLLMService()

        qa_pipeline = QAPipeline(
            retriever=retriever,
            reranker=reranker,
            llm_service=llm_service,
        )

        st.session_state.rag_services = {
            "embedding_service": embedding_service,
            "vector_store": vector_store,
            "ingestion_pipeline": ingestion_pipeline,
            "retriever": retriever,
            "reranker": reranker,
            "llm_service": llm_service,
            "qa_pipeline": qa_pipeline,
        }

    return st.session_state.rag_services

# ---------------------------------------------------------------------------
# Uploaded-file utilities
# ---------------------------------------------------------------------------

def calculate_file_fingerprint(uploaded_files: list[Any]) -> str:
    """Generate a stable fingerprint for the uploaded PDF set."""

    file_details: list[tuple[str, str]] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        file_details.append(
            (uploaded_file.name, content_hash)
        )

    fingerprint_input = repr(sorted(file_details)).encode("utf-8")

    return hashlib.sha256(fingerprint_input).hexdigest()

def save_uploaded_files(uploaded_files: list[Any]) -> list[Path]:
    """Save uploaded PDFs using safe, content-derived filenames."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        original_name = Path(uploaded_file.name).name

        safe_name = re.sub(
            r"[^A-Za-z0-9._-]",
            "_",
            original_name,
        )

        file_bytes = uploaded_file.getvalue()
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        saved_name = f"{content_hash[:12]}_{safe_name}"
        destination = UPLOAD_DIR / saved_name

        destination.write_bytes(file_bytes)
        saved_paths.append(destination)

    return saved_paths

# ---------------------------------------------------------------------------
# Document display utilities
# ---------------------------------------------------------------------------

def get_document_source(document: Document) -> str:
    """Return a readable source filename from document metadata."""

    source = document.metadata.get("source", "Unknown source")

    return Path(str(source)).name

def get_document_page(document: Document) -> str:
    """Return a human-readable page number."""

    page = document.metadata.get("page")

    if isinstance(page, int):
        # PyPDFLoader normally stores zero-based page numbers.
        return str(page + 1)

    return "Unknown"

def display_documents(
    documents: list[Document],
    section_title: str,
) -> None:
    """Display retrieved or reranked document chunks."""

    st.subheader(section_title)

    if not documents:
        st.info("No document chunks available.")
        return

    for index, document in enumerate(documents, start=1):
        source = get_document_source(document)
        page = get_document_page(document)

        with st.expander(
            f"Chunk {index} — {source} — Page {page}",
            expanded=False,
        ):
            st.markdown(document.page_content)

            st.caption(
                f"Source: {source} | Page: {page}"
            )

def display_qa_result(result: QAResult) -> None:
    """Display the complete RAG question-answering result."""

    st.divider()

    st.subheader("Question")
    st.write(result.question)

    st.subheader("Final Answer")
    st.markdown(result.answer)

    st.divider()

    display_documents(
        result.retrieved_documents,
        "Retrieved Documents — Top 5",
    )

    display_documents(
        result.reranked_documents,
        "Reranked Documents — Top 3",
    )

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the Streamlit document-based question-answering application."""

    st.title("📚 Document-Based RAG Question Answering")
    st.write(
        "Upload PDF documents, process them into a searchable knowledge base, "
        "and ask questions grounded in their content."
    )

    services = initialize_services()

    # -------------------------------------------------------
    # PDF upload
    # -------------------------------------------------------

    st.header("1. Upload PDF Documents")

    uploaded_files = st.file_uploader(
        "Choose 2–3 PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload between 2 and 3 PDF files.",
    )

    current_fingerprint: str | None = None

    if uploaded_files:
        if len(uploaded_files) > MAX_PDF_FILES:
            st.error(
                f"You can upload a maximum of {MAX_PDF_FILES} PDF files."
            )
        else:
            current_fingerprint = calculate_file_fingerprint(
                uploaded_files
            )

            st.write("Selected documents:")

            for uploaded_file in uploaded_files:
                st.write(
                    f"📄 {uploaded_file.name} "
                    f"({uploaded_file.size / 1024:.1f} KB)"
                )

    # -------------------------------------------------------
    # Invalidate an old index if the document set changes
    # -------------------------------------------------------

    active_fingerprint = st.session_state.get(
        "active_fingerprint"
    )

    if current_fingerprint != active_fingerprint:
        st.session_state.active_fingerprint = None
        st.session_state.pop("last_qa_result", None)

        # Prevent questions from using a previous document set.
        services["vector_store"].vector_store = None

    # -------------------------------------------------------
    # Process documents
    # -------------------------------------------------------

    can_process = (
        uploaded_files is not None
        and MIN_PDF_FILES <= len(uploaded_files) <= MAX_PDF_FILES
    )

    if uploaded_files and not can_process:
        st.warning(
            f"Please upload between {MIN_PDF_FILES} and "
            f"{MAX_PDF_FILES} PDF documents."
        )

    process_clicked = st.button(
        "Process Documents",
        type="primary",
        disabled=not can_process,
    )

    if process_clicked and can_process:
        try:
            with st.spinner(
                "Loading PDFs, splitting text, generating embeddings, "
                "and creating the FAISS index..."
            ):
                pdf_paths = save_uploaded_files(uploaded_files)

                ingestion_result = (
                    services["ingestion_pipeline"].ingest_documents(
                        pdf_paths
                    )
                )

                st.session_state.active_fingerprint = (
                    current_fingerprint
                )

                st.session_state.ingestion_result = ingestion_result
                st.session_state.pop("last_qa_result", None)

            st.success("Documents processed successfully!")

        except Exception as exc:
            services["vector_store"].vector_store = None
            st.session_state.active_fingerprint = None

            st.error(
                "Document processing failed. "
                "Please check that the PDFs are valid and try again."
            )
            print(f"Application error: {type(exc).__name__}: {exc}")

    # -------------------------------------------------------
    # Ingestion summary
    # -------------------------------------------------------

    if (
        st.session_state.get("active_fingerprint")
        == current_fingerprint
        and current_fingerprint is not None
    ):
        ingestion_result = st.session_state.get(
            "ingestion_result",
            {},
        )

        st.success("Your document knowledge base is ready.")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "PDF Files",
            ingestion_result.get("files_processed", 0),
        )

        col2.metric(
            "Pages Loaded",
            ingestion_result.get("pages_loaded", 0),
        )

        col3.metric(
            "Chunks Indexed",
            ingestion_result.get("chunks_generated", 0),
        )

        # ---------------------------------------------------
        # Question answering
        # ---------------------------------------------------

        st.divider()
        st.header("2. Ask a Question")

        with st.form("question_form"):
            question = st.text_input(
                "Enter your question",
                placeholder="Ask something about your uploaded documents...",
            )

            ask_clicked = st.form_submit_button(
                "Get Answer",
                type="primary",
            )

        if ask_clicked:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner(
                        "Retrieving relevant passages and generating an answer..."
                    ):
                        result = services[
                            "qa_pipeline"
                        ].answer_question(question)

                        st.session_state.last_qa_result = result

                except QAPipelineError as exc:
                    st.error(
                        "The question-answering pipeline encountered an error."
                    )
                    st.exception(exc)

                except Exception as exc:
                    st.error(
                        "An unexpected error occurred while answering your question."
                    )
                    print(f"Application error: {type(exc).__name__}: {exc}")

        # Display the most recent successful result.
        last_result = st.session_state.get("last_qa_result")

        if last_result is not None:
            display_qa_result(last_result)

    else:
        st.info(
            "Upload 2–3 PDF documents and click "
            "'Process Documents' to start asking questions."
        )

    # -------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------

    with st.sidebar:
        st.header("About This Application")

        st.markdown(
            """
            **RAG workflow**

            1. PDF document loading
            2. Text chunking
            3. Hugging Face embeddings
            4. FAISS similarity retrieval
            5. Cross-Encoder reranking
            6. Grounded prompt construction
            7. Groq LLM response generation
            """
        )

        st.divider()

        st.caption(
            "Answers are generated using the uploaded documents "
            "as the retrieval context."
        )

if __name__ == "__main__":
    main()