"""Unit tests for src/qa_pipeline.py."""

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.documents import Document

from src.llm_service import LLMGenerationError, LLMEmptyResponseError
from src.prompt_template import FALLBACK_RESPONSE
from src.qa_pipeline import QAPipeline, QAResult, QAPipelineError


# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

SAMPLE_QUESTION = "What is retrieval-augmented generation?"

RETRIEVED_DOCS = [
    Document(
        page_content="RAG combines retrieval with language model generation.",
        metadata={"source": "rag_paper.pdf", "page": 1},
    ),
    Document(
        page_content="FAISS enables fast approximate nearest-neighbour search.",
        metadata={"source": "faiss_guide.pdf", "page": 3},
    ),
    Document(
        page_content="LangChain provides modular abstractions for LLM pipelines.",
        metadata={"source": "langchain_docs.pdf", "page": 7},
    ),
    Document(
        page_content="Cross-Encoder models score relevance of (query, passage) pairs.",
        metadata={"source": "reranking_survey.pdf", "page": 2},
    ),
    Document(
        page_content="Groq LPUs accelerate token generation significantly.",
        metadata={"source": "groq_overview.pdf", "page": 1},
    ),
]

RERANKED_DOCS = RETRIEVED_DOCS[:3]  # top-3 after reranking

SAMPLE_PROMPT = (
    "You are a helpful assistant.\n"
    "=== BEGIN RETRIEVED CONTEXT ===\nSome context\n=== END RETRIEVED CONTEXT ===\n"
    "Question: What is retrieval-augmented generation?\nAnswer:"
)

SAMPLE_ANSWER = (
    "RAG augments LLM generation by first retrieving relevant documents from a "
    "vector store and injecting them into the prompt as grounded context."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_retriever() -> MagicMock:
    """Fixture providing a mock DocumentRetriever that returns RETRIEVED_DOCS."""
    retriever = MagicMock()
    retriever.retrieve.return_value = list(RETRIEVED_DOCS)
    return retriever


@pytest.fixture
def mock_reranker() -> MagicMock:
    """Fixture providing a mock DocumentReranker that returns RERANKED_DOCS."""
    reranker = MagicMock()
    reranker.rerank.return_value = list(RERANKED_DOCS)
    return reranker


@pytest.fixture
def mock_prompt_builder() -> MagicMock:
    """Fixture providing a mock RAGPromptBuilder that returns SAMPLE_PROMPT."""
    builder = MagicMock()
    builder.build_prompt.return_value = SAMPLE_PROMPT
    return builder


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Fixture providing a mock GroqLLMService that returns SAMPLE_ANSWER."""
    service = MagicMock()
    service.generate_response.return_value = SAMPLE_ANSWER
    return service


@pytest.fixture
def pipeline(
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
) -> QAPipeline:
    """Fixture providing a fully mocked QAPipeline."""
    return QAPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_successful_end_to_end_orchestration(pipeline: QAPipeline):
    """Test Case 1: Full pipeline returns a correctly populated QAResult."""
    result = pipeline.answer_question(SAMPLE_QUESTION)

    assert isinstance(result, QAResult)
    assert result.question == SAMPLE_QUESTION
    assert result.answer == SAMPLE_ANSWER
    assert len(result.retrieved_documents) == len(RETRIEVED_DOCS)
    assert len(result.reranked_documents) == len(RERANKED_DOCS)


def test_correct_question_passed_to_retriever(
    pipeline: QAPipeline, mock_retriever: MagicMock
):
    """Test Case 2: Retriever is called with the stripped question string."""
    pipeline.answer_question("  " + SAMPLE_QUESTION + "  ")

    mock_retriever.retrieve.assert_called_once_with(SAMPLE_QUESTION)


def test_retrieved_documents_passed_to_reranker(
    pipeline: QAPipeline,
    mock_reranker: MagicMock,
    mock_retriever: MagicMock,
):
    """Test Case 3: Reranker receives the exact list returned by the retriever."""
    pipeline.answer_question(SAMPLE_QUESTION)

    args, _ = mock_reranker.rerank.call_args
    assert args[0] == SAMPLE_QUESTION
    assert args[1] == mock_retriever.retrieve.return_value


def test_top_3_reranked_documents_passed_to_prompt_builder(
    pipeline: QAPipeline,
    mock_prompt_builder: MagicMock,
    mock_reranker: MagicMock,
):
    """Test Case 4: Prompt builder receives the question and the reranked document list."""
    pipeline.answer_question(SAMPLE_QUESTION)

    mock_prompt_builder.build_prompt.assert_called_once_with(
        SAMPLE_QUESTION,
        mock_reranker.rerank.return_value,
    )


def test_generated_prompt_passed_to_llm_service(
    pipeline: QAPipeline,
    mock_llm_service: MagicMock,
    mock_prompt_builder: MagicMock,
):
    """Test Case 5: LLM service receives the exact prompt string from the prompt builder."""
    pipeline.answer_question(SAMPLE_QUESTION)

    mock_llm_service.generate_response.assert_called_once_with(SAMPLE_PROMPT)


def test_qaresult_fields_are_correct(pipeline: QAPipeline):
    """Test Case 6: All QAResult fields are populated with the correct objects."""
    result = pipeline.answer_question(SAMPLE_QUESTION)

    assert result.question == SAMPLE_QUESTION
    assert result.retrieved_documents == RETRIEVED_DOCS
    assert result.reranked_documents == RERANKED_DOCS
    assert result.answer == SAMPLE_ANSWER


def test_empty_retrieval_returns_exact_fallback(
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 7: Empty retrieval returns the exact FALLBACK_RESPONSE string."""
    empty_retriever = MagicMock()
    empty_retriever.retrieve.return_value = []

    pipeline = QAPipeline(
        retriever=empty_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    result = pipeline.answer_question(SAMPLE_QUESTION)

    assert result.answer == FALLBACK_RESPONSE
    assert result.answer == "Information not available in the provided documents."
    assert result.retrieved_documents == []
    assert result.reranked_documents == []


def test_empty_retrieval_skips_reranker_prompt_and_llm(
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 8: When retrieval is empty, reranker, prompt builder, and LLM are not called."""
    empty_retriever = MagicMock()
    empty_retriever.retrieve.return_value = []

    pipeline = QAPipeline(
        retriever=empty_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    pipeline.answer_question(SAMPLE_QUESTION)

    mock_reranker.rerank.assert_not_called()
    mock_prompt_builder.build_prompt.assert_not_called()
    mock_llm_service.generate_response.assert_not_called()


def test_empty_reranking_returns_exact_fallback(
    mock_retriever: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 9: Empty reranking returns the exact FALLBACK_RESPONSE string."""
    empty_reranker = MagicMock()
    empty_reranker.rerank.return_value = []

    pipeline = QAPipeline(
        retriever=mock_retriever,
        reranker=empty_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    result = pipeline.answer_question(SAMPLE_QUESTION)

    assert result.answer == FALLBACK_RESPONSE
    assert result.answer == "Information not available in the provided documents."
    assert result.retrieved_documents == RETRIEVED_DOCS
    assert result.reranked_documents == []


def test_empty_reranking_skips_prompt_builder_and_llm(
    mock_retriever: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 10: When reranking is empty, prompt builder and LLM are not called."""
    empty_reranker = MagicMock()
    empty_reranker.rerank.return_value = []

    pipeline = QAPipeline(
        retriever=mock_retriever,
        reranker=empty_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    pipeline.answer_question(SAMPLE_QUESTION)

    mock_prompt_builder.build_prompt.assert_not_called()
    mock_llm_service.generate_response.assert_not_called()



def test_empty_llm_response_uses_fallback(
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
):
    """Test empty LLM response returns the exact FALLBACK_RESPONSE."""

    llm_raises_empty = MagicMock()
    llm_raises_empty.generate_response.side_effect = LLMEmptyResponseError(
        "Received empty response content from LLM."
    )

    pipeline = QAPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=llm_raises_empty,
    )

    result = pipeline.answer_question(SAMPLE_QUESTION)

    assert result.answer == FALLBACK_RESPONSE
    assert result.retrieved_documents == RETRIEVED_DOCS
    assert result.reranked_documents == RERANKED_DOCS



def test_invalid_question_type_raises_type_error(pipeline: QAPipeline):
    """Test Case 12a: Non-string questions raise TypeError."""
    with pytest.raises(TypeError, match="Expected question to be a string"):
        pipeline.answer_question(None)  # type: ignore

    with pytest.raises(TypeError, match="Expected question to be a string"):
        pipeline.answer_question(42)  # type: ignore

    with pytest.raises(TypeError, match="Expected question to be a string"):
        pipeline.answer_question(["list"])  # type: ignore


def test_empty_question_raises_value_error(pipeline: QAPipeline):
    """Test Case 12b: Empty or whitespace-only questions raise ValueError."""
    with pytest.raises(ValueError, match="question cannot be empty or whitespace only"):
        pipeline.answer_question("")

    with pytest.raises(ValueError, match="question cannot be empty or whitespace only"):
        pipeline.answer_question("   \t\n  ")


def test_dependency_injection_uses_provided_mocks(
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 13: Injected mocks are used without constructing any real components."""
    with (
        patch("src.qa_pipeline.DocumentRetriever") as mock_ret_cls,
        patch("src.qa_pipeline.DocumentReranker") as mock_rer_cls,
        patch("src.qa_pipeline.RAGPromptBuilder") as mock_pb_cls,
        patch("src.qa_pipeline.GroqLLMService") as mock_llm_cls,
    ):
        pipeline = QAPipeline(
            retriever=mock_retriever,
            reranker=mock_reranker,
            prompt_builder=mock_prompt_builder,
            llm_service=mock_llm_service,
        )

        assert pipeline.retriever is mock_retriever
        assert pipeline.reranker is mock_reranker
        assert pipeline.prompt_builder is mock_prompt_builder
        assert pipeline.llm_service is mock_llm_service

        mock_ret_cls.assert_not_called()
        mock_rer_cls.assert_not_called()
        mock_pb_cls.assert_not_called()
        mock_llm_cls.assert_not_called()


def test_retriever_failure_propagates_as_runtime_error(
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 14a: RuntimeError from retriever propagates with its original cause."""
    failing_retriever = MagicMock()
    failing_retriever.retrieve.side_effect = RuntimeError(
        "Failed to retrieve documents for question 'query': FAISS index corrupted"
    )

    pipeline = QAPipeline(
        retriever=failing_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    with pytest.raises(RuntimeError, match="Failed to retrieve documents"):
        pipeline.answer_question(SAMPLE_QUESTION)


def test_reranker_failure_propagates_as_runtime_error(
    mock_retriever: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 14b: RuntimeError from reranker propagates with its original cause."""
    failing_reranker = MagicMock()
    failing_reranker.rerank.side_effect = RuntimeError(
        "Failed to rerank documents using Cross-Encoder: CUDA error"
    )

    pipeline = QAPipeline(
        retriever=mock_retriever,
        reranker=failing_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    with pytest.raises(RuntimeError, match="Failed to rerank documents"):
        pipeline.answer_question(SAMPLE_QUESTION)



def test_non_empty_llm_generation_error_propagates(
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
):
    """Test Case 14c: Non-empty LLMGenerationError is wrapped as QAPipelineError."""

    failing_llm = MagicMock()
    original_error = LLMGenerationError(
        "Failed to generate response from LLM: connection timeout"
    )
    failing_llm.generate_response.side_effect = original_error

    pipeline = QAPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=failing_llm,
    )

    with pytest.raises(
        QAPipelineError,
        match="LLM generation failed"
    ) as exc_info:
        pipeline.answer_question(SAMPLE_QUESTION)

    # Verify that the original LLM exception is preserved.
    assert isinstance(exc_info.value.__cause__, LLMGenerationError)
    assert exc_info.value.__cause__ is original_error



def test_document_metadata_remains_intact(pipeline: QAPipeline):
    """Test Case 15: Retrieved and reranked documents preserve source and page metadata."""
    result = pipeline.answer_question(SAMPLE_QUESTION)

    for doc in result.retrieved_documents:
        assert "source" in doc.metadata
        assert "page" in doc.metadata

    for doc in result.reranked_documents:
        assert "source" in doc.metadata
        assert "page" in doc.metadata

    # Spot-check specific metadata on the first retrieved document
    assert result.retrieved_documents[0].metadata["source"] == "rag_paper.pdf"
    assert result.retrieved_documents[0].metadata["page"] == 1

    # Spot-check specific metadata on the first reranked document
    assert result.reranked_documents[0].metadata["source"] == "rag_paper.pdf"
    assert result.reranked_documents[0].metadata["page"] == 1


def test_llm_not_invoked_when_retrieval_empty(
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 16a: LLM is never called when retrieval returns no documents."""
    empty_retriever = MagicMock()
    empty_retriever.retrieve.return_value = []

    pipeline = QAPipeline(
        retriever=empty_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    pipeline.answer_question(SAMPLE_QUESTION)

    mock_llm_service.generate_response.assert_not_called()


def test_llm_not_invoked_when_reranking_empty(
    mock_retriever: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 16b: LLM is never called when reranking returns no documents."""
    empty_reranker = MagicMock()
    empty_reranker.rerank.return_value = []

    pipeline = QAPipeline(
        retriever=mock_retriever,
        reranker=empty_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    pipeline.answer_question(SAMPLE_QUESTION)

    mock_llm_service.generate_response.assert_not_called()


def test_qaresult_is_dataclass_with_correct_types(pipeline: QAPipeline):
    """Test Case 17: QAResult is a proper dataclass with the expected field types."""
    result = pipeline.answer_question(SAMPLE_QUESTION)

    assert isinstance(result, QAResult)
    assert isinstance(result.question, str)
    assert isinstance(result.retrieved_documents, list)
    assert isinstance(result.reranked_documents, list)
    assert isinstance(result.answer, str)
    assert all(isinstance(d, Document) for d in result.retrieved_documents)
    assert all(isinstance(d, Document) for d in result.reranked_documents)


def test_question_whitespace_is_stripped_before_passing_downstream(
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_service: MagicMock,
):
    """Test Case 18: Leading/trailing whitespace is stripped; cleaned question flows through all steps."""
    padded_question = "   " + SAMPLE_QUESTION + "   "

    pipeline = QAPipeline(
        retriever=mock_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_service=mock_llm_service,
    )

    result = pipeline.answer_question(padded_question)

    # The stored question in QAResult is the cleaned version
    assert result.question == SAMPLE_QUESTION

    # Every downstream component received the stripped question
    mock_retriever.retrieve.assert_called_once_with(SAMPLE_QUESTION)
    rerank_args, _ = mock_reranker.rerank.call_args
    assert rerank_args[0] == SAMPLE_QUESTION
    build_args, _ = mock_prompt_builder.build_prompt.call_args
    assert build_args[0] == SAMPLE_QUESTION
