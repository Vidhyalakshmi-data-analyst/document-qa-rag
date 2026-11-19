
"""QA orchestration pipeline for the Document-Based RAG system.

Connects document retrieval, Cross-Encoder reranking, prompt construction,
and LLM generation into a single question-answering workflow.
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from src.llm_service import (
    GroqLLMService,
    LLMEmptyResponseError,
)
from src.prompt_template import FALLBACK_RESPONSE, RAGPromptBuilder
from src.reranker import DocumentReranker
from src.retriever import DocumentRetriever


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class QAPipelineError(RuntimeError):
    """Raised when a QA pipeline component fails unexpectedly."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    """Structured result returned by QAPipeline.answer_question()."""

    question: str
    retrieved_documents: list[Document] = field(default_factory=list)
    reranked_documents: list[Document] = field(default_factory=list)
    answer: str = ""


# ---------------------------------------------------------------------------
# QA Pipeline
# ---------------------------------------------------------------------------

class QAPipeline:
    """Orchestrate retrieval, reranking, prompting, and LLM generation."""

    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        reranker: Optional[DocumentReranker] = None,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        llm_service: Optional[GroqLLMService] = None,
    ) -> None:
        """Initialize the pipeline with injectable dependencies."""

        self.retriever = (
            retriever if retriever is not None else DocumentRetriever()
        )
        self.reranker = (
            reranker if reranker is not None else DocumentReranker()
        )
        self.prompt_builder = (
            prompt_builder if prompt_builder is not None else RAGPromptBuilder()
        )
        self.llm_service = (
            llm_service if llm_service is not None else GroqLLMService()
        )

    @staticmethod
    def _validate_documents(
        documents: object,
        component_name: str,
    ) -> list[Document]:
        """Validate that a component returned a list of LangChain Documents."""

        if not isinstance(documents, list):
            raise QAPipelineError(
                f"{component_name} must return a list of Document objects; "
                f"received {type(documents).__name__}."
            )

        for index, document in enumerate(documents):
            if not isinstance(document, Document):
                raise QAPipelineError(
                    f"{component_name} returned an invalid item at index "
                    f"{index}: expected Document, received "
                    f"{type(document).__name__}."
                )

        return documents

    def answer_question(self, question: str) -> QAResult:
        """Execute the complete document-based question-answering workflow.

        Returns the exact fallback response when retrieval, reranking, or
        LLM response content is empty.

        Raises:
            TypeError: If question is not a string.
            ValueError: If question is empty or whitespace only.
            QAPipelineError: If a component fails or returns invalid output.
        """

        # Step 1: Validate question.
        if not isinstance(question, str):
            raise TypeError(
                f"Expected question to be a string, got: "
                f"{type(question).__name__}"
            )

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "question cannot be empty or whitespace only."
            )

        # Step 2: Retrieve candidate documents.
        try:
            retrieved_result = self.retriever.retrieve(cleaned_question)
        except Exception as exc:
            raise QAPipelineError(
                f"Document retrieval failed: {exc}"
            ) from exc

        retrieved_documents = self._validate_documents(
            retrieved_result,
            "Retriever",
        )

        # Step 3: No retrieved documents → exact fallback.
        if not retrieved_documents:
            return QAResult(
                question=cleaned_question,
                answer=FALLBACK_RESPONSE,
            )

        # Step 4: Rerank retrieved documents.
        try:
            reranked_result = self.reranker.rerank(
                cleaned_question,
                retrieved_documents,
            )
        except Exception as exc:
            raise QAPipelineError(
                f"Document reranking failed: {exc}"
            ) from exc

        reranked_documents = self._validate_documents(
            reranked_result,
            "Reranker",
        )

        # Step 5: No reranked documents → exact fallback.
        if not reranked_documents:
            return QAResult(
                question=cleaned_question,
                retrieved_documents=retrieved_documents,
                answer=FALLBACK_RESPONSE,
            )

        # Step 6: Build grounded prompt.
        try:
            prompt = self.prompt_builder.build_prompt(
                cleaned_question,
                reranked_documents,
            )
        except Exception as exc:
            raise QAPipelineError(
                f"Prompt construction failed: {exc}"
            ) from exc

        if not isinstance(prompt, str) or not prompt.strip():
            raise QAPipelineError(
                "Prompt builder returned an empty or invalid prompt."
            )

        # Step 7: Generate the answer.
        try:
            answer = self.llm_service.generate_response(prompt)

        except LLMEmptyResponseError:
            # An empty model response is a fallback condition.
            answer = FALLBACK_RESPONSE

        except Exception as exc:
            # API, network, configuration, and other generation errors
            # must not be confused with missing document information.
            raise QAPipelineError(
                f"LLM generation failed: {exc}"
            ) from exc

        # Step 8: Validate LLM output.
        if not isinstance(answer, str):
            raise QAPipelineError(
                "LLM service returned an invalid response type: "
                f"{type(answer).__name__}."
            )

        answer = answer.strip()

        if not answer:
            answer = FALLBACK_RESPONSE

        # Step 9: Return complete QA result.
        return QAResult(
            question=cleaned_question,
            retrieved_documents=retrieved_documents,
            reranked_documents=reranked_documents,
            answer=answer,
        )
