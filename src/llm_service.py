
"""LLM service module for the Document-Based RAG system.

This module provides integration with Groq language models using LangChain's ChatGroq.
It supports lazy initialization, dependency injection for unit testing, strict input
and output validation, and custom exception handling without exposing API keys.
"""

from typing import Any, Optional

from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, validate_groq_api_key

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class LLMServiceError(Exception):
    """Base exception for all LLM service failures."""

class LLMConfigurationError(LLMServiceError):
    """Raised when LLM configuration or credentials are missing or invalid."""

class LLMInitializationError(LLMServiceError):
    """Raised when LLM client or model initialization fails."""

class LLMGenerationError(LLMServiceError):
    """Raised when LLM inference, API invocation, or response parsing fails."""

class LLMEmptyResponseError(LLMGenerationError):
    """Raised specifically when the LLM returns empty response content."""

# ---------------------------------------------------------------------------
# Groq LLM Service
# ---------------------------------------------------------------------------

class GroqLLMService:
    """Service responsible for invoking the Groq LLM to generate answers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        model: Optional[Any] = None,
    ) -> None:
        """Initialize GroqLLMService with configuration or injected model."""

        self.api_key: Optional[str] = (
            api_key if api_key is not None else GROQ_API_KEY
        )
        self.model_name: str = (
            model_name if model_name is not None else GROQ_MODEL
        )
        self.temperature: float = temperature
        self._model: Optional[Any] = model

    @property
    def model(self) -> Any:
        """Get the active LLM, lazily initializing ChatGroq if necessary."""

        if self._model is None:
            if not self.model_name or not self.model_name.strip():
                raise LLMConfigurationError(
                    "GROQ_MODEL is not configured or empty."
                )

            try:
                validated_key = validate_groq_api_key(self.api_key)
            except ValueError as exc:
                raise LLMConfigurationError(str(exc)) from exc

            try:
                self._model = ChatGroq(
                    api_key=validated_key,
                    model=self.model_name.strip(),
                    temperature=self.temperature,
                )
            except Exception as exc:
                raise LLMInitializationError(
                    f"Failed to initialize ChatGroq model "
                    f"'{self.model_name}': {exc}"
                ) from exc

        return self._model

    def generate_response(self, prompt: str) -> str:
        """Generate and validate a response from the configured Groq model.

        Raises:
            TypeError: If prompt is not a string.
            ValueError: If prompt is empty or whitespace only.
            LLMEmptyResponseError: If the model returns no usable content.
            LLMGenerationError: If invocation or response parsing fails.
        """

        # Validate input
        if prompt is None or not isinstance(prompt, str):
            raise TypeError(
                f"Expected prompt to be a string, got: {type(prompt).__name__}"
            )

        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            raise ValueError(
                "Prompt cannot be empty or whitespace only."
            )

        # Invoke LLM
        try:
            response = self.model.invoke(cleaned_prompt)
        except Exception as exc:
            raise LLMGenerationError(
                f"Failed to generate response from LLM: {exc}"
            ) from exc

        # Validate response
        if response is None:
            raise LLMEmptyResponseError(
                "Received None response from LLM."
            )

        if isinstance(response, str):
            content = response.strip()

        elif hasattr(response, "content"):
            raw_content = response.content
            content = (
                str(raw_content).strip()
                if raw_content is not None
                else ""
            )

        else:
            raise LLMGenerationError(
                f"Unexpected response format from LLM: "
                f"received {type(response).__name__} "
                f"without a readable 'content' attribute."
            )

        if not content:
            raise LLMEmptyResponseError(
                "Received empty response content from LLM."
            )

        return content