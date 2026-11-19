
import pytest
from unittest.mock import MagicMock

from src.llm_service import (
    GroqLLMService,
    LLMEmptyResponseError,
    LLMGenerationError,
)

@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        "   ",
        "\n\t",
        MagicMock(content=None),
        MagicMock(content="   "),
    ],
)
def test_generate_response_raises_empty_response_error(response):
    """Empty or whitespace-only model responses must raise a specific error."""

    model = MagicMock()
    model.invoke.return_value = response

    service = GroqLLMService(model=model)

    with pytest.raises(LLMEmptyResponseError):
        service.generate_response("What is RAG?")

def test_empty_response_error_is_generation_error():
    """The dedicated empty-response exception must retain parent compatibility."""

    assert issubclass(LLMEmptyResponseError, LLMGenerationError)

def test_generate_response_returns_valid_response():
    """A valid model response must be returned after trimming whitespace."""

    model = MagicMock()
    model.invoke.return_value = MagicMock(
        content="  RAG combines retrieval and generation.  "
    )

    service = GroqLLMService(model=model)

    result = service.generate_response("What is RAG?")

    assert result == "RAG combines retrieval and generation."