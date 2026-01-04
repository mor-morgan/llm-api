"""
API routes for the LLM service.

This module defines the HTTP endpoints exposed by the application and
delegates all business logic to the LLMService layer.

Responsibilities:
- Request/response validation via Pydantic schemas
- HTTP routing and response formatting
- Mapping domain-level exceptions to API-level error handling

The actual model logic (generation, encoding, decoding) is handled
by the LLMService class.
"""
from fastapi import APIRouter, HTTPException
from llm_api.schemas.models import (
    GenerateRequest, GenerateResponse,
    EncodeRequest, EncodeResponse,
    DecodeRequest, DecodeResponse
)
from llm_api.services.llm_service import LLMService
from llm_api.exceptions.llm_exceptions import (
    ModelLoadError,
    TokenizationError,
    GenerationError,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Singleton LLM service
# ---------------------------------------------------------------------------

# The model is loaded once at startup and reused across all requests.
llm = LLMService()


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Generate text from a prompt using the language model.

    This endpoint receives a text prompt and a maximum token limit,
    then delegates the generation process to the LLMService.

    Args:
        req (GenerateRequest): Request payload containing:
            - prompt: Input text prompt
            - max_tokens: Maximum number of tokens to generate

    Returns:
        GenerateResponse: The generated text.

    Raises:
        GenerationError: If text generation fails during inference.
        ModelLoadError: If the model is not available or failed to load.
    """
    try:
        result = llm.generate(
        prompt=req.prompt,
        max_tokens=req.max_tokens
        )
        return GenerateResponse(text=result)
    except GenerationError:
        raise
    except ModelLoadError:
        raise


@router.post("/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest):
    """
    Encode text into token IDs.

    This endpoint converts an input string into a list of integer token IDs
    using the tokenizer provided by the LLMService.

    Args:
        req (EncodeRequest): Request payload containing:
            - text: Input text to tokenize

    Returns:
        EncodeResponse: A list of token IDs representing the input text.

    Raises:
        TokenizationError: If the input text cannot be tokenized.
    """
    try:
        tokens = llm.encode(req.text)
        return EncodeResponse(tokens=tokens)
    except TokenizationError:
        raise


@router.post("/decode", response_model=DecodeResponse)
def decode(req: DecodeRequest):
    """
    Decode token IDs back into text.

    This endpoint converts a list of integer token IDs back into
    a human-readable string using the LLMService tokenizer.

    Args:
        req (DecodeRequest): Request payload containing:
            - tokens: List of token IDs

    Returns:
        DecodeResponse: The decoded text string.

    Raises:
        TokenizationError: If decoding the tokens fails.
    """
    try:
        text = llm.decode(req.tokens)
        return DecodeResponse(text=text)
    except TokenizationError:
        raise