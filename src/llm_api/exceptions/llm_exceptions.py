"""
Custom exceptions for the LLM API.

These exceptions allow us to distinguish between different failure modes
(model loading, tokenization, generation, decoding) and map them to
appropriate HTTP error responses.
"""

class LLMError(Exception):
    """Base class for all LLM-related errors"""

class ModelLoadError(LLMError):
    """Raised when model or tokenizer cannot be loaded"""


class TokenizationError(LLMError):
    """Raised when tokenization or detokenization fails"""


class GenerationError(LLMError):
    """Raised when text generation fails"""