# tests/unit_tests/test_llm_service.py

"""
Unit tests for the LLMService class.

These tests validate the core, deterministic behavior of the service layer
without involving the HTTP / FastAPI layer.

Covered functionality:
- Text generation returns a non-empty string
- Encoding text into token IDs
- Decoding token IDs back into text
- Encode/decode round-trip consistency

Notes:
- These tests run against a real HuggingFace tokenizer and model (GPT-2).
- The tests intentionally use fixed inputs and expected token values
  to ensure deterministic behavior of encode/decode.
- Model loading is performed once at import time to reduce test runtime.
"""
from llm_api.services.llm_service import LLMService

# Create a single shared service instance for all tests.
# This avoids reloading the model for every test function.
llm = LLMService()


def test_generate():
    """
    Verify that text generation returns a non-empty string.

    This test ensures that:
    - The generate() method executes successfully
    - The returned value is a string
    - The generated output is not empty
    """
    out = llm.generate("Hello, my name is", max_tokens=20)
    assert isinstance(out, str)
    assert len(out) > 0


def test_basic_encode():
    """
    Verify that encoding text returns a list of integer token IDs.

    This test checks:
    - encode() returns a list
    - The list is non-empty
    - All elements are integers
    - The token IDs match the expected GPT-2 encoding
    """
    tokens = llm.encode("Hello, my name is")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(x, int) for x in tokens)
    assert tokens == [15496, 11, 616, 1438, 318]


def test_basic_decode():
    """
    Verify that decoding token IDs returns the expected text.

    This test ensures that:
    - decode() returns a string
    - The returned text is non-empty
    - The decoded text matches the expected original string
    """
    test = llm.decode([15496, 11, 616, 1438, 318])
    assert isinstance(test, str)
    assert len(test) > 0
    assert test == "Hello, my name is"


def test_encode_decode_roundtrip():
    """
    Verify encode/decode round-trip consistency.

    This test checks that:
    - Encoding text produces valid token IDs
    - Decoding those tokens back to text
    - Re-encoding the decoded text produces the same token IDs

    This guarantees tokenizer stability and reversibility
    for normal input text.
    """
    out = llm.encode("Hello, my name is Dani")
    assert isinstance(out, list)
    assert len(out) > 0
    assert all(isinstance(x, int) for x in out)
    assert out == llm.encode(llm.decode(out))


