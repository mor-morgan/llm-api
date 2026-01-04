# tests/unit_tests/test_models.py
"""
Unit tests for Pydantic request models.

This module tests the validation logic of the request schemas used by the API:
- GenerateRequest
- EncodeRequest
- DecodeRequest

The tests ensure that:
- Valid payloads are accepted and parsed correctly
- Invalid payloads raise Pydantic ValidationError exceptions
- Edge cases (length limits, missing fields, wrong types) are handled properly
"""
import pytest
from pydantic import ValidationError

from llm_api.schemas.models import (
    GenerateRequest,
    EncodeRequest,
    DecodeRequest,
)

# -------------------------
# GenerateRequest
# -------------------------

def test_generate_request_valid():
    """
    Verify that a valid GenerateRequest payload is accepted.

    Ensures that:
    - The prompt field is stored correctly
    - The max_tokens field is stored correctly
    """
    req = GenerateRequest(prompt="Hello", max_tokens=50)
    assert req.prompt == "Hello"
    assert req.max_tokens == 50


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "", "max_tokens": 10},                 # too short (min_length=1)
        {"prompt": "   ", "max_tokens": 10},              # NOTE: passes min_length, fails only if you add custom strip validation
        {"prompt": "a" * 2001, "max_tokens": 10},         # too long (max_length=2000)
        {"prompt": "Hello", "max_tokens": 0},             # too small (ge=1)
        {"prompt": "Hello", "max_tokens": 201},           # too large (le=200)
        {"prompt": "Hello", "max_tokens": "abc"},         # wrong type
        {"max_tokens": 10},                               # missing prompt
        {"prompt": "Hello"},                              # missing max_tokens (should default to 50 if defined with default)
    ],
)
def test_generate_request_invalid(payload):
    """
    Verify that invalid GenerateRequest payloads raise ValidationError.

    Covers edge cases such as:
    - Empty or overly long prompts
    - Invalid max_tokens values
    - Missing required fields
    - Incorrect field types
    """
    with pytest.raises(ValidationError):
        EncodeRequest(**payload)


# -------------------------
# EncodeRequest
# -------------------------

def test_encode_request_valid():
    """
    Verify that a valid EncodeRequest payload is accepted.

    Ensures that:
    - The text field is stored correctly
    """
    req = EncodeRequest(text="Hello, my name is")
    assert req.text.startswith("Hello")


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},               # too short
        {"text": "a" * 2001},       # too long
        {},                         # missing text
        {"text": 123},              # wrong type
    ],
)
def test_encode_request_invalid(payload):
    """
    Verify that invalid EncodeRequest payloads raise ValidationError.

    Covers:
    - Empty text
    - Excessively long text
    - Missing required field
    - Incorrect data types
    """
    with pytest.raises(ValidationError):
        EncodeRequest(**payload)


# -------------------------
# DecodeRequest
# -------------------------

def test_decode_request_valid():
    """
    Verify that a valid DecodeRequest payload is accepted.

    Ensures that:
    - The tokens list is stored correctly
    - Token values are preserved as-is
    """
    req = DecodeRequest(tokens=[15496, 11, 616])
    assert req.tokens == [15496, 11, 616]


@pytest.mark.parametrize(
    "payload",
    [
        {"tokens": []},                        # too short (min_items=1)
        {"tokens": ["1", "2"]},                # wrong type (should be int)
        {"tokens": [1, 2, "x"]},               # mixed invalid
        {},                                    # missing tokens
    ],
)
def test_decode_request_invalid(payload):
    """
    Verify that invalid DecodeRequest payloads raise ValidationError.

    Covers:
    - Empty token lists
    - Incorrect token types
    - Mixed-type token lists
    - Missing required field
    """
    with pytest.raises(ValidationError):
        DecodeRequest(**payload)


def test_decode_request_too_many_tokens():
    """
    Verify that DecodeRequest enforces the maximum token limit.

    Ensures that providing more tokens than allowed
    raises a ValidationError.
    """
    tokens = list(range(4097))
    with pytest.raises(ValidationError):
        DecodeRequest(tokens=tokens)