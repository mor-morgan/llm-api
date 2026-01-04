# tests/unit_tests/test_exceptions.py

"""
Unit tests for API exception handlers.

This module verifies that FastAPI exception handler functions
return the correct HTTP status codes and JSON error payloads
when specific LLM-related exceptions are raised.

The handlers tested here are pure functions (request + exception -> response),
making them suitable for true unit testing without running the server.
"""

from fastapi.responses import JSONResponse
from starlette.requests import Request

from llm_api.app.main import handle_model_load_error, handle_tokenization_error, handle_generation_error
from llm_api.exceptions.llm_exceptions import ModelLoadError, TokenizationError, GenerationError

import json

def _dummy_request() -> Request:
    """
    Create a minimal dummy HTTP Request object.

    The exception handlers require a Request instance as part of their
    signature, but do not actually use any request attributes.
    This helper provides the smallest valid Request object needed
    for unit testing.

    Returns:
        Request: A minimal Starlette Request instance.
    """
    return Request({"type": "http", "method": "GET", "path": "/"})

def test_handle_model_load_error():
    """
    Test handling of ModelLoadError exceptions.

    Verifies that:
    - The handler returns a JSONResponse
    - HTTP status code is 503 (Service Unavailable)
    - The response body contains the correct error code and detail message
    """
    req = _dummy_request()

    exc = ModelLoadError("model failed to load")

    response = handle_model_load_error(req, exc)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert response.body == b'{"error":"MODEL_LOAD_FAILED","detail":"model failed to load"}'

def test_handle_tokenization_error():
    """
    Test handling of TokenizationError exceptions.

    Verifies that:
    - HTTP status code is 400 (Bad Request)
    - The error code is TOKENIZATION_FAILED
    - The detail message matches the exception message
    """
    req = _dummy_request()
    exc = TokenizationError("bad input")

    response = handle_tokenization_error(req, exc)

    assert response.status_code == 400
    body = json.loads(response.body.decode("utf-8"))
    assert body["error"] == "TOKENIZATION_FAILED"
    assert body["detail"] == "bad input"

def test_handle_generation_error():
    """
    Test handling of GenerationError exceptions.

    Verifies that:
    - HTTP status code is 500 (Internal Server Error)
    - The error code is GENERATION_FAILED
    - The detail message matches the exception message
    """
    req = _dummy_request()
    exc = GenerationError("generation crashed")

    response = handle_generation_error(req, exc)

    assert response.status_code == 500
    body = json.loads(response.body.decode("utf-8"))
    assert body["error"] == "GENERATION_FAILED"
    assert body["detail"] == "generation crashed"

