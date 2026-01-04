# tests/integration_tests/test_api_exceptions.py
"""
Integration tests for API-level exception handling.

This module verifies that the FastAPI application correctly translates
internal domain exceptions into proper HTTP responses.

The tests focus on:
- Ensuring the correct HTTP status codes are returned
- Ensuring the error payload follows the expected API contract
- Verifying that exception handlers are correctly wired to the routes

These tests operate at the API level using TestClient and monkeypatching
the LLM service methods to simulate failures.
"""
from fastapi.testclient import TestClient
from llm_api.app.main import app

def test_generate_model_load_error(monkeypatch):
    """
    Verify that a ModelLoadError raised during text generation
    is converted into a 503 Service Unavailable response.

    This test simulates a failure in the model loading phase by
    monkeypatching the llm.generate method to raise ModelLoadError.

    Expected behavior:
    - HTTP status code: 503
    - Error code: MODEL_LOAD_FAILED
    - Error detail contains the original exception message
    """
    import llm_api.api.routes as routes
    from llm_api.exceptions.llm_exceptions import ModelLoadError

    def boom(*args, **kwargs):
        raise ModelLoadError("model not found")

    monkeypatch.setattr(routes.llm, "generate", boom)

    client = TestClient(app)
    res = client.post("/generate", json={"prompt": "hi", "max_tokens": 5})

    assert res.status_code == 503
    body = res.json()
    assert body["error"] == "MODEL_LOAD_FAILED"
    assert "model not found" in body["detail"]


def test_encode_tokenization_error(monkeypatch):
    """
    Verify that a TokenizationError raised during encoding
    is converted into a 400 Bad Request response.

    This test simulates a tokenization failure by monkeypatching
    the llm.encode method to raise TokenizationError.

    Expected behavior:
    - HTTP status code: 400
    - Error code: TOKENIZATION_FAILED
    - Error detail contains the original exception message
    """
    import llm_api.api.routes as routes
    from llm_api.exceptions.llm_exceptions import TokenizationError

    def boom(*args, **kwargs):
        raise TokenizationError("bad input")

    monkeypatch.setattr(routes.llm, "encode", boom)

    client = TestClient(app)
    res = client.post("/encode", json={"text": "hi"})

    assert res.status_code == 400
    body = res.json()
    assert body["error"] == "TOKENIZATION_FAILED"
    assert "bad input" in body["detail"]


def test_generate_generation_error(monkeypatch):
    """
    Verify that a GenerationError raised during text generation
    is converted into a 500 Internal Server Error response.

    This test simulates a runtime generation failure (e.g. GPU OOM)
    by monkeypatching the llm.generate method to raise GenerationError.

    Expected behavior:
    - HTTP status code: 500
    - Error code: GENERATION_FAILED
    - Error detail contains the original exception message
    """
    import llm_api.api.routes as routes
    from llm_api.exceptions.llm_exceptions import GenerationError

    def boom(*args, **kwargs):
        raise GenerationError("gpu oom")

    monkeypatch.setattr(routes.llm, "generate", boom)

    client = TestClient(app)
    res = client.post("/generate", json={"prompt": "hi", "max_tokens": 5})

    assert res.status_code == 500
    body = res.json()
    assert body["error"] == "GENERATION_FAILED"
    assert "gpu oom" in body["detail"]