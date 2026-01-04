# tests/integration_tests/test_api_routes.py
"""
Integration tests for the core API routes.

This module exercises the FastAPI application endpoints end-to-end using
TestClient, verifying:

- Health endpoint responds as expected
- Encode / Decode / Generate endpoints return successful responses when the
  underlying LLM service methods are mocked
- Request schema validation errors are returned as HTTP 422 responses when
  invalid payloads are sent (handled by FastAPI/Pydantic)

The tests use monkeypatch to replace the real LLM behavior with deterministic
functions, preventing heavy model loading and making the tests fast and stable.
"""

from fastapi.testclient import TestClient
from llm_api.app.main import app
import llm_api.api.routes as routes


def test_health():
    """
    Ensure the health endpoint is reachable and returns the expected payload.

    Expected behavior:
    - HTTP status code: 200
    - JSON body: {"status": "ok"}
    """
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_encode_success(monkeypatch):
    """
    Ensure /encode returns token IDs when the LLM encode method succeeds.

    The test monkeypatches routes.llm.encode to return a deterministic token list.

    Expected behavior:
    - HTTP status code: 200
    - JSON body: {"tokens": [1, 2, 3]}
    """
    monkeypatch.setattr(routes.llm, "encode", lambda text: [1, 2, 3])

    client = TestClient(app)
    res = client.post("/encode", json={"text": "Hello"})
    assert res.status_code == 200
    assert res.json() == {"tokens": [1, 2, 3]}


def test_decode_success(monkeypatch):
    """
    Ensure /decode returns decoded text when the LLM decode method succeeds.

    The test monkeypatches routes.llm.decode to return a deterministic string.

    Expected behavior:
    - HTTP status code: 200
    - JSON body: {"text": "Hello"}
    """
    monkeypatch.setattr(routes.llm, "decode", lambda tokens: "Hello")

    client = TestClient(app)
    res = client.post("/decode", json={"tokens": [1, 2, 3]})
    assert res.status_code == 200
    assert res.json() == {"text": "Hello"}


def test_generate_success(monkeypatch):
    """
    Ensure /generate returns generated text when the LLM generate method succeeds.

    The test monkeypatches routes.llm.generate to return a deterministic string.

    Expected behavior:
    - HTTP status code: 200
    - JSON body contains the generated text under "text"
    """
    monkeypatch.setattr(
        routes.llm,
        "generate",
        lambda prompt, max_tokens=50: "Hello, my name is Dani",
    )

    client = TestClient(app)
    res = client.post("/generate", json={"prompt": "Hello, my name is", "max_tokens": 10})
    assert res.status_code == 200
    assert res.json() == {"text": "Hello, my name is Dani"}


def test_generate_validation_error_empty_prompt():
    """
    Ensure invalid /generate payloads are rejected with a 422 validation error.

    This test sends an empty prompt, which violates the request schema.
    FastAPI/Pydantic should reject the request before reaching the route handler.

    Expected behavior:
    - HTTP status code: 422
    - Response body contains a "detail" field describing validation issues
    """
    client = TestClient(app)
    res = client.post("/generate", json={"prompt": "", "max_tokens": 10})
    assert res.status_code == 422  # RequestValidationError
    body = res.json()
    assert body["details"]  # list of validation issues


def test_decode_validation_error_wrong_type():
    """
    Ensure invalid /decode payloads are rejected with a 422 validation error.

    The decode endpoint expects a list of integers for "tokens".
    This test sends strings instead, which should fail schema validation.

    Expected behavior:
    - HTTP status code: 422
    - Response body contains a "detail" field describing validation issues
    """
    client = TestClient(app)
    res = client.post("/decode", json={"tokens": ["1", "2"]})
    assert res.status_code == 422


def test_request_id_middleware_adds_header():
    """
    Verify that the request ID middleware adds X-Request-Id to responses.

    This test sends a request to the /health endpoint with a predefined
    X-Request-Id header and asserts that:

    - The request succeeds with HTTP 200
    - The response contains the same X-Request-Id header value

    This ensures proper request tracing support across the API.
    """
    client = TestClient(app)
    res = client.get("/health", headers={"X-Request-Id": "test-id-123"})
    assert res.status_code == 200
    assert res.headers.get("X-Request-Id") == "test-id-123"