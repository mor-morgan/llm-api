# tests/integration_tests/test_api_with_service_mock.py
"""
Integration tests for API routes using a mocked LLM service.

This module verifies the behavior of the FastAPI routes while replacing
the real LLMService implementation with lightweight mock functions.
The goal is to test request/response wiring, HTTP status codes, and
response schemas without loading or running a real language model.

These tests operate at the API layer and ensure that:
- Routes are correctly wired
- Request schemas are validated
- Responses match the expected API contract
"""
from fastapi.testclient import TestClient
from llm_api.app.main import app
import llm_api.api.routes as routes


client = TestClient(app)


def test_health_ok():
    """
    Verify that the health endpoint is reachable.

    This test ensures that:
    - The /health endpoint responds successfully
    - The HTTP status code is 200
    - The returned payload matches the expected health response
    """
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_generate_ok(monkeypatch):
    """
    Verify successful text generation via the /generate endpoint.

    The LLMService.generate method is mocked to return a fixed string
    regardless of input. This allows testing the API behavior without
    loading or running a real language model.

    This test validates that:
    - The endpoint accepts valid input
    - The HTTP response status is 200
    - The generated text is returned in the expected response format
    """
    def fake_generate(prompt: str, max_tokens: int = 50):
        return "Hello world"

    monkeypatch.setattr(routes.llm, "generate", fake_generate)

    res = client.post("/generate", json={"prompt": "hi", "max_tokens": 5})
    assert res.status_code == 200
    assert res.json() == {"text": "Hello world"}
    
def test_encode_ok(monkeypatch):
    """
    Verify successful tokenization via the /encode endpoint.

    The real LLMService.encode method is replaced with a fake
    implementation to avoid loading an actual model.

    This test asserts that:
    - The endpoint returns HTTP 200
    - The response contains the mocked token list
    """
    def fake_encode(text: str):
        return [1, 2, 3]

    monkeypatch.setattr(routes.llm, "encode", fake_encode)

    res = client.post("/encode", json={"text": "Hello"})
    assert res.status_code == 200
    assert res.json() == {"tokens": [1, 2, 3]}


def test_decode_ok(monkeypatch):
    """
    Verify successful decoding via the /decode endpoint.

    The real LLMService.decode method is mocked to return a fixed string.
    This avoids invoking any real decoding logic.

    This test ensures that:
    - The endpoint responds with HTTP 200
    - The decoded text matches the mocked output
    """
    def fake_decode(tokens):
        return "Hello"

    monkeypatch.setattr(routes.llm, "decode", fake_decode)

    res = client.post("/decode", json={"tokens": [1, 2, 3]})
    assert res.status_code == 200
    assert res.json() == {"text": "Hello"}
