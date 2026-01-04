LLM API Service
---------------
A production-style FastAPI-based LLM service that wraps a Hugging Face causal 
language model behind a clean HTTP API, with proper layering, error handling, testing, and Docker packaging.
This project was built as a structured backend exercise and demonstrates clean architecture, testing practices, and containerization.

Features
--------
 * FastAPI HTTP API
 * Hugging Face Transformers integration (gpt2 by default)
 * Centralized LLM service layer
 * Domain-specific exceptions
 * Structured logging with middleware
 * Unit tests and integration tests
 * Fully Dockerized application

Project Structure
-----------------
llm-api/
├── src/
│   └── llm_api/
│       ├── api/
│       │   └── routes.py            # API endpoints
│       ├── app/
│       │   ├── main.py              # FastAPI application entrypoint
│       │   └── _init_.py
│       ├── exceptions/
│       │   └── llm_exceptions.py    # Domain-specific exceptions
│       ├── logs/
│       │   ├── logging_config.py    # Logging configuration
│       │   └── middleware.py        # Logging middleware
│       ├── schemas/
│       │   └── models.py            # Pydantic request/response models
│       ├── services/
│       │   └── llm_service.py       # LLM service layer
│       └── _init_.py
│
├── tests/
│   ├── unit_tests/
│   │   ├── test_llm_service.py
│   │   ├── test_exception_handlers.py
│   │   └── test_models.py
│   └── integration_tests/
│       ├── test_api_routes.py
│       ├── test_api_exceptions.py
│       └── test_api_with_service_mock.py
│
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── poetry.lock
└── README.md

Architecture Overview
---------------------

LLM Service Layer
-----------------
The LLMService encapsulates all interactions with the language model:
 * Loading the model and tokenizer
 * Text generation
 * Token encoding and decoding
This keeps the API layer clean and decoupled from model-specific logic.

API Layer
---------
The FastAPI layer:
 * Validates input and output using Pydantic schemas
 * Delegates all business logic to the service layer
 * Translates domain-specific exceptions into HTTP responses

Exception Handling
------------------
Custom exceptions (e.g. ModelLoadError, TokenizationError, GenerationError) allow:
 * Clear separation between domain errors and HTTP concerns
 * Consistent and testable API error responses


Testing
-------
Unit Tests
----------
Unit tests focus on isolated logic:
 * LLM service behavior (with mocked tokenizer/model)
 * Exception handlers
 * Schema validation
    Run:
        $ poetry run pytest tests/unit_tests

Integration Tests
-----------------
Integration tests validate the API behavior:
 * FastAPI routes using TestClient
 * Error handling paths
 * API behavior with mocked services
    Run:
        $ poetry run pytest tests/integration_tests

Run all tests:
    $ poetry run pytest -q -vv


Running with Docker
-------------------
Build the Docker image
$ docker build --no-cache -t llm-api .

    Note:
    The first build may take several minutes due to dependency installation
    and Hugging Face model downloads.

Run the container
-----------------
$ docker run --rm -p 8000:8000 llm-api

Once running, the API is available at:
 * Swagger UI: http://localhost:8000/docs
 * OpenAPI JSON: http://localhost:8000/openapi.json

Cold Start Behavior
-------------------
The application loads the language model during startup.
For models like gpt2, startup can take 1–3 minutes, depending on:
 * Network speed
 * Local hardware
 * Docker image cache
This behavior is expected and clearly logged during startup.

Configuration
-------------
 * Python version: 3.11
 * Dependency management: Poetry
 * LLM backend: Hugging Face Transformers
 * Default model: gpt2 (configurable in LLMService)
 * Container runtime: Docker

Design Decisions
----------------
 * Service isolation: LLM logic is not coupled to HTTP
 * Fail-fast startup: Model loading errors are surfaced immediately
 * Deterministic tests: External dependencies are mocked where needed
 * Docker-first approach: The project can run without local Python setup

How This Project Is Submitted
-----------------------------
This project is intended to be submitted as:
 * A Git repository or ZIP archive
 * Containing the full source code and Dockerfile

The evaluator can run the project using:
$ docker build --no-cache -t llm-api .
$ docker run --rm -p 8000:8000 llm-api

or
$ poetry run uvicorn llm_api.app.main:app --reload

No additional setup is required.