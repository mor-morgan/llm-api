"""
This module creates and configures the FastAPI application. It wires together:
- Logging initialization
- Middleware registration
- API router registration
- A basic health check endpoint
- Global exception handlers that convert domain and framework errors into
  consistent JSON responses.

The exception handlers here act as the last line of defense, ensuring the API
returns predictable error payloads for both expected (domain) and unexpected
(server) failures.
"""

from llm_api.logs.logging_config import setup_logging
# Initialize logging as early as possible so startup and import-time logs are captured.
setup_logging()
from fastapi import FastAPI, Request
from llm_api.api.routes import router
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from llm_api.exceptions.llm_exceptions import (
    LLMError,
    ModelLoadError,
    TokenizationError,
    GenerationError,
)
import logging

from llm_api.logs.middleware import RequestIdMiddleware


logger = logging.getLogger()
logger.info("LLM Application startup completed")


# ---------------------------------------------------------------------------
# FastAPI app configuration
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM API")

# Add middleware to enrich requests/responses with a request id and enable
# request/response tracing across logs.
app.add_middleware(RequestIdMiddleware)

# Register API routes
app.include_router(router)

@app.get("/health")
def health():
    """
    Health check endpoint.

    Provides a simple readiness signal for uptime checks, load balancers, and
    orchestration systems.

    Returns:
        dict: A small payload indicating the service is running.
    """
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Domain exception handlers (LLM-related)
# ---------------------------------------------------------------------------
@app.exception_handler(ModelLoadError)
def handle_model_load_error(request, exc: ModelLoadError):
    """
    Handle model loading failures.

    This handler is invoked when the underlying model cannot be loaded or is not
    available. It returns HTTP 503 to indicate a temporary server-side
    unavailability.

    Args:
        request (Request): The incoming HTTP request.
        exc (ModelLoadError): The domain exception raised by the service layer.

    Returns:
        JSONResponse: A standardized JSON error response.
    """
    return JSONResponse(
        status_code=503,
        content={"error": "MODEL_LOAD_FAILED", "detail": str(exc)},
    )

@app.exception_handler(TokenizationError)
def handle_tokenization_error(request, exc: TokenizationError):
    """
    Handle tokenization/encoding/decoding failures.

    This handler covers failures when encoding text into tokens or decoding tokens
    back into text. It returns HTTP 400 because the request payload is considered
    invalid for the tokenizer.

    Args:
        request (Request): The incoming HTTP request.
        exc (TokenizationError): The domain exception raised by the service layer.

    Returns:
        JSONResponse: A standardized JSON error response.
    """
    return JSONResponse(
        status_code=400,
        content={"error": "TOKENIZATION_FAILED", "detail": str(exc)},
    )

@app.exception_handler(GenerationError)
def handle_generation_error(request, exc: GenerationError):
    """
    Handle text generation failures.

    This handler is invoked when model inference fails during text generation.
    It returns HTTP 500 since the failure is server-side and not caused directly
    by request validation.

    Args:
        request (Request): The incoming HTTP request.
        exc (GenerationError): The domain exception raised by the service layer.

    Returns:
        JSONResponse: A standardized JSON error response.
    """
    return JSONResponse(
        status_code=500,
        content={"error": "GENERATION_FAILED", "detail": str(exc)},
    )



# ---------------------------------------------------------------------------
# Framework / validation error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors raised by FastAPI/Pydantic.

    FastAPI raises RequestValidationError when request bodies, query params, or
    path params do not match the expected schema.

    Args:
        request (Request): The incoming HTTP request.
        exc (RequestValidationError): Validation error containing details.

    Returns:
        JSONResponse: A standardized JSON error response with validation details.
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "details": exc.errors(),
        },
    )
 
@app.exception_handler(Exception)
async def handle_unhandled_exception_error(request: Request, exc: Exception):
    """
    Handle HTTP exceptions raised by Starlette/FastAPI.

    These exceptions are raised explicitly (e.g., via HTTPException) or by
    underlying routing/middleware layers.

    Args:
        request (Request): The incoming HTTP request.
        exc (StarletteHTTPException): The HTTP exception containing status and detail.

    Returns:
        JSONResponse: A standardized JSON error response.
    """
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_SERVER_ERROR", "message": "Something went wrong"},
    )

# ---------------------------------------------------------------------------
# Catch-all handler
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, exc: StarletteHTTPException):
    """
    Handle unexpected/unhandled server errors.

    This is a safety net that ensures the API always returns a JSON response,
    even for unanticipated exceptions. It deliberately does not expose internal
    exception details to clients.

    Args:
        request (Request): The incoming HTTP request.
        exc (Exception): The unexpected exception.

    Returns:
        JSONResponse: A generic 500 error response.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP_ERROR", "message": exc.detail},
    )



