"""
Request middleware utilities.

This module provides middleware components used for request-level concerns,
such as request tracing and structured logging.

Currently includes:
- RequestIdMiddleware: assigns a unique request_id to each request
  and propagates it through logs and response headers.
"""

import logging
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("llm_api.http")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that attaches a request ID to each incoming HTTP request.

    The request ID is:
    - Read from the incoming X-Request-Id header if provided
    - Otherwise generated automatically (UUID4)

    The request ID is:
    - Stored on request.state.request_id
    - Added to the response headers
    - Included in structured log messages
    """
    async def dispatch(self, request: Request, call_next):
        """
        Process an incoming HTTP request and attach a request ID.

        This method:
        - Extracts or generates a request ID
        - Logs request start and completion
        - Adds the request ID to response headers
        - Logs unhandled exceptions with request context

        Args:
            request (Request): Incoming FastAPI request object.
            call_next (Callable): Function that forwards the request to the next middleware/handler.

        Returns:
            Response: The HTTP response returned by the downstream handler.
        """
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id

        logger.info(
            "request_started method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )

        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed method=%s path=%s request_id=%s",
                request.method,
                request.url.path,
                request_id,
            )
            raise

        response.headers["X-Request-Id"] = request_id
        logger.info(
            "request_finished method=%s path=%s status=%s request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            request_id,
        )
        return response