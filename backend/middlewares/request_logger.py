from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from time import time
import logging

logger = logging.getLogger("roster-neural")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time()
        response = await call_next(request)
        duration = time() - start
        logger.info(
            "%s %s - %s - %.3f ms",
            request.method,
            request.url.path,
            response.status_code,
            duration * 1000,
        )
        return response