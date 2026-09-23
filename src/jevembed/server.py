import json
import threading
from dataclasses import dataclass

from .errors import BackendError, ValidationError
from .schemas import validate_request


@dataclass(frozen=True)
class HTTPServiceLimits:
    """Per-process admission limits for the optional HTTP API."""

    max_body_bytes: int = 2 * 1024 * 1024
    max_questions: int = 64
    max_embedding_inputs: int = 4096
    max_concurrent_requests: int = 4

    def __post_init__(self):
        for name in ("max_body_bytes", "max_questions", "max_embedding_inputs", "max_concurrent_requests"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def create_app(client, *, enable_debug=False, limits=None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    limits = HTTPServiceLimits() if limits is None else limits
    if not isinstance(limits, HTTPServiceLimits):
        raise TypeError("limits must be an HTTPServiceLimits instance")
    slots = threading.BoundedSemaphore(limits.max_concurrent_requests)
    app = FastAPI(title="JevEmbed", version="0.1.0")

    @app.exception_handler(ValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(BackendError)
    async def backend_error(request, exc):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    async def read_request(request):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > limits.max_body_bytes:
                    raise HTTPException(413, detail=f"Request body exceeds {limits.max_body_bytes} bytes")
            except ValueError:
                pass  # The streamed byte count remains authoritative.
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > limits.max_body_bytes:
                raise HTTPException(413, detail=f"Request body exceeds {limits.max_body_bytes} bytes")
            raw.extend(chunk)
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValidationError("Request body must be valid JSON") from exc
        try:
            if isinstance(body, dict) and isinstance(body.get("questions"), dict) and len(body["questions"]) > limits.max_questions:
                raise HTTPException(413, detail=f"Request exceeds {limits.max_questions} questions")
            validate_request(body)  # HTTP model is mandatory.
            # Reject large candidate lists before materializing rendered plans.
            minimum_inputs = sum(1 + len(question["criteria"]) if question["type"] in ("choice", "score")
                                 else 2 for question in body["questions"].values())
            if minimum_inputs > limits.max_embedding_inputs:
                raise HTTPException(413, detail=f"Request exceeds {limits.max_embedding_inputs} embedding inputs")
            _, plans = client._prepare(body)
        except RecursionError as exc:
            raise ValidationError("Request JSON is too deeply nested") from exc
        if sum(len(plan.inputs) for plan in plans) > limits.max_embedding_inputs:
            raise HTTPException(413, detail=f"Request exceeds {limits.max_embedding_inputs} embedding inputs")
        return body

    async def admitted_request(request, operation):
        if not slots.acquire(blocking=False):
            raise HTTPException(429, detail="Too many concurrent requests", headers={"Retry-After": "1"})
        try:
            body = await read_request(request)
            return await run_in_threadpool(operation, body)
        finally:
            slots.release()

    @app.post("/v1/systemone")
    async def system_one(request: Request):
        return await admitted_request(request, client.evaluate)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": client.models()}

    if enable_debug:
        @app.post("/debug/explain")
        async def explain(request: Request):
            return await admitted_request(request, client.explain)

    return app
