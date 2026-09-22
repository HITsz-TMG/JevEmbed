from .errors import BackendError, ValidationError
from .schemas import validate_request


def create_app(client, *, enable_debug=False):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="JevEmbed", version="0.1.0")

    @app.exception_handler(ValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(BackendError)
    async def backend_error(request, exc):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    async def read_request(request):
        try:
            body = await request.json()
        except (ValueError, UnicodeError) as exc:
            raise ValidationError("Request body must be valid JSON") from exc
        validate_request(body)  # HTTP model is mandatory.
        return body

    @app.post("/v1/systemone")
    async def system_one(request: Request):
        body = await read_request(request)
        return await run_in_threadpool(client.evaluate, body)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": client.models()}

    if enable_debug:
        @app.post("/debug/explain")
        async def explain(request: Request):
            return client.explain(await read_request(request))

    return app
