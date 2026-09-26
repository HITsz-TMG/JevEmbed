from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import request
from jevembed import JevEmbed, ModelConfig, BackendError, ValidationError
from jevembed.backends import HTTPEmbeddingBackend, EmbeddingInput
from jevembed.server import HTTPServiceLimits, create_app


def http_config(**kwargs):
    return ModelConfig(model_id="http", backend="http", model_name_or_path="remote", server_enforces_length=True, **kwargs)


def test_http_batching_reordering_usage_templates_and_auth(monkeypatch):
    monkeypatch.setenv("TEST_EMBED_KEY", "test-value")
    calls = []
    def handler(r):
        data = json.loads(r.content)
        assert r.url.path == "/v1/embeddings"
        assert r.headers["authorization"] == "Bearer test-value"
        assert set(data) == {"model", "input", "encoding_format"}
        calls.append(data["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [i+1, 1]} for i in reversed(range(len(data["input"])))], "usage": {"prompt_tokens": len(data["input"])*11}})
    cfg = http_config(batch_size=2, api_key_env="TEST_EMBED_KEY")
    backend = HTTPEmbeddingBackend(cfg, transport=httpx.MockTransport(handler))
    batch = backend.encode([EmbeddingInput("query", "Original", "state"), EmbeddingInput("document", "", "doc"), EmbeddingInput("document", "", "third")])
    assert batch.vectors == [[1, 1], [2, 1], [1, 1]] and batch.input_tokens == 33
    assert calls == [["Instruct: Original\nQuery: state", "doc"], ["third"]]


@pytest.mark.parametrize("rows", [[], [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [1]}],
    [{"index": 1, "embedding": [1]}], [{"index": True, "embedding": [1]}], [{"embedding": [1]}]])
def test_http_bad_indices(rows):
    b = HTTPEmbeddingBackend(http_config(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": rows, "usage": {"prompt_tokens": 1}})))
    with pytest.raises(BackendError):
        b.encode([EmbeddingInput("document", "", "x")])


def test_http_retry_timeout_and_permanent_error(monkeypatch):
    monkeypatch.setattr("jevembed.backends.http.time.sleep", lambda seconds: None)
    attempts = []
    def handler(r):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("timeout")
        if len(attempts) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}], "usage": {"prompt_tokens": 1}})
    b = HTTPEmbeddingBackend(http_config(), transport=httpx.MockTransport(handler))
    assert b.encode([EmbeddingInput("document", "", "x")]).input_tokens == 1
    assert len(attempts) == 3
    attempts.clear()
    def bad(r):
        attempts.append(1)
        return httpx.Response(401)
    b.transport = httpx.MockTransport(bad)
    with pytest.raises(BackendError, match="401"):
        b.encode([EmbeddingInput("document", "", "x")])
    assert len(attempts) == 1
    attempts.clear()
    def timeout(r):
        attempts.append(1)
        raise httpx.ReadTimeout("timeout")
    b.transport = httpx.MockTransport(timeout)
    with pytest.raises(BackendError, match="transport"):
        b.encode([EmbeddingInput("document", "", "x")])
    assert len(attempts) == 3


def test_http_usage_modes_concurrent():
    def handler(r):
        data = json.loads(r.content)
        text = data["input"][0]
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}], "usage": {"prompt_tokens": len(text)}})
    b = HTTPEmbeddingBackend(http_config(), transport=httpx.MockTransport(handler))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda n: b.encode([EmbeddingInput("document", "", "x"*n)]).input_tokens, range(1, 9)))
    assert results == list(range(1, 9))
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}]}))
    with pytest.raises(BackendError, match="usage"):
        HTTPEmbeddingBackend(http_config(), transport=transport).encode([EmbeddingInput("document", "", "hello")])
    batch = HTTPEmbeddingBackend(http_config(usage_mode="estimate"), transport=transport).encode([EmbeddingInput("document", "", "hello")])
    assert batch.input_tokens == 2 and "estimated" in batch.usage_source


def test_same_model_http_requests_encode_concurrently():
    barrier = threading.Barrier(2)

    def handler(r):
        barrier.wait(timeout=5)
        count = len(json.loads(r.content)["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [1, 0]} for i in range(count)],
                                         "usage": {"prompt_tokens": count}})

    cfg = http_config(cache_capacity=0)
    client = JevEmbed(config=cfg, backend=HTTPEmbeddingBackend(cfg, transport=httpx.MockTransport(handler)))
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(client.evaluate, [{**request(), "state": state} for state in ("one", "two")]))
    assert all(result["answers"]["q"]["type"] == "choice" for result in results)


def test_http_requires_explicit_length_and_prompt_contract():
    with pytest.raises(ValidationError):
        HTTPEmbeddingBackend(ModelConfig(backend="http"))
    with pytest.raises(ValidationError):
        HTTPEmbeddingBackend(http_config(template_owner="server"))
    with pytest.raises(ValidationError):
        HTTPEmbeddingBackend(http_config(overflow_policy="truncate"))


def test_server_python_parity_and_validation(client):
    app = TestClient(create_app(client, enable_debug=True))
    r = {**request(), "model": "test"}
    expected = client.evaluate(r)
    client.clear_cache()
    response = app.post("/v1/systemone", json=r)
    assert response.status_code == 200 and response.json() == expected
    assert app.get("/v1/models").json()["data"][0]["id"] == "test"
    assert app.post("/debug/explain", json=r).status_code == 200
    assert app.post("/v1/systemone", json=request()).status_code == 422
    assert app.post("/v1/systemone", json={**r, "model": "unknown"}).status_code == 422
    assert app.post("/v1/systemone", content="not JSON").status_code == 422
    assert app.post("/v1/systemone", content='{"model":"test","state":[NaN],"questions":{}}').status_code == 422
    r = {**request("noul"), "model": "test"}
    assert set(app.post("/v1/systemone", json=r).json()["answers"]["q"]) == {"type", "noul"}


def test_server_backend_failure(client, backend):
    def broken(items):
        raise BackendError("backend unavailable")
    backend.encode = broken
    response = TestClient(create_app(client)).post("/v1/systemone", json={**request(), "model": "test"})
    assert response.status_code == 502 and "answers" not in response.json()


def test_server_limits_work_without_limiting_python_api(client):
    large = {**request(), "model": "test"}
    large["questions"]["q"]["criteria"] = {str(i): f"candidate {i}" for i in range(300)}
    assert len(client.evaluate(large)["answers"]["q"]["probabilities"]) == 300

    app = TestClient(create_app(client, limits=HTTPServiceLimits(max_embedding_inputs=301)))
    assert app.post("/v1/systemone", json=large).status_code == 200
    app = TestClient(create_app(client, limits=HTTPServiceLimits(max_embedding_inputs=300)))
    response = app.post("/v1/systemone", json=large)
    assert response.status_code == 413 and "embedding inputs" in response.json()["detail"]

    app = TestClient(create_app(client, limits=HTTPServiceLimits(max_questions=1)))
    doubled = {**large, "questions": {"q": large["questions"]["q"], "q2": large["questions"]["q"]}}
    assert app.post("/v1/systemone", json=doubled).status_code == 413


def test_server_rejects_oversized_body_even_if_content_length_is_small(client):
    app = TestClient(create_app(client, limits=HTTPServiceLimits(max_body_bytes=100)))
    large = json.dumps({**request(), "model": "test", "state": "x" * 200}).encode()
    assert app.post("/v1/systemone", content=large).status_code == 413
    assert app.post("/v1/systemone", content=large, headers={"Content-Length": "1"}).status_code == 413


def test_server_rejects_requests_when_inference_slots_are_busy():
    from conftest import FixedBackend

    started, release = threading.Event(), threading.Event()

    class BlockingBackend(FixedBackend):
        def encode(self, items):
            started.set()
            assert release.wait(5)
            return super().encode(items)

    client = JevEmbed(backend=BlockingBackend(), model="test")
    app = TestClient(create_app(client, limits=HTTPServiceLimits(max_concurrent_requests=1)))
    body = {**request(), "model": "test"}
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(app.post, "/v1/systemone", json=body)
        try:
            assert started.wait(5)
            busy = app.post("/v1/systemone", json=body)
            assert busy.status_code == 429 and busy.headers["Retry-After"] == "1"
        finally:
            release.set()
        assert first.result().status_code == 200


@pytest.mark.parametrize("path", ["/v1/systemone", "/debug/explain"])
def test_server_times_out_stalled_body_and_releases_slot(client, path):
    app = create_app(client, enable_debug=True,
                     limits=HTTPServiceLimits(max_concurrent_requests=1, body_read_timeout_seconds=0.1))
    body = {**request(), "model": "test"}

    async def check():
        stalled = asyncio.Event()

        async def upload():
            yield b"{"
            stalled.set()
            await asyncio.Event().wait()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            first = asyncio.create_task(http.post(path, content=upload()))
            try:
                await asyncio.wait_for(stalled.wait(), 2)
                busy = await http.post(path, json=body)
                assert busy.status_code == 429
                timed_out = await asyncio.wait_for(first, 2)
                assert timed_out.status_code == 408
                assert "body" in timed_out.json()["detail"]
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as recovered:
                    assert (await asyncio.wait_for(recovered.post(path, json=body), 2)).status_code == 200
            finally:
                if not first.done():
                    first.cancel()
                    await asyncio.gather(first, return_exceptions=True)

    asyncio.run(check())


def test_server_body_timeout_is_overall_not_per_chunk(client):
    app = create_app(client, limits=HTTPServiceLimits(body_read_timeout_seconds=0.08))
    body = json.dumps({**request(), "model": "test"}).encode()

    async def upload():
        for byte in body:
            yield bytes([byte])
            await asyncio.sleep(0.02)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await asyncio.wait_for(http.post("/v1/systemone", content=upload()), 2)
            assert response.status_code == 408
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as recovered:
                assert (await asyncio.wait_for(recovered.post("/v1/systemone", json={**request(), "model": "test"}), 2)).status_code == 200

    asyncio.run(check())


def test_server_cancelled_upload_releases_slot(client):
    app = create_app(client, limits=HTTPServiceLimits(max_concurrent_requests=1, body_read_timeout_seconds=5))
    body = {**request(), "model": "test"}

    async def check():
        stalled = asyncio.Event()

        async def upload():
            yield b"{"
            stalled.set()
            await asyncio.Event().wait()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            first = asyncio.create_task(http.post("/v1/systemone", content=upload()))
            await asyncio.wait_for(stalled.wait(), 2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first, 2)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as recovered:
                assert (await asyncio.wait_for(recovered.post("/v1/systemone", json=body), 2)).status_code == 200

    asyncio.run(check())


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_server_limit_values_must_be_positive_integers(value):
    with pytest.raises(ValueError):
        HTTPServiceLimits(max_concurrent_requests=value)


@pytest.mark.parametrize("value", [0, -1, False, float("nan"), float("inf"), -float("inf"), "30"])
def test_server_body_timeout_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="body_read_timeout_seconds"):
        HTTPServiceLimits(body_read_timeout_seconds=value)


def test_local_embedding_server_over_tcp():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            response = json.dumps({"data": [{"index": i, "embedding": [1, i+1]} for i in reversed(range(len(body["input"])))],
                                   "usage": {"prompt_tokens": 17*len(body["input"])}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = http_config(base_url=f"http://127.0.0.1:{server.server_port}/v1", batch_size=2)
        c = JevEmbed(config=cfg)
        result = c.evaluate(request())
        assert result["usage"]["input_tokens"] == 51 and len(calls) == 2
        assert set(result["answers"]["q"]["probabilities"]) == {"a", "b"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
