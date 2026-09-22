from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import request
from jevembed import JevEmbed, ModelConfig, BackendError, ValidationError
from jevembed.backends import HTTPEmbeddingBackend, EmbeddingInput
from jevembed.server import create_app


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
