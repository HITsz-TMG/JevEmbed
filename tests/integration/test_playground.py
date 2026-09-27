"""The optional Playground uses the existing decision API and its admission limits."""

from fastapi.testclient import TestClient

from conftest import request
from jevembed.server import HTTPServiceLimits, create_app


def test_playground_is_opt_in_and_serves_its_assets(client):
    plain = TestClient(create_app(client))
    assert plain.get("/").status_code == 404
    assert plain.get("/playground/").status_code == 404

    enabled = TestClient(create_app(client, enable_playground=True))
    redirect = enabled.get("/", follow_redirects=False)
    assert redirect.status_code in (307, 308)
    assert redirect.headers["location"].endswith("playground/")

    page = enabled.get("/playground/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "JevEmbed" in page.text
    for name, content_type in (("app.js", "javascript"), ("logic.mjs", "javascript"),
                               ("games.js", "javascript"), ("style.css", "text/css"),
                               ("games.css", "text/css")):
        asset = enabled.get(f"/playground/{name}")
        assert asset.status_code == 200
        assert content_type in asset.headers["content-type"]


def test_game_lab_uses_small_stateless_turns(client):
    app = TestClient(create_app(client, enable_playground=True))
    for kind in ("dino", "blocks", "snake", "racing"):
        opened = app.get("/playground/api/games/new", params={"kind": kind})
        assert opened.status_code == 200
        game = opened.json()
        assert game["question"]["type"] == "choice"
        action = next(iter(game["question"]["criteria"]))
        advanced = app.post("/playground/api/games/step", json={"state": game["state"], "action": action})
        assert advanced.status_code == 200
        assert advanced.json()["stats"]["turn"] == 1
    assert app.post("/playground/api/games/step", json={"state": game["state"], "action": "Fly"}).status_code == 422
    assert app.post("/playground/api/games/step", content=b"x" * (16 * 1024 + 1)).status_code == 413
    disabled = TestClient(create_app(client))
    assert disabled.get("/playground/api/games/new").status_code == 404


def test_playground_keeps_model_api_and_resource_limits(client):
    app = TestClient(create_app(client, enable_playground=True,
                                limits=HTTPServiceLimits(max_embedding_inputs=2, max_body_bytes=1024)))
    body = {**request("noul"), "model": "test"}
    expected = client.evaluate(body)
    client.clear_cache()

    assert app.get("/v1/models").json()["data"][0]["id"] == "test"
    response = app.post("/v1/systemone", json=body)
    assert response.status_code == 200
    assert response.json() == expected
    assert app.post("/v1/systemone", json=request("noul")).status_code == 422

    too_much_work = {**request("choice"), "model": "test"}
    response = app.post("/v1/systemone", json=too_much_work)
    assert response.status_code == 413
    assert "embedding inputs" in response.json()["detail"]
    oversized = {**body, "state": "x" * 1024}
    assert app.post("/v1/systemone", json=oversized).status_code == 413
