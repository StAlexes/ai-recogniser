from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from airec.api import rag as rag_api
from airec.api.app import create_app
from airec.core.config import Settings, save_settings
from airec.core.paths import AppPaths
from airec.rag.client import RagError, RagKbClient, normalize_base_url

TOKEN = "test-token"
HITS = [
    {
        "id": 7,
        "source": "docs.md",
        "breadcrumb": "Раздел > Термин",
        "text": "Фрагмент документа",
        "position": 42,
        "score": 0.75,
        "dense": 0.7,
        "lexical": 0.2,
        "exact": True,
        "match": "exact",
        "extra": "ignored",
    }
]


def test_client_status_search_and_ask() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"chunks": 11959, "indexing": False, "gen_model": "gemma3:4b"})
        if request.url.path == "/search":
            assert request.url.params["q"] == "API"
            assert request.url.params["k"] == "3"
            return httpx.Response(200, json={"results": HITS})
        if request.url.path == "/ask":
            payload = json.loads(request.content)
            assert payload == {"question": "Что такое API?", "k": 2}
            return httpx.Response(200, json={"answer": "Ответ", "sources": HITS})
        return httpx.Response(404)

    client = RagKbClient("http://127.0.0.1:8000/", transport=httpx.MockTransport(handler))

    status = client.status()
    assert status["reachable"] is True
    assert status["chunks"] == 11959
    assert status["gen_model"] == "gemma3:4b"
    assert status["error"] is None

    search = client.search("API", 3)
    assert search["results"][0]["source"] == "docs.md"
    assert search["results"][0]["score"] == 0.75
    assert search["results"][0]["exact"] is True
    assert "extra" not in search["results"][0]

    asked = client.ask("Что такое API?", 2)
    assert asked["answer"] == "Ответ"
    assert asked["sources"][0]["breadcrumb"] == "Раздел > Термин"


def test_unreachable_rag_is_reported_not_raised_by_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = RagKbClient("http://127.0.0.1:8000", transport=httpx.MockTransport(handler))
    status = client.status()
    assert status["reachable"] is False
    assert "недоступен" in status["error"]

    with pytest.raises(RagError) as excinfo:
        client.search("API", 3)
    assert excinfo.value.status_code == 502


def test_base_url_validation() -> None:
    assert normalize_base_url(" http://127.0.0.1:8000/ ") == "http://127.0.0.1:8000"
    for value in ("", "127.0.0.1:8000", "ftp://127.0.0.1:8000", "http://user:pass@host"):
        with pytest.raises(RagError):
            normalize_base_url(value)


def make_client(tmp_path: Path, settings: Settings | None = None) -> TestClient:
    paths = AppPaths(
        root=tmp_path,
        data=tmp_path,
        models=tmp_path / "models",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
        results=tmp_path / "results",
        temp=tmp_path / "temp",
        bin=tmp_path / "bin",
        ui=tmp_path / "ui-dist",
    )
    paths.ensure()
    active = settings or Settings()
    save_settings(paths.config_file, active)
    app = create_app(settings=active, paths=paths, token=TOKEN, auth_enabled=True, port=12345)
    return TestClient(app)


def test_rag_api_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def status(self) -> dict[str, object]:
            return {"enabled": True, "reachable": True, "chunks": 5, "error": None}

        def search(self, query: str, k: int) -> dict[str, object]:
            assert query == "API"
            assert k == 5
            return {"query": query, "results": HITS}

        def ask(self, question: str, k: int) -> dict[str, object]:
            assert question == "Что такое API?"
            assert k == 5
            return {"question": question, "answer": "Ответ", "sources": HITS}

    monkeypatch.setattr(rag_api, "RagKbClient", FakeClient)
    with make_client(tmp_path) as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        status = client.get("/api/v1/rag/status", headers=headers)
        assert status.status_code == 200
        assert status.json()["reachable"] is True

        search = client.get("/api/v1/rag/search?q=API", headers=headers)
        assert search.status_code == 200
        assert search.json()["results"][0]["source"] == "docs.md"

        ask = client.post(
            "/api/v1/rag/ask",
            headers=headers,
            json={"question": "Что такое API?"},
        )
        assert ask.status_code == 200
        assert ask.json()["answer"] == "Ответ"
