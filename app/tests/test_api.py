from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from airec.api.app import create_app
from airec.core.config import Settings
from airec.core.paths import AppPaths
from airec.version import __version__

TOKEN = "test-token"


def make_client(tmp_path: Path, ui_dir: Path | None = None) -> TestClient:
    monkeypatch_paths = AppPaths(
        root=tmp_path,
        data=tmp_path,
        models=tmp_path / "models",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
        results=tmp_path / "results",
        temp=tmp_path / "temp",
        bin=tmp_path / "bin",
        ui=ui_dir or (tmp_path / "ui-dist"),
    )
    monkeypatch_paths.ensure()
    app = create_app(
        settings=Settings(),
        paths=monkeypatch_paths,
        token=TOKEN,
        auth_enabled=True,
        port=12345,
    )
    return TestClient(app)


def test_health_requires_token(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        assert client.get("/api/v1/health").status_code == 401
        response = client.get("/api/v1/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200
        assert response.json()["version"] == __version__
        assert response.json()["status"] == "ok"


def test_healthz_is_public(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_static_ui_is_public_but_api_is_not(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        assert client.get("/").status_code == 200
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/api/v1/settings").status_code == 401


def test_settings_update_persists(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        current = client.get("/api/v1/settings", headers=headers).json()
        current["theme"] = "dark"
        current["close_to_tray"] = False

        response = client.put("/api/v1/settings", json=current, headers=headers)
        assert response.status_code == 200
        assert response.json()["theme"] == "dark"
        assert (tmp_path / "config.json").exists()


def test_jobs_stub(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        assert client.get("/api/v1/jobs", headers=headers).json()["total"] == 0
        assert client.post("/api/v1/jobs", headers=headers).status_code == 501


def test_models_catalog(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get("/api/v1/models", headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200
        ids = {item["id"] for item in response.json()["items"]}
        assert "gigaam-v3-e2e-ctc" in ids


def test_spa_fallback_does_not_capture_api(tmp_path: Path) -> None:
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<html>ui-shell</html>", encoding="utf-8")

    with make_client(tmp_path, ui_dir=ui) as client:
        assert client.get("/queue").text == "<html>ui-shell</html>"
        assert client.get("/api/nope", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 404
