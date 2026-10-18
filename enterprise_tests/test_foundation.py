"""Foundation startup without any model/database imports or live services."""

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from backend.enterprise import app


def test_no_heavy_imports_on_startup():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import backend.enterprise; import sys; assert not {'torch', 'sentence_transformers', 'groq', 'sqlalchemy'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_live_and_ready_are_honest_about_development_state(monkeypatch):
    monkeypatch.setenv("RAGBASE_PROFILE", "dev-lite")
    with TestClient(app) as client:
        assert client.get("/health/live").json()["development_only"]
        status = client.get("/health/ready").json()
        assert status["generation"] == "disabled"
        assert status["documents"] == "not_implemented"
        assert client.post("/query", json={"query": "secret"}).status_code == 404


def test_production_profile_cannot_launch_mock_foundation(monkeypatch):
    monkeypatch.setenv("RAGBASE_PROFILE", "server-local")
    with pytest.raises(RuntimeError, match="Only the development"), TestClient(app):
        pass
