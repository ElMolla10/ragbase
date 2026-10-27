"""The browser shell must not become a path around authentication or storage."""

from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.enterprise import create_app
from backend.enterprise_config import EnterpriseSettings
from backend.library import router
from backend.ui import asset_file, install_browser, ui_root_is_safe


def _shell(tmp_path):
    root = tmp_path / "ui"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(
        "<!doctype html><title>enterprise shell</title>", encoding="utf-8"
    )
    (root / "assets" / "app.js").write_text("console.log('shell')", encoding="utf-8")
    (root / "favicon.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8"
    )
    leaked = root / "api"
    leaked.mkdir()
    (leaked / "workspaces").write_text('{"documents":"leaked"}', encoding="utf-8")
    return root


def test_static_mount_does_not_bypass_authentication(tmp_path):
    root = _shell(tmp_path)
    token = "t" * 32
    app = FastAPI()
    app.state.settings = EnterpriseSettings(dev_tokens={token: uuid4()})
    app.include_router(router)
    assert install_browser(app, root, storage_root=None)

    with TestClient(app, raise_server_exceptions=False) as client:
        shell = client.get("/")
        assert shell.status_code == 200
        assert "enterprise shell" in shell.text
        assert "default-src 'self'" in shell.headers["content-security-policy"]

        asset = client.get("/assets/app.js")
        assert asset.status_code == 200
        assert "shell" in asset.text

        denied = client.get("/api/workspaces")
        assert denied.status_code == 401
        assert b"leaked" not in denied.content
        assert "enterprise shell" not in denied.text

        planted = client.get(
            "/api/workspaces", headers={"Authorization": "Bearer " + token}
        )
        assert planted.status_code != 200
        assert b"leaked" not in planted.content

        escaped = client.get("/assets/..%2F..%2Fsecret.txt")
        assert escaped.status_code == 404
        assert b"leaked" not in escaped.content


def test_create_app_shell_does_not_publish_api_files(tmp_path):
    root = _shell(tmp_path)
    app = create_app(EnterpriseSettings(ui_root=root))
    with TestClient(app) as client:
        assert "enterprise shell" in client.get("/").text
        hidden = client.get("/api/workspaces")
        assert hidden.status_code == 404
        assert b"leaked" not in hidden.content
        assert client.get("/health/ready").status_code == 200


def test_shell_refuses_to_serve_the_artifact_store(tmp_path):
    store = tmp_path / "data"
    (store / "assets").mkdir(parents=True)
    (store / "index.html").write_text("ORIGINAL", encoding="utf-8")
    (store / "assets" / "secret.js").write_text("SECRET", encoding="utf-8")
    app = create_app(EnterpriseSettings(storage_root=store, ui_root=store))
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 503
        assert b"ORIGINAL" not in response.content
        asset = client.get("/assets/secret.js")
        assert asset.status_code == 404
        assert b"SECRET" not in asset.content


def test_parent_of_storage_is_not_a_browser_root(tmp_path):
    store = tmp_path / "data"
    store.mkdir()
    assert ui_root_is_safe(tmp_path, store) is False
    assert ui_root_is_safe(store, store) is False
    ui = tmp_path / "ui"
    ui.mkdir()
    assert ui_root_is_safe(ui, store) is True


def test_asset_symlink_cannot_escape(tmp_path):
    root = _shell(tmp_path)
    secret = tmp_path / "secret.js"
    secret.write_text("SECRET", encoding="utf-8")
    link = root / "assets" / "escaped.js"
    link.symlink_to(secret)
    app = FastAPI()
    assert install_browser(app, root, storage_root=None)
    with TestClient(app) as client:
        response = client.get("/assets/escaped.js")
        assert response.status_code == 404
        assert b"SECRET" not in response.content
    try:
        asset_file(root, "../secret.js")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 404
    else:
        raise AssertionError("traversal was accepted")


def test_index_symlink_disables_the_shell(tmp_path):
    root = tmp_path / "ui"
    root.mkdir()
    secret = tmp_path / "secret.html"
    secret.write_text("SECRET", encoding="utf-8")
    (root / "index.html").symlink_to(secret)
    app = FastAPI()
    assert install_browser(app, root, storage_root=None) is False
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 404
        assert b"SECRET" not in response.content
