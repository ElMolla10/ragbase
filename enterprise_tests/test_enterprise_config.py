"""Enterprise identity/configuration never inherits the legacy demo credentials."""

from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from backend.auth.development import authenticate
from backend.enterprise_config import EnterpriseSettings


def test_ui_root_comes_only_from_enterprise_configuration(monkeypatch, tmp_path):
    missing = tmp_path / "absent-shell"
    monkeypatch.setenv("ENTERPRISE_UI_ROOT", str(missing))
    assert EnterpriseSettings.from_env().ui_root == missing
    monkeypatch.setenv("ENTERPRISE_UI_ROOT", "")
    assert EnterpriseSettings.from_env().ui_root is None


def test_legacy_database_url_is_not_inherited(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://sensitive-legacy")
    monkeypatch.delenv("ENTERPRISE_DATABASE_URL", raising=False)
    assert EnterpriseSettings.from_env().database_url is None


def test_secrets_are_not_in_settings_repr(tmp_path):
    config = EnterpriseSettings(
        database_url="postgresql+asyncpg://user:private@db/enterprise",
        storage_root=tmp_path,
        dev_tokens={"secret" * 8: uuid4()},
    )
    assert config.database_url not in repr(config)
    assert next(iter(config.dev_tokens)) not in repr(config)


@pytest.mark.parametrize("profile", ["server-local", "dev-remote"])
def test_dev_identity_never_enables_production_profile(profile):
    with pytest.raises(ValueError, match="development"):
        EnterpriseSettings(profile=profile)


@pytest.mark.parametrize(
    "authorization", [None, "Basic abc", "Bearer wrong", "Bearer " + "a" * 300]
)
def test_missing_or_forged_tokens_fail(authorization):
    with pytest.raises(HTTPException) as error:
        authenticate(authorization, {"real" * 10: uuid4()})
    assert error.value.status_code == 401


def test_subject_is_server_owned():
    actor = UUID("11111111-1111-4111-8111-111111111111")
    assert authenticate("Bearer " + "a" * 40, {"a" * 40: actor}).user_id == actor


@pytest.mark.parametrize("limit", [0, 5_000_001])
def test_upload_limit_cannot_exceed_parser_and_proxy_boundary(limit):
    with pytest.raises(ValueError, match="bounded parser"):
        EnterpriseSettings(max_upload_bytes=limit)


@pytest.mark.parametrize("tokens", ["[]", "null", '{"short":"not-a-uuid"}'])
def test_invalid_token_configuration_fails_without_exposing_secrets(
    monkeypatch, tokens
):
    monkeypatch.setenv("ENTERPRISE_DEV_TOKENS", tokens)
    with pytest.raises(ValueError, match="Invalid enterprise"):
        EnterpriseSettings.from_env()
